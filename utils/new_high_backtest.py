"""신고가 돌파 전략 백테스트.

이벤트 기반이라 월간 리밸런싱과 리듬이 다르다 — 돌파한 날 사고, 손절선이나 이탈
이동평균에 걸린 날 판다. 판정은 종가, 체결은 다음 거래일 **시가**다. 종가 체결은
쓰지 않는다: 마감 동시호가에 대량을 던지면 체결가를 모른 채 거래하게 된다.

자산은 **현금과 주수로** 들고 간다. 진입할 때 그 시점 자산의 1/slots 를 배정하고,
보유 중에는 비중을 건드리지 않는다(오른 종목은 커진 채로 간다). 살 현금이 모자라면
있는 만큼만 산다 — 팔지 않은 평가익으로는 새 종목을 살 수 없기 때문이다.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from config import CACHE_TTL_COMPUTE
from utils.new_high_service import (
    DEFAULT_BACKTEST_MONTHS,
    HIGH_WINDOW,
    HIGH_WINDOW_WEEKS,
    benchmark_info,
    build_price_panel,
    compute_signals,
    load_price_frames,
    load_settings,
    load_universe,
    validate_settings,
)
from utils.pool_settings_store import get_pool_slippage
from utils.share_allocation import ShareTarget, allocate_integer_shares, backtest_initial_capital
from utils.stock_memo_store import attach_stock_memos
from utils.trade_stats import summarize_trades
from utils.ttl_cache import TtlCache

logger = logging.getLogger(__name__)

# 백테스트 기간 상한 — 신고가 창(52주)만큼 앞선 데이터가 있어야 판정이 된다.
MAX_BACKTEST_MONTHS = 60


def _drawdown_pct(series: pd.Series) -> float:
    return float(((series / series.cummax()) - 1).min() * 100)


def _cagr_pct(total_pct: float, months: int) -> float | None:
    if months <= 0:
        return None
    return ((1 + total_pct / 100) ** (12 / months) - 1) * 100


def _sortino(returns: pd.Series) -> float | None:
    """하방 변동성 대비 수익. 하락일이 없으면 나눌 수 없어 None."""
    downside = returns[returns < 0]
    if downside.empty or len(returns) < 2:
        return None
    dd = float((downside**2).mean() ** 0.5)
    if dd <= 0:
        return None
    return float(returns.mean() / dd * (252**0.5))


def load_context(settings: dict[str, Any]) -> dict[str, Any]:
    """가격 패널·신호를 한 번만 만들어 재사용한다.

    패널 생성이 이 계산에서 가장 비싼 부분(종목 수백 개의 캐시 역직렬화)이라,
    같은 요청에서 백테스트와 보유 재구성이 각자 만들면 시간이 두 배가 된다.
    """
    pool = settings["pool"]
    universe = load_universe(pool)
    panel = build_price_panel(universe, load_price_frames(universe))
    return {
        "pool": pool,
        "universe": universe,
        "name_by": {row["ticker"]: row["name"] for row in universe},
        "industry_by": {row["ticker"]: row.get("industry", "") for row in universe},
        "panel": panel,
        "signals": compute_signals(panel, int(settings["exit_ma_days"])),
    }


def run_backtest(
    months: int | None = None,
    settings: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """돌파 전략 백테스트. 일별 자산곡선과 체결 내역을 함께 돌려준다.

    """
    settings = validate_settings(settings or load_settings())
    months = int(months or DEFAULT_BACKTEST_MONTHS)
    if not 1 <= months <= MAX_BACKTEST_MONTHS:
        raise ValueError(f"'months' 는 1~{MAX_BACKTEST_MONTHS} 사이여야 합니다.")

    pool = settings["pool"]
    slots = int(settings["top_n"])
    stop_pct = float(settings["stop_loss_pct"])
    # 슬리피지는 종목풀 설정을 단일 소스로 쓴다 — 매수·매도 편도값을 각각 적용한다.
    buy_slippage, sell_slippage = get_pool_slippage(pool)

    context = context or load_context(settings)
    name_by, industry_by = context["name_by"], context["industry_by"]

    # ADR 진입 게이트 — 판정일 ADR 이 하한 미만이면 그날은 **신규 진입만** 건너뛴다.
    # 보유 청산(손절·이탈)은 그대로 돈다. 이력 이전 날짜는 게이트 미적용.
    from utils.momentum_service import adr_market_of_pool, load_adr_series

    adr_floor = settings.get("adr_floor")
    adr_series = pd.Series(dtype=float)
    if adr_floor is not None:
        adr_market = adr_market_of_pool(pool)
        adr_series = load_adr_series(adr_market) if adr_market else pd.Series(dtype=float)

    def entry_blocked(stamp: pd.Timestamp) -> bool:
        if adr_floor is None or adr_series.empty:
            return False
        value = adr_series.asof(pd.Timestamp(stamp))
        return pd.notna(value) and float(value) < float(adr_floor)
    panel, signals = context["panel"], context["signals"]

    close_df, open_df = panel["close"], panel["open"]
    breakout, below_ma = signals["breakout"], signals["below_ma"]

    # 진입 우선순위 — 거래대금 급증 배수가 큰 쪽부터 (자리가 모자랄 때의 정렬 기준).
    min_mult = settings["min_value_mult"]
    value_mult = signals["value_mult"]

    def priority_of(day: pd.Timestamp, ticker: str) -> float:
        score = value_mult.at[day, ticker]
        return float(score) if pd.notna(score) else 0.0

    dates = close_df.index
    span = [d for d in dates if d >= dates[-1] - pd.DateOffset(months=months)]
    if len(span) < 2:
        raise RuntimeError("백테스트할 구간의 가격 데이터가 부족합니다.")

    # 자산은 현금 + 보유 주수로 들고 간다. 포지션 손익을 자산에 곱하면 동시에 들고 있던
    # 종목의 손익이 합산이 아니라 곱으로 쌓여 수익이 부풀려진다.
    #
    # 시작 자본은 통화별 상수(config.BACKTEST_INITIAL_CAPITAL)다. 예전에는 1.0 상대곡선이라
    # 주수가 소수(4.503주)로 나왔는데, 실제로는 정수 주수만 살 수 있어 운용 현황과 결과가
    # 어긋났다. 곡선은 마지막에 시작 자본으로 나눠 예전과 같은 배수로 돌려준다.
    initial_capital = backtest_initial_capital(pool)
    cash = float(initial_capital)
    holdings: dict[str, dict[str, Any]] = {}
    trades: list[dict[str, Any]] = []
    curve: list[float] = []
    last_day = span[-1]

    def _value_at(day: pd.Timestamp) -> float:
        """그날 종가로 평가한 총자산 — 현금 + 보유 평가액."""
        total = cash
        for ticker, position in holdings.items():
            price = close_df.at[day, ticker]
            if pd.notna(price):
                total += position["shares"] * float(price)
        return total

    for index, day in enumerate(span[:-1]):
        nxt = span[index + 1]

        # 1) 청산 판정 (오늘 종가) → 내일 시가 체결
        for ticker in list(holdings):
            position = holdings[ticker]
            price = close_df.at[day, ticker]
            if pd.isna(price):
                continue
            hit_stop = (price / position["entry"] - 1) * 100 <= stop_pct
            hit_ma = bool(below_ma.at[day, ticker])
            if not (hit_stop or hit_ma):
                continue
            exit_price = open_df.at[nxt, ticker]
            if pd.isna(exit_price):
                exit_price = price  # 다음 날 시가가 없으면(거래정지) 오늘 종가로 본다
            ret = (float(exit_price) * (1 - sell_slippage / 100)) / position["entry"] - 1
            cash += position["shares"] * float(exit_price) * (1 - sell_slippage / 100)
            trades.append(
                {
                    "ticker": ticker,
                    "name": name_by.get(ticker, ticker),
                    "industry": industry_by.get(ticker, ""),
                    "entry_date": str(position["date"].date()),
                    # 표시는 **실제 체결 시가**다. 슬리피지는 가격이 아니라 비용이라
                    # 손익률에만 반영한다 — 가격에 섞으면 시장에 없는 호가가 찍힌다.
                    "entry_price": round(position["open"], 2),
                    "exit_date": str(nxt.date()),
                    "exit_price": round(float(exit_price), 2),
                    "return_pct": round(ret * 100, 2),
                    "days": len(close_df.loc[position["date"] : day]) - 1,
                    "reason": "손절" if hit_stop else "이탈",
                }
            )
            del holdings[ticker]

        # 2) 진입 — 빈 자리만큼, 거래대금 급증이 큰 순 (ADR 게이트에 걸린 날은 건너뜀)
        free = slots - len(holdings)
        if free > 0 and not entry_blocked(day):
            # 배정 기준은 **체결 시점(다음 거래일 시가)의 자산**이다. 청산 대금이 이미
            # 현금에 들어와 있으므로 파는 쪽과 사는 쪽이 같은 시점으로 맞는다.
            fill_value = cash
            for held_ticker, held_position in holdings.items():
                held_price = open_df.at[nxt, held_ticker]
                if pd.isna(held_price):
                    held_price = close_df.at[day, held_ticker]
                if pd.notna(held_price):
                    fill_value += held_position["shares"] * float(held_price)
            row = breakout.loc[day]
            picks = [
                t
                for t in row[row].index
                if t not in holdings
                and not pd.isna(open_df.at[nxt, t])
                and _meets_min_mult(value_mult.at[day, t], min_mult)
            ]
            picks.sort(key=lambda t: priority_of(day, t), reverse=True)
            picks = picks[:free]
            # 주수 배분은 운용 현황과 **같은 함수**를 쓴다 — 규칙이 갈라지면 백테스트가
            # 실제로 못 내는 성과를 내게 된다. 예산은 살 수 있는 현금까지만(팔지 않은
            # 평가익으로는 못 산다). 손익 계산에는 슬리피지를 얹은 값을 쓴다.
            fill_price_by_ticker = {t: float(open_df.at[nxt, t]) * (1 + buy_slippage / 100) for t in picks}
            slot_amount = fill_value / slots if slots else 0.0
            quantities = allocate_integer_shares(
                [
                    ShareTarget(key=ticker, target_amount=slot_amount, price=price)
                    for ticker, price in fill_price_by_ticker.items()
                    if price > 0
                ],
                budget=min(slot_amount * len(picks), cash),
            )
            for ticker in picks:
                shares = quantities.get(ticker, 0)
                if shares <= 0:
                    continue
                fill_price = fill_price_by_ticker[ticker]
                holdings[ticker] = {
                    "open": float(open_df.at[nxt, ticker]),
                    "entry": fill_price,
                    "date": nxt,
                    "shares": shares,
                }
                cash -= shares * fill_price

        curve.append(_value_at(day))

    # 마지막 날은 판정·체결이 없다(체결할 다음 거래일이 없어서). 다만 그날 종가로
    # **평가**는 해야 곡선이 하루 짧아지지 않는다 — 모멘텀 엔진과 같은 기준.
    curve.append(_value_at(last_day))

    # 아직 청산하지 않은 종목 — 성과에는 평가손익으로 이미 반영돼 있지만 체결 내역에는 없다.
    # 슬리브 안에서의 현재 비중도 함께 담는다. 진입할 때 1/slots 였다가 시세대로 흘러간
    # 값이라, 합성 화면이 '지금 이 종목이 몇 % 여야 하는지' 를 이 값으로 잡는다.
    sleeve_value = _value_at(last_day)
    open_positions = []
    for ticker, position in holdings.items():
        price = close_df.at[last_day, ticker]
        if pd.isna(price):
            continue
        value = position["shares"] * float(price)
        open_positions.append(
            {
                "ticker": ticker,
                "name": name_by.get(ticker, ticker),
                "industry": industry_by.get(ticker, ""),
                "entry_date": str(position["date"].date()),
                "entry_price": round(position["open"], 2),
                "price": float(price),
                # 표시용 평가손익 — 아직 안 팔았으니 매도 슬리피지는 빼지 않는다.
                "return_pct": round((float(price) / position["open"] - 1) * 100, 2),
                "days": len(close_df.loc[position["date"] : last_day]) - 1,
                # 오늘 편입된 종목은 목록에서 따로 표시한다.
                "is_new": position["date"] == last_day,
                # 이 슬리브 안에서의 비중(%) — 슬리브 전체를 100 으로 본다.
                "sleeve_weight_pct": round(value / sleeve_value * 100, 4) if sleeve_value > 0 else 0.0,
            }
        )
    open_positions.sort(key=lambda row: row["entry_date"], reverse=True)
    # 빈 슬롯·잔여 현금 비중 — 종목 비중과 합쳐 100 이 된다.
    sleeve_cash_weight_pct = round(cash / sleeve_value * 100, 4) if sleeve_value > 0 else 100.0
    exited_today = [t for t in trades if t["exit_date"] == str(last_day.date())]

    # 곡선은 시작 1.0 배수로 되돌린다 — 시작 자본은 정수 주수를 세기 위한 것이고,
    # 성과 지표(수익률·MDD·벤치마크 대비)는 예전과 같은 배수 기준으로 읽어야 한다.
    strategy = pd.Series(curve, index=span) / initial_capital  # 마지막 날은 평가만 한 값이 들어간다
    # 벤치마크는 **시작일 시가**를 1 로 둔다 — 전략도 그날 시가에 사기 때문이다(공용 함수).
    from utils.benchmark_curve import benchmark_growth

    benchmark = benchmark_growth(pool, strategy.index)

    strategy_total = float((strategy.iloc[-1] - 1) * 100)
    benchmark_total = float((benchmark.iloc[-1] - 1) * 100)

    return {
        "start_date": str(strategy.index[0].date()),
        "end_date": str(strategy.index[-1].date()),
        "months": months,
        "strategy_total_pct": round(strategy_total, 2),
        "strategy_cagr_pct": round(_cagr_pct(strategy_total, months) or 0.0, 2),
        "strategy_mdd_pct": round(_drawdown_pct(strategy), 2),
        "strategy_sortino": _sortino(strategy.pct_change().dropna()),
        "benchmark_total_pct": round(benchmark_total, 2),
        "benchmark_cagr_pct": round(_cagr_pct(benchmark_total, months) or 0.0, 2),
        "benchmark_mdd_pct": round(_drawdown_pct(benchmark), 2),
        "benchmark_sortino": _sortino(benchmark.pct_change().dropna()),
        "benchmark_name": benchmark_info(pool)["name"],
        **summarize_trades(trades),
        "trades": sorted(trades, key=lambda t: t["exit_date"], reverse=True),
        "as_of": str(last_day.date()),
        "open_positions": open_positions,
        "sleeve_cash_weight_pct": sleeve_cash_weight_pct,
        "exited_today": exited_today,
        "daily": [
            {
                "date": str(d.date()),
                "strategy_pct": round((v - 1) * 100, 2),
                "benchmark_pct": round((float(benchmark.loc[d]) - 1) * 100, 2),
            }
            for d, v in strategy.items()
        ],
    }


def _meets_min_mult(mult: Any, minimum: float | None) -> bool:
    """진입 자격 — 거래대금 급증 배수가 하한 이상인가.

    하한이 없으면 항상 통과. 배수를 모르면(상장 직후 등) 미달로 본다 — 추정하지 않는다.
    """
    if minimum is None:
        return True
    return bool(pd.notna(mult)) and float(mult) >= minimum


def _market_caps(pool: str) -> dict[str, float]:
    """티커 → 시가총액. 배치 B 가 메타 캐시에 적어 둔 값을 읽기만 한다.

    한국 개별주는 예전에 여기서 네이버 시세표를 직접 순회했다(424종목에 4초). 그런데 그
    목록은 시총 **순위**를 매기려고 배치가 이미 받아 오는 값이라, 배치가 금액까지 적게
    하고(`utils/market_cap_rank`) 화면은 DB 만 읽는다. 국가별 분기도 함께 사라졌다.

    값이 없는 종목은 맵에서 빠진다 — 화면은 '-' 로 둔다(임의 보정 없음).
    현재 값만 있고 과거 이력이 없다. 그래서 백테스트 우선순위에는 쓰지 않는다.
    """
    from utils.db_manager import get_db_connection

    db = get_db_connection()
    if db is None:
        return {}
    caps: dict[str, float] = {}
    for doc in db["stock_cache_meta"].find({"ticker_type": pool}, {"ticker": 1, "meta_cache": 1}):
        value = (doc.get("meta_cache") or {}).get("total_net_assets")
        if value:
            caps[str(doc.get("ticker") or "").strip().upper()] = float(value)
    return caps


# 보유를 재구성할 때 돌리는 구간. 관측된 최장 보유가 100거래일 안쪽이라 1년이면 충분하다.
_HOLDINGS_LOOKBACK_MONTHS = 12


def _live_quotes(pool: str, tickers: list[str], cached_last: pd.Timestamp) -> dict[str, Any]:
    """진행 중인 세션의 실시간 시세. 캐시에 아직 안 들어온 날일 때만 의미가 있다.

    반환 ``{"live": bool, "pre_market": bool, "traded_at": str|None,
    "by_ticker": {티커: {price, high, change_pct}}}``.
    ``live`` 는 마지막 체결일이 가격 캐시의 마지막 거래일보다 **뒤**라는 뜻 —
    그날 종가가 아직 확정되지 않았으므로 화면은 '돌파중'처럼 잠정 상태로 표시한다.
    캐시와 같은 날이면 이미 확정된 세션이라 실시간을 쓰지 않는다.

    장전(동시호가) 구간은 ``live`` 로 보지 않는다. 그 시각 스냅샷의 고가·저가·시가는
    아직 **직전 세션의 값**이고 현재가만 오늘 예상체결가라, 둘을 섞으면 어제 확정된
    돌파가 오늘 예상가에 밀려 '터치 후 밀림'으로 뒤집힌다. 오늘 값이 다 갖춰지는
    정규장부터 쓴다.
    """
    from utils.settings_loader import get_ticker_type_settings

    country = str((get_ticker_type_settings(pool) or {}).get("country_code") or "").strip().lower()
    if not country or not tickers:
        return {"live": False, "pre_market": False, "traded_at": None, "by_ticker": {}}

    from services.price_service import get_realtime_snapshot

    try:
        snapshot = get_realtime_snapshot(country, tickers)
    except Exception:
        logger.exception("[new_high] 실시간 시세 조회 실패 (%s)", pool)
        return {"live": False, "pre_market": False, "traded_at": None, "by_ticker": {}}

    by_ticker: dict[str, dict[str, float]] = {}
    traded_at: str | None = None
    pre_market = False
    for ticker, quote in snapshot.items():
        price = quote.get("nowVal")
        if price is None or float(price) <= 0:
            continue
        # 오늘 시가 — 어제 확정된 진입·청산이 체결된 가격이다. ETF 는 이 값이 안 와서
        # None 이 되고, 그런 종목은 체결로 처리하지 않는다(가격을 지어내지 않는다).
        open_val = quote.get("open")
        by_ticker[ticker] = {
            "price": float(price),
            "high": float(quote.get("high") or price),
            "open": float(open_val) if open_val is not None and float(open_val) > 0 else None,
            "change_pct": float(quote.get("changeRate")) if quote.get("changeRate") is not None else None,
        }
        if quote.get("is_pre_market"):
            pre_market = True
        stamp = str(quote.get("localTradedAt") or "")
        if stamp and (traded_at is None or stamp > traded_at):
            traded_at = stamp

    live = bool(traded_at) and not pre_market and str(traded_at)[:10] > str(cached_last.date())
    return {
        "live": live,
        "pre_market": pre_market,
        "traded_at": traded_at,
        # 시세는 항상 담는다. 현재가·등락률은 어느 구간이든 오늘 값이라 표시에 쓰고,
        # 돌파 판정은 `live` 일 때만 한다 — ETF 처럼 체결 시각·고가를 안 주는 종목도
        # 일간(%) 은 정상으로 보여야 한다.
        "by_ticker": by_ticker,
    }


# 장전에 화면을 주기적으로 다시 받기 시작할 시점 — 개장 몇 분 전부터인가.
# 실제로 예상체결가가 움직이는 구간은 동시호가(개장 30분 전~개장)라 한 시간이면 넉넉하다.
# 시세 제공처의 '장전' 플래그는 새벽부터 켜져 있을 수 있어 그것만 믿고 돌리지 않는다.
_PRE_MARKET_REFRESH_LEAD_MINUTES = 60


def _should_auto_refresh(pool: str, quotes: dict[str, Any]) -> bool:
    """화면이 주기 갱신을 걸어야 하는 시점인지.

    장중이면 늘 참이고, 장전이면 개장이 가까울 때만 참이다. 개장 시각은 시장마다 달라
    화면이 알 수 없으므로 여기서 판단해 내려준다.
    """
    if quotes["live"]:
        return True
    if not quotes["pre_market"]:
        return False

    from config import MARKET_SCHEDULES
    from utils.settings_loader import get_ticker_type_settings

    country = str((get_ticker_type_settings(pool) or {}).get("country_code") or "").strip().lower()
    schedule = (MARKET_SCHEDULES or {}).get(country)
    if not isinstance(schedule, dict):
        return False
    tz_name = str(schedule.get("timezone") or "").strip()
    open_time = schedule.get("open")
    if not tz_name or open_time is None:
        return False
    try:
        now_local = pd.Timestamp.now(tz=tz_name)
        opens_at = pd.Timestamp(f"{now_local.date()} {open_time.hour:02d}:{open_time.minute:02d}", tz=tz_name)
    except Exception:
        return False
    return opens_at - pd.Timedelta(minutes=_PRE_MARKET_REFRESH_LEAD_MINUTES) <= now_local <= opens_at


def _next_session(pool: str, last: pd.Timestamp) -> str | None:
    """캐시 마지막 거래일 **다음**의 거래일 — 진입·청산이 체결되는 날.

    화면이 '오늘 매수'인지 '내일 매수'인지 가리는 데 쓴다. 장 시작 전에는 캐시의
    마지막 거래일이 아직 어제라, '다음 거래일' 이 곧 오늘이다.
    캘린더가 답할 수 없으면 None 을 돌려준다 — 날짜를 지어내지 않는다.
    """
    from utils.settings_loader import get_ticker_type_settings
    from utils.trading_calendar import get_trading_days

    country = str((get_ticker_type_settings(pool) or {}).get("country_code") or "").strip().lower()
    if not country:
        return None
    try:
        days = get_trading_days(
            str((last + pd.Timedelta(days=1)).date()),
            str((last + pd.Timedelta(days=14)).date()),
            country,
        )
    except Exception:
        logger.exception("[new_high] 다음 거래일 조회 실패 (%s)", pool)
        return None
    return str(days[0].date()) if days else None


def _apply_display_quotes(
    rows: list[dict[str, Any]],
    holdings: list[dict[str, Any]],
    by_ticker: dict[str, dict[str, Any]],
) -> None:
    """현재가·일간(%)·보유 수익률만 실시간으로 바꾼다. **판정에는 쓰지 않는다.**

    돌파 거리·터치·진입 예정은 확정 종가로 정해지고, 이 함수는 사람이 보는 숫자만 바꾼다.
    그래서 체결 시각이나 고가를 안 주는 종목(국내 ETF)도 일간(%) 은 정상으로 나온다.
    """
    for row in rows:
        quote = by_ticker.get(row["ticker"])
        if not quote:
            continue
        row["price"] = quote["price"]
        if quote["change_pct"] is not None:
            row["change_pct"] = round(quote["change_pct"], 2)
    for held in holdings:
        quote = by_ticker.get(held["ticker"])
        if not quote:
            continue
        held["price"] = quote["price"]
        held["return_pct"] = round((quote["price"] / held["entry_price"] - 1) * 100, 2)


def _apply_live_trade_values(
    rows: list[dict[str, Any]],
    pool: str,
    value_df: pd.DataFrame,
    last: pd.Timestamp,
    min_value_mult: float | None,
    *,
    confirmed_today: bool,
) -> None:
    """진행 중인 세션의 누적 거래대금으로 `trade_value`·`value_mult`·`qualifies` 를 다시 쓴다.

    분모는 확정된 직전 19거래일 합에 오늘을 더한 20일 평균이다 — 백테스트가 쓰는 식과 같다.
    백테스트는 확정된 과거만 보므로 여기서 바꾼 값이 성과 계산에 섞이지 않는다.
    국내 상장이 아니거나 조회에 실패하면 아무것도 바꾸지 않는다(캐시 값 유지).

    실시간 배수는 `value_mult_live` 에 **항상** 따로 남긴다 — 토스는 대체거래소(NXT)
    거래분까지 합산해 주므로 KRX 확정값보다 크고, 화면이 둘을 나란히 보여준다.
    """
    from utils.settings_loader import get_ticker_type_settings

    settings = get_ticker_type_settings(pool) or {}
    if str(settings.get("country_code") or "").strip().lower() != "kor":
        return
    tickers = [row["ticker"] for row in rows]
    try:
        from utils.data_loader import fetch_toss_kr_stock_snapshot

        snapshot = fetch_toss_kr_stock_snapshot(tickers)
    except Exception:
        logger.exception("[new_high] 실시간 거래대금 조회 실패 (%s)", pool)
        return
    if not snapshot:
        return

    # 오늘을 뺀 직전 19거래일 — 확정된 값만 쓴다.
    recent = value_df.loc[:last].tail(19)
    for row in rows:
        today = (snapshot.get(row["ticker"]) or {}).get("tradeValue")
        if today is None or float(today) <= 0 or row["ticker"] not in recent.columns:
            continue
        history = recent[row["ticker"]].dropna()
        if len(history) < 19:
            continue
        base = (float(history.sum()) + float(today)) / 20
        if base <= 0:
            continue
        live_mult = round(float(today) / base, 2)
        row["value_mult_live"] = live_mult
        if not confirmed_today:
            # 오늘 확정값이 아직 없다 — 실시간이 유일한 오늘 값이라 판정에도 쓴다.
            row["trade_value"] = float(today)
            row["value_mult"] = live_mult
            row["qualifies"] = _meets_min_mult(live_mult, min_value_mult)


def _market_today(pool: str) -> str:
    """그 시장의 오늘 날짜(YYYY-MM-DD). 캐시가 오늘까지 확정됐는지 판단하는 기준이다."""
    from config import MARKET_SCHEDULES
    from utils.settings_loader import get_ticker_type_settings

    country = str((get_ticker_type_settings(pool) or {}).get("country_code") or "").strip().lower()
    tz_name = str((MARKET_SCHEDULES.get(country) or {}).get("timezone") or "Asia/Seoul")
    return str(pd.Timestamp.now(tz=tz_name).date())


def _cache_refreshed_at(pool: str) -> str | None:
    """이 종목풀 가격 캐시의 마지막 갱신 시각(ISO). 배치가 안 돌았으면 None."""
    from utils.cache_utils import get_cache_refresh_completed_at

    completed = get_cache_refresh_completed_at(pool)
    return completed.isoformat() if completed else None


# 유니버스 전체를 현재까지 돌리는 계산이라 수십 초 걸린다 — 설정이 같으면
# 결과도 같으므로 짧게 재사용한다(설정을 바꾸면 키가 달라져 새로 계산한다).
_POSITIONS_CACHE = TtlCache(CACHE_TTL_COMPUTE, name="new_high_positions")


def current_positions(settings: dict[str, Any] | None = None) -> dict[str, Any]:
    """지금 들고 있어야 할 종목(보유·오늘 이탈)과 오늘 신호(돌파·후보).

    보유는 **백테스트와 같은 엔진**을 현재까지 돌린 마지막 상태다. 화면과 백테스트가
    다른 코드로 갈라지면 표시된 보유와 성과가 어긋나므로 계산을 나누지 않는다.
    """
    settings = validate_settings(settings or load_settings())
    cache_key = _POSITIONS_CACHE.make_key(settings)
    result = _POSITIONS_CACHE.get_or_compute(cache_key, lambda: _current_positions(settings))
    # 종목 메모는 **캐시 밖**에서 붙인다 — 다른 화면에서 고친 값이 즉시 보여야 한다.
    attach_stock_memos(
        result["breakouts"], result["candidates"], result["holdings"], result["planned_entries"], result["exited_today"]
    )
    return result


def _current_positions(settings: dict[str, Any]) -> dict[str, Any]:
    pool = settings["pool"]
    context = load_context(settings)
    universe = context["universe"]
    name_by, industry_by = context["name_by"], context["industry_by"]
    panel, signals = context["panel"], context["signals"]

    close_df = panel["close"]
    last = close_df.index[-1]
    prior_high = signals["prior_high"].loc[last]
    prior_high_intraday = signals["prior_high_intraday"].loc[last]
    today_high = panel["high"].loc[last]
    value_mult = signals["value_mult"].loc[last]
    trade_value = panel["value"].loc[last]
    market_cap_by = _market_caps(pool)
    # 시총 순위 — 배치 B 가 메타 캐시에 적어 둔 시장 전체 순위(개별주 풀만 값 있음).
    from utils.market_cap_rank import market_cap_rank_of
    from utils.stock_cache_meta_io import get_stock_cache_meta_docs

    try:
        meta_docs = get_stock_cache_meta_docs(pool, [row["ticker"] for row in universe])
    except Exception:
        meta_docs = {}
    rank_by_ticker = {t: market_cap_rank_of((doc or {}).get("meta_cache")) for t, doc in meta_docs.items()}
    # 일간 등락률 — 다른 화면(순위·시장추세)과 같은 기준으로 직전 거래일 종가 대비.
    prev_close = close_df.loc[close_df.index[-2]] if len(close_df.index) >= 2 else None

    rows = []
    for ticker in close_df.columns:
        price = close_df.at[last, ticker]
        high = prior_high.get(ticker)
        if pd.isna(price) or pd.isna(high) or high <= 0:
            continue
        gap_pct = (float(price) / float(high) - 1) * 100
        intraday = prior_high_intraday.get(ticker)
        has_intraday = pd.notna(intraday) and float(intraday) > 0
        # 장중에 최고 종가선을 건드렸는데 종가는 그 아래로 밀린 상태.
        # 화면이 '터치 후 밀림'으로 따로 표시한다 — 그냥 '임박'으로 묻으면 흐름이 안 보인다.
        day_high = today_high.get(ticker)
        touched = bool(pd.notna(day_high) and float(day_high) >= float(high) and float(price) < float(high))
        before = None if prev_close is None else prev_close.get(ticker)
        change_pct = (
            round((float(price) / float(before) - 1) * 100, 2)
            if before is not None and pd.notna(before) and float(before) > 0
            else None
        )
        rows.append(
            {
                "ticker": ticker,
                "name": name_by.get(ticker, ticker),
                "industry": industry_by.get(ticker, ""),
                "change_pct": change_pct,
                "market_cap": market_cap_by.get(ticker),
                "market_cap_rank": rank_by_ticker.get(ticker),
                "trade_value": float(trade_value.get(ticker)) if pd.notna(trade_value.get(ticker)) else None,
                "price": float(price),
                "prior_high": float(high),
                # 관례상의 52주 신고가(장중 고가) — 참고 표시용. 판정에는 쓰지 않는다.
                "prior_high_intraday": float(intraday) if has_intraday else None,
                "gap_high_pct": round((float(price) / float(intraday) - 1) * 100, 2) if has_intraday else None,
                # 0 이상이면 돌파, 음수면 최고 종가까지 남은 거리.
                "gap_pct": round(gap_pct, 2),
                "touched": touched,
                "value_mult": round(float(value_mult.get(ticker)), 2) if pd.notna(value_mult.get(ticker)) else None,
                # 돌파했더라도 이 값이 거짓이면 사지 않는다 (백테스트와 같은 판정).
                "qualifies": _meets_min_mult(value_mult.get(ticker), settings["min_value_mult"]),
            }
        )

    # 보유·이탈은 백테스트 엔진의 마지막 상태를 그대로 쓴다.
    simulated = run_backtest(_HOLDINGS_LOOKBACK_MONTHS, settings, context)
    holdings = simulated["open_positions"]

    quotes = _live_quotes(pool, [r["ticker"] for r in rows] + [h["ticker"] for h in holdings], last)
    # 마지막 거래일 종가로 '다음 시가에 할 일' 을 판정한다. 백테스트 루프는 마지막 날을
    # 판정하지 않는다(체결할 다음 날이 없어서). 그래서 여기서 한 번 더 본다 — 이게 없으면
    # 화면에 살 종목만 보이고 팔 종목이 안 보인다.
    stop_pct = float(settings["stop_loss_pct"])
    exit_ma_days = int(settings["exit_ma_days"])
    below_ma_last = signals["below_ma"].loc[last]

    def mark_exits(price_of) -> None:
        """청산 여부를 표시한다. price_of 가 None 을 돌려주면 판정하지 않는다."""
        for held in holdings:
            price = price_of(held["ticker"])
            if price is None:
                continue
            hit_stop = (price / held["entry_price"] - 1) * 100 <= stop_pct
            hit_ma = bool(below_ma_last.get(held["ticker"]))
            held["status"] = "sell" if (hit_stop or hit_ma) else "hold"
            held["exit_reason"] = "손절" if hit_stop else ("이탈" if hit_ma else None)

    # ADR 진입 게이트 — 발동하면 오늘은 신규 진입이 없다(보유 관리는 그대로).
    adr_gate: dict[str, Any] | None = None
    if settings.get("adr_floor") is not None:
        from utils.momentum_service import adr_gate_blocked, adr_market_of_pool, load_adr_series

        gate_market = adr_market_of_pool(pool)
        gate_series = load_adr_series(gate_market) if gate_market else pd.Series(dtype=float)
        gate_value = gate_series.asof(last) if not gate_series.empty else None
        adr_gate = {
            "market": gate_market,
            "floor": settings["adr_floor"],
            "value": round(float(gate_value), 1) if gate_value is not None and pd.notna(gate_value) else None,
            "blocked": adr_gate_blocked(settings, last),
        }

    def pick_entries() -> list[dict[str, Any]]:
        """자리·자격·우선순위를 적용해 다음 시가에 살 종목을 고른다."""
        if adr_gate is not None and adr_gate["blocked"]:
            return []
        planned_exits = sum(1 for h in holdings if h.get("status") == "sell")
        free = int(settings["top_n"]) - (len(holdings) - planned_exits)
        if free <= 0:
            return []
        ready = [
            row
            for row in rows
            if row["gap_pct"] >= 0 and row["qualifies"] and row["ticker"] not in {h["ticker"] for h in holdings}
        ]
        ready.sort(key=lambda row: row["value_mult"] or 0.0, reverse=True)
        return ready[:free]

    def confirmed_close(ticker: str) -> float | None:
        price = close_df.at[last, ticker]
        return None if pd.isna(price) else float(price)

    # 거래대금 배수 — 실시간(토스, KRX+NXT 합산) 값을 **항상** 함께 담는다. 화면이 확정값
    # 옆에 괄호로 보여준다. 오늘 확정값이 아직 없으면(장중) 실시간이 판정 기준이 된다 —
    # 오늘 돌파한 종목을 어제 자금 유입으로 판정할 수는 없기 때문이다.
    _apply_live_trade_values(
        rows,
        pool,
        panel["value"],
        last,
        settings["min_value_mult"],
        confirmed_today=str(last.date()) >= str(_market_today(pool)),
    )

    mark_exits(confirmed_close)
    # 확정 종가 기준 판정 — 장중이면 아래에서 잠정 종가로 다시 판정하며 예상으로 바꾼다.
    for held in holdings:
        held["is_exit_forecast"] = False
    entries = pick_entries()

    if quotes["live"]:
        # 장이 열려 있다는 것은 위에서 고른 진입·청산이 **오늘 시가에 이미 체결됐다**는 뜻이다.
        # 그 결과를 반영하지 않으면 마감 후 배치가 돌 때까지(한국은 16시 이후) 하루 종일
        # '진입 예정' 으로 남아 실제 계좌와 어긋난다.
        session = str(quotes["traded_at"])[:10]
        # ① 청산 — 오늘 시가에 나갔다. 보유일은 청산 신호가 난 어제까지로 세므로 그대로 쓴다.
        #    시가를 모르는 종목(ETF 등)은 체결로 처리하지 않고 그대로 둔다.
        stayed = []
        for held in holdings:
            open_px = (quotes["by_ticker"].get(held["ticker"]) or {}).get("open")
            if held.get("status") != "sell" or not open_px:
                stayed.append(held)
                continue
            simulated["exited_today"].append(
                {
                    "ticker": held["ticker"],
                    "name": held["name"],
                    "industry": held["industry"],
                    "entry_date": held["entry_date"],
                    "entry_price": held["entry_price"],
                    "exit_date": session,
                    "exit_price": float(open_px),
                    "return_pct": round((float(open_px) / held["entry_price"] - 1) * 100, 2),
                    "days": held["days"],
                    "reason": held.get("exit_reason") or "이탈",
                }
            )
        holdings[:] = stayed

        # ② 남은 보유의 보유일을 하루 늘린다. 백테스트가 매긴 값은 마지막 확정 거래일
        #    기준이고, 오늘은 그 다음 거래일이다. 이걸 안 하면 어제 산 종목이 계속 '진입' 으로 보인다.
        for held in holdings:
            held["days"] = int(held["days"]) + 1
            held["is_new"] = False

        # ③ 진입 — 어제 종가로 확정된 목록이 오늘 시가에 체결됐다.
        for row in entries:
            open_px = (quotes["by_ticker"].get(row["ticker"]) or {}).get("open")
            if not open_px:
                continue
            holdings.append(
                {
                    "ticker": row["ticker"],
                    "name": row["name"],
                    "industry": row["industry"],
                    "entry_date": session,
                    "entry_price": float(open_px),
                    "price": float(open_px),
                    "return_pct": 0.0,
                    "days": 0,
                    "is_new": True,
                    "status": "hold",
                    "exit_reason": None,
                }
            )

        _apply_display_quotes(rows, holdings, quotes["by_ticker"])
        # 기준선도 하루 앞당긴다 — 오늘이 새 거래일이므로 '직전' 최고는 어제 종가까지다.
        # `prior_high` 는 캐시 마지막 날 자신을 뺀 값이라(`shift(1)`) 그대로 쓰면 어제 종가가
        # 빠져, 어제 신고가를 찍은 종목의 돌파 거리가 그만큼 부풀려진다.
        window = close_df.loc[(close_df.index > last - pd.Timedelta(HIGH_WINDOW)) & (close_df.index <= last)]
        base_close = window.max()
        base_intraday = panel["high"].loc[window.index].max()
        for row in rows:
            live = quotes["by_ticker"].get(row["ticker"])
            if not live:
                continue
            high = base_close.get(row["ticker"])
            if pd.notna(high) and float(high) > 0:
                row["prior_high"] = float(high)
            intraday = base_intraday.get(row["ticker"])
            if pd.notna(intraday) and float(intraday) > 0:
                row["prior_high_intraday"] = float(intraday)
            price = live["price"]
            row["gap_pct"] = round((price / row["prior_high"] - 1) * 100, 2)
            if row["prior_high_intraday"]:
                row["gap_high_pct"] = round((price / row["prior_high_intraday"] - 1) * 100, 2)
            # 장중 고가가 선을 건드렸는지도 실시간 고가로 다시 본다.
            row["touched"] = bool(live["high"] >= row["prior_high"] and price < row["prior_high"])

        # 남은 보유·후보는 이제 **오늘 잠정 종가** 기준으로 다시 판정한다 — 내일 할 일이다.
        # 이탈 이평선도 오늘 잠정 종가를 넣어 다시 계산한다(어제 선으로 보면 하루 뒤처진다).
        recent = close_df.loc[:last].tail(exit_ma_days - 1)
        for held in holdings:
            live = quotes["by_ticker"].get(held["ticker"])
            if not live or held["ticker"] not in recent.columns:
                continue
            window = recent[held["ticker"]].tolist() + [live["price"]]
            if len(window) == exit_ma_days and not any(pd.isna(v) for v in window):
                below_ma_last[held["ticker"]] = live["price"] < sum(window) / exit_ma_days
        mark_exits(lambda ticker: (quotes["by_ticker"].get(ticker) or {}).get("price"))
        # 장중 판정은 **오늘 잠정 종가** 기준이라 확정이 아니다. 종가가 바뀌면 결과도 바뀌므로
        # 화면·합성이 '(예상)' 으로 구분할 수 있게 표시한다(모멘텀의 `is_exit_forecast` 와 같은 뜻).
        for held in holdings:
            held["is_exit_forecast"] = held.get("status") == "sell"
        entries = pick_entries()

    else:
        # 장중으로 인정되지 않는 구간(장전, 또는 ETF 처럼 체결 시각을 안 주는 종목).
        # 현재가·등락률·보유 수익률만 얹는다 — 종목풀 순위 화면과 같은 기준이라 두 화면의
        # '일간(%)' 이 어긋나지 않는다. 고가·시가가 없거나 직전 세션 값이라 돌파 거리·터치·
        # 진입 예정은 확정 종가 기준 그대로 둔다.
        _apply_display_quotes(rows, holdings, quotes["by_ticker"])

    # 이미 보유 중인 종목은 다시 사지 않는다(백테스트도 같다). 목록에는 남기되 표시를 구분한다 —
    # 보유 종목이 아직 신고가를 갱신 중인지가 추세 판단에 쓸모 있다.
    held = {row["ticker"] for row in holdings}
    for row in rows:
        row["is_held"] = row["ticker"] in held

    rows.sort(key=lambda r: r["gap_pct"], reverse=True)

    # 보유·이탈 행에도 일간 등락률을 붙인다 — 후보 표와 같은 값을 쓰도록 rows 에서 가져온다
    # (장중이면 위 실시간 덮어쓰기까지 반영된 값이다). rows 에 없는 종목은 값 없음으로 둔다.
    change_pct_by = {row["ticker"]: row["change_pct"] for row in rows}
    for item in [*holdings, *simulated["exited_today"]]:
        item["change_pct"] = change_pct_by.get(item["ticker"])

    # 실계좌 보유 여부 — 전략상 보유(is_held)와 별개로 "지금 계좌에 실제로 있는가".
    # 시장 화면(`/kor-market-stock` 등)의 '보유' 컬럼과 같은 소스·같은 기준(그 국가의 모든 계좌).
    from utils.portfolio_io import load_all_holding_tickers
    from utils.settings_loader import get_ticker_type_settings as _pool_settings

    country = str((_pool_settings(pool) or {}).get("country_code") or "").strip().lower()
    try:
        account_held = load_all_holding_tickers(country_code=country or None)
    except Exception:
        logger.warning("[신고가] 실계좌 보유 조회 실패 — '보유' 컬럼을 비운다", exc_info=True)
        account_held = set()
    for item in [*rows, *holdings, *simulated["exited_today"]]:
        item["account_held"] = str(item.get("ticker") or "").strip().upper() in account_held

    # 이탈 행은 청산가와 별개로 현재 시세도 담는다 — 판 뒤로 얼마나 더 갔는지 보이게.
    price_by = {row["ticker"]: row["price"] for row in rows}
    for item in simulated["exited_today"]:
        item["price"] = price_by.get(item["ticker"])
    # 보유·이탈 행에도 시가총액·시총 순위(화면 공용 컬럼).
    # 시총은 원래 진입 우선순위 판정용이라 후보 행에만 채웠는데, 화면은 보유 표에도 같은
    # 컬럼을 두고 있어 통째로 비어 보였다.
    # 거래대금 배수도 후보 행과 같은 값을 붙인다 — 화면 표준 배치(현재가 뒤 거래대금)용.
    value_mult_by = {row["ticker"]: (row.get("value_mult"), row.get("value_mult_live")) for row in rows}
    for item in [*holdings, *simulated["exited_today"]]:
        item["market_cap"] = market_cap_by.get(item["ticker"])
        item["market_cap_rank"] = rank_by_ticker.get(item["ticker"])
        item["value_mult"], item["value_mult_live"] = value_mult_by.get(item["ticker"], (None, None))

    # 장이 열려 있으면 오늘 시가 체결은 이미 끝났으므로, 다음 체결일은 오늘 다음 거래일이다.
    fill_base = pd.Timestamp(str(quotes["traded_at"])[:10]) if quotes["live"] else last
    from utils.settings_loader import get_ticker_type_settings

    return {
        "as_of": str(last.date()),
        # 화면이 표시용 시세를 60초마다 갱신할 때 쓰는 국가 코드(시세 소스가 국가별로 다르다).
        "country": str((get_ticker_type_settings(pool) or {}).get("country_code") or "").strip().lower(),
        # 진입 예정·매도 예정이 실제로 체결되는 날. 화면이 '오늘/내일' 을 이 값으로 가른다.
        "next_session": _next_session(pool, fill_base),
        "holdings": holdings,
        # 내일 시가에 살 종목 (자리·자격·우선순위를 모두 적용한 결과).
        "planned_entries": entries,
        # ADR 게이트 — 하한 미설정이면 None. blocked=True 면 오늘은 신규 진입이 없다.
        "adr_gate": adr_gate,
        "exited_today": simulated["exited_today"],
        "pool": pool,
        "universe_count": len(rows),
        "window_weeks": HIGH_WINDOW_WEEKS,
        "min_value_mult": settings["min_value_mult"],
        # 가격 캐시가 마지막으로 갱신된 시각 — 화면이 "언제 기준인지"를 알린다.
        "refreshed_at": _cache_refreshed_at(pool),
        # 진행 중인 세션의 시세를 얹었는지 — 화면이 '돌파중/돌파성공'을 가르는 데 쓴다.
        "live": quotes["live"],
        # 장전 구간 — 판정에는 안 쓰고 현재가·등락률만 얹었다는 표시.
        "pre_market": quotes["pre_market"],
        # 화면이 주기 갱신을 걸어야 하는지. 장중이거나 개장이 가까운 장전이면 참.
        "auto_refresh": _should_auto_refresh(pool, quotes),
        "quote_at": quotes["traded_at"],
        "breakouts": [r for r in rows if r["gap_pct"] >= 0],
        # 임박(-3% 이내) → 근접(-7% 이내) 순으로 후보를 보여준다.
        "candidates": [r for r in rows if -7 <= r["gap_pct"] < 0][:30],
    }


__all__ = ["MAX_BACKTEST_MONTHS", "current_positions", "load_context", "run_backtest"]
