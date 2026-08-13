"""신고가 돌파 전략 백테스트.

이벤트 기반이라 월간 리밸런싱과 리듬이 다르다 — 돌파한 날 사고, 손절선이나 이탈
이동평균에 걸린 날 판다. 판정은 종가, 체결은 다음 거래일 **시가**다. 종가 체결은
쓰지 않는다: 마감 동시호가에 대량을 던지면 체결가를 모른 채 거래하게 된다.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from utils.new_high_service import (
    HIGH_WINDOW_WEEKS,
    benchmark_info,
    build_price_panel,
    compute_signals,
    load_benchmark_close,
    load_price_frames,
    load_settings,
    load_universe,
    validate_settings,
)

logger = logging.getLogger(__name__)

# 백테스트 기간 상한 — 신고가 창(240거래일)만큼 앞선 데이터가 있어야 판정이 된다.
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
    as_of: str | None = None,
) -> dict[str, Any]:
    """돌파 전략 백테스트. 일별 자산곡선과 체결 내역을 함께 돌려준다.

    ``as_of`` 를 주면 그 거래일까지만 돌린다 — 과거 시점의 보유·신호를 재현할 때 쓴다.
    """
    settings = validate_settings(settings or load_settings())
    months = int(months or settings["backtest_months"])
    if not 1 <= months <= MAX_BACKTEST_MONTHS:
        raise ValueError(f"'months' 는 1~{MAX_BACKTEST_MONTHS} 사이여야 합니다.")

    pool = settings["pool"]
    slots = int(settings["top_n"])
    stop_pct = float(settings["stop_loss_pct"])
    slippage = float(settings["slippage_pct"])

    context = context or load_context(settings)
    universe = context["universe"]
    name_by, industry_by = context["name_by"], context["industry_by"]
    panel, signals = context["panel"], context["signals"]

    close_df, open_df = panel["close"], panel["open"]
    breakout, below_ma = signals["breakout"], signals["below_ma"]

    # 진입 우선순위. `market_cap` 은 **현재 시가총액**을 전 구간에 그대로 쓴다.
    # 과거 시가총액 이력이 없어(상장주식수 이력 부재) 이렇게 할 수밖에 없는데,
    # 그 결과 "나중에 대형주가 되는 종목"이 과거에도 우대받는 미래 정보가 섞인다.
    # 사용자가 이 왜곡을 알고 선택한 방식이다 — 결과를 실제 기대값으로 읽으면 안 된다.
    min_mult = settings["min_value_mult"]
    use_market_cap = settings["entry_priority"] == "market_cap"
    caps = _market_caps(pool, [row["ticker"] for row in universe]) if use_market_cap else {}
    value_mult = signals["value_mult"]

    def priority_of(day: pd.Timestamp, ticker: str) -> float:
        if use_market_cap:
            return float(caps.get(ticker, 0.0))
        score = value_mult.at[day, ticker]
        return float(score) if pd.notna(score) else 0.0

    dates = close_df.index
    if as_of:
        dates = dates[dates <= pd.Timestamp(as_of)]
        if len(dates) == 0:
            raise RuntimeError(f"{as_of} 이전의 가격 데이터가 없습니다.")
    span = [d for d in dates if d >= dates[-1] - pd.DateOffset(months=months)]
    if len(span) < 2:
        raise RuntimeError("백테스트할 구간의 가격 데이터가 부족합니다.")

    equity = 1.0
    holdings: dict[str, dict[str, Any]] = {}
    trades: list[dict[str, Any]] = []
    curve: list[float] = []
    last_day = span[-1]

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
            ret = (float(exit_price) * (1 - slippage / 100)) / position["entry"] - 1
            equity *= 1 + ret / slots
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

        # 2) 진입 — 빈 자리만큼, 거래대금 급증이 큰 순
        free = slots - len(holdings)
        if free > 0:
            row = breakout.loc[day]
            picks = [
                t for t in row[row].index
                if t not in holdings
                and not pd.isna(open_df.at[nxt, t])
                and _meets_min_mult(value_mult.at[day, t], min_mult)
            ]
            picks.sort(key=lambda t: priority_of(day, t), reverse=True)
            for ticker in picks[:free]:
                entry_open = float(open_df.at[nxt, ticker])
                holdings[ticker] = {
                    "open": entry_open,
                    # 손익 계산에는 슬리피지를 얹은 값을 쓴다(표시용 가격과 구분).
                    "entry": entry_open * (1 + slippage / 100),
                    "date": nxt,
                }

        open_pnl = sum(
            (float(close_df.at[day, t]) / p["entry"] - 1) / slots
            for t, p in holdings.items()
            if pd.notna(close_df.at[day, t])
        )
        curve.append(equity * (1 + open_pnl))

    # 아직 청산하지 않은 종목 — 성과에는 평가손익으로 이미 반영돼 있지만 체결 내역에는 없다.
    open_positions = []
    for ticker, position in holdings.items():
        price = close_df.at[last_day, ticker]
        if pd.isna(price):
            continue
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
            }
        )
    open_positions.sort(key=lambda row: row["entry_date"], reverse=True)
    exited_today = [t for t in trades if t["exit_date"] == str(last_day.date())]

    strategy = pd.Series(curve, index=span[:-1])
    benchmark = load_benchmark_close(pool).reindex(strategy.index).ffill()
    benchmark = benchmark / benchmark.iloc[0]

    wins = [t for t in trades if t["return_pct"] > 0]
    losses = [t for t in trades if t["return_pct"] <= 0]
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
        "trade_count": len(trades),
        "win_rate_pct": round(len(wins) / len(trades) * 100, 1) if trades else None,
        "avg_win_pct": round(sum(t["return_pct"] for t in wins) / len(wins), 2) if wins else None,
        "avg_loss_pct": round(sum(t["return_pct"] for t in losses) / len(losses), 2) if losses else None,
        "stop_count": sum(1 for t in trades if t["reason"] == "손절"),
        "exit_ma_count": sum(1 for t in trades if t["reason"] == "이탈"),
        "trades": sorted(trades, key=lambda t: t["exit_date"], reverse=True),
        "as_of": str(last_day.date()),
        "open_positions": open_positions,
        "exited_today": exited_today,
        "daily": [
            {"date": str(d.date()), "strategy_pct": round((v - 1) * 100, 2),
             "benchmark_pct": round((float(benchmark.loc[d]) - 1) * 100, 2)}
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


def _market_caps(pool: str, tickers: list[str]) -> dict[str, float]:
    """티커 → 시가총액(원). 소스는 국가마다 다르므로 여기서 갈라 쓴다.

    한국 개별주는 네이버 시세표(`/kor-market-stock` 과 같은 소스, 억 원 단위)를,
    그 외에는 종목 캐시 메타의 순자산총액(`/pools-rank` 와 같은 값)을 쓴다.
    값이 없는 종목은 맵에서 빠진다 — 화면은 '-' 로 둔다(임의 보정 없음).

    현재 값만 있고 과거 이력이 없다. 그래서 백테스트 우선순위에는 쓰지 않는다.
    """
    from utils.settings_loader import get_ticker_type_settings

    country = str((get_ticker_type_settings(pool) or {}).get("country_code") or "").strip().lower()
    if country == "kor":
        from utils.kor_stock_market_service import load_kor_market_caps

        # 네이버는 억 원 단위로 준다 — 다른 소스와 맞춰 원 단위로 되돌린다.
        return {ticker: float(cap) * 1_0000_0000 for ticker, cap in load_kor_market_caps(tickers).items()}

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
        by_ticker[ticker] = {
            "price": float(price),
            "high": float(quote.get("high") or price),
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
        "by_ticker": by_ticker if live else {},
    }


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


def _cache_refreshed_at(pool: str) -> str | None:
    """이 종목풀 가격 캐시의 마지막 갱신 시각(ISO). 배치가 안 돌았으면 None."""
    from utils.cache_utils import get_cache_refresh_completed_at

    completed = get_cache_refresh_completed_at(pool)
    return completed.isoformat() if completed else None


def available_dates(
    context: dict[str, Any], min_value_mult: float | None, limit: int = 20
) -> list[dict[str, Any]]:
    """최근 거래일 목록과 그날의 후보·돌파 종목 수. 날짜 셀렉트가 이 값을 쓴다.

    돌파 수에는 **자격을 통과한 것만** 센다. 거래대금 하한에 걸려 사지 못하는 종목까지
    세면 "돌파 2"인데 실제로는 살 게 없는 날이 생겨 목록이 사람을 오해하게 만든다.
    """
    panel, signals = context["panel"], context["signals"]
    close_df, prior_high, value_mult = panel["close"], signals["prior_high"], signals["value_mult"]
    gap = (close_df / prior_high - 1) * 100
    qualified = value_mult >= min_value_mult if min_value_mult is not None else value_mult.notna() | True

    rows: list[dict[str, Any]] = []
    for day in close_df.index[-limit:]:
        row, ok = gap.loc[day], qualified.loc[day]
        breakout = row >= 0
        rows.append(
            {
                "date": str(day.date()),
                "candidate_count": int((breakout | ((row >= -12) & (row < 0))).sum()),
                "breakout_count": int((breakout & ok).sum()),
            }
        )
    return rows[::-1]


def current_positions(settings: dict[str, Any] | None = None, as_of: str | None = None) -> dict[str, Any]:
    """지금 들고 있어야 할 종목(보유·오늘 이탈)과 오늘 신호(돌파·후보).

    보유는 **백테스트와 같은 엔진**을 현재까지 돌린 마지막 상태다. 화면과 백테스트가
    다른 코드로 갈라지면 표시된 보유와 성과가 어긋나므로 계산을 나누지 않는다.
    """
    settings = validate_settings(settings or load_settings())
    pool = settings["pool"]
    context = load_context(settings)
    universe = context["universe"]
    name_by, industry_by = context["name_by"], context["industry_by"]
    panel, signals = context["panel"], context["signals"]

    close_df = panel["close"]
    dates = close_df.index
    if as_of:
        dates = dates[dates <= pd.Timestamp(as_of)]
        if len(dates) == 0:
            raise RuntimeError(f"{as_of} 이전의 가격 데이터가 없습니다.")
    last = dates[-1]
    prior_high = signals["prior_high"].loc[last]
    prior_high_intraday = signals["prior_high_intraday"].loc[last]
    today_high = panel["high"].loc[last]
    value_mult = signals["value_mult"].loc[last]
    trade_value = panel["value"].loc[last]
    market_cap_by = _market_caps(pool, [row["ticker"] for row in universe])
    # 일간 등락률 — 다른 화면(순위·시장추세)과 같은 기준으로 직전 거래일 종가 대비.
    prev_close = close_df.loc[dates[-2]] if len(dates) >= 2 else None

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
    simulated = run_backtest(_HOLDINGS_LOOKBACK_MONTHS, settings, context, as_of=str(last.date()))
    holdings = simulated["open_positions"]

    # 과거 날짜를 보는 중이면 실시간을 섞지 않는다 — 그날 상태를 재현하는 화면이다.
    quotes = (
        _live_quotes(pool, [r["ticker"] for r in rows] + [h["ticker"] for h in holdings], last)
        if not as_of
        else {"live": False, "pre_market": False, "traded_at": None, "by_ticker": {}}
    )
    if quotes["live"]:
        for row in rows:
            live = quotes["by_ticker"].get(row["ticker"])
            if not live:
                continue
            price = live["price"]
            row["price"] = price
            row["change_pct"] = round(live["change_pct"], 2) if live["change_pct"] is not None else row["change_pct"]
            row["gap_pct"] = round((price / row["prior_high"] - 1) * 100, 2)
            if row["prior_high_intraday"]:
                row["gap_high_pct"] = round((price / row["prior_high_intraday"] - 1) * 100, 2)
            # 장중 고가가 선을 건드렸는지도 실시간 고가로 다시 본다.
            row["touched"] = bool(live["high"] >= row["prior_high"] and price < row["prior_high"])
        for held in holdings:
            live = quotes["by_ticker"].get(held["ticker"])
            if not live:
                continue
            held["price"] = live["price"]
            held["return_pct"] = round((live["price"] / held["entry_price"] - 1) * 100, 2)

    # 마지막 거래일 종가로 '내일 할 일'을 판정한다. 백테스트 루프는 마지막 날을 판정하지
    # 않는다(체결할 다음 날이 없어서). 그래서 여기서 한 번 더 본다 — 이게 없으면
    # 화면에 살 종목만 보이고 팔 종목이 안 보인다.
    stop_pct = float(settings["stop_loss_pct"])
    below_ma_last = signals["below_ma"].loc[last]
    for held in holdings:
        price = close_df.at[last, held["ticker"]]
        if pd.isna(price):
            continue
        hit_stop = (float(price) / held["entry_price"] - 1) * 100 <= stop_pct
        hit_ma = bool(below_ma_last.get(held["ticker"]))
        held["status"] = "sell" if (hit_stop or hit_ma) else "hold"
        held["exit_reason"] = "손절" if hit_stop else ("이탈" if hit_ma else None)

    # 내일 살 종목 — 오늘 청산 예정분만큼 자리가 더 생긴다(둘 다 내일 시가에 체결).
    planned_exits = sum(1 for h in holdings if h.get("status") == "sell")
    free = int(settings["top_n"]) - (len(holdings) - planned_exits)
    entries: list[dict[str, Any]] = []
    if free > 0:
        use_market_cap = settings["entry_priority"] == "market_cap"
        caps = _market_caps(pool, [row["ticker"] for row in universe]) if use_market_cap else {}
        ready = [
            row for row in rows
            if row["gap_pct"] >= 0 and row["qualifies"] and row["ticker"] not in {h["ticker"] for h in holdings}
        ]
        ready.sort(
            key=lambda row: caps.get(row["ticker"], 0.0) if use_market_cap else (row["value_mult"] or 0.0),
            reverse=True,
        )
        entries = ready[:free]

    # 이미 보유 중인 종목은 다시 사지 않는다(백테스트도 같다). 목록에는 남기되 표시를 구분한다 —
    # 보유 종목이 아직 신고가를 갱신 중인지가 추세 판단에 쓸모 있다.
    held = {row["ticker"] for row in holdings}
    for row in rows:
        row["is_held"] = row["ticker"] in held

    rows.sort(key=lambda r: r["gap_pct"], reverse=True)
    return {
        "as_of": str(last.date()),
        # 진입 예정·매도 예정이 실제로 체결되는 날. 화면이 '오늘/내일' 을 이 값으로 가른다.
        "next_session": _next_session(pool, last),
        "holdings": holdings,
        # 내일 시가에 살 종목 (자리·자격·우선순위를 모두 적용한 결과).
        "planned_entries": entries,
        "exited_today": simulated["exited_today"],
        "pool": pool,
        "universe_count": len(rows),
        "window_weeks": HIGH_WINDOW_WEEKS,
        "min_value_mult": settings["min_value_mult"],
        "available_dates": available_dates(context, settings["min_value_mult"]),
        # 가격 캐시가 마지막으로 갱신된 시각 — 화면이 "언제 기준인지"를 알린다.
        "refreshed_at": _cache_refreshed_at(pool),
        # 진행 중인 세션의 시세를 얹었는지 — 화면이 '돌파중/돌파성공'을 가르는 데 쓴다.
        "live": quotes["live"],
        # 장전 구간 — 실시간을 섞지 않았다는 표시. 화면이 '장중' 대신 '장전'으로 알린다.
        "pre_market": quotes["pre_market"],
        "quote_at": quotes["traded_at"],
        "breakouts": [r for r in rows if r["gap_pct"] >= 0],
        # 임박(-3% 이내) → 근접(-7% 이내) → 관찰(-12% 이내) 순으로 후보를 보여준다.
        "candidates": [r for r in rows if -12 <= r["gap_pct"] < 0][:30],
    }


__all__ = ["MAX_BACKTEST_MONTHS", "current_positions", "load_context", "run_backtest"]
