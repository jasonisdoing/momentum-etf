"""신고가 전략 백테스트.

이벤트 기반이라 월간 리밸런싱과 리듬이 다르다 — 돌파한 날 사고, 이탈 이동평균에
걸린 날 판다. 판정은 종가, 체결은 다음 거래일 **시가**다. 종가 체결은
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
    build_price_panel,
    compute_signals,
    load_price_frames,
    load_settings,
    load_universe,
    validate_settings,
)
from utils.slot_backtest import run_slot_backtest
from utils.slot_positions import (
    _apply_display_quotes,
    _cache_refreshed_at,
    _live_quotes,
    _market_caps,
    _market_today,
    _next_session,
    _pool_country,
    _should_auto_refresh,
)
from utils.stock_memo_store import attach_stock_memos
from utils.ttl_cache import TtlCache

logger = logging.getLogger(__name__)

# 백테스트 기간 상한 — 신고가 창(52주)만큼 앞선 데이터가 있어야 판정이 된다.
MAX_BACKTEST_MONTHS = 60


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
    *,
    start_date: str | None = None,
) -> dict[str, Any]:
    """돌파 전략 백테스트. 일별 자산곡선과 체결 내역을 함께 돌려준다."""
    settings = validate_settings(settings or load_settings())
    months = int(months or DEFAULT_BACKTEST_MONTHS)
    if not 1 <= months <= MAX_BACKTEST_MONTHS:
        raise ValueError(f"'months' 는 1~{MAX_BACKTEST_MONTHS} 사이여야 합니다.")

    pool = settings["pool"]
    slots = int(settings["top_n"])

    context = context or load_context(settings)
    signals = context["signals"]
    breakout, below_ma, value_mult = signals["breakout"], signals["below_ma"], signals["value_mult"]

    # 진입 자격 — 돌파한 날 중 거래대금 급증 배수가 하한 이상인 종목만. 자리가 모자라면
    # 배수가 큰 쪽(돌파에 자금이 실린 종목)부터 담는다.
    min_mult = settings["min_value_mult"]
    qualifies = value_mult.notna() & (value_mult >= min_mult) if min_mult is not None else breakout.notna()

    return run_slot_backtest(
        pool=pool,
        months=months,
        start_date=start_date,
        panel=context["panel"],
        entry=breakout & qualifies,
        exit_signal=below_ma,
        priority=value_mult.fillna(0.0),
        slots=slots,
        adr_floor=settings.get("adr_floor"),
        name_by=context["name_by"],
        industry_by=context["industry_by"],
        exit_reason="이탈",
    )


def _meets_min_mult(mult: Any, minimum: float | None) -> bool:
    """진입 자격 — 거래대금 급증 배수가 하한 이상인가.

    하한이 없으면 항상 통과. 배수를 모르면(상장 직후 등) 미달로 본다 — 추정하지 않는다.
    """
    if minimum is None:
        return True
    return bool(pd.notna(mult)) and float(mult) >= minimum


# 장전에 화면을 주기적으로 다시 받기 시작할 시점 — 개장 몇 분 전부터인가.
# 실제로 예상체결가가 움직이는 구간은 동시호가(개장 30분 전~개장)라 한 시간이면 넉넉하다.
# 시세 제공처의 '장전' 플래그는 새벽부터 켜져 있을 수 있어 그것만 믿고 돌리지 않는다.
_PRE_MARKET_REFRESH_LEAD_MINUTES = 60


# 유니버스 전체를 현재까지 돌리는 계산이라 수십 초 걸린다 — 설정이 같으면
# 결과도 같으므로 짧게 재사용한다(설정을 바꾸면 키가 달라져 새로 계산한다).
_POSITIONS_CACHE = TtlCache(CACHE_TTL_COMPUTE, name="new_high_positions")


def current_positions(settings: dict[str, Any] | None = None) -> dict[str, Any]:
    """지금 들고 있어야 할 종목(보유·오늘 이탈)과 오늘 신호(돌파·후보).

    보유는 **백테스트와 같은 엔진**을 현재까지 돌린 마지막 상태다. 화면과 백테스트가
    다른 코드로 갈라지면 표시된 보유와 성과가 어긋나므로 계산을 나누지 않는다.
    """
    settings = validate_settings(settings or load_settings())
    from utils.strategy_settings import require_start_date

    start_date = require_start_date(settings)
    cache_key = _POSITIONS_CACHE.make_key(settings, start_date)
    result = _POSITIONS_CACHE.get_or_compute(cache_key, lambda: _current_positions(settings, start_date=start_date))
    # 종목 메모는 **캐시 밖**에서 붙인다 — 다른 화면에서 고친 값이 즉시 보여야 한다.
    attach_stock_memos(
        result["breakouts"], result["candidates"], result["holdings"], result["planned_entries"], result["exited_today"]
    )
    return result


def _current_positions(settings: dict[str, Any], *, start_date: str | None) -> dict[str, Any]:
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
    market_cap_by = _market_caps(pool)
    # 시총 순위 — 배치 B 가 메타 캐시에 적어 둔 시장 전체 순위(개별주 풀만 값 있음).
    from utils.market_cap_rank import market_cap_rank_of
    from utils.stock_cache_meta_io import get_stock_cache_meta_docs

    try:
        meta_docs = get_stock_cache_meta_docs(pool, [row["ticker"] for row in universe])
    except Exception:
        meta_docs = {}
    rank_by_ticker = {t: market_cap_rank_of((doc or {}).get("meta_cache")) for t, doc in meta_docs.items()}
    # 현재 화면과 진입 자격은 모멘텀 화면과 동일하게 배치 메타·공용 실시간 경로를 쓴다.
    # 과거 백테스트는 아래의 공용 순수 계산으로 만든 signals 값을 계속 사용한다.
    from utils.rank_service import _load_trade_value_mult
    from utils.trade_value import live_min_value_mult

    value_mult_by, value_mult_live_by = _load_trade_value_mult(pool, list(close_df.columns))
    min_value_mult = settings["min_value_mult"]
    # 장중 누적 배수에 맞춘 하한 — 그 시장의 세션 경과 비율만큼 낮춘다.
    live_required = live_min_value_mult(min_value_mult, _pool_country(pool))
    # 일간 등락률 — 다른 화면(순위·시장추세)과 같은 기준으로 직전 거래일 종가 대비.
    prev_close = close_df.loc[close_df.index[-2]] if len(close_df.index) >= 2 else None

    def high_drawdown(ticker: str) -> float | None:
        """고점 대비(%) — 모멘텀 운용 현황·순위 화면과 **같은 공용 계산**."""
        from core.strategy.scoring import drawdown_from_high_pct

        value = drawdown_from_high_pct(close_df[ticker].dropna())
        return None if value is None else round(value, 2)

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
                "price": float(price),
                "prior_high": float(high),
                # 관례상의 52주 신고가(장중 고가) — 참고 표시용. 판정에는 쓰지 않는다.
                "prior_high_intraday": float(intraday) if has_intraday else None,
                "gap_high_pct": round((float(price) / float(intraday) - 1) * 100, 2) if has_intraday else None,
                # 0 이상이면 돌파, 음수면 최고 종가까지 남은 거리.
                "gap_pct": round(gap_pct, 2),
                "high_drawdown_pct": high_drawdown(ticker),
                "touched": touched,
                "value_mult": round(float(value_mult_by[ticker]), 2) if ticker in value_mult_by else None,
                "value_mult_live": value_mult_live_by.get(ticker),
                # 돌파했더라도 이 값이 거짓이면 사지 않는다 (백테스트와 같은 판정).
                "qualifies": _meets_min_mult(
                    value_mult_by.get(ticker), live_required if ticker in value_mult_live_by else min_value_mult
                ),
            }
        )

    # 보유·이탈은 백테스트 엔진의 마지막 상태를 그대로 쓴다.
    simulated = run_backtest(DEFAULT_BACKTEST_MONTHS, settings, context, start_date=start_date)
    holdings = simulated["open_positions"]

    quotes = _live_quotes(pool, [r["ticker"] for r in rows] + [h["ticker"] for h in holdings], last)
    # 마지막 거래일 종가로 '다음 시가에 할 일' 을 판정한다. 백테스트 루프는 마지막 날을
    # 판정하지 않는다(체결할 다음 날이 없어서). 그래서 여기서 한 번 더 본다 — 이게 없으면
    # 화면에 살 종목만 보이고 팔 종목이 안 보인다.
    exit_ma_days = int(settings["exit_ma_days"])
    below_ma_last = signals["below_ma"].loc[last]
    # 이탈 이평선 값 — 화면이 "이탈까지 얼마 남았는지"를 보여준다. **판정에 쓰는 그 선**이라
    # 화면 숫자와 매도 판정이 갈리지 않는다. 장중이면 아래에서 잠정 종가로 다시 계산한다.
    exit_ma_last = signals["exit_ma"].loc[last]

    def attach_exit_ma_gap() -> None:
        """보유·이탈 행에 이탈선 값과 현재가 대비 여유(%)를 붙인다. 값이 없으면 채우지 않는다.

        이탈 행도 표의 같은 칸을 채운다 — 판 뒤 그 종목이 어디에 있는지 같은 기준으로 본다.
        """
        for held in [*holdings, *simulated["exited_today"]]:
            line = exit_ma_last.get(held["ticker"])
            price = held.get("price")
            if line is None or pd.isna(line) or float(line) <= 0 or price is None:
                held["exit_ma"] = None
                held["exit_ma_gap_pct"] = None
                continue
            held["exit_ma"] = round(float(line), 2)
            held["exit_ma_gap_pct"] = round((float(price) / float(line) - 1) * 100, 2)

    def mark_exits(price_of) -> None:
        """**장중 잠정 종가**로 청산 여부를 다시 본다 — 확정 판정은 엔진 값을 쓴다.
        price_of 가 None 을 돌려주면 판정하지 않는다."""
        for held in holdings:
            price = price_of(held["ticker"])
            if price is None:
                continue
            hit_ma = bool(below_ma_last.get(held["ticker"]))
            held["status"] = "sell" if hit_ma else "hold"
            held["exit_reason"] = "이탈" if hit_ma else None

    # ADR 진입 게이트 — 발동하면 오늘은 신규 진입이 없다(보유 관리는 그대로). 백테스트가
    # 쓰는 공용 게이트와 **같은 함수**라 화면 표시와 실제 판정이 갈리지 않는다.
    adr_gate: dict[str, Any] | None = None
    if settings.get("adr_floor") is not None:
        from utils.momentum_service import adr_market_of_pool
        from utils.slot_backtest import adr_entry_gate

        gate_blocked, gate_at = adr_entry_gate(pool, settings["adr_floor"])
        adr_gate = {
            "market": adr_market_of_pool(pool),
            "floor": settings["adr_floor"],
            "value": gate_at(last),
            "blocked": gate_blocked(last),
        }

    def pick_entries() -> list[dict[str, Any]]:
        """**장중 잠정 종가**로 다시 고른 진입 후보 — 확정 판정은 엔진 값을 쓴다."""
        if adr_gate is not None and adr_gate["blocked"]:
            return []
        planned = sum(1 for h in holdings if h.get("status") == "sell")
        free = int(settings["top_n"]) - (len(holdings) - planned)
        if free <= 0:
            return []
        ready = [
            row
            for row in rows
            if row["gap_pct"] >= 0 and row["qualifies"] and row["ticker"] not in {h["ticker"] for h in holdings}
        ]
        ready.sort(key=lambda row: row["value_mult"] or 0.0, reverse=True)
        return ready[:free]

    # 확정 판정('다음 시가에 할 일')은 **엔진이 낸 값**을 그대로 쓴다 — 화면이 다시 판정하면
    # 백테스트와 갈라진다(tests/test_screen_matches_backtest.py 가 이 관계를 지킨다).
    engine_exits = set(simulated["planned_exits"])
    for held in holdings:
        held["status"] = "sell" if held["ticker"] in engine_exits else "hold"
        held["exit_reason"] = "이탈" if held["status"] == "sell" else None
    attach_exit_ma_gap()
    # 확정 종가 기준 판정 — 장중이면 아래에서 잠정 종가로 다시 판정하며 예상으로 바꾼다.
    for held in holdings:
        held["is_exit_forecast"] = False
    row_by_ticker = {row["ticker"]: row for row in rows}
    entries = [row_by_ticker[ticker] for ticker in simulated["planned_entries"] if ticker in row_by_ticker]

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
                live_line = sum(window) / exit_ma_days
                below_ma_last[held["ticker"]] = live["price"] < live_line
                exit_ma_last[held["ticker"]] = live_line
        mark_exits(lambda ticker: (quotes["by_ticker"].get(ticker) or {}).get("price"))
        attach_exit_ma_gap()
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

    # ── 지난 세션의 청산분은 버린다 ─────────────────────────────────────────
    # 그 세션이 이미 마감했으면 보유 표에 있을 이유가 없다 — 내역은 「체결」 탭에 남는다.
    # 캐시가 하루 늦게 채워지는 동안(미국 종가는 한국 시간 새벽) 어제 팔린 종목이 계속
    # '이탈' 로 보이던 것을 여기서 끊는다. 실시간 시세 유무와 무관하게 시장 현지 날짜로만 본다.
    market_today = _market_today(pool)
    if market_today:
        simulated["exited_today"] = [t for t in simulated["exited_today"] if str(t["exit_date"]) >= market_today]

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
    drawdown_by = {row["ticker"]: row["high_drawdown_pct"] for row in rows}
    for item in [*holdings, *simulated["exited_today"]]:
        item["market_cap"] = market_cap_by.get(item["ticker"])
        item["market_cap_rank"] = rank_by_ticker.get(item["ticker"])
        item["value_mult"], item["value_mult_live"] = value_mult_by.get(item["ticker"], (None, None))
        item["high_drawdown_pct"] = drawdown_by.get(item["ticker"])

    # 장이 열려 있으면 오늘 시가 체결은 이미 끝났으므로, 다음 체결일은 오늘 다음 거래일이다.
    fill_base = pd.Timestamp(str(quotes["traded_at"])[:10]) if quotes["live"] else last
    from utils.settings_loader import get_ticker_type_settings

    return {
        "as_of": str(last.date()),
        # 화면이 표시용 시세를 60초마다 갱신할 때 쓰는 국가 코드(시세 소스가 국가별로 다르다).
        "country": str((get_ticker_type_settings(pool) or {}).get("country_code") or "").strip().lower(),
        # 진입 예정·매도 예정이 실제로 체결되는 날. 화면이 '오늘/내일' 을 이 값으로 가른다.
        "next_session": _next_session(pool, fill_base),
        # 동시 보유 상한 — 화면이 '빈 슬롯' 행 수를 이 값으로 센다. 설정 초안이 아니라
        # **이 결과를 만든 값**을 그대로 내려, 저장 전 화면 값과 어긋나지 않게 한다.
        "top_n": int(settings["top_n"]),
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
