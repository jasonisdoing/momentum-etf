"""모멘텀 전략 백테스트 — 신고가와 **같은 슬롯 엔진**(`utils.slot_backtest`) 위에 얹는다.

전략 규칙
--------
1. 유니버스: 설정에서 고른 종목풀 1개. 제외(exclude_from_ranking)는 제외.
2. 진입: 그날 종가로 **장기 이격 > 0 그리고 단기 이격 >= 0**. 다음 거래일 **시가** 체결.
   자리가 모자라면 **장기 이격률**이 큰 순으로 담는다(순위 화면과 같은 `rank_score`).
3. 청산: 둘 중 하나라도 깨진 날. 판정은 종가, 체결은 다음 거래일 시가.
4. 자리 배분: 동시 보유 상한 top_n, 균등 배분(정수 주수).
   자리가 꽉 차 있으면 더 좋은 후보가 와도 **교체하지 않는다** — 신고가와 같은 결론이다.
5. ADR 하한: 그날 시장 ADR 이 하한 미만이면 **신규 진입만** 건너뛴다. 보유는 그대로 둔다.

예전에는 주 1회 교체(판정일 종가 → 다음 주 첫 거래일 시가)에 '자격 유지' 규칙을 얹고,
ADR 만 주중에 매일 봐서 하한 미달이면 전량 매도했다. 주간 전략에 일간 예외가 붙은 꼴이라
화면(주간 기준)에 그 매도가 드러나지 않았고, 경로 재생·3중 날짜 같은 구조가 딸려 왔다.
2026-09-04 비교에서 8개 풀 중 7개가 일간 슬롯 쪽이 좋아 구조를 통일했다.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from config import CACHE_TTL_COMPUTE
from core.strategy.scoring import calculate_maps_score, drawdown_from_high_pct, hold_eligible, rank_score
from utils.logger import get_app_logger
from utils.momentum_service import (
    adr_market_of_pool,
    load_price_frames,
    load_settings,
    load_universe,
    pool_info,
    validate_settings,
)
from utils.moving_averages import calculate_moving_average
from utils.new_high_service import build_price_panel
from utils.slot_backtest import adr_entry_gate, run_slot_backtest
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

logger = get_app_logger()

MAX_BACKTEST_MONTHS = 60
DEFAULT_BACKTEST_MONTHS = 12


def compute_signals(panel: dict[str, pd.DataFrame], short_ma_days: int, long_ma_days: int) -> dict[str, pd.DataFrame]:
    """이평선 두 개로 만드는 신호 표(행 = 거래일, 열 = 종목) — 백테스트·화면이 같은 값을 본다.

    ``short``/``long`` 은 이격률(%)이다. 이평선을 못 채운 날은 NaN 이고, 그런 날은 ``known``
    이 거짓이라 사고팔지 않는다 — 값을 추정하지 않는다.
    """
    close_df = panel["close"]
    short_gap, long_gap, ready = {}, {}, {}
    for ticker in close_df.columns:
        close = close_df[ticker]
        short_ma = calculate_moving_average(close, short_ma_days, min_periods=short_ma_days)
        long_ma = calculate_moving_average(close, long_ma_days, min_periods=long_ma_days)
        short_gap[ticker] = calculate_maps_score(close, short_ma)
        long_gap[ticker] = calculate_maps_score(close, long_ma)
        # 판정 가능 여부는 **이평선 자체**로 본다 — `calculate_maps_score` 는 못 채운 날을
        # 0 으로 메우므로(fillna) 이격만 보면 워밍업 구간이 '이격 0' 인 정상 값처럼 보인다.
        ready[ticker] = close.notna() & short_ma.notna() & long_ma.notna()
    short_frame = pd.DataFrame(short_gap, index=close_df.index)
    long_frame = pd.DataFrame(long_gap, index=close_df.index)
    known = pd.DataFrame(ready, index=close_df.index)
    # 보유 자격은 순위 화면·종목풀 백테스트와 **같은 공용 규칙**(`hold_eligible`)이다.
    eligible = known & hold_eligible(long_frame, short_frame)
    return {
        "short": short_frame,
        "long": long_frame,
        "known": known,
        "eligible": eligible,
        # 청산은 '자격을 잃었다고 판정할 수 있는 날'만 — 데이터가 없으면 유지한다.
        "exit": known & ~eligible,
        # 진입 우선순위 — 순위 화면과 같은 단일 기준(`rank_score`, 정의는 장기 이격률).
        "priority": rank_score(long_frame, short_frame),
    }


def load_context(settings: dict[str, Any]) -> dict[str, Any]:
    """가격 패널·신호를 한 번만 만들어 재사용한다.

    패널 생성이 이 계산에서 가장 비싼 부분(종목 수백 개의 캐시 역직렬화)이라, 같은 요청에서
    백테스트와 운용 현황이 각자 만들면 시간이 두 배가 된다. 신호는 이평선 두 개에만
    의존하므로 튜닝은 이평선 조합마다 한 번씩만 만들면 된다.
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
        "signals": compute_signals(panel, int(settings["short_ma_days"]), int(settings["long_ma_days"])),
    }


def run_backtest(
    months: int | None = None,
    settings: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
    *,
    start_date: str | None = None,
) -> dict[str, Any]:
    """모멘텀 백테스트. 일별 자산곡선과 체결 내역을 함께 돌려준다(신고가와 같은 형태)."""
    settings = validate_settings(settings or load_settings())
    months = int(months or DEFAULT_BACKTEST_MONTHS)
    if not 1 <= months <= MAX_BACKTEST_MONTHS:
        raise ValueError(f"'months' 는 1~{MAX_BACKTEST_MONTHS} 사이여야 합니다.")

    context = context or load_context(settings)
    signals = context["signals"]
    return run_slot_backtest(
        pool=settings["pool"],
        months=months,
        start_date=start_date,
        panel=context["panel"],
        entry=signals["eligible"],
        exit_signal=signals["exit"],
        # 순위를 모르는 종목은 맨 뒤로 — 자리 경쟁에서 밀린다(0 으로 채우면 음수 이격보다 앞선다).
        priority=signals["priority"].fillna(float("-inf")),
        slots=int(settings["top_n"]),
        adr_floor=settings.get("adr_floor"),
        name_by=context["name_by"],
        industry_by=context["industry_by"],
        exit_reason="이탈",
    )


# ── 운용 현황 ─────────────────────────────────────────────────────────────
# 보유 재구성은 전략·종목풀에 저장된 시작일부터 이어 계산한다.
_POSITIONS_CACHE = TtlCache(CACHE_TTL_COMPUTE, name="momentum_positions")


def current_positions(settings: dict[str, Any] | None = None) -> dict[str, Any]:
    """지금 들고 있어야 할 종목(보유·이탈)과 진입 후보.

    보유는 **백테스트와 같은 엔진**을 현재까지 돌린 마지막 상태다. 화면과 백테스트가 다른
    코드로 갈라지면 표시된 보유와 성과가 어긋나므로 계산을 나누지 않는다.
    """
    settings = validate_settings(settings or load_settings())
    from utils.strategy_settings import require_start_date

    start_date = require_start_date(settings)
    cache_key = _POSITIONS_CACHE.make_key(settings, start_date)
    result = _POSITIONS_CACHE.get_or_compute(cache_key, lambda: _current_positions(settings, start_date=start_date))
    # 종목 메모는 **캐시 밖**에서 붙인다 — 다른 화면에서 고친 값이 즉시 보여야 한다.
    attach_stock_memos(result["candidates"], result["holdings"], result["planned_entries"], result["exited_today"])
    return result


def _current_positions(settings: dict[str, Any], *, start_date: str | None) -> dict[str, Any]:
    pool = settings["pool"]
    context = load_context(settings)
    name_by, industry_by = context["name_by"], context["industry_by"]
    panel, signals = context["panel"], context["signals"]
    close_df = panel["close"]
    last = close_df.index[-1]
    slots = int(settings["top_n"])
    info = pool_info(pool)

    short_gap, long_gap = signals["short"].loc[last], signals["long"].loc[last]
    market_cap_by = _market_caps(pool)
    from utils.market_cap_rank import market_cap_rank_of
    from utils.rank_service import _load_trade_value_mult
    from utils.stock_cache_meta_io import get_stock_cache_meta_docs

    tickers = list(close_df.columns)
    try:
        meta_docs = get_stock_cache_meta_docs(pool, tickers)
    except Exception:
        meta_docs = {}
    rank_by_ticker = {t: market_cap_rank_of((doc or {}).get("meta_cache")) for t, doc in meta_docs.items()}
    value_mult_by, value_mult_live_by = _load_trade_value_mult(pool, tickers)
    # 일간 등락률 — 다른 화면(순위·시장추세)과 같은 기준으로 직전 거래일 종가 대비.
    prev_close = close_df.loc[close_df.index[-2]] if len(close_df.index) >= 2 else None

    def high_drawdown(ticker: str) -> float | None:
        value = drawdown_from_high_pct(close_df[ticker].dropna())
        return None if value is None else round(value, 2)

    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        price = close_df.at[last, ticker]
        short_value, long_value = short_gap.get(ticker), long_gap.get(ticker)
        if pd.isna(price) or pd.isna(short_value) or pd.isna(long_value):
            continue  # 판정할 수 없는 종목은 후보로도 세지 않는다 — 값을 추정하지 않는다.
        before = None if prev_close is None else prev_close.get(ticker)
        rows.append(
            {
                "ticker": ticker,
                "name": name_by.get(ticker, ticker),
                "industry": industry_by.get(ticker, ""),
                "change_pct": (
                    round((float(price) / float(before) - 1) * 100, 2)
                    if before is not None and pd.notna(before) and float(before) > 0
                    else None
                ),
                "price": float(price),
                "market_cap": market_cap_by.get(ticker),
                "market_cap_rank": rank_by_ticker.get(ticker),
                "value_mult": round(float(value_mult_by[ticker]), 2) if ticker in value_mult_by else None,
                "value_mult_live": value_mult_live_by.get(ticker),
                # 이탈까지 남은 여유(%) — 둘 중 하나라도 0 이하가 되면 다음 거래일 시가에 판다.
                "short_gap_pct": round(float(short_value), 2),
                "long_gap_pct": round(float(long_value), 2),
                "high_drawdown_pct": high_drawdown(ticker),
                # 진입 자격 — 백테스트와 같은 공용 규칙.
                "eligible": bool(hold_eligible(float(long_value), float(short_value))),
            }
        )
    # 우선순위 — 장기 이격률이 큰 순(백테스트의 `priority` 와 같은 기준).
    rows.sort(key=lambda row: row["long_gap_pct"], reverse=True)

    # 보유·이탈은 백테스트 엔진의 마지막 상태를 그대로 쓴다.
    simulated = run_backtest(DEFAULT_BACKTEST_MONTHS, settings, context, start_date=start_date)
    holdings = simulated["open_positions"]
    exited_today = simulated["exited_today"]
    row_by_ticker = {row["ticker"]: row for row in rows}

    quotes = _live_quotes(pool, tickers, last)
    # '다음 시가에 할 일' 은 **엔진이 판정한 값**을 그대로 쓴다 — 화면이 다시 판정하면
    # 백테스트와 갈라진다(tests/test_screen_matches_backtest.py 가 이 관계를 지킨다).
    planned_exits = set(simulated["planned_exits"])
    for held in holdings:
        held["status"] = "sell" if held["ticker"] in planned_exits else "hold"
        held["exit_reason"] = "이탈" if held["status"] == "sell" else None

    entry_blocked, adr_at = adr_entry_gate(pool, settings.get("adr_floor"))
    adr_gate: dict[str, Any] | None = None
    if settings.get("adr_floor") is not None:
        adr_gate = {
            "market": adr_market_of_pool(pool),
            "floor": settings["adr_floor"],
            "value": adr_at(last),
            "blocked": entry_blocked(last),
        }

    # 다음 시가에 살 종목 — 엔진이 고른 티커에 화면 표시용 값만 붙인다.
    entries = [row_by_ticker[ticker] for ticker in simulated["planned_entries"] if ticker in row_by_ticker]

    # ── 지난 세션의 청산분은 버린다 ─────────────────────────────────────────
    # 그 세션이 이미 마감했으면 보유 표에 있을 이유가 없다 — 내역은 「체결」 탭에 남는다.
    # 들어온 쪽(대체 매수)은 보유라 남고 나간 쪽만 사라지지만, 짝을 맞추자고 지난 매도를
    # 계속 세워 두면 오늘 할 일과 섞인다.
    market_today = _market_today(pool)
    if market_today:
        exited_today = [t for t in exited_today if str(t["exit_date"]) >= market_today]

    # 표시용 시세·부가 정보 — 보유·이탈 행에도 후보 표와 같은 값을 붙인다.
    for item in [*holdings, *exited_today]:
        row = row_by_ticker.get(item["ticker"])
        item["change_pct"] = (row or {}).get("change_pct")
        item["market_cap_rank"] = (row or {}).get("market_cap_rank")
        item["value_mult"] = (row or {}).get("value_mult")
        item["value_mult_live"] = (row or {}).get("value_mult_live")
        item["short_gap_pct"] = (row or {}).get("short_gap_pct")
        item["long_gap_pct"] = (row or {}).get("long_gap_pct")
        item["high_drawdown_pct"] = (row or {}).get("high_drawdown_pct")
    for item in exited_today:
        item["price"] = (row_by_ticker.get(item["ticker"]) or {}).get("price")
    _apply_display_quotes(rows, holdings, quotes["by_ticker"])

    # 실계좌 보유 여부 — 전략상 보유와 별개로 "지금 계좌에 실제로 있는가".
    from utils.portfolio_io import load_all_holding_tickers

    try:
        account_held = load_all_holding_tickers(country_code=_pool_country(pool) or None)
    except Exception:
        logger.warning("[모멘텀] 실계좌 보유 조회 실패 — '보유' 표시를 비운다", exc_info=True)
        account_held = set()
    for item in [*rows, *holdings, *exited_today]:
        item["account_held"] = str(item.get("ticker") or "").strip().upper() in account_held

    held_tickers = {h["ticker"] for h in holdings}
    entry_tickers = {row["ticker"] for row in entries}
    # 순위 — 우선순위(장기 이격률) 순 자리. 진입 예정과 후보가 **같은 번호 체계**를 쓴다.
    # 자격 미달 종목은 자리를 차지하지 않는다 — 세면 화면 순위가 6·7·9 처럼 건너뛴다.
    rank_by_ticker = {row["ticker"]: index for index, row in enumerate([r for r in rows if r["eligible"]], start=1)}
    # 진입 후보 — 우선순위 순 top_n 개. 이미 담은(보유·진입 예정) 종목은 표에서 뺀다.
    candidates = [
        {**row, "rank": rank_by_ticker[row["ticker"]]}
        for row in rows
        if row["eligible"] and row["ticker"] not in held_tickers and row["ticker"] not in entry_tickers
    ][:slots]

    # 장이 열려 있으면 오늘 시가 체결은 이미 끝났으므로, 다음 체결일은 오늘 다음 거래일이다.
    fill_base = pd.Timestamp(str(quotes["traded_at"])[:10]) if quotes["live"] else last
    return {
        "as_of": str(last.date()),
        "pool": pool,
        "country": info["country"],
        "currency": info["currency"],
        "top_n": slots,
        "next_session": _next_session(pool, fill_base),
        "holdings": holdings,
        "planned_entries": [{**row, "rank": rank_by_ticker[row["ticker"]]} for row in entries],
        "exited_today": exited_today,
        "candidates": candidates,
        "adr_gate": adr_gate,
        "universe_count": len(rows),
        "refreshed_at": _cache_refreshed_at(pool),
        "live": quotes["live"],
        "pre_market": quotes["pre_market"],
        "auto_refresh": _should_auto_refresh(pool, quotes),
        "quote_at": quotes["traded_at"],
    }


__all__ = ["MAX_BACKTEST_MONTHS", "compute_signals", "current_positions", "load_context", "run_backtest"]
