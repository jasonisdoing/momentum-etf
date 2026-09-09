"""모멘텀 전략 백테스트 — 신고가와 **같은 슬롯 엔진**(`core.strategy.slot_backtest`) 위에 얹는다.

전략 규칙
--------
1. 유니버스: 설정에서 고른 종목풀 1개. 제외(exclude_from_ranking)는 제외.
2. 진입: 그날 종가로 **장기 이격 > 0 그리고 단기 이격 >= 0**. 다음 거래일 **시가** 체결.
   자리가 모자라면 **장기 이격률**이 큰 순으로 담는다(순위 화면과 같은 `rank_score`).
3. 청산: 둘 중 하나라도 깨진 날. 판정은 종가, 체결은 다음 거래일 시가.
4. 자리 배분: 동시 보유 상한 top_n, 균등 배분(정수 주수).
   자리가 꽉 차 있으면 더 좋은 후보가 와도 **교체하지 않는다** — 신고가와 같은 결론이다.
5. ADR 하한: 그날 시장 ADR 이 하한 미만이면 **신규 진입만** 건너뛴다. 보유는 그대로 둔다.
6. 장중 화면: 실시간 가격을 **마지막 봉**으로 쓴 같은 신호 계산으로 판정을 보여준다
   (AGENTS.md §10-6). 규칙은 그대로고 입력만 잠정이라, 종가가 확정되면 백테스트와 일치한다.

예전에는 주 1회 교체(판정일 종가 → 다음 주 첫 거래일 시가)에 '자격 유지' 규칙을 얹고,
ADR 만 주중에 매일 봐서 하한 미달이면 전량 매도했다. 주간 전략에 일간 예외가 붙은 꼴이라
화면(주간 기준)에 그 매도가 드러나지 않았고, 경로 재생·3중 날짜 같은 구조가 딸려 왔다.
2026-09-04 비교에서 8개 풀 중 7개가 일간 슬롯 쪽이 좋아 구조를 통일했다.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from config import CACHE_TTL_COMPUTE
from core.strategy.intraday import effective_close_frame, mark_engine_statuses
from core.strategy.momentum import signals as momentum_signals
from core.strategy.price_panel import build_price_panel
from core.strategy.scoring import drawdown_from_high_pct, is_new_listing
from core.strategy.slot_backtest import run_slot_backtest
from utils.logger import get_app_logger
from utils.momentum_service import (
    adr_market_of_pool,
    load_price_frames,
    load_settings,
    load_universe,
    pool_info,
    validate_settings,
)
from utils.pool_signal_backtest_service import validate_backtest_months
from utils.slot_positions import (
    _apply_display_quotes,
    _cache_refreshed_at,
    _live_quotes,
    _market_caps,
    _market_today,
    _next_session,
    _pool_country,
    _should_auto_refresh,
    load_slot_market,
)
from utils.stock_memo_store import attach_stock_memos
from utils.ttl_cache import TtlCache

logger = get_app_logger()

DEFAULT_BACKTEST_MONTHS = 12


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
        "signals": momentum_signals.compute_signals(
            panel, int(settings["short_ma_days"]), int(settings["long_ma_days"])
        ),
    }


def run_backtest(
    months: int | None = None,
    settings: dict[str, Any] | None = None,
    context: dict[str, Any] | None = None,
    *,
    start_date: str | None = None,
    market: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """모멘텀 백테스트. 일별 자산곡선과 체결 내역을 함께 돌려준다(신고가와 같은 형태).

    ``market`` 은 슬리피지·시작 자본·ADR 게이트·벤치마크 묶음(`load_slot_market`) — 엔진은
    조회 없이 이 값들만 받는다. 운용 현황이 캐시 키에 쓰려고 미리 수집해 넘긴다.
    """
    settings = validate_settings(settings or load_settings())
    months = int(months or DEFAULT_BACKTEST_MONTHS)
    validate_backtest_months(months)

    context = context or load_context(settings)
    market = market or load_slot_market(settings["pool"], settings.get("adr_floor"))
    signals = context["signals"]
    return run_slot_backtest(
        months=months,
        start_date=start_date,
        panel=context["panel"],
        # 진입은 보유 자격에 변동성 문턱(설정)을 얹은 신호 — 청산은 아래 0선 그대로다.
        entry=momentum_signals.entry_signal(context["panel"]["close"], signals, settings["entry_vol_mult"]),
        exit_signal=signals["exit"],
        # 순위를 모르는 종목은 맨 뒤로 — 자리 경쟁에서 밀린다(0 으로 채우면 음수 이격보다 앞선다).
        priority=signals["priority"].fillna(float("-inf")),
        slots=int(settings["top_n"]),
        name_by=context["name_by"],
        industry_by=context["industry_by"],
        exit_reason="이탈",
        **market,
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
    market = load_slot_market(settings["pool"], settings.get("adr_floor"))
    # 키에 슬리피지·시작 자본까지 넣는다 — 풀 설정을 바꾸면 즉시 새 값으로 계산돼야 한다.
    cache_key = _POSITIONS_CACHE.make_key(
        settings, start_date, market["buy_slippage"], market["sell_slippage"], market["initial_capital"]
    )
    result = _POSITIONS_CACHE.get_or_compute(
        cache_key, lambda: _current_positions(settings, start_date=start_date, market=market)
    )
    # 종목 메모는 **캐시 밖**에서 붙인다 — 다른 화면에서 고친 값이 즉시 보여야 한다.
    attach_stock_memos(result["candidates"], result["holdings"], result["planned_entries"], result["exited_today"])
    return result


def _current_positions(settings: dict[str, Any], *, start_date: str | None, market: dict[str, Any]) -> dict[str, Any]:
    pool = settings["pool"]
    context = load_context(settings)
    # 확정 실행 — 이 결과의 daily 를 페이로드에 실어 합성 슬리브 몫이 같은 실행을 읽는다.
    try:
        simulated = run_backtest(DEFAULT_BACKTEST_MONTHS, settings, context, start_date=start_date, market=market)
    except RuntimeError as error:
        if "이후의 가격 데이터가 아직 없습니다" not in str(error):
            raise
        # 시작일이 오늘인데 오늘 확정 봉이 아직 없는 경우 — 확정 구간은 「시작 전」이라
        # 빈 상태로 두고, 장중이면 아래 잠정 실행이 오늘 잠정 봉으로 첫 판정을 낸다.
        # 값을 지어내는 게 아니라 '아직 아무 것도 하지 않은 전략'의 실제 상태다.
        simulated = {
            "open_positions": [],
            "exited_today": [],
            "planned_exits": [],
            "planned_entries": [],
            "planned_entry_weights": {},
            "daily": [],
        }
    name_by, industry_by = context["name_by"], context["industry_by"]
    panel, signals = context["panel"], context["signals"]
    close_df = panel["close"]
    last = close_df.index[-1]
    slots = int(settings["top_n"])
    info = pool_info(pool)

    short_gap, long_gap = signals["short"].loc[last], signals["long"].loc[last]
    # 진입 자격 — 백테스트와 같은 신호(보유 자격 + 변동성 문턱). 후보 표가 이 값으로 순위를 매긴다.
    entry_last = momentum_signals.entry_signal(close_df, signals, settings["entry_vol_mult"]).loc[last]
    # 변동성(%) — 진입 문턱 판정과 같은 값(20일 일간 수익률 표준편차). 화면 공용 컬럼용.
    vol_last = momentum_signals.daily_volatility_pct(close_df).loc[last]
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
                "volatility_pct": round(float(vol_last[ticker]), 2) if pd.notna(vol_last.get(ticker)) else None,
                # 신규상장(🆕) — 전 화면 공용 판정(core.strategy.scoring.is_new_listing).
                "new_listing": is_new_listing(close_df[ticker].dropna()),
                # 이탈까지 남은 여유(%) — 둘 중 하나라도 0 이하가 되면 다음 거래일 시가에 판다.
                "short_gap_pct": round(float(short_value), 2),
                "long_gap_pct": round(float(long_value), 2),
                "high_drawdown_pct": high_drawdown(ticker),
                # 진입 자격 — 백테스트와 같은 신호(보유 자격 + 진입 문턱).
                "eligible": bool(entry_last.get(ticker, False)),
            }
        )
    # 우선순위 — 장기 이격률이 큰 순(백테스트의 `priority` 와 같은 기준).
    rows.sort(key=lambda row: row["long_gap_pct"], reverse=True)

    # 보유·이탈은 백테스트 엔진의 마지막 상태를 그대로 쓴다(위 확정 실행).
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

    adr_gate: dict[str, Any] | None = None
    if settings.get("adr_floor") is not None:
        adr_gate = {
            "market": adr_market_of_pool(pool),
            "floor": settings["adr_floor"],
            "value": market["adr_at"](last),
            "blocked": market["entry_blocked"](last),
        }

    # 다음 시가에 살 종목 — 엔진이 고른 티커에 화면 표시용 값만 붙인다.
    entries = [
        {**row_by_ticker[ticker], "sleeve_weight_pct": simulated["planned_entry_weights"][ticker]}
        for ticker in simulated["planned_entries"]
        if ticker in row_by_ticker
    ]
    # 실시간 표시·장중 판정이 기준일 목표의 가격과 판정을 바꾸지 못하게 분리한다.
    target_holdings = [dict(row) for row in holdings]
    target_entries = [dict(row) for row in entries]
    for held in holdings:
        held["is_exit_forecast"] = False
    # 합성 슬리브 몫이 읽는 일별 곡선 — 이 현황을 만든 **같은 실행**의 값이다(§10-6).
    # 장중이면 아래에서 잠정 실행의 곡선(오늘 잠정 봉 포함)으로 바뀐다.
    engine_daily = simulated["daily"]

    if quotes["live"]:
        # ── 장중 실행 — 실시간 가격을 **마지막 봉**으로 붙인 같은 엔진의 잠정 실행이다
        # (AGENTS.md §10-6). 어제 확정 판정의 오늘 시가 체결(시가를 모르면 체결 예정),
        # 오늘 잠정 봉의 재판정, 진입 예정 비중까지 전부 엔진이 낸다 — 화면은 표시만 한다.
        session = str(quotes["traded_at"])[:10]
        session_ts = pd.Timestamp(session)
        live_prices = {t: q["price"] for t, q in quotes["by_ticker"].items() if t in close_df.columns}
        live_opens = {
            t: float(q["open"]) for t, q in quotes["by_ticker"].items() if t in close_df.columns and q.get("open")
        }
        eff_close = effective_close_frame(close_df, live_prices, session_ts)
        eff = momentum_signals.compute_signals(
            {"close": eff_close}, int(settings["short_ma_days"]), int(settings["long_ma_days"])
        )
        eff_entry = momentum_signals.entry_signal(eff_close, eff, settings["entry_vol_mult"])
        live_result = run_slot_backtest(
            months=DEFAULT_BACKTEST_MONTHS,
            start_date=start_date,
            panel={"close": eff_close, "open": effective_close_frame(panel["open"], live_opens, session_ts)},
            entry=eff_entry,
            exit_signal=eff["exit"],
            priority=eff["priority"].fillna(float("-inf")),
            slots=slots,
            name_by=name_by,
            industry_by=industry_by,
            exit_reason="이탈",
            provisional_last_bar=True,
            **market,
        )
        holdings = live_result["open_positions"]
        exited_today = live_result["exited_today"]
        engine_daily = live_result["daily"]
        mark_engine_statuses(holdings, live_result["planned_exits"])

        # 이격·자격 표시를 잠정 봉 기준으로 갱신한다. 실시간 시세가 없는 종목은 판정 불가라
        # 확정값을 유지한다(엔진의 known 규칙과 같다).
        eff_short = eff["short"].loc[session_ts]
        eff_long = eff["long"].loc[session_ts]
        eff_eligible = eff_entry.loc[session_ts]
        eff_vol = momentum_signals.daily_volatility_pct(eff_close).loc[session_ts]
        for row in rows:
            ticker = row["ticker"]
            if ticker not in live_prices:
                continue
            short_value, long_value = eff_short.get(ticker), eff_long.get(ticker)
            if pd.notna(short_value):
                row["short_gap_pct"] = round(float(short_value), 2)
            if pd.notna(long_value):
                row["long_gap_pct"] = round(float(long_value), 2)
            if pd.notna(eff_vol.get(ticker)):
                row["volatility_pct"] = round(float(eff_vol[ticker]), 2)
            row["eligible"] = bool(eff_eligible.get(ticker, False))
            # 고점 대비 — 순위 화면과 같은 실시간 기준(잠정 봉 포함). 확정 기준으로 두면
            # 어제 신고점(⭐)이 오늘 장중에 내리는 중에도 그대로 남는다.
            live_drawdown = drawdown_from_high_pct(eff_close[ticker].dropna())
            if live_drawdown is not None:
                row["high_drawdown_pct"] = round(live_drawdown, 2)
        rows.sort(key=lambda row: row["long_gap_pct"], reverse=True)

        # 진입 예정 — 체결 예정(오늘 시가, fill_date)이 앞자리, 잠정 예정(내일 시가)이 뒷자리.
        entries = [
            {**row_by_ticker[row["ticker"]], "sleeve_weight_pct": row["sleeve_weight_pct"], "fill_date": session}
            for row in live_result["pending_entries"]
            if row["ticker"] in row_by_ticker
        ] + [
            {**row_by_ticker[ticker], "sleeve_weight_pct": live_result["planned_entry_weights"][ticker]}
            for ticker in live_result["planned_entries"]
            if ticker in row_by_ticker
        ]

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
        item["volatility_pct"] = (row or {}).get("volatility_pct")
        item["short_gap_pct"] = (row or {}).get("short_gap_pct")
        item["long_gap_pct"] = (row or {}).get("long_gap_pct")
        item["high_drawdown_pct"] = (row or {}).get("high_drawdown_pct")
        item["new_listing"] = (row or {}).get("new_listing")
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
    # 진입 예정도 포함한다 — 확정 진입 예정 행은 복사본(fill_date 부착)이라 rows 쪽 갱신이
    # 반영되지 않는다. 실계좌에 있으면 어떤 행이든 보유 표시(녹색)가 붙어야 한다.
    for item in [*rows, *holdings, *exited_today, *entries]:
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
        "target_holdings": target_holdings,
        "target_entries": target_entries,
        # 이 현황을 만든 엔진 실행의 일별 곡선 — 합성 슬리브 몫이 같은 실행 결과를 읽는다.
        "daily": engine_daily,
        # 순위는 잠정 자격 기준이라 체결 예정(어제 확정) 종목이 오늘 자격 밖이면 값이 없다.
        "planned_entries": [{**row, "rank": rank_by_ticker.get(row["ticker"])} for row in entries],
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


__all__ = ["current_positions", "load_context", "run_backtest"]
