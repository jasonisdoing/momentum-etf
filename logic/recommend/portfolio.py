"""전략 중립적인 포트폴리오 추천 생성 로직."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Set

import pandas as pd

# 순환 import 방지를 위해 TYPE_CHECKING 사용
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from strategies.maps.rules import StrategyRules
from utils.logger import get_app_logger
from utils.data_loader import count_trading_days

logger = get_app_logger()


def _normalize_category_value(category: Optional[str]) -> Optional[str]:
    """카테고리 값을 정규화합니다."""
    if category is None:
        return None
    category_str = str(category).strip()
    if not category_str:
        return None
    return category_str.upper()


def _resolve_entry_price(series: Any, buy_date: Optional[pd.Timestamp]) -> Optional[float]:
    """매수일 이후 첫 종가를 반환합니다."""
    if buy_date is None:
        return None

    if not isinstance(series, pd.Series) or series.empty:
        return None

    try:
        buy_ts = pd.to_datetime(buy_date).normalize()
    except Exception:
        return None

    cleaned = series.dropna().copy()
    if cleaned.empty:
        return None

    try:
        cleaned.index = pd.to_datetime(cleaned.index).normalize()
    except Exception:
        return None

    future_slice = cleaned.loc[cleaned.index >= buy_ts]
    if future_slice.empty:
        return float(cleaned.iloc[-1])

    entry_val = future_slice.iloc[0]
    try:
        return float(entry_val)
    except (TypeError, ValueError):
        return None


def _calculate_cooldown_blocks(
    trade_cooldown_info: Dict[str, Dict[str, Optional[pd.Timestamp]]],
    cooldown_days: int,
    base_date: pd.Timestamp,
    country_code: str,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """쿨다운 블록을 계산합니다."""
    sell_cooldown_block: Dict[str, Dict[str, Any]] = {}
    buy_cooldown_block: Dict[str, Dict[str, Any]] = {}
    base_date_norm = base_date.normalize()

    if cooldown_days and cooldown_days > 0:
        for tkr, trade_info in (trade_cooldown_info or {}).items():
            if not isinstance(trade_info, dict):
                continue

            last_buy = trade_info.get("last_buy")
            last_sell = trade_info.get("last_sell")

            if last_buy is not None:
                last_buy_ts = pd.to_datetime(last_buy).normalize()
                if last_buy_ts <= base_date_norm:
                    days_since_buy = max(
                        count_trading_days(country_code, last_buy_ts, base_date_norm),
                        0,
                    )
                    if days_since_buy < cooldown_days:
                        logger.info(
                            f"[COOLDOWN BLOCK] {tkr}: last_buy={last_buy_ts.strftime('%Y-%m-%d')}, base_date={base_date_norm.strftime('%Y-%m-%d')}, days_since={days_since_buy}, cooldown_days={cooldown_days}"
                        )
                        sell_cooldown_block[tkr] = {
                            "last_buy": last_buy_ts,
                            "days_since": days_since_buy,
                        }

            if last_sell is not None:
                last_sell_ts = pd.to_datetime(last_sell).normalize()
                if last_sell_ts <= base_date_norm:
                    days_since_sell = max(
                        count_trading_days(country_code, last_sell_ts, base_date_norm),
                        0,
                    )
                    if days_since_sell < cooldown_days:
                        buy_cooldown_block[tkr] = {
                            "last_sell": last_sell_ts,
                            "days_since": days_since_sell,
                        }

    return sell_cooldown_block, buy_cooldown_block


def _parse_score_value(score_raw: Any) -> Optional[float]:
    """점수 값을 파싱합니다."""
    if isinstance(score_raw, (int, float)):
        return float(score_raw)
    try:
        return float(score_raw)
    except (TypeError, ValueError):
        return None


def _create_decision_entry(
    tkr: str,
    name: str,
    data: Dict[str, Any],
    is_held: bool,
    holdings: Dict,
    etf_meta: Dict,
    full_etf_meta: Dict,
    consecutive_holding_info: Dict,
    sell_cooldown_block: Dict,
    buy_cooldown_block: Dict,
    base_date: pd.Timestamp,
    country_code: str,
    current_equity: float,
    stop_loss_threshold: Optional[float],
    rsi_sell_threshold: float = 10.0,
) -> Dict[str, Any]:
    """개별 종목의 의사결정 엔트리를 생성합니다."""
    # 순환 import 방지
    from strategies.maps.constants import DECISION_MESSAGES, DECISION_NOTES

    price = data.get("price", 0.0)
    score_value = _parse_score_value(data.get("score", 0.0))
    rsi_score_value = _parse_score_value(data.get("rsi_score", 0.0))

    buy_signal = False
    state = "HOLD" if is_held else "WAIT"
    phrase = ""

    if price == 0.0 and is_held:
        phrase = DECISION_NOTES["PRICE_DATA_FAIL"]

    sell_block_info = sell_cooldown_block.get(tkr)
    buy_block_info = buy_cooldown_block.get(tkr)

    # 보유일 계산
    buy_date = None
    holding_days = 0
    consecutive_info = consecutive_holding_info.get(tkr)
    buy_date = consecutive_info.get("buy_date") if consecutive_info else None

    # DEBUG: 쿨다운 문제 디버깅
    if sell_block_info and tkr == "473640":
        logger.info(
            f"[DEBUG 473640] buy_date(consecutive)={buy_date}, last_buy(cooldown)={sell_block_info.get('last_buy')}, days_since={sell_block_info.get('days_since')}"
        )

    evaluation_date = max(base_date.normalize(), pd.Timestamp.now().normalize())

    if is_held and buy_date:
        buy_date_norm = pd.to_datetime(buy_date).normalize()
        if buy_date_norm <= evaluation_date:
            holding_days = count_trading_days(country_code, buy_date_norm, evaluation_date)

    # 보유 수익률 계산
    holding_return_pct: Optional[float] = None
    if is_held:
        entry_price = _resolve_entry_price(data.get("close"), buy_date)
        if entry_price and entry_price > 0 and price and price > 0:
            holding_return_pct = ((price / entry_price) - 1.0) * 100.0

    # 매매 의사결정
    if state == "HOLD":
        price_ma, ma = data["price"], data["s1"]

        # RSI 과매수 매도 조건 체크
        if holding_return_pct is not None and stop_loss_threshold is not None and holding_return_pct <= float(stop_loss_threshold):
            state = "CUT_STOPLOSS"
            phrase = DECISION_MESSAGES.get("CUT_STOPLOSS", "손절매도")
        elif rsi_score_value <= rsi_sell_threshold:
            state = "SELL_RSI"
            phrase = f"{DECISION_MESSAGES.get('SELL_RSI', 'RSI 과매수 매도')} (RSI점수: {rsi_score_value:.1f})"
        elif not pd.isna(price_ma) and not pd.isna(ma) and price_ma < ma:
            state = "SELL_TREND"
            phrase = DECISION_NOTES["TREND_BREAK"]

        if sell_block_info and state in ("SELL_TREND", "SELL_RSI"):
            state = "HOLD"
            days_since = sell_block_info.get("days_since", 0)
            last_buy = sell_block_info.get("last_buy")
            if last_buy:
                last_buy_str = last_buy.strftime("%m/%d")
                phrase = f"쿨다운 대기중 (매수일: {last_buy_str}, {days_since}일 경과)"
            else:
                phrase = f"쿨다운 대기중 ({days_since}일 경과)"

    elif state == "WAIT":
        # 점수 기반 매수 시그널 판단
        from logic.common import has_buy_signal

        score_value = data.get("score", 0.0)
        if has_buy_signal(score_value):
            buy_signal = True
            if buy_block_info:
                buy_signal = False
                phrase = "쿨다운 대기중"

    # 메타 정보
    meta = etf_meta.get(tkr) or full_etf_meta.get(tkr, {}) or {}
    display_name = str(meta.get("name") or tkr)
    raw_category = meta.get("category")
    display_category = str(raw_category) if raw_category and str(raw_category).upper() != "TBD" else "-"

    if holding_days == 0 and state in {"BUY", "BUY_REPLACE"}:
        holding_days = 1

    # 일간 수익률
    prev_close = data.get("prev_close", 0.0)
    day_ret = ((price / prev_close) - 1.0) * 100.0 if pd.notna(price) and pd.notna(prev_close) and prev_close > 0 else 0.0
    day_ret = round(day_ret, 2)

    holding_days_display = str(holding_days) if holding_days > 0 else "-"
    amount = price if is_held else 0.0
    position_weight_pct = (amount / current_equity) * 100.0 if current_equity > 0 else 0.0
    position_weight_pct = round(position_weight_pct, 2)

    current_row = [
        0,
        tkr,
        display_name,
        display_category,
        state,
        holding_days_display,
        price,
        day_ret,
        1 if is_held else 0,
        amount,
        round(holding_return_pct, 2) if holding_return_pct is not None else 0.0,
        position_weight_pct,
        (f"{data.get('drawdown_from_peak'):.1f}%" if data.get("drawdown_from_peak") is not None else "-"),
        data.get("score"),
        f"{data['filter']}일" if data.get("filter") is not None else "-",
        phrase,
    ]
    current_row[4] = state

    return {
        "state": state,
        "weight": position_weight_pct,
        "score": score_value if score_value is not None else 0.0,
        "rsi_score": rsi_score_value if rsi_score_value is not None else 0.0,
        "tkr": tkr,
        "row": current_row,
        "buy_signal": buy_signal,
        "sell_cooldown_info": sell_block_info,
        "buy_cooldown_info": buy_block_info,
        "is_held": is_held,
        "filter": data.get("filter"),
        "recommend_enabled": bool(etf_meta.get(tkr, {}).get("recommend_enabled", True)),
        "hold_return_pct": holding_return_pct,
    }


def generate_daily_recommendations_for_portfolio(
    account_id: str,
    country_code: str,
    base_date: pd.Timestamp,
    strategy_rules: Any,  # StrategyRules 타입 (순환 import 방지)
    data_by_tkr: Dict[str, Any],
    holdings: Dict[str, Dict[str, float]],
    etf_meta: Dict[str, Any],
    full_etf_meta: Dict[str, Any],
    regime_info: Optional[Dict],
    current_equity: float,
    total_cash: float,
    pairs: List[Tuple[str, str]],
    consecutive_holding_info: Dict[str, Dict],
    trade_cooldown_info: Dict[str, Dict[str, Optional[pd.Timestamp]]],
    cooldown_days: int,
    risk_off_equity_ratio: int = 100,
    rsi_sell_threshold: float = 10.0,
) -> List[Dict[str, Any]]:
    """
    주어진 데이터를 기반으로 포트폴리오의 일일 매매 추천을 생성합니다.

    이 함수는 전략 중립적(strategy-agnostic)입니다.
    data_by_tkr에 포함된 모든 전략의 점수를 사용하여 포트폴리오 의사결정을 수행합니다.
    """
    # 순환 import 방지를 위해 함수 내부에서 import
    from strategies.maps.constants import DECISION_MESSAGES, DECISION_NOTES
    from strategies.maps.messages import build_buy_replace_note
    from logic.common import select_candidates_by_category, sort_decisions_by_order_and_score

    # 전략 설정
    denom = strategy_rules.portfolio_topn
    if denom <= 0:
        raise ValueError(f"'{account_id}' 계좌의 최대 보유 종목 수(portfolio_topn)는 0보다 커야 합니다.")

    replace_threshold = strategy_rules.replace_threshold
    try:
        stop_loss_threshold = -abs(float(denom))
    except (TypeError, ValueError):
        stop_loss_threshold = None

    # 핵심 보유 종목 (강제 보유, TOPN 포함)
    core_holdings_tickers = set(strategy_rules.core_holdings or [])

    # Universe 유효성 검증
    universe_tickers_set = {ticker for ticker, _ in pairs}
    invalid_core_tickers = core_holdings_tickers - universe_tickers_set
    if invalid_core_tickers:
        logger.warning(f"[{account_id.upper()}] CORE_HOLDINGS에 Universe에 없는 종목이 포함됨: {invalid_core_tickers}")

    # 유효한 핵심 보유 종목만 사용
    valid_core_holdings = core_holdings_tickers & universe_tickers_set
    if valid_core_holdings:
        logger.info(f"[{account_id.upper()}] 핵심 보유 종목 (TOPN 포함): {sorted(valid_core_holdings)}")

    # 현재 보유 종목의 카테고리
    held_categories = set()
    held_category_keys = set()
    for tkr in holdings.keys():
        category = etf_meta.get(tkr, {}).get("category")
        if category and category != "TBD":
            held_categories.add(category)
            normalized = _normalize_category_value(category)
            if normalized:
                held_category_keys.add(normalized)

    decisions = []

    # 쿨다운 블록 계산
    sell_cooldown_block, buy_cooldown_block = _calculate_cooldown_blocks(trade_cooldown_info, cooldown_days, base_date, country_code)

    # 각 종목에 대한 의사결정 생성
    for tkr, name in pairs:
        d = data_by_tkr.get(tkr)
        is_effectively_held = tkr in holdings

        if not d and not is_effectively_held:
            continue

        if not d:
            d = {
                "price": 0.0,
                "prev_close": 0.0,
                "s1": float("nan"),
                "s2": float("nan"),
                "score": 0.0,
                "rsi_score": 0.0,
                "filter": 0,
                "close": pd.Series(),
            }

        decision = _create_decision_entry(
            tkr,
            name,
            d,
            is_effectively_held,
            holdings,
            etf_meta,
            full_etf_meta,
            consecutive_holding_info,
            sell_cooldown_block,
            buy_cooldown_block,
            base_date,
            country_code,
            current_equity,
            stop_loss_threshold,
            rsi_sell_threshold,
        )
        decisions.append(decision)

    # 1. 핵심 보유 종목 매도 신호 무시 (강제 HOLD_CORE)
    for decision in decisions:
        ticker = decision["tkr"]
        if ticker in valid_core_holdings:
            # 매도 신호를 HOLD_CORE로 강제 변경
            if decision["state"] in {"SELL_TREND", "SELL_RSI", "CUT_STOPLOSS", "SELL_REPLACE"}:
                decision["state"] = "HOLD_CORE"
                decision["row"][4] = "HOLD_CORE"
                decision["row"][-1] = DECISION_MESSAGES.get("HOLD_CORE", "🔒 핵심 보유")
            # 이미 보유 중인 핵심 종목도 HOLD_CORE로 표시
            elif decision["state"] == "HOLD":
                decision["state"] = "HOLD_CORE"
                decision["row"][4] = "HOLD_CORE"
                if not decision["row"][-1] or decision["row"][-1] == "":
                    decision["row"][-1] = DECISION_MESSAGES.get("HOLD_CORE", "🔒 핵심 보유")

    # 2. 핵심 보유 종목 미보유 시 자동 매수
    for core_ticker in valid_core_holdings:
        if core_ticker not in holdings:
            # 이미 decision이 있는지 확인
            existing_decision = next((d for d in decisions if d["tkr"] == core_ticker), None)
            if existing_decision:
                # WAIT 상태를 BUY로 변경
                if existing_decision["state"] == "WAIT":
                    existing_decision["state"] = "BUY"
                    existing_decision["row"][4] = "BUY"
                    existing_decision["row"][-1] = "🔒 핵심 보유 (자동 매수)"
                    existing_decision["buy_signal"] = True
            else:
                # decision이 없으면 새로 생성 (data_by_tkr에 있는 경우만)
                if core_ticker in data_by_tkr:
                    core_data = data_by_tkr[core_ticker]
                    core_name = etf_meta.get(core_ticker, {}).get("name") or core_ticker
                    core_decision = _create_decision_entry(
                        core_ticker,
                        core_name,
                        core_data,
                        False,  # is_held
                        holdings,
                        etf_meta,
                        full_etf_meta,
                        consecutive_holding_info,
                        sell_cooldown_block,
                        buy_cooldown_block,
                        base_date,
                        country_code,
                        current_equity,
                        stop_loss_threshold,
                        rsi_sell_threshold,
                    )
                    core_decision["state"] = "BUY"
                    core_decision["row"][4] = "BUY"
                    core_decision["row"][-1] = "🔒 핵심 보유 (자동 매수)"
                    core_decision["buy_signal"] = True
                    decisions.append(core_decision)

    universe_tickers = {etf["ticker"] for etf in full_etf_meta.values()}

    # 리스크 오프 처리
    is_risk_off = regime_info and regime_info.get("is_risk_off", False)
    if risk_off_equity_ratio is None:
        raise ValueError("risk_off_equity_ratio 값이 필요합니다.")

    try:
        risk_off_target_ratio = int(risk_off_equity_ratio)
    except (TypeError, ValueError) as exc:
        raise ValueError("risk_off_equity_ratio 값은 정수여야 합니다.") from exc

    if not (0 <= risk_off_target_ratio <= 100):
        raise ValueError("risk_off_equity_ratio 값은 0부터 100 사이여야 합니다.")

    risk_off_effective = is_risk_off and risk_off_target_ratio < 100
    full_risk_off_exit = risk_off_effective and risk_off_target_ratio <= 0
    partial_risk_off = risk_off_effective and risk_off_target_ratio > 0

    wait_candidates_raw: List[Dict] = [
        d for d in decisions if d["state"] == "WAIT" and d.get("buy_signal") and d["tkr"] in universe_tickers and d.get("recommend_enabled", True)
    ]

    # 점수순으로 정렬 (높은 점수가 우선)
    wait_candidates_raw.sort(key=lambda x: x.get("score", 0.0), reverse=True)

    # SELL_RSI로 매도하는 카테고리 추적 (같은 날 매수 금지)
    # 매도 전에도 RSI 과매수 경고가 있는 보유 종목의 카테고리는 차단
    sell_rsi_categories_today: Set[str] = set()
    for d in decisions:
        # 1. 이미 SELL_RSI 상태인 경우
        if d["state"] == "SELL_RSI":
            category = etf_meta.get(d["tkr"], {}).get("category")
            if category and category != "TBD":
                sell_rsi_categories_today.add(category)
                logger.info(f"[SELL_RSI CATEGORY] {d['tkr']} 매도로 인해 '{category}' 카테고리 매수 차단")
        # 2. 보유 중이지만 RSI 과매수 경고가 있는 경우 (매도 전 예방)
        elif d["state"] in {"HOLD", "HOLD_CORE"} and d.get("rsi_score", 100.0) <= rsi_sell_threshold:
            category = etf_meta.get(d["tkr"], {}).get("category")
            if category and category != "TBD":
                sell_rsi_categories_today.add(category)
                logger.info(
                    f"[RSI WARNING CATEGORY] {d['tkr']} RSI 과매수 경고로 '{category}' 카테고리 매수 차단 (RSI점수: {d.get('rsi_score', 0):.1f})"
                )

    # 실제 보유 중인 종목 수 계산 (CORE 포함)
    # HOLD + HOLD_CORE = 전체 보유 종목
    held_count = sum(1 for d in decisions if d["state"] in {"HOLD", "HOLD_CORE"})
    # 추가 매수 가능 슬롯 = TOPN - 전체 보유 수
    slots_to_fill = denom - held_count

    if risk_off_effective:
        for decision in decisions:
            decision["risk_off_target_ratio"] = risk_off_target_ratio
            if decision["state"] == "HOLD":
                note_text = DECISION_NOTES["RISK_OFF_TRIM"]
                if partial_risk_off:
                    note_text = f"{note_text} (보유목표 {risk_off_target_ratio}%)"
                decision["row"][-1] = note_text
                decision["row"][4] = "HOLD"

            if decision.get("buy_signal") and full_risk_off_exit:
                decision["buy_signal"] = False
                if decision["state"] == "WAIT":
                    decision["row"][-1] = f"{DECISION_NOTES['RISK_OFF_TRIM']} (보유목표 {risk_off_target_ratio}%)"

    # 신규 매수 로직 (리스크 오프 상태에서도 허용, 비중만 제한)
    if slots_to_fill > 0:
        # 매도 예정 종목을 제외한 held_categories 재계산
        from logic.common import get_held_categories_excluding_sells

        held_categories_for_buy = get_held_categories_excluding_sells(
            decisions,
            get_category_func=lambda d: etf_meta.get(d["tkr"], {}).get("category"),
            get_state_func=lambda d: d["state"],
            get_ticker_func=lambda d: d["tkr"],
            holdings=set(holdings.keys()),
        )

        # 점수가 양수인 모든 매수 시그널 종목을 순서대로 시도 (이미 점수순 정렬됨)
        successful_buys = 0
        for cand in wait_candidates_raw:
            if successful_buys >= slots_to_fill:
                break

            cand_category = etf_meta.get(cand["tkr"], {}).get("category")
            cand_category_key = _normalize_category_value(cand_category)

            # 카테고리 중복 체크
            if cand_category and cand_category != "TBD" and cand_category in held_categories_for_buy:
                continue

            # SELL_RSI로 매도한 카테고리는 같은 날 매수 금지
            if cand_category and cand_category != "TBD" and cand_category in sell_rsi_categories_today:
                logger.info(f"[BUY BLOCKED] {cand['tkr']} 매수 차단 - '{cand_category}' 카테고리가 SELL_RSI로 매도됨")
                cand["state"], cand["row"][4] = "WAIT", "WAIT"
                cand["row"][-1] = f"RSI 과매수 매도 카테고리 ({cand_category})"
                cand["buy_signal"] = False
                continue

            # RSI 과매수 종목 매수 차단
            cand_rsi_score = cand.get("rsi_score", 100.0)

            if cand_rsi_score <= rsi_sell_threshold:
                cand["state"], cand["row"][4] = "WAIT", "WAIT"
                cand["row"][-1] = f"RSI 과매수 (RSI점수: {cand_rsi_score:.1f})"
                cand["buy_signal"] = False
                continue

            cand["state"], cand["row"][4] = "BUY", "BUY"
            buy_price = float(data_by_tkr.get(cand["tkr"], {}).get("price", 0))
            if buy_price > 0:
                budget = (current_equity / denom) if denom > 0 else 0
                if budget > total_cash:
                    budget = total_cash

                if budget > 0:
                    cand["row"][-1] = DECISION_MESSAGES["NEW_BUY"]
                    if cand_category and cand_category != "TBD":
                        held_categories.add(cand_category)
                        held_categories_for_buy.add(cand_category)
                        if cand_category_key:
                            held_category_keys.add(cand_category_key)
                    successful_buys += 1
                else:
                    cand["row"][-1] = DECISION_NOTES["INSUFFICIENT_CASH"]
            else:
                cand["row"][-1] = DECISION_NOTES["NO_PRICE"]

    # 교체 매매 로직
    replacement_candidates, _ = select_candidates_by_category(
        [cand for cand in wait_candidates_raw if cand.get("state") != "BUY"],
        etf_meta,
        held_categories=None,
        max_count=None,
        skip_held_categories=False,
    )

    # 핵심 보유 종목은 교체 매매 대상에서 제외 (HOLD만 대상, HOLD_CORE 제외)
    current_held_stocks = [d for d in decisions if d["state"] == "HOLD"]
    # MAPS 점수 사용
    current_held_stocks.sort(key=lambda x: x.get("score", 0.0) if pd.notna(x.get("score")) else -float("inf"))

    for best_new in replacement_candidates:
        if not current_held_stocks:
            break

        # RSI 과매수 종목 교체 매수 차단
        best_new_rsi_score = best_new.get("rsi_score", 100.0)
        if best_new_rsi_score <= rsi_sell_threshold:
            best_new["state"], best_new["row"][4] = "WAIT", "WAIT"
            best_new["row"][-1] = f"RSI 과매수 (RSI점수: {best_new_rsi_score:.1f})"
            best_new["buy_signal"] = False
            continue

        wait_stock_category = etf_meta.get(best_new["tkr"], {}).get("category")
        wait_stock_category_key = _normalize_category_value(wait_stock_category)

        # 핵심 보유 종목의 카테고리는 교체 매수 차단
        core_holdings_categories = {
            etf_meta.get(ticker, {}).get("category")
            for ticker in valid_core_holdings
            if etf_meta.get(ticker, {}).get("category") and etf_meta.get(ticker, {}).get("category") != "TBD"
        }
        if wait_stock_category and wait_stock_category != "TBD" and wait_stock_category in core_holdings_categories:
            best_new["state"], best_new["row"][4] = "WAIT", "WAIT"
            best_new["row"][-1] = f"핵심 보유 카테고리 ({wait_stock_category})"
            best_new["buy_signal"] = False
            continue

        held_stock_same_category = next(
            (
                s
                for s in current_held_stocks
                if wait_stock_category and wait_stock_category != "TBD" and etf_meta.get(s["tkr"], {}).get("category") == wait_stock_category
            ),
            None,
        )

        # SELL_RSI로 매도한 카테고리는 같은 날 교체 매수 금지
        if wait_stock_category and wait_stock_category != "TBD" and wait_stock_category in sell_rsi_categories_today:
            best_new["state"], best_new["row"][4] = "WAIT", "WAIT"
            best_new["row"][-1] = f"RSI 과매수 매도 카테고리 ({wait_stock_category})"
            best_new["buy_signal"] = False
            continue

        ticker_to_sell = None
        # MAPS 점수 사용
        best_new_score = best_new.get("score")

        if held_stock_same_category:
            held_score = held_stock_same_category.get("score")

            if pd.notna(best_new_score) and pd.notna(held_score) and best_new_score > held_score + replace_threshold:
                ticker_to_sell = held_stock_same_category["tkr"]
        else:
            if current_held_stocks:
                weakest_held = current_held_stocks[0]
                weakest_score = weakest_held.get("score")

                if pd.notna(best_new_score) and pd.notna(weakest_score) and best_new_score > weakest_score + replace_threshold:
                    ticker_to_sell = weakest_held["tkr"]

        if ticker_to_sell:
            sell_block_for_candidate = sell_cooldown_block.get(ticker_to_sell)
            logger.info(
                f"[REPLACE CHECK] ticker_to_sell={ticker_to_sell}, "
                f"sell_block_for_candidate={sell_block_for_candidate}, "
                f"cooldown_days={cooldown_days}"
            )
            if sell_block_for_candidate and cooldown_days > 0:
                blocked_name = etf_meta.get(ticker_to_sell, {}).get("name") or ticker_to_sell
                best_new["state"], best_new["row"][4] = "WAIT", "WAIT"
                best_new["row"][-1] = f"쿨다운 {cooldown_days}일 대기중 - {blocked_name}"
                best_new["buy_signal"] = False
                logger.info(f"[REPLACE BLOCKED] {ticker_to_sell} 교체 차단 (쿨다운)")
                continue

            d_weakest = data_by_tkr.get(ticker_to_sell)
            if d_weakest:
                replacement_name = best_new.get("row", [None, None, None])[2]
                if not replacement_name:
                    replacement_name = (
                        etf_meta.get(best_new["tkr"], {}).get("name") or full_etf_meta.get(best_new["tkr"], {}).get("name") or best_new["tkr"]
                    )
                sell_base = DECISION_MESSAGES.get("SELL_REPLACE", DECISION_NOTES.get("REPLACE_SELL", "교체 매도"))

                for d_item in decisions:
                    if d_item["tkr"] == ticker_to_sell:
                        pl_raw = d_item.get("hold_return_pct")
                        try:
                            pl_pct = float(pl_raw)
                        except (TypeError, ValueError):
                            pl_pct = 0.0
                        sell_phrase = f"{sell_base} 손익률 {pl_pct:+.2f}% - {replacement_name}({best_new['tkr']}) 교체"
                        d_item["state"], d_item["row"][4], d_item["row"][-1] = ("SELL_REPLACE", "SELL_REPLACE", sell_phrase)
                        break

            best_new["state"], best_new["row"][4] = "BUY_REPLACE", "BUY_REPLACE"
            buy_price = float(data_by_tkr.get(best_new["tkr"], {}).get("price", 0))
            if buy_price > 0:
                best_new["row"][-1] = build_buy_replace_note(
                    ticker_to_sell,
                    full_etf_meta.get(ticker_to_sell, {}).get("name", ticker_to_sell),
                )
            else:
                best_new["row"][-1] = f"{ticker_to_sell}(을)를 대체 (가격정보 없음)"

            sold_category = etf_meta.get(ticker_to_sell, {}).get("category")
            if sold_category and sold_category in held_categories:
                held_categories.discard(sold_category)
                sold_key = _normalize_category_value(sold_category)
                if sold_key:
                    held_category_keys.discard(sold_key)
            if wait_stock_category and wait_stock_category != "TBD":
                held_categories.add(wait_stock_category)
            if wait_stock_category_key:
                held_category_keys.add(wait_stock_category_key)

            current_held_stocks = [s for s in current_held_stocks if s["tkr"] != ticker_to_sell]
            best_new_as_held = best_new.copy()
            best_new_as_held["state"] = "HOLD"
            current_held_stocks.append(best_new_as_held)
            # MAPS 점수 사용
            current_held_stocks.sort(key=lambda x: x.get("score", 0.0) if pd.notna(x.get("score")) else -float("inf"))

    # 쿨다운 최종 적용
    SELL_STATE_SET = {"SELL_TREND", "SELL_REPLACE", "CUT_STOPLOSS", "SELL_RSI"}
    BUY_STATE_SET = {"BUY", "BUY_REPLACE"}

    if cooldown_days and cooldown_days > 0:
        for d in decisions:
            tkr = d["tkr"]
            sell_info = sell_cooldown_block.get(tkr)
            buy_info = buy_cooldown_block.get(tkr)

            if sell_info and d["state"] in SELL_STATE_SET:
                if d["state"] == "SELL_REPLACE":
                    continue
                d["state"] = "HOLD"
                d["row"][4] = "HOLD"
                phrase_str = str(d["row"][-1] or "")
                if "시장위험회피" not in phrase_str and "시장 위험 회피" not in phrase_str:
                    d["row"][-1] = "쿨다운 대기중"
                d["buy_signal"] = False

            if buy_info and d["state"] in BUY_STATE_SET:
                d["state"] = "WAIT"
                d["row"][4] = "WAIT"
                phrase_str = str(d["row"][-1] or "")
                if "시장위험회피" not in phrase_str and "시장 위험 회피" not in phrase_str:
                    d["row"][-1] = "쿨다운 대기중"
                d["buy_signal"] = False

    final_decisions = list(decisions)

    # RSI 과매수 종목 문구 추가 (WAIT 상태)
    for d in final_decisions:
        if d["state"] == "WAIT" and d.get("buy_signal"):
            rsi_score = d.get("rsi_score", 100.0)
            if rsi_score <= rsi_sell_threshold:
                if not d["row"][-1] or d["row"][-1] == "":
                    d["row"][-1] = f"RSI 과매수 (RSI점수: {rsi_score:.1f})"

    # 포트폴리오 가득 찬 경우 처리
    if slots_to_fill <= 0:
        for d in final_decisions:
            if d["state"] == "WAIT" and d.get("buy_signal"):
                if not d["row"][-1]:
                    d["row"][-1] = DECISION_NOTES["PORTFOLIO_FULL"]

    sort_decisions_by_order_and_score(final_decisions)
    return final_decisions


def safe_generate_daily_recommendations_for_portfolio(*args, **kwargs) -> List[Dict[str, Any]]:
    """안전하게 generate_daily_recommendations_for_portfolio 함수를 실행합니다."""
    try:
        return generate_daily_recommendations_for_portfolio(*args, **kwargs)
    except Exception as e:
        logger.exception("generate_daily_recommendations_for_portfolio 실행 중 오류: %s", e)
        return []


__all__ = [
    "generate_daily_recommendations_for_portfolio",
    "safe_generate_daily_recommendations_for_portfolio",
]
