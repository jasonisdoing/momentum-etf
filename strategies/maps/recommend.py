"""Momentum 전략 추천 생성기."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


from .rules import StrategyRules
from .constants import DECISION_MESSAGES, DECISION_NOTES
from .messages import build_buy_replace_note
from .shared import select_candidates_by_category, sort_decisions_by_order_and_score
from logic.recommend.formatting import load_account_precision
from utils.logger import get_app_logger

logger = get_app_logger()
from utils.data_loader import count_trading_days


def _resolve_entry_price(series: Any, buy_date: Optional[pd.Timestamp]) -> Optional[float]:
    """Return the entry price by locating the first available close on/after the buy date."""

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


def generate_daily_recommendations_for_portfolio(
    account_id: str,
    country_code: str,
    base_date: pd.Timestamp,
    strategy_rules: StrategyRules,
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
) -> List[Dict[str, Any]]:
    """
    주어진 데이터를 기반으로 포트폴리오의 일일 매매 추천를 생성합니다.
    """

    def _format_kr_price(p):
        return f"{int(round(p)):,}"

    def _aud_money_formatter(amount, precision: int):
        # 호주달러 표기는 A$로 표시
        return f"A${amount:,.{precision}f}"

    def _aud_price_formatter(p, precision: int):
        return f"${p:,.{precision}f}"

    # 표시 통화와 정밀도 결정: precision.json(country) 기준 사용
    precision_settings = load_account_precision(account_id) or {}
    qty_precision = int(precision_settings.get("stock_qty_precision", 0))
    price_precision = int(precision_settings.get("stock_price_precision", 0))
    amt_precision = int(precision_settings.get("stock_amt_precision", 0))

    if country_code == "aus":

        def money_formatter(amount):
            return _aud_money_formatter(amount, amt_precision)

        def price_formatter(p):
            return _aud_price_formatter(p, price_precision)

    else:  # 기본(₩) 국가 -> 가격을 정수(또는 설정값)로 표기

        def price_formatter(p):
            # 한국 단가는 기본 정수, 설정에 값이 있으면 적용
            if price_precision and price_precision > 0:
                return f"{p:,.{price_precision}f}"
            return _format_kr_price(p)

    def format_shares(quantity):
        # 모든 국가 공통: precision.json의 수량 정밀도 적용
        if qty_precision and qty_precision > 0:
            return f"{quantity:,.{qty_precision}f}".rstrip("0").rstrip(".")
        return f"{int(round(quantity)):,d}"

    def _format_cooldown_phrase(action: str, last_dt: Optional[pd.Timestamp]) -> str:
        if last_dt is None:
            return DECISION_NOTES["COOLDOWN_GENERIC"].format(days=cooldown_days)
        return DECISION_NOTES["COOLDOWN_WITH_ACTION"].format(days=cooldown_days, action=action, date=last_dt.strftime("%Y-%m-%d"))

    # 전략 설정 로드
    denom = strategy_rules.portfolio_topn
    if denom <= 0:
        raise ValueError(f"'{account_id}' 계좌의 최대 보유 종목 수(portfolio_topn)는 0보다 커야 합니다.")
    replace_threshold = strategy_rules.replace_threshold
    try:
        stop_loss_threshold = -abs(float(denom))
    except (TypeError, ValueError):
        stop_loss_threshold = None

    # 현재 보유 종목의 카테고리 (TBD 제외)
    held_categories = set()
    for tkr in holdings.keys():
        category = etf_meta.get(tkr, {}).get("category")
        if category and category != "TBD":
            held_categories.add(category)

    # 포지션 비중 가이드라인: 모든 국가 동일 규칙 적용 (min_pos는 현재 신규 매수 로직에서 미사용)
    # min_pos = 1.0 / (denom * 2.0)  # 최소 편입 비중
    max_pos = 1.0 / denom  # 목표/최대 비중 # noqa: F841

    # 현재 보유 종목 수 계산
    held_count = len(holdings)

    decisions = []

    base_date_norm = base_date.normalize()
    sell_cooldown_block: Dict[str, Dict[str, Any]] = {}
    buy_cooldown_block: Dict[str, Dict[str, Any]] = {}

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
                        count_trading_days(country_code, last_buy_ts, base_date_norm) - 1,
                        0,
                    )
                    if days_since_buy < cooldown_days:
                        sell_cooldown_block[tkr] = {
                            "last_buy": last_buy_ts,
                            "days_since": days_since_buy,
                        }

            if last_sell is not None:
                last_sell_ts = pd.to_datetime(last_sell).normalize()
                if last_sell_ts <= base_date_norm:
                    days_since_sell = max(
                        count_trading_days(country_code, last_sell_ts, base_date_norm) - 1,
                        0,
                    )
                    if days_since_sell < cooldown_days:
                        buy_cooldown_block[tkr] = {
                            "last_sell": last_sell_ts,
                            "days_since": days_since_sell,
                        }

    evaluation_date = max(base_date.normalize(), pd.Timestamp.now().normalize())

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
                "filter": 0,
                "close": pd.Series(),  # 추가
            }

        price = d.get("price", 0.0)
        score_raw = d.get("score", 0.0)
        score_value: Optional[float]
        if isinstance(score_raw, (int, float)):
            score_value = float(score_raw)
        else:
            try:
                score_value = float(score_raw)
            except (TypeError, ValueError):
                score_value = None

        buy_signal = False
        state = "HOLD" if is_effectively_held else "WAIT"
        phrase = ""
        if price == 0.0 and is_effectively_held:
            phrase = DECISION_NOTES["PRICE_DATA_FAIL"]

        sell_block_info = sell_cooldown_block.get(tkr)
        buy_block_info = buy_cooldown_block.get(tkr)

        buy_date = None
        holding_days = 0
        consecutive_info = consecutive_holding_info.get(tkr)
        buy_date = consecutive_info.get("buy_date") if consecutive_info else None

        if is_effectively_held and buy_date:
            buy_date_norm = pd.to_datetime(buy_date).normalize()
            if buy_date_norm <= evaluation_date:
                holding_days = count_trading_days(
                    country_code,
                    buy_date_norm,
                    evaluation_date,
                )

        holding_return_pct: Optional[float] = None
        if is_effectively_held:
            entry_price = _resolve_entry_price(d.get("close"), buy_date)
            if entry_price and entry_price > 0 and price and price > 0:
                holding_return_pct = ((price / entry_price) - 1.0) * 100.0

        if state == "HOLD":
            price_ma, ma = d["price"], d["s1"]
            if holding_return_pct is not None and stop_loss_threshold is not None and holding_return_pct <= float(stop_loss_threshold):
                state = "CUT_STOPLOSS"
                phrase = DECISION_MESSAGES.get("CUT_STOPLOSS", "손절매도")
            elif not pd.isna(price_ma) and not pd.isna(ma) and price_ma < ma:
                state = "SELL_TREND"
                phrase = DECISION_NOTES["TREND_BREAK"]

            if sell_block_info and state == "SELL_TREND":
                state = "HOLD"
                phrase = _format_cooldown_phrase("최근 매수", sell_block_info.get("last_buy"))

        elif state == "WAIT":
            buy_signal_days_today = d["filter"]
            if buy_signal_days_today > 0:
                buy_signal = True
                if buy_block_info:
                    buy_signal = False
                    phrase = _format_cooldown_phrase("최근 매도", buy_block_info.get("last_sell"))

        amount = price if is_effectively_held else 0.0

        meta = etf_meta.get(tkr) or full_etf_meta.get(tkr, {}) or {}
        display_name = str(meta.get("name") or tkr)
        raw_category = meta.get("category")
        display_category = str(raw_category) if raw_category and str(raw_category).upper() != "TBD" else "-"

        if holding_days == 0 and state in {"BUY", "BUY_REPLACE"}:
            holding_days = 1

        # 일간 수익률 계산
        prev_close = d.get("prev_close", 0.0)
        day_ret = ((price / prev_close) - 1.0) * 100.0 if pd.notna(price) and pd.notna(prev_close) and prev_close > 0 else 0.0
        day_ret = round(day_ret, 2)

        holding_days_display = str(holding_days) if holding_days > 0 else "-"

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
            1 if is_effectively_held else 0,
            amount,
            round(holding_return_pct, 2) if holding_return_pct is not None else 0.0,
            position_weight_pct,
            (f"{d.get('drawdown_from_peak'):.1f}%" if d.get("drawdown_from_peak") is not None else "-"),
            d.get("score"),
            f"{d['filter']}일" if d.get("filter") is not None else "-",
            phrase,
        ]
        current_row[4] = state

        decisions.append(
            {
                "state": state,
                "weight": position_weight_pct,
                "score": score_value if score_value is not None else 0.0,
                "tkr": tkr,
                "row": current_row,
                "buy_signal": buy_signal,
                "sell_cooldown_info": sell_block_info,
                "buy_cooldown_info": buy_block_info,
                "is_held": is_effectively_held,
                "filter": d.get("filter"),
                "recommend_enabled": bool(etf_meta.get(tkr, {}).get("recommend_enabled", True)),
                "hold_return_pct": holding_return_pct,
            }
        )

    universe_tickers = {etf["ticker"] for etf in full_etf_meta.values()}  # Use full_etf_meta for universe

    is_risk_off = regime_info and regime_info.get("is_risk_off", False)
    try:
        risk_off_target_ratio = int(risk_off_equity_ratio)
    except (TypeError, ValueError):
        risk_off_target_ratio = 100
    risk_off_target_ratio = min(100, max(0, risk_off_target_ratio))
    risk_off_effective = is_risk_off and risk_off_target_ratio < 100
    full_risk_off_exit = risk_off_effective and risk_off_target_ratio <= 0
    partial_risk_off = risk_off_effective and risk_off_target_ratio > 0

    # WAIT 후보 목록과 남은 슬롯 수는 모든 시나리오에서 참조되므로 기본값을 미리 정의합니다.
    wait_candidates_raw: List[Dict] = []
    slots_to_fill = denom - held_count

    if risk_off_effective:
        for decision in decisions:
            if decision["state"] == "HOLD":
                decision["state"] = "SELL_REGIME_FILTER"
                decision["row"][4] = "SELL_REGIME_FILTER"

                d_sell = data_by_tkr.get(decision["tkr"])
                if d_sell:
                    note_text = DECISION_NOTES["RISK_OFF_SELL"]
                    if partial_risk_off:
                        note_text = f"{note_text} (목표 {risk_off_target_ratio}%)"
                    decision["row"][-1] = note_text
                decision["risk_off_target_ratio"] = risk_off_target_ratio

            if decision.get("buy_signal") and full_risk_off_exit:
                decision["buy_signal"] = False
                if decision["state"] == "WAIT":
                    original_phrase = decision["row"][-1]
                    if original_phrase and "추세진입" in original_phrase:
                        decision["row"][-1] = f"{DECISION_NOTES['RISK_OFF']} ({original_phrase})"
                    else:
                        decision["row"][-1] = DECISION_NOTES["RISK_OFF"]
    else:
        # 모든 'WAIT' 상태의 매수 후보 목록을 미리 정의합니다.
        wait_candidates_raw = [
            d for d in decisions if d["state"] == "WAIT" and d.get("buy_signal") and d["tkr"] in universe_tickers and d.get("recommend_enabled", True)
        ]

        # 신규 매수 로직: 빈 슬롯이 있을 때 실행
        if slots_to_fill > 0:
            selected_candidates, rejected_candidates = select_candidates_by_category(
                wait_candidates_raw,
                etf_meta,
                held_categories=held_categories,
                max_count=slots_to_fill,
                skip_held_categories=True,
            )

            for cand, reason in rejected_candidates:
                if reason == "category_held":
                    cand["row"][-1] = DECISION_NOTES["CATEGORY_DUP"]

            for cand in selected_candidates:
                cand_category = etf_meta.get(cand["tkr"], {}).get("category")
                # 매수 실행
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
                    else:
                        cand["row"][-1] = DECISION_NOTES["INSUFFICIENT_CASH"]
                else:
                    cand["row"][-1] = DECISION_NOTES["NO_PRICE"]

        # 교체 매매 로직: 포트폴리오에 빈 슬롯이 있더라도, 더 좋은 종목으로 교체할 기회가 있으면 실행
        replacement_candidates, _ = select_candidates_by_category(
            [cand for cand in wait_candidates_raw if cand.get("state") != "BUY"],
            etf_meta,
            held_categories=None,
            max_count=None,
            skip_held_categories=False,
        )

        # 2. 교체 로직 실행
        current_held_stocks = [d for d in decisions if d["state"] == "HOLD"]
        current_held_stocks.sort(key=lambda x: x["score"] if pd.notna(x["score"]) else -float("inf"))

        for best_new in replacement_candidates:
            if not current_held_stocks:
                break

            wait_stock_category = etf_meta.get(best_new["tkr"], {}).get("category")

            # 2-1. 동일 카테고리 보유 종목과 비교
            held_stock_same_category = next(
                (
                    s
                    for s in current_held_stocks
                    if wait_stock_category and wait_stock_category != "TBD" and etf_meta.get(s["tkr"], {}).get("category") == wait_stock_category
                ),
                None,
            )

            ticker_to_sell = None
            if held_stock_same_category:
                # 동일 카테고리 보유 종목이 있으면, 점수만 비교 (임계값 미적용)
                if (
                    pd.notna(best_new["score"])
                    and pd.notna(held_stock_same_category["score"])
                    and best_new["score"] > held_stock_same_category["score"] + replace_threshold
                ):
                    ticker_to_sell = held_stock_same_category["tkr"]
                else:
                    # 점수가 더 높지 않으면 교체하지 않음. 루프는 계속 진행하여 다른 카테고리 교체 가능성 확인
                    pass
            else:
                # 2-2. 동일 카테고리가 없으면, 가장 약한 보유 종목과 비교 (임계값 적용)
                if current_held_stocks:
                    weakest_held = current_held_stocks[0]
                    if (
                        pd.notna(best_new["score"])
                        and pd.notna(weakest_held["score"])
                        and best_new["score"] > weakest_held["score"] + replace_threshold
                    ):
                        ticker_to_sell = weakest_held["tkr"]

            if ticker_to_sell:
                sell_block_for_candidate = sell_cooldown_block.get(ticker_to_sell)
                if sell_block_for_candidate and cooldown_days > 0:
                    blocked_name = etf_meta.get(ticker_to_sell, {}).get("name") or ticker_to_sell
                    best_new["state"], best_new["row"][4] = "WAIT", "WAIT"
                    best_new["row"][-1] = f"쿨다운 {cooldown_days}일 대기중 - {blocked_name}"
                    best_new["buy_signal"] = False
                    continue

                # 3. 교체 실행
                d_weakest = data_by_tkr.get(ticker_to_sell)
                if d_weakest:
                    sell_phrase = DECISION_NOTES["REPLACE_SELL"]

                    for d_item in decisions:
                        if d_item["tkr"] == ticker_to_sell:
                            d_item["state"], d_item["row"][4], d_item["row"][-1] = (
                                "SELL_REPLACE",
                                "SELL_REPLACE",
                                sell_phrase,
                            )
                            break

                # (b) 매수 추천 생성
                best_new["state"], best_new["row"][4] = "BUY_REPLACE", "BUY_REPLACE"
                buy_price = float(data_by_tkr.get(best_new["tkr"], {}).get("price", 0))
                if buy_price > 0:
                    best_new["row"][-1] = build_buy_replace_note(
                        ticker_to_sell,
                        full_etf_meta.get(ticker_to_sell, {}).get("name", ticker_to_sell),
                    )
                else:
                    best_new["row"][-1] = f"{ticker_to_sell}(을)를 대체 (가격정보 없음)"
                current_held_stocks = [s for s in current_held_stocks if s["tkr"] != ticker_to_sell]
                best_new_as_held = best_new.copy()
                best_new_as_held["state"] = "HOLD"
                current_held_stocks.append(best_new_as_held)
                current_held_stocks.sort(key=lambda x: x["score"] if pd.notna(x["score"]) else -float("inf"))

    SELL_STATE_SET = {"SELL_TREND", "SELL_REPLACE", "CUT_STOPLOSS", "SELL_REGIME_FILTER"}
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
                d["row"][-1] = _format_cooldown_phrase("최근 매수", sell_info.get("last_buy"))
                d["buy_signal"] = False

            if buy_info and d["state"] in BUY_STATE_SET:
                d["state"] = "WAIT"
                d["row"][4] = "WAIT"
                d["row"][-1] = _format_cooldown_phrase("최근 매도", buy_info.get("last_sell"))
                d["buy_signal"] = False

    # 최종 목록에서는 WAIT 종목도 모두 노출해, 카테고리별 대표만 보여주는 형태를 피한다.
    final_decisions = list(decisions)

    # 포트폴리오가 가득 찼을 때, 매수 추천되지 않은 WAIT 종목에 사유 기록
    if slots_to_fill <= 0:
        held_categories = {etf_meta.get(d["tkr"], {}).get("category") for d in decisions if d["state"] == "HOLD"}
        for d in final_decisions:
            if d["state"] == "WAIT" and d.get("buy_signal"):
                # 이미 교체매매 로직에서 사유가 기록된 경우는 제외
                if not d["row"][-1]:
                    wait_category = etf_meta.get(d["tkr"], {}).get("category")
                    if wait_category and wait_category != "TBD" and wait_category in held_categories:
                        # 동일 카테고리 보유로 인한 중복
                        d["row"][-1] = DECISION_NOTES["CATEGORY_DUP"]
                    else:
                        # 그 외의 경우 (점수 미달 등)
                        d["row"][-1] = DECISION_NOTES["PORTFOLIO_FULL"]

    # 최종 정렬
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
