"""
전략 중립적인 포트폴리오 추천 생성 로직 (Refactored).

`logic/backtest/portfolio.py`의 구조를 따르며, 단일 일자(추천일)에 대한 의사결정을 수행합니다.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

import config
from logic.common import (
    check_buy_candidate_filters,
    get_held_categories_excluding_sells,
    is_category_exception,
    select_candidates_by_category,
    sort_decisions_by_order_and_score,
    validate_core_holdings,
    validate_portfolio_topn,
)
from logic.common.notes import format_min_score_phrase
from strategies.maps.constants import DECISION_MESSAGES, DECISION_NOTES
from strategies.maps.evaluator import StrategyEvaluator
from strategies.maps.messages import build_buy_replace_note
from utils.data_loader import count_trading_days
from utils.logger import get_app_logger

if TYPE_CHECKING:
    from strategies.maps.rules import StrategyRules

logger = get_app_logger()


from logic.common.portfolio import calculate_cooldown_blocks
from logic.common.price import resolve_entry_price, resolve_highest_price_since_buy


def _create_decision_entry(
    tkr: str,
    data: dict[str, Any],
    is_held: bool,
    etf_meta: dict,
    full_etf_meta: dict,
    consecutive_holding_info: dict,
    sell_cooldown_block: dict,
    buy_cooldown_block: dict,
    base_date: pd.Timestamp,
    country_code: str,
    current_equity: float,
    stop_loss_threshold: float | None,
    cooldown_days: int | None,
    min_buy_score: float,
    rsi_sell_threshold: float,
    trailing_stop_pct: float,
    evaluator: StrategyEvaluator,
) -> dict[str, Any]:
    """개별 종목의 의사결정 엔트리를 생성합니다."""

    price_raw = data.get("price", 0.0)
    price = float(price_raw) if pd.notna(price_raw) else 0.0
    score_raw = data.get("score", 0.0)
    score_value = float(score_raw) if pd.notna(score_raw) else 0.0
    rsi_raw = data.get("rsi_score", 0.0)
    rsi_score_value = float(rsi_raw) if pd.notna(rsi_raw) else 0.0

    state = "HOLD" if is_held else "WAIT"
    phrase = ""

    if price == 0.0 and is_held:
        phrase = DECISION_NOTES["PRICE_DATA_FAIL"]

    sell_block_info = sell_cooldown_block.get(tkr)
    buy_block_info = buy_cooldown_block.get(tkr)

    # 보유 정보 로드
    buy_date = None
    holding_days = 0
    consecutive_info = consecutive_holding_info.get(tkr)
    if consecutive_info:
        buy_date = consecutive_info.get("buy_date")

    evaluation_date = max(base_date.normalize(), pd.Timestamp.now().normalize())

    if is_held and buy_date:
        buy_date_norm = pd.to_datetime(buy_date).normalize()
        if buy_date_norm <= evaluation_date:
            holding_days = count_trading_days(country_code, buy_date_norm, evaluation_date)

    # 수익률 및 고점 계산
    holding_return_pct: float | None = None
    highest_price: float | None = None
    avg_cost = 0.0

    if is_held:
        entry_price = resolve_entry_price(data.get("close"), buy_date)
        if entry_price and entry_price > 0:
            avg_cost = entry_price
            if price > 0:
                holding_return_pct = ((price / entry_price) - 1.0) * 100.0

        highest_price = resolve_highest_price_since_buy(data.get("close"), buy_date)

    # Evaluator 실행
    if state == "HOLD":
        ma_val = float(data.get("s1") or 0.0)

        state, phrase = evaluator.evaluate_sell_decision(
            current_state=state,
            price=price,
            avg_cost=avg_cost,
            highest_price=highest_price if highest_price is not None else 0.0,
            ma_value=ma_val,
            ma_period=data.get("ma_period") or 20,
            score=score_value,
            rsi_score=rsi_score_value,
            is_core_holding=False,  # 상위 레벨에서 override 예정
            stop_loss_threshold=stop_loss_threshold,
            rsi_sell_threshold=rsi_sell_threshold,
            trailing_stop_pct=trailing_stop_pct,
            min_buy_score=min_buy_score,
            sell_cooldown_info=sell_block_info,  # 쿨다운 정보 전달
            cooldown_days=cooldown_days or 0,
        )
    elif state == "WAIT":
        buy_signal, phrase = evaluator.check_buy_signal(
            score=score_value,
            min_buy_score=min_buy_score,
            buy_cooldown_info=buy_block_info,
            cooldown_days=cooldown_days or 0,
        )
    else:
        buy_signal = False

    # Buy signal 확인 (WAIT 상태일 때만 의미 있음)
    buy_signal = (
        state == "WAIT"
        and evaluator.check_buy_signal(
            score=score_value,
            min_buy_score=min_buy_score,
            buy_cooldown_info=buy_block_info,
            cooldown_days=cooldown_days or 0,
        )[0]
    )

    # 메타데이터 및 포맷팅
    meta = etf_meta.get(tkr) or full_etf_meta.get(tkr, {}) or {}
    display_name = str(meta.get("name") or tkr)
    raw_category = meta.get("category")
    display_category = str(raw_category) if raw_category else "-"

    if holding_days == 0 and state in {"BUY", "BUY_REPLACE"}:
        holding_days = 1

    prev_close_raw = data.get("prev_close", 0.0)
    prev_close = float(prev_close_raw) if pd.notna(prev_close_raw) else 0.0
    day_ret = 0.0
    if price > 0 and prev_close > 0:
        day_ret = round(((price / prev_close) - 1.0) * 100.0, 2)

    holding_days_display = str(holding_days) if holding_days > 0 else "-"
    amount = price if is_held else 0.0

    equity_base = current_equity if pd.notna(current_equity) and current_equity > 0 else 1.0
    position_weight_pct = round((amount / equity_base) * 100.0, 2)

    # Row 데이터 구성 (Reporting용)
    current_row = [
        0,  # Order (나중에 채움)
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

    return {
        "state": state,
        "weight": position_weight_pct,
        "score": score_value,
        "rsi_score": rsi_score_value,
        "tkr": tkr,
        "row": current_row,
        "buy_signal": buy_signal,
        "sell_cooldown_info": sell_block_info,
        "buy_cooldown_info": buy_block_info,
        "is_held": is_held,
        "filter": data.get("filter"),
        "hold_return_pct": holding_return_pct,
    }


def run_portfolio_recommend(
    account_id: str,
    country_code: str,
    base_date: pd.Timestamp,
    strategy_rules: StrategyRules,
    data_by_tkr: dict[str, Any],
    holdings: dict[str, dict[str, float]],
    etf_meta: dict[str, Any],
    full_etf_meta: dict[str, Any],
    current_equity: float,
    total_cash: float,
    pairs: list[tuple[str, str]],
    consecutive_holding_info: dict[str, dict],
    trade_cooldown_info: dict[str, dict[str, pd.Timestamp | None]],
    cooldown_days: int,
    rsi_sell_threshold: float,
) -> list[dict[str, Any]]:
    """일일 포트폴리오 추천 로직 실행"""

    # 1. 설정 검증
    denom = strategy_rules.portfolio_topn
    validate_portfolio_topn(denom, account_id)

    replace_threshold = strategy_rules.replace_threshold
    stop_loss_pct = strategy_rules.stop_loss_pct
    stop_loss_threshold = -abs(float(stop_loss_pct)) if stop_loss_pct is not None else -abs(float(denom))
    min_buy_score = float(strategy_rules.min_buy_score)
    trailing_stop_pct = getattr(strategy_rules, "trailing_stop_pct", 0.0)

    # 2. 핵심 보유 종목 및 카테고리 정보 준비
    core_holdings_tickers = set(strategy_rules.core_holdings or [])
    universe_tickers_set = {ticker for ticker, _ in pairs}
    valid_core_holdings = validate_core_holdings(core_holdings_tickers, universe_tickers_set, account_id)

    # 3. 쿨다운 정보 계산
    sell_cooldown_block, buy_cooldown_block = calculate_cooldown_blocks(
        trade_cooldown_info, cooldown_days, base_date, country_code
    )

    evaluator = StrategyEvaluator()
    decisions = []

    # 4. 각 종목별 1차 의사결정 (SELL, HOLD, WAIT 등 판단)
    for tkr, _ in pairs:
        d = data_by_tkr.get(tkr)
        is_effectively_held = tkr in holdings

        if not d and not is_effectively_held:
            continue

        # 데이터 부족 종목 필터링
        if config.ENABLE_DATA_SUFFICIENCY_CHECK and d and d.get("data_insufficient"):
            if tkr not in valid_core_holdings and not is_effectively_held:
                continue

        if not d:
            # 더미 데이터 생성 (보유중인데 데이터가 없는 경우 등)
            d = {
                "price": 0.0,
                "score": 0.0,
                "rsi_score": 0.0,
                "close": pd.Series(),
                "ma_period": strategy_rules.ma_period,
            }

        decision = _create_decision_entry(
            tkr,
            d,
            is_effectively_held,
            etf_meta,
            full_etf_meta,
            consecutive_holding_info,
            sell_cooldown_block,
            buy_cooldown_block,
            base_date,
            country_code,
            current_equity,
            stop_loss_threshold,
            cooldown_days,
            min_buy_score,
            rsi_sell_threshold,
            trailing_stop_pct,
            evaluator,
        )
        decisions.append(decision)

    # 5. 후처리: 핵심 보유 종목 강제 HOLD/BUY 처리
    for decision in decisions:
        ticker = decision["tkr"]
        if ticker in valid_core_holdings:
            # 이미 보유중이면 강제 HOLD
            if decision["is_held"] or decision["state"] in {
                "SELL_TREND",
                "SELL_RSI",
                "CUT_STOPLOSS",
                "SELL_REPLACE",
                "HOLD",
            }:
                decision["state"] = "HOLD_CORE"
                decision["row"][4] = "HOLD_CORE"
                decision["row"][-1] = DECISION_MESSAGES.get("HOLD_CORE", "🔒 핵심 보유")
            # 미보유면 자동 매수 처리 (아래에서 추가됨, 여기선 상태만 정리)

    # 핵심 보유 종목 미보유 시 자동 매수 Entry 추가/수정
    for core_ticker in valid_core_holdings:
        if core_ticker not in holdings:
            existing = next((d for d in decisions if d["tkr"] == core_ticker), None)
            if existing:
                existing["state"] = "BUY"
                existing["row"][4] = "BUY"
                existing["row"][-1] = "🔒 핵심 보유 (자동 매수)"
                existing["buy_signal"] = True
            elif core_ticker in data_by_tkr:
                # decisions에 없었다면 추가 (거의 없을 듯)
                core_data = data_by_tkr[core_ticker]
                new_decision = _create_decision_entry(
                    core_ticker,
                    core_data,
                    False,
                    etf_meta,
                    full_etf_meta,
                    consecutive_holding_info,
                    sell_cooldown_block,
                    buy_cooldown_block,
                    base_date,
                    country_code,
                    current_equity,
                    stop_loss_threshold,
                    cooldown_days,
                    min_buy_score,
                    rsi_sell_threshold,
                    trailing_stop_pct,
                    evaluator,
                )
                new_decision["state"] = "BUY"
                new_decision["row"][4] = "BUY"
                new_decision["row"][-1] = "🔒 핵심 보유 (자동 매수)"
                new_decision["buy_signal"] = True
                decisions.append(new_decision)

    # 6. 매수 후보 선정 (Wait Candidates)
    wait_candidates = [
        d for d in decisions if d["state"] == "WAIT" and d.get("buy_signal") and d["tkr"] in universe_tickers_set
    ]
    # 점수순 정렬
    wait_candidates.sort(key=lambda x: x.get("score", 0.0), reverse=True)

    # 7. 매도 예정 종목 확인 및 슬롯 계산
    sell_rsi_categories_today = set()
    for d in decisions:
        # SELL_RSI이거나 RSI 과매수인 HOLD 종목의 카테고리 수집
        cat = etf_meta.get(d["tkr"], {}).get("category")
        if not cat or is_category_exception(cat):
            continue

        if d["state"] == "SELL_RSI":
            sell_rsi_categories_today.add(cat)
        elif d["state"] in {"HOLD", "HOLD_CORE"} and d.get("rsi_score", 0.0) >= rsi_sell_threshold:
            sell_rsi_categories_today.add(cat)

    # 보유 예정 수 (HOLD 계열 + 쿨다운 중인 SELL_RSI 등 사실상 보유로 치는 것들)
    # logic/recommend/old_portfolio.py 의 로직 참조:
    # SELL_RSI는 쿨다운으로 안 팔릴 수도 있으니 일단은 held_count에 포함?
    # old logic: SELL_RSI는 항상 포함. 다른 SELL은 쿨다운 중일때만 포함.
    sell_state_set_for_count = {"SELL_TREND", "SELL_REPLACE", "CUT_STOPLOSS"}
    held_count = 0
    for d in decisions:
        if d["state"] in {"HOLD", "HOLD_CORE", "SELL_RSI"}:
            held_count += 1
        elif d["state"] in sell_state_set_for_count and d["tkr"] in sell_cooldown_block:
            held_count += 1

    slots_to_fill = denom - held_count

    # 8. 신규 매수 (Slots 채우기)
    if slots_to_fill > 0:
        held_categories_for_buy = get_held_categories_excluding_sells(
            decisions,
            get_category_func=lambda d: etf_meta.get(d["tkr"], {}).get("category"),
            get_state_func=lambda d: d["state"],
            get_ticker_func=lambda d: d["tkr"],
            holdings=set(holdings.keys()),
        )
        # Normalize categories in set
        held_categories_for_buy = {str(c).strip().upper() for c in held_categories_for_buy if c}

        successful_buys = 0
        for cand in wait_candidates:
            # 슬롯이 꽉 찼으면 더 이상 매수하지 않고 나머지는 대기 처리
            if successful_buys >= slots_to_fill:
                cand["row"][-1] = ""
                # buy_signal 유지 -> Step 9에서 사용
                continue

            raw_cat = etf_meta.get(cand["tkr"], {}).get("category")
            cand_cat = str(raw_cat).strip().upper() if raw_cat else None
            cand_rsi = cand.get("rsi_score", 100.0)

            # check_buy_candidate_filters 내부에서도 normalization을 할 수 있으나
            # 여기서 normalized된 'held_categories_for_buy'를 넘기려면
            # check_buy_candidate_filters가 normalized된 set을 받을 수 있어야 함.
            # logic/common/portfolio.py의 check 함수는 `category in held_categories`를 함.
            # 따라서 여기서도 cand_cat을 맞춰서 넘겨줘야 함. (위에서 normalize 함)

            # 단, is_category_exception은 원본 카테고리(혹은 매칭)를 필요로 할 수 있으니 주의.
            # check_buy_candidate_filters는 내부에서 is_category_exception 호출함.
            # 임시로 raw_cat을 넘기되, held_categories 검사는 로직 내부 확인 필요.
            # logic/common 확인 결과: `category` 인자를 그대로 `is_category_exception`과 `category in held_categories`에 씀.
            # 일관성을 위해 logic/common 함수도 수정하거나, 여기서 set과 input 모두 normalize해야 함.
            # 여기서는 set을 normalize했으므로 input(cand_cat)도 normalize해서 넘겨야 매칭됨.
            # 하지만 is_category_exception은 정확한 string match를 원할 수 있음.
            # -> is_category_exception은 보통 한글이라 strip 정도만 함.
            # 안전하게: check_buy_candidate_filters 호출 시 'category'는 raw_cat 사용,
            # 'held_categories'는 normalized set.
            # -> 이러면 mismatch 발생.
            # **Better approach**: Don't change `check_buy_candidate_filters` signature too much locally.
            # Let's verify `check_buy_candidate_filters` in common.

            can_buy, block_reason = check_buy_candidate_filters(
                category=cand_cat,  # Normalized passing
                held_categories=held_categories_for_buy,
                sell_rsi_categories_today=sell_rsi_categories_today,
                rsi_score=cand_rsi,
                rsi_sell_threshold=rsi_sell_threshold,
            )

            if not can_buy:
                cand["state"], cand["row"][4] = "WAIT", "WAIT"
                cand["row"][-1] = block_reason
                cand["buy_signal"] = False
                continue

            # 가격 및 예산 체크
            price = float(data_by_tkr.get(cand["tkr"], {}).get("price") or 0.0)
            if price <= 0:
                cand["row"][-1] = DECISION_NOTES["NO_PRICE"]
                continue

            budget = (current_equity / denom) if denom > 0 else 0
            if budget > total_cash:
                budget = total_cash  # 현금 부족 시 제한

            if budget > 0:
                cand["state"], cand["row"][4] = "BUY", "BUY"
                cand["row"][-1] = DECISION_MESSAGES["NEW_BUY"]

                if cand_cat and not is_category_exception(cand_cat):
                    held_categories_for_buy.add(cand_cat)
                successful_buys += 1
            else:
                cand["row"][-1] = DECISION_NOTES["INSUFFICIENT_CASH"]
    else:
        # 슬롯이 처음부터 없으면 모든 후보 대기 처리 (Replacement 후보로 넘김)
        for cand in wait_candidates:
            cand["row"][-1] = ""
            # buy_signal 유지 -> Step 9에서 사용

    # 9. 교체 매매 (Replace)
    # Buy signal이 있지만 선택되지 못한 후보들 (WAIT 상태)
    replace_candidates_pool = [
        cand for cand in wait_candidates if cand.get("state") != "BUY" and cand.get("buy_signal")
    ]

    # 교체 후보 선정 (logic/common 활용)
    replacement_candidates, _ = select_candidates_by_category(
        replace_candidates_pool, etf_meta, held_categories=None, max_count=None, skip_held_categories=False
    )

    # 현재 보유중인 종목 (교체 대상) - 점수 낮은 순
    current_held_stocks = [d for d in decisions if d["state"] == "HOLD"]  # HOLD_CORE 제외
    current_held_stocks.sort(key=lambda x: x.get("score", 0.0) if pd.notna(x.get("score")) else -float("inf"))

    for best_new in replacement_candidates:
        if not current_held_stocks:
            break

        # 교체 매수 필터링 (RSI, 핵심보유 카테고리 등)
        new_rsi = best_new.get("rsi_score", 0.0)
        if new_rsi >= rsi_sell_threshold:
            best_new["row"][-1] = f"RSI 과매수 ({new_rsi:.1f})"
            continue

        new_cat = etf_meta.get(best_new["tkr"], {}).get("category")

        # 핵심 보유 카테고리 체크
        core_cats = {etf_meta.get(t, {}).get("category") for t in valid_core_holdings}
        if new_cat and not is_category_exception(new_cat) and new_cat in core_cats:
            best_new["row"][-1] = f"핵심 보유 카테고리 ({new_cat})"
            continue

        # 같은 카테고리 보유 종목 찾기
        held_same_cat = next(
            (
                s
                for s in current_held_stocks
                if new_cat
                and not is_category_exception(new_cat)
                and etf_meta.get(s["tkr"], {}).get("category") == new_cat
            ),
            None,
        )

        target_sell = None

        if held_same_cat:
            # 같은 카테고리가 있으면 점수 비교
            score_diff = best_new.get("score", 0) - held_same_cat.get("score", 0)
            if score_diff >= replace_threshold:
                target_sell = held_same_cat
        else:
            # 다른 카테고리면 가장 점수 낮은 종목과 비교
            weakest = current_held_stocks[0]
            # 카테고리 중복 아니어야 함 (이미 위에서 select_candidates가 잘 골라줬겠지만 확인)
            # -> select_candidates_by_category는 단순히 점수순 정렬만 함.
            # 중복 체크는 여기서 다시 해야 함?
            # 아니, held_categories_for_buy 체크가 필요함.

            # 보유 중인 카테고리와 겹치면 안됨 (예외 카테고리 제외)
            is_dup = False
            for h in current_held_stocks:
                h_cat = etf_meta.get(h["tkr"], {}).get("category")
                if h_cat == new_cat and not is_category_exception(new_cat):
                    is_dup = True  # 이미 위에서 (held_same_cat) 잡혔어야 함.
                    break

            if not is_dup:
                score_diff = best_new.get("score", 0) - weakest.get("score", 0)
                if score_diff >= replace_threshold:
                    target_sell = weakest

        if target_sell:
            # 교체 실행
            # 1. 매도 처리
            current_held_stocks.remove(target_sell)
            target_sell["state"] = "SELL_REPLACE"
            target_sell["row"][4] = "SELL_REPLACE"

            # 매도 문구 (수익률 포함 등) - 간단히 처리하거나 함수 호출
            # 여기서는 문구 포맷팅 로직을 직접 구현하거나 HELPER 쓰기
            # old_pipeline의 _format_sell_replace_phrase 는 pipeline에 있었음.
            # 여기서는 row 메시지를 직접 구성.

            hold_ret = target_sell.get("hold_return_pct") or 0.0

            # 2. 매수 처리
            best_new["state"] = "BUY_REPLACE"
            best_new["row"][4] = "BUY_REPLACE"

            # 문구 생성
            sell_name = etf_meta.get(target_sell["tkr"], {}).get("name") or target_sell["tkr"]
            new_name = etf_meta.get(best_new["tkr"], {}).get("name") or best_new["tkr"]

            target_sell["row"][-1] = f"🔄 교체매도 손익률 {hold_ret:+.2f}% - {new_name}({best_new['tkr']}) 교체"
            best_new["row"][-1] = build_buy_replace_note(sell_name, target_sell["tkr"])  # 수정된 common 함수 사용

    # 10. Wait 상태 메시지 정리 (Category Dup, Low Score 등)
    # _apply_wait_note_if_empty 로직과 유사하게
    current_held_tickers = {d["tkr"] for d in decisions if d["state"] in {"HOLD", "HOLD_CORE", "BUY", "BUY_REPLACE"}}
    held_scores = [d.get("score", 0) for d in decisions if d["tkr"] in current_held_tickers]
    weakest_score = min(held_scores) if held_scores else 0.0

    held_cats_final = set()
    for d in decisions:
        if d["state"] in {"HOLD", "HOLD_CORE", "BUY", "BUY_REPLACE"}:
            cat = etf_meta.get(d["tkr"], {}).get("category")
            if cat and not is_category_exception(cat):
                held_cats_final.add(cat)

    for d in decisions:
        if d["state"] == "WAIT":
            # 이미 메시지가 있으면 스킵
            if d["row"][-1] and "부족" not in str(d["row"][-1]):
                continue

            score_val = d.get("score", 0.0)
            if score_val <= min_buy_score:
                d["row"][-1] = format_min_score_phrase(score_val, min_buy_score)
            else:
                # 포트폴리오 꽉 참 -> 점수 부족
                if not held_scores:
                    d["row"][-1] = ""  # PORTFOLIO_FULL (removed) -> Empty
                else:
                    req = weakest_score + replace_threshold
                    d["row"][-1] = DECISION_NOTES["REPLACE_SCORE"].format(min_buy_score=req)

    # 데이터 부족 메시지 (한번 더 체크)
    for d in decisions:
        # data_by_tkr 원본 확인
        orig = data_by_tkr.get(d["tkr"])
        if orig and orig.get("data_insufficient"):
            note = DECISION_NOTES.get("DATA_INSUFFICIENT", "⚠️ 거래일 부족")
            if d["row"][-1]:
                if note not in d["row"][-1]:
                    d["row"][-1] += f" | {note}"
            else:
                d["row"][-1] = note

    # 11. 중복 카테고리 필터링 (화면 표시용)
    # logic: 예외 카테고리가 아니면, Active 상태(보유/매수/매도)인 종목만 남기고,
    #        Active 종목이 없으면 점수가 가장 높은 대기 종목 1개만 남김.
    final_filtered = []

    # 1. 카테고리별 그룹화
    decisions_by_cat = {}
    for d in decisions:
        cat = etf_meta.get(d["tkr"], {}).get("category")
        # 카테고리가 없거나 "-"인 경우 "Uncategorized" 혹은 그대로 처리
        cat_key = cat if cat else "Uncategorized"
        decisions_by_cat.setdefault(cat_key, []).append(d)

    active_states = {
        "HOLD",
        "HOLD_CORE",
        "BUY",
        "BUY_REPLACE",
        "SELL_REPLACE",
        "SELL_RSI",
        "SELL_TREND",
        "CUT_STOPLOSS",
    }

    for cat_key, items in decisions_by_cat.items():
        # 예외 카테고리는 모두 표시
        # check if cat_key matches any exception (config.CATEGORY_EXCEPTIONS)
        # Assuming exact match or if exception in cat name? strict check better.
        is_exc = is_category_exception(cat_key)

        if is_exc:
            final_filtered.extend(items)
            continue

        # Active item 찾기
        active_items = [d for d in items if d["state"] in active_states]

        if active_items:
            # Active 상태인 종목들은 모두 표시 (예: 같은 카테고리 내 교체 매매 등)
            final_filtered.extend(active_items)
        else:
            # Active가 없으면 (모두 WAIT 등), 점수 가장 높은 1개만 표시
            # items는 점수가 있을수도 없을수도.
            scored_items = [d for d in items if isinstance(d.get("score"), (int, float))]
            if scored_items:
                best = max(scored_items, key=lambda x: x.get("score", -999))
                final_filtered.append(best)
            elif items:
                # 점수도 없으면 그냥 첫번째
                final_filtered.append(items[0])

    decisions = final_filtered

    # 12. 최종 정렬
    sort_decisions_by_order_and_score(decisions)

    # 순위 할당 (row[0] 업데이트)
    for i, d in enumerate(decisions):
        d["row"][0] = i + 1
        d["rank"] = i + 1

    # Reporting Compatibility
    for d in decisions:
        row = d["row"]
        d["ticker"] = row[1]
        d["name"] = row[2]
        d["category"] = row[3]

        try:
            d["holding_days"] = int(row[5])
        except (ValueError, TypeError):
            d["holding_days"] = 0

        d["daily_pct"] = row[7]
        d["evaluation_pct"] = row[10]
        d["price"] = row[6]
        d["phrase"] = row[15]

        try:
            d["streak"] = int(str(row[14]).replace("일", ""))
        except (ValueError, TypeError):
            d["streak"] = 0

    return decisions


def safe_run_portfolio_recommend(*args, **kwargs) -> list[dict[str, Any]]:
    """Exception safe wrapper"""
    try:
        return run_portfolio_recommend(*args, **kwargs)
    except Exception as e:
        logger.exception(f"run_portfolio_recommend failed: {e}")
        return []


# Aliases
generate_daily_recommendations_for_portfolio = run_portfolio_recommend
safe_generate_daily_recommendations_for_portfolio = safe_run_portfolio_recommend
