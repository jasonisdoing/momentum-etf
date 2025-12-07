"""
포트폴리오 백테스트 실행 모듈

전략 중립적인 포트폴리오 백테스트 로직을 제공합니다.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pandas as pd

from logic.common import calculate_held_categories, is_category_exception, select_candidates_by_category
from logic.common.notes import format_min_score_phrase
from logic.common.price import calculate_trade_price
from strategies.maps.constants import DECISION_CONFIG, DECISION_NOTES
from strategies.maps.evaluator import StrategyEvaluator
from strategies.maps.labeler import compute_net_trade_note
from strategies.maps.metrics import process_ticker_data
from utils.logger import get_app_logger
from utils.report import format_kr_money

logger = get_app_logger()


def _execute_individual_sells(
    position_state: dict,
    valid_core_holdings: set[str],
    metrics_by_ticker: dict,
    today_prices: dict[str, float],
    score_today: dict[str, float],
    rsi_score_today: dict[str, float],
    ticker_to_category: dict[str, str],
    sell_rsi_categories_today: set[str],
    sell_trades_today_map: dict,
    daily_records_by_ticker: dict,
    i: int,
    total_days: int,
    country_code: str,
    stop_loss_threshold: float | None,
    rsi_sell_threshold: float,
    trailing_stop_pct: float,
    cooldown_days: int,
    cash: float,
    current_holdings_value: float,
    ma_period: int,
    min_buy_score: float,
    evaluator: StrategyEvaluator,
) -> tuple[float, float]:
    """개별 종목 매도 로직 (StrategyEvaluator 사용)"""
    for ticker, ticker_metrics in metrics_by_ticker.items():
        ticker_state, price = position_state[ticker], today_prices.get(ticker)

        if ticker_state["shares"] > 0 and pd.notna(price) and metrics_by_ticker[ticker]["available_mask"][i]:
            # 최고가 갱신 (트레일링 스탑용)
            if price > ticker_state.get("highest_price", 0.0):
                ticker_state["highest_price"] = price

            in_cooldown = i < ticker_state["sell_block_until"]

            # 매도 의사결정 (StrategyEvaluator)
            ma_val_today = ticker_metrics["ma_values"][i]
            ma_val = float(ma_val_today) if not pd.isna(ma_val_today) else 0.0
            ticker_ma_period = ticker_metrics.get("ma_period", ma_period)

            current_score = score_today.get(ticker, 0.0)
            if pd.isna(current_score):
                current_score = -float("inf")

            # 쿨다운 정보 구성 (Evaluator 호환성)
            # sell_cooldown_info는 현재 백테스트 로직에서 simple index check로 대체되므로 None 전달
            # in_cooldown 변수가 이미 체크됨
            pass

            decision, phrase = evaluator.evaluate_sell_decision(
                current_state="HOLD",
                price=price,
                avg_cost=ticker_state["avg_cost"],
                highest_price=ticker_state.get("highest_price", 0.0),
                ma_value=ma_val,
                ma_period=ticker_ma_period,
                score=current_score,
                rsi_score=rsi_score_today.get(ticker, 0.0),
                is_core_holding=(ticker in valid_core_holdings),
                stop_loss_threshold=stop_loss_threshold,
                rsi_sell_threshold=rsi_sell_threshold,
                trailing_stop_pct=trailing_stop_pct,
                min_buy_score=min_buy_score,
                sell_cooldown_info=None,  # 백테스트 루프 내 제어
                cooldown_days=cooldown_days,
            )

            # 백테스트 고유 쿨다운 처리 (Evaluator는 상태만 반환, 실행 여부는 여기서)
            if decision == "HOLD_CORE":
                decision = None

            if not decision or decision == "HOLD":
                continue

            # 손절매가 아닌데 쿨다운 중이면 스킵
            if in_cooldown and decision != "CUT_STOPLOSS":
                continue

            if decision:
                # 다음날 시초가 + 슬리피지로 매도 가격 계산
                sell_price = calculate_trade_price(
                    i,
                    total_days,
                    metrics_by_ticker[ticker]["open_values"],
                    metrics_by_ticker[ticker]["close_values"],
                    country_code,
                    is_buy=False,
                )
                if sell_price <= 0:
                    continue

                qty = ticker_state["shares"]
                trade_amount = qty * sell_price
                trade_profit = (sell_price - ticker_state["avg_cost"]) * qty if ticker_state["avg_cost"] > 0 else 0.0
                hold_ret = (
                    (sell_price / ticker_state["avg_cost"] - 1.0) * 100.0 if ticker_state["avg_cost"] > 0 else 0.0
                )

                # 순매도 집계
                sell_trades_today_map.setdefault(ticker, []).append({"shares": float(qty), "price": float(sell_price)})

                # SELL_RSI인 경우 해당 카테고리 추적
                if decision == "SELL_RSI":
                    sold_category = ticker_to_category.get(ticker)
                    if sold_category and not is_category_exception(sold_category):
                        sell_rsi_categories_today.add(sold_category)

                cash += trade_amount
                current_holdings_value = max(0.0, current_holdings_value - trade_amount)
                ticker_state["shares"], ticker_state["avg_cost"] = 0, 0.0
                ticker_state["highest_price"] = 0.0  # 매도 후 최고가 초기화

                # 매도 후 재매수 금지 기간만 설정 (매수 쿨다운)
                if cooldown_days > 0:
                    ticker_state["buy_block_until"] = i + cooldown_days + 1

                # 행 업데이트
                row = daily_records_by_ticker[ticker][-1]
                row.update(
                    {
                        "decision": decision,
                        "trade_amount": trade_amount,
                        "trade_profit": trade_profit,
                        "trade_pl_pct": hold_ret,
                        "shares": 0,
                        "pv": 0,
                        "avg_cost": 0,
                    }
                )
                if decision == "SELL_TREND":
                    row["note"] = phrase
                elif decision == "SELL_TRAILING":
                    row["note"] = (
                        f"고점({ticker_state.get('highest_price', 0.0):,.0f}) 대비 {trailing_stop_pct}% 하락"  # phrase에 이미 포함되어 있지만 백테스트 형식 유지
                    )

    return cash, current_holdings_value


def _rank_buy_candidates(
    tickers_available_today: set[str],
    position_state: dict,
    buy_signal_today: dict[str, int],
    score_today: dict[str, float],
    i: int,
) -> list[tuple[float, str]]:
    """매수 후보를 점수 순으로 정렬

    Returns:
        [(score, ticker), ...] 점수 내림차순 정렬
    """
    buy_ranked_candidates = []
    for candidate_ticker in tickers_available_today:
        ticker_state_cand = position_state[candidate_ticker]
        buy_signal_days_today = buy_signal_today.get(candidate_ticker, 0)

        if ticker_state_cand["shares"] == 0 and i >= ticker_state_cand["buy_block_until"] and buy_signal_days_today > 0:
            # MAPS 점수 사용
            score_cand = score_today.get(candidate_ticker, float("nan"))
            final_score = score_cand if not pd.isna(score_cand) else -float("inf")
            buy_ranked_candidates.append((final_score, candidate_ticker))

    buy_ranked_candidates.sort(reverse=True)
    return buy_ranked_candidates


def _update_ticker_note(
    daily_records_by_ticker: dict,
    ticker: str,
    dt: pd.Timestamp,
    note: str,
) -> None:
    """티커의 노트를 업데이트하는 헬퍼 함수

    Args:
        daily_records_by_ticker: 일별 기록 딕셔너리
        ticker: 티커
        dt: 날짜
        note: 노트 내용
    """
    if daily_records_by_ticker.get(ticker) and daily_records_by_ticker[ticker][-1]["date"] == dt:
        daily_records_by_ticker[ticker][-1]["note"] = note


def _apply_wait_note_if_empty(
    daily_records_by_ticker: dict,
    ticker: str,
    dt: pd.Timestamp,
    ticker_to_category: dict[str, str],
    held_categories: set[str],
    held_categories_normalized: set[str],
    position_state: dict = None,
    score_today: dict[str, float] = None,
    replace_threshold: float = 0.0,
) -> None:
    """WAIT 상태 종목에 대해 카테고리 중복 여부에 따라 노트를 설정합니다."""

    records = daily_records_by_ticker.get(ticker)
    if not (records and records[-1]["date"] == dt):
        return

    current_note = str(records[-1].get("note") or "").strip()
    if current_note:
        return
    # Calculate minimum required score for replacement
    if position_state and score_today is not None:
        held_scores = [
            score_today.get(t, 0.0) for t, state in position_state.items() if state.get("shares", 0) > 0 and t != "CASH"
        ]
        if held_scores:
            weakest_score = min(held_scores)
            required_score = weakest_score + replace_threshold
            records[-1]["note"] = DECISION_NOTES["REPLACE_SCORE"].format(min_buy_score=required_score)


def _execute_new_buys(
    buy_ranked_candidates: list[tuple[float, str]],
    position_state: dict,
    valid_core_holdings: set[str],
    ticker_to_category: dict[str, str],
    sell_rsi_categories_today: set[str],
    rsi_score_today: dict[str, float],
    today_prices: dict[str, float],
    metrics_by_ticker: dict,
    daily_records_by_ticker: dict,
    buy_trades_today_map: dict,
    cash: float,
    current_holdings_value: float,
    top_n: int,
    rsi_sell_threshold: float,
    cooldown_days: int,
    replace_threshold: float,
    score_today: dict[str, float],
    i: int,
    total_days: int,
    dt: pd.Timestamp,
    country_code: str,
    initial_capital: float = 0.0,
) -> tuple[float, float, set[str], set[str]]:
    """신규 매수 실행

    Returns:
        (cash, current_holdings_value, purchased_today, held_categories)
    """
    from logic.common import (
        calculate_buy_budget,
        calculate_held_categories,
        calculate_held_count,
        check_buy_candidate_filters,
    )

    held_count = calculate_held_count(position_state)
    slots_to_fill = max(0, top_n - held_count)
    purchased_today: set[str] = set()

    if slots_to_fill <= 0 or not buy_ranked_candidates:
        held_categories = calculate_held_categories(position_state, ticker_to_category, valid_core_holdings)
        if slots_to_fill <= 0 and buy_ranked_candidates:
            held_categories_normalized = {str(cat).strip().upper() for cat in held_categories if isinstance(cat, str)}
            for _, candidate_ticker in buy_ranked_candidates:
                _apply_wait_note_if_empty(
                    daily_records_by_ticker,
                    candidate_ticker,
                    dt,
                    ticker_to_category,
                    held_categories,
                    held_categories_normalized,
                    position_state,
                    score_today,
                    replace_threshold,
                )
        return cash, current_holdings_value, purchased_today, held_categories

    # 보유 중인 카테고리 (매수 시 중복 체크용, 고정 종목 카테고리 포함)
    held_categories = calculate_held_categories(position_state, ticker_to_category, valid_core_holdings)
    held_categories_normalized = {str(cat).strip().upper() for cat in held_categories if isinstance(cat, str)}

    # 점수가 양수인 모든 매수 시그널 종목을 candidates에 넣기 (이미 정렬됨)
    successful_buys = 0
    for score, ticker_to_buy in buy_ranked_candidates:
        if successful_buys >= slots_to_fill:
            break
        if cash <= 0:
            break

        price = today_prices.get(ticker_to_buy)
        if pd.isna(price):
            continue

        # 매수 후보 필터링 체크
        category = ticker_to_category.get(ticker_to_buy)
        rsi_score_buy_candidate = rsi_score_today.get(ticker_to_buy, 0.0)

        can_buy, block_reason = check_buy_candidate_filters(
            category=category,
            held_categories=held_categories,
            sell_rsi_categories_today=sell_rsi_categories_today,
            rsi_score=rsi_score_buy_candidate,
            rsi_sell_threshold=rsi_sell_threshold,
        )

        if not can_buy:
            _update_ticker_note(daily_records_by_ticker, ticker_to_buy, dt, block_reason)
            continue

        # 매수 예산 계산 (총자산 / TOPN 기준)
        budget = calculate_buy_budget(
            cash=cash,
            current_holdings_value=current_holdings_value,
            top_n=top_n,
        )

        if budget <= 0:
            continue

        # 다음날 시초가 + 슬리피지로 매수 가격 계산
        buy_price = calculate_trade_price(
            i,
            total_days,
            metrics_by_ticker[ticker_to_buy]["open_values"],
            metrics_by_ticker[ticker_to_buy]["close_values"],
            country_code,
            is_buy=True,
        )
        if buy_price <= 0:
            continue

        req_qty = budget / buy_price if buy_price > 0 else 0
        trade_amount = budget

        if trade_amount <= cash + 1e-9 and req_qty > 0:
            ticker_state = position_state[ticker_to_buy]
            cash -= trade_amount
            current_holdings_value += trade_amount
            ticker_state["shares"] += req_qty
            ticker_state["avg_cost"] = buy_price

            # 매도 쿨다운 설정: 매수 후 N일간 매도 금지 (손절 제외)
            if cooldown_days > 0:
                ticker_state["sell_block_until"] = i + cooldown_days + 1

            if category and not is_category_exception(category):
                held_categories.add(category)
                normalized_category = str(category).strip().upper()
                if normalized_category:
                    held_categories_normalized.add(normalized_category)

            if daily_records_by_ticker[ticker_to_buy] and daily_records_by_ticker[ticker_to_buy][-1]["date"] == dt:
                row = daily_records_by_ticker[ticker_to_buy][-1]
                row.update(
                    {
                        "decision": "BUY",
                        "trade_amount": trade_amount,
                        "shares": ticker_state["shares"],
                        "pv": ticker_state["shares"] * price,
                        "avg_cost": ticker_state["avg_cost"],
                    }
                )
            purchased_today.add(ticker_to_buy)
            # 순매수 집계
            buy_trades_today_map.setdefault(ticker_to_buy, []).append({"shares": float(req_qty), "price": float(price)})
            successful_buys += 1

    return cash, current_holdings_value, purchased_today, held_categories


def run_portfolio_backtest(
    stocks: list[dict],
    initial_capital: float = 100_000_000.0,
    core_start_date: pd.Timestamp | None = None,
    top_n: int = 10,
    date_range: list[str] | None = None,
    country: str = "kor",
    prefetched_data: dict[str, pd.DataFrame] | None = None,
    prefetched_metrics: Mapping[str, dict[str, Any]] | None = None,
    trading_calendar: Sequence[pd.Timestamp] | None = None,
    ma_period: int = 20,
    ma_type: str = "SMA",
    replace_threshold: float = 0.0,
    stop_loss_pct: float = -10.0,
    trailing_stop_pct: float = 0.0,
    cooldown_days: int = 5,
    rsi_sell_threshold: float = 10.0,
    core_holdings: list[str] | None = None,
    quiet: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
    missing_ticker_sink: set[str] | None = None,
    *,
    min_buy_score: float,
) -> dict[str, pd.DataFrame]:
    """
    이동평균 기반 모멘텀 전략으로 포트폴리오 백테스트를 실행합니다.

    Args:
        stocks: 백테스트할 종목 목록
        initial_capital: 초기 자본금
        core_start_date: 백테스트 시작일
        top_n: 포트폴리오 최대 보유 종목 수
        date_range: 백테스트 기간 [시작일, 종료일]
        country: 시장 국가 코드 (예: kor)
        prefetched_data: 미리 로드된 가격 데이터
        ma_period: 이동평균 기간
        replace_threshold: 종목 교체 임계값
        stop_loss_pct: 손절 비율 (%)
        trailing_stop_pct: 트레일링 스탑 비율 (%)
        cooldown_days: 거래 쿨다운 기간

    Returns:
        Dict[str, pd.DataFrame]: 종목별 백테스트 결과
    """

    country_code = (country or "").strip().lower() or "kor"

    def _log(message: str) -> None:
        if quiet:
            logger.debug(message)
        else:
            logger.info(message)

    etf_ma_period = ma_period
    stock_ma_period = ma_period
    stop_loss_threshold = stop_loss_pct

    from logic.common import validate_core_holdings, validate_portfolio_topn

    validate_portfolio_topn(top_n)

    # 핵심 보유 종목 (강제 보유, TOPN 포함)

    core_holdings_tickers = set(core_holdings or [])
    universe_tickers_set = {stock["ticker"] for stock in stocks}
    valid_core_holdings = validate_core_holdings(core_holdings_tickers, universe_tickers_set)

    # ETF와 주식을 구분하여 처리
    etf_tickers = {stock["ticker"] for stock in stocks if stock.get("type") == "etf"}

    # 이동평균 계산에 필요한 과거 데이터를 확보하기 위한 추가 조회 범위(웜업)
    # (실제 데이터 요청은 상위 프리패치 단계에서 수행)

    # 개별 종목 데이터 로딩 및 지표 계산
    # 티커별 카테고리 매핑 생성 (성능 최적화를 위해 딕셔너리로 변환)
    ticker_to_category = {stock["ticker"]: stock.get("category") for stock in stocks}
    etf_meta = {stock["ticker"]: stock for stock in stocks if stock.get("ticker")}
    metrics_by_ticker = {}
    tickers_to_process = [s["ticker"] for s in stocks]

    # StrategyEvaluator 인스턴스 생성
    evaluator = StrategyEvaluator()

    for ticker in tickers_to_process:
        df = None
        if prefetched_data and ticker in prefetched_data:
            df = prefetched_data[ticker]

        if df is None:
            raise RuntimeError(f"[백테스트] '{ticker}' 데이터가 프리패치에 없습니다. 튜닝 프리패치 단계를 확인하세요.")

        precomputed_entry = prefetched_metrics.get(ticker) if prefetched_metrics else None
        ticker_metrics = process_ticker_data(
            ticker,
            df,
            etf_tickers,
            etf_ma_period,
            stock_ma_period,
            ma_type=ma_type,
            precomputed_entry=precomputed_entry,
            min_buy_score=min_buy_score,
        )
        if ticker_metrics:
            metrics_by_ticker[ticker] = ticker_metrics

    missing_metrics = [t for t in tickers_to_process if t not in metrics_by_ticker]
    if missing_metrics:
        missing_set = {
            str(ticker).strip().upper() for ticker in missing_metrics if isinstance(ticker, str) and str(ticker).strip()
        }
        if missing_ticker_sink is not None:
            missing_ticker_sink.update(missing_set)
        else:
            logger.warning("가격 데이터 부족으로 제외된 종목: %s", ", ".join(sorted(missing_set)))

    cores_before_filter = len(valid_core_holdings)
    valid_core_holdings = {ticker for ticker in valid_core_holdings if ticker in metrics_by_ticker}
    if cores_before_filter != len(valid_core_holdings):
        dropped = cores_before_filter - len(valid_core_holdings)
        logger.warning("[백테스트] 핵심 보유 종목 중 %d개는 가격 데이터가 없어 제외되었습니다.", dropped)

    if not quiet:
        logger.info(f"[백테스트] metrics_by_ticker: {len(metrics_by_ticker)}개 종목 처리 완료")

    # 모든 종목의 거래일을 합집합하여 전체 백테스트 기간을 설정합니다.
    union_index = pd.DatetimeIndex([])
    for ticker, ticker_metrics in metrics_by_ticker.items():
        union_index = union_index.union(ticker_metrics["close"].index)

    if union_index.empty:
        return {}

    # 요청된 시작일 이후로 인덱스를 필터링합니다.
    if core_start_date:
        union_index = union_index[union_index >= core_start_date]
        if not quiet:
            logger.info(
                f"[백테스트] union_index: {len(union_index)}일 (core_start_date={core_start_date.strftime('%Y-%m-%d')})"
            )

    if union_index.empty:
        logger.warning(
            f"[백테스트] union_index가 비어있습니다. core_start_date={core_start_date}, "
            f"metrics_by_ticker={len(metrics_by_ticker)}"
        )
        return {}

    for ticker, ticker_metrics in metrics_by_ticker.items():
        close_series = ticker_metrics["close"].reindex(union_index)
        open_series = ticker_metrics["open"].reindex(union_index)
        ma_series = ticker_metrics["ma"].reindex(union_index)
        ma_score_series = ticker_metrics["ma_score"].reindex(union_index)
        rsi_score_series = ticker_metrics.get("rsi_score", pd.Series(dtype=float)).reindex(union_index)
        buy_signal_series = ticker_metrics["buy_signal_days"].reindex(union_index).fillna(0).astype(int)

        ticker_metrics["close_series"] = close_series
        ticker_metrics["close_values"] = close_series.to_numpy()
        ticker_metrics["open_series"] = open_series
        ticker_metrics["open_values"] = open_series.to_numpy()
        ticker_metrics["available_mask"] = close_series.notna().to_numpy()
        ticker_metrics["ma_values"] = ma_series.to_numpy()
        ticker_metrics["ma_score_values"] = ma_score_series.to_numpy()
        ticker_metrics["rsi_score_values"] = rsi_score_series.to_numpy()
        ticker_metrics["buy_signal_series"] = buy_signal_series
        ticker_metrics["buy_signal_values"] = buy_signal_series.to_numpy()

    # 시뮬레이션 상태 변수 초기화
    position_state = {
        ticker: {
            "shares": 0,
            "avg_cost": 0.0,
            "highest_price": 0.0,  # 트레일링 스탑용 최고가
            "buy_block_until": -1,
            "sell_block_until": -1,
        }
        for ticker in metrics_by_ticker.keys()
    }
    cash = float(initial_capital)
    daily_records_by_ticker = {ticker: [] for ticker in metrics_by_ticker.keys()}
    out_cash = []
    if trading_calendar is None:
        raise RuntimeError("trading_calendar must be provided to run_portfolio_backtest.")

    # 일별 루프를 돌며 시뮬레이션을 실행합니다.
    total_days = len(union_index)
    _log(f"[백테스트] 총 {total_days}일의 데이터를 처리합니다...")

    for i, dt in enumerate(union_index):
        # 진행률 표시 (10% 단위로)
        if i % max(1, total_days // 10) == 0 or i == total_days - 1:
            progress_pct = int((i + 1) / total_days * 100)
            _log(f"[백테스트] 진행률: {progress_pct}% ({i + 1}/{total_days}일)")
        if progress_callback is not None:
            progress_callback(i + 1, total_days)

        # 디버깅: 첫 3일만 로그
        if i < 3 and not quiet:
            logger.info(f"[백테스트] Day {i}: {dt}, metrics_by_ticker={len(metrics_by_ticker)}")

        # 당일 시작 시점 보유 수량 스냅샷(순매수/순매도 판단용)
        buy_trades_today_map: dict[str, list[dict[str, float]]] = {}
        sell_trades_today_map: dict[str, list[dict[str, float]]] = {}

        # SELL_RSI로 매도한 카테고리 추적 (같은 날 매수 금지)
        sell_rsi_categories_today: set[str] = set()

        tickers_available_today: list[str] = []
        today_prices: dict[str, float] = {}
        score_today: dict[str, float] = {}
        rsi_score_today: dict[str, float] = {}
        buy_signal_today: dict[str, int] = {}

        for ticker, ticker_metrics in metrics_by_ticker.items():
            available = bool(ticker_metrics["available_mask"][i])
            price_val = ticker_metrics["close_values"][i]
            price_float = float(price_val) if not pd.isna(price_val) else float("nan")
            today_prices[ticker] = price_float

            ma_val = ticker_metrics["ma_values"][i]
            score_val = ticker_metrics["ma_score_values"][i]
            rsi_score_val = ticker_metrics.get("rsi_score_values", [float("nan")] * len(union_index))[i]
            buy_signal_val = ticker_metrics["buy_signal_values"][i]

            score_today[ticker] = float(score_val) if not pd.isna(score_val) else 0.0
            rsi_score_today[ticker] = float(rsi_score_val) if not pd.isna(rsi_score_val) else 0.0
            buy_signal_today[ticker] = int(buy_signal_val) if not pd.isna(buy_signal_val) else 0

            if available:
                tickers_available_today.append(ticker)

        # RSI 과매수 경고 카테고리도 추적 (쿨다운으로 아직 매도 안 했지만 RSI 높은 경우)
        for ticker, ticker_state in position_state.items():
            if ticker_state["shares"] > 0:
                rsi_val = rsi_score_today.get(ticker, 0.0)
                if rsi_val >= rsi_sell_threshold:
                    # 쿨다운으로 매도하지 못한 경우에도 카테고리 차단
                    if i < ticker_state["sell_block_until"]:
                        category = ticker_to_category.get(ticker)
                        if category and not is_category_exception(category):
                            sell_rsi_categories_today.add(category)

        # 현재 총 보유 자산 가치를 계산합니다.
        current_holdings_value = 0
        for held_ticker, held_state in position_state.items():
            if held_state["shares"] > 0:
                price_h = today_prices.get(held_ticker)
                if pd.notna(price_h):
                    current_holdings_value += held_state["shares"] * price_h

        # --- 1. 기본 정보 및 출력 행 생성 ---
        records_added_this_day = 0
        for ticker, ticker_metrics in metrics_by_ticker.items():
            position_snapshot = position_state[ticker]
            price = today_prices.get(ticker, float("nan"))
            available_today = ticker in tickers_available_today and not pd.isna(price)

            # 핵심 보유 종목은 HOLD_CORE로 표시
            if position_snapshot["shares"] > 0:
                decision_out = "HOLD_CORE" if ticker in valid_core_holdings else "HOLD"
            else:
                decision_out = "WAIT"

            note = ""
            if decision_out in ("WAIT", "HOLD", "HOLD_CORE"):
                if position_snapshot["shares"] > 0 and i < position_snapshot["sell_block_until"]:
                    remaining = int(position_snapshot["sell_block_until"] - i)
                    note = f"쿨다운 대기중({remaining}일 후 매도 가능)" if remaining > 0 else "쿨다운 종료"
                elif position_snapshot["shares"] == 0 and i < position_snapshot["buy_block_until"]:
                    remaining_buy = int(position_snapshot["buy_block_until"] - i)
                    note = f"쿨다운 대기중({remaining_buy}일 후 매수 가능)" if remaining_buy > 0 else "쿨다운 종료"
                elif decision_out == "WAIT":
                    score_check = score_today.get(ticker, float("nan"))
                    if pd.isna(score_check) or score_check <= min_buy_score:
                        note = format_min_score_phrase(score_check, min_buy_score)

            # 핵심 보유 종목 표시
            if decision_out == "HOLD_CORE" and not note:
                note = "🔒 핵심 보유"

            ma_val = ticker_metrics["ma_values"][i]
            ma_value = float(ma_val) if not pd.isna(ma_val) else float("nan")
            score_value = score_today.get(ticker, 0.0)
            rsi_score_value = rsi_score_today.get(ticker, 0.0)
            filter_value = buy_signal_today.get(ticker, 0)

            if available_today:
                pv_value = position_snapshot["shares"] * price
                record = {
                    "date": dt,
                    "price": price,
                    "shares": position_snapshot["shares"],
                    "pv": pv_value,
                    "decision": decision_out,
                    "avg_cost": position_snapshot["avg_cost"],
                    "trade_amount": 0.0,
                    "trade_profit": 0.0,
                    "trade_pl_pct": 0.0,
                    "note": note,
                    "signal1": ma_value if not pd.isna(ma_value) else None,
                    "signal2": None,
                    "score": score_value if not pd.isna(score_value) else None,
                    "rsi_score": rsi_score_value if not pd.isna(rsi_score_value) else None,
                    "filter": filter_value,
                }
            else:
                avg_cost = position_snapshot["avg_cost"]
                pv_value = position_snapshot["shares"] * (avg_cost if pd.notna(avg_cost) else 0.0)
                rsi_score_value = rsi_score_today.get(ticker, 0.0)
                record = {
                    "date": dt,
                    "price": avg_cost,
                    "shares": position_snapshot["shares"],
                    "pv": pv_value,
                    "decision": "HOLD" if position_snapshot["shares"] > 0 else "WAIT",
                    "avg_cost": avg_cost,
                    "trade_amount": 0.0,
                    "trade_profit": 0.0,
                    "trade_pl_pct": 0.0,
                    "note": "데이터 없음",
                    "signal1": ma_value if not pd.isna(ma_value) else None,
                    "signal2": None,
                    "score": score_value if not pd.isna(score_value) else None,
                    "rsi_score": rsi_score_value if not pd.isna(rsi_score_value) else None,
                    "filter": filter_value,
                }

            daily_records_by_ticker[ticker].append(record)
            records_added_this_day += 1

        # --- 2. 매도 로직 ---
        cash, current_holdings_value = _execute_individual_sells(
            position_state=position_state,
            valid_core_holdings=valid_core_holdings,
            metrics_by_ticker=metrics_by_ticker,
            today_prices=today_prices,
            score_today=score_today,
            rsi_score_today=rsi_score_today,
            ticker_to_category=ticker_to_category,
            sell_rsi_categories_today=sell_rsi_categories_today,
            sell_trades_today_map=sell_trades_today_map,
            daily_records_by_ticker=daily_records_by_ticker,
            i=i,
            total_days=total_days,
            country_code=country_code,
            stop_loss_threshold=stop_loss_threshold,
            rsi_sell_threshold=rsi_sell_threshold,
            trailing_stop_pct=trailing_stop_pct,
            cooldown_days=cooldown_days,
            cash=cash,
            current_holdings_value=current_holdings_value,
            ma_period=ma_period,
            min_buy_score=min_buy_score,
            evaluator=evaluator,
        )

        # --- 3-1. 핵심 보유 종목 자동 매수 (최우선) ---
        for core_ticker in valid_core_holdings:
            if position_state[core_ticker]["shares"] == 0:
                # 핵심 보유 종목이 미보유 상태면 자동 매수
                if core_ticker in tickers_available_today:
                    price = today_prices.get(core_ticker)
                    if pd.notna(price) and price > 0 and cash > 0:
                        # 무조건 균등 비중: 현재 총자산 / TOPN
                        current_total_equity = cash + current_holdings_value
                        budget = current_total_equity / top_n if top_n > 0 else 0

                        budget = min(budget, cash)  # 현금 부족 시 현금만큼만
                        shares_to_buy = budget / price if price > 0 else 0

                        if shares_to_buy > 0 and budget <= cash:
                            trade_amount = shares_to_buy * price
                            cash -= trade_amount
                            position_state[core_ticker]["shares"] = shares_to_buy
                            position_state[core_ticker]["avg_cost"] = price
                            # 매도 후 재매수 금지 기간만 설정 (매수 쿨다운)
                            position_state[core_ticker]["buy_block_until"] = i + cooldown_days + 1

                            buy_trades_today_map.setdefault(core_ticker, []).append(
                                {"shares": float(shares_to_buy), "price": float(price)}
                            )

                            # 레코드 업데이트
                            if (
                                daily_records_by_ticker[core_ticker]
                                and daily_records_by_ticker[core_ticker][-1]["date"] == dt
                            ):
                                row = daily_records_by_ticker[core_ticker][-1]
                                row.update(
                                    {
                                        "decision": "HOLD_CORE",
                                        "shares": shares_to_buy,
                                        "pv": shares_to_buy * price,
                                        "avg_cost": price,
                                        "trade_amount": trade_amount,
                                        "note": "🔒 핵심 보유 (자동 매수)",
                                    }
                                )

                            current_holdings_value += trade_amount

        # --- 3. 매수 로직 ---
        # 1. 매수 후보 선정 (종합 점수 기준)
        buy_ranked_candidates = _rank_buy_candidates(
            tickers_available_today=tickers_available_today,
            position_state=position_state,
            buy_signal_today=buy_signal_today,
            score_today=score_today,
            i=i,
        )

        # 2. 매수 실행 (신규 매수)
        cash, current_holdings_value, purchased_today, held_categories = _execute_new_buys(
            buy_ranked_candidates=buy_ranked_candidates,
            position_state=position_state,
            valid_core_holdings=valid_core_holdings,
            ticker_to_category=ticker_to_category,
            sell_rsi_categories_today=sell_rsi_categories_today,
            rsi_score_today=rsi_score_today,
            today_prices=today_prices,
            metrics_by_ticker=metrics_by_ticker,
            daily_records_by_ticker=daily_records_by_ticker,
            buy_trades_today_map=buy_trades_today_map,
            cash=cash,
            current_holdings_value=current_holdings_value,
            top_n=top_n,
            rsi_sell_threshold=rsi_sell_threshold,
            cooldown_days=cooldown_days,
            replace_threshold=replace_threshold,
            score_today=score_today,
            i=i,
            total_days=total_days,
            dt=dt,
            country_code=country_code,
            initial_capital=initial_capital,
        )

        # 3. 교체 매수 실행 (포트폴리오가 가득 찬 경우)
        if len(purchased_today) == 0 and buy_ranked_candidates:
            from logic.common import calculate_buy_budget

            # 종합 점수를 사용 (buy_ranked_candidates는 이미 종합 점수로 정렬됨)
            helper_candidates = [
                {"tkr": ticker, "score": score}
                for score, ticker in buy_ranked_candidates
                if ticker not in purchased_today
            ]

            replacement_candidates, _ = select_candidates_by_category(
                helper_candidates,
                etf_meta,
                held_categories=None,
                max_count=None,
                skip_held_categories=False,
            )

            held_stocks_with_scores = []
            for held_ticker, held_position in position_state.items():
                # 핵심 보유 종목은 교체 매매 대상에서 제외
                if held_ticker in valid_core_holdings:
                    continue
                if held_position["shares"] > 0:
                    # MAPS 점수 사용
                    score_h = score_today.get(held_ticker, float("nan"))

                    if not pd.isna(score_h):
                        held_stocks_with_scores.append(
                            {
                                "ticker": held_ticker,
                                "score": score_h,
                                "category": ticker_to_category.get(held_ticker),
                            }
                        )

            held_stocks_with_scores.sort(key=lambda x: x["score"])

            # 고정 종목 카테고리 미리 계산 (성능 최적화)
            core_categories = set()
            for core_ticker in valid_core_holdings:
                core_cat = ticker_to_category.get(core_ticker)
                if core_cat and not is_category_exception(core_cat):
                    core_categories.add(core_cat)

            for candidate in replacement_candidates:
                replacement_ticker = candidate["tkr"]
                wait_stock_category = ticker_to_category.get(replacement_ticker)
                best_new_score_raw = candidate.get("score")
                try:
                    best_new_score = float(best_new_score_raw)
                except (TypeError, ValueError):
                    best_new_score = float("-inf")

                # 교체 대상이 될 수 있는 보유 종목을 찾습니다.
                # 1. 같은 카테고리의 종목이 있는지 확인
                held_stock_same_category = next(
                    (s for s in held_stocks_with_scores if s["category"] == wait_stock_category),
                    None,
                )

                weakest_held_stock = held_stocks_with_scores[0] if held_stocks_with_scores else None

                # 교체 여부 및 대상 종목 결정
                ticker_to_sell = None
                replacement_note = ""

                if held_stock_same_category:
                    # 같은 카테고리 종목이 있는 경우: 점수만 비교
                    if best_new_score > held_stock_same_category["score"] + replace_threshold:
                        ticker_to_sell = held_stock_same_category["ticker"]
                        replacement_note = f"{ticker_to_sell}(을)를 {replacement_ticker}(으)로 교체 (동일 카테고리)"
                    else:
                        # 점수가 더 높지 않으면 교체하지 않고 다음 대기 종목으로 넘어감
                        required_score = held_stock_same_category["score"] + replace_threshold
                        _update_ticker_note(
                            daily_records_by_ticker,
                            replacement_ticker,
                            dt,
                            DECISION_NOTES["REPLACE_SCORE"].format(min_buy_score=required_score),
                        )
                        continue  # 다음 buy_ranked_candidate로 넘어감
                elif weakest_held_stock:
                    # 같은 카테고리 종목이 없는 경우: 가장 약한 종목과 임계값 포함 비교
                    if best_new_score > weakest_held_stock["score"] + replace_threshold:
                        ticker_to_sell = weakest_held_stock["ticker"]
                        replacement_note = f"{ticker_to_sell}(을)를 {replacement_ticker}(으)로 교체 (새 카테고리)"
                    else:
                        # 임계값을 넘지 못하면 교체하지 않고 다음 대기 종목으로 넘어감
                        required_score = weakest_held_stock["score"] + replace_threshold
                        _update_ticker_note(
                            daily_records_by_ticker,
                            replacement_ticker,
                            dt,
                            DECISION_NOTES["REPLACE_SCORE"].format(min_buy_score=required_score),
                        )
                        continue  # 다음 buy_ranked_candidate로 넘어감
                else:
                    # 보유 종목이 없으면 교체할 수 없음
                    continue  # 다음 buy_ranked_candidate로 넘어감

                # 교체할 종목이 결정되었으면 매도/매수 진행
                if ticker_to_sell:
                    # SELL_RSI로 매도한 카테고리는 같은 날 교체 매수 금지
                    replacement_category = ticker_to_category.get(replacement_ticker)
                    if (
                        replacement_category
                        and not is_category_exception(replacement_category)
                        and replacement_category in sell_rsi_categories_today
                    ):
                        if (
                            daily_records_by_ticker[replacement_ticker]
                            and daily_records_by_ticker[replacement_ticker][-1]["date"] == dt
                        ):
                            daily_records_by_ticker[replacement_ticker][-1]["note"] = (
                                f"RSI 과매수 매도 카테고리 ({replacement_category})"
                            )
                        continue  # 다음 교체 후보로 넘어감

                    # RSI 과매수 종목 교체 매수 차단
                    rsi_score_replace_candidate = rsi_score_today.get(replacement_ticker, 0.0)

                    if rsi_score_replace_candidate >= rsi_sell_threshold:
                        # RSI 과매수 종목은 교체 매수하지 않음
                        if (
                            daily_records_by_ticker[replacement_ticker]
                            and daily_records_by_ticker[replacement_ticker][-1]["date"] == dt
                        ):
                            daily_records_by_ticker[replacement_ticker][-1]["note"] = (
                                f"RSI 과매수 (RSI점수: {rsi_score_replace_candidate:.1f})"
                            )
                        continue  # 다음 교체 후보로 넘어감

                    sell_price = today_prices.get(ticker_to_sell)
                    buy_price = today_prices.get(replacement_ticker)

                    if pd.notna(sell_price) and sell_price > 0 and pd.notna(buy_price) and buy_price > 0:
                        # (a) 교체 대상 종목 매도
                        weakest_state = position_state[ticker_to_sell]
                        sell_qty = weakest_state["shares"]
                        sell_amount = sell_qty * sell_price
                        hold_ret = (
                            (sell_price / weakest_state["avg_cost"] - 1.0) * 100.0
                            if weakest_state["avg_cost"] > 0
                            else 0.0
                        )
                        trade_profit = (
                            (sell_price - weakest_state["avg_cost"]) * sell_qty
                            if weakest_state["avg_cost"] > 0
                            else 0.0
                        )

                        cash += sell_amount
                        current_holdings_value = max(0.0, current_holdings_value - sell_amount)
                        weakest_state["shares"], weakest_state["avg_cost"] = 0, 0.0
                        # 매도 후 재매수 금지 기간만 설정 (매수 쿨다운)
                        if cooldown_days > 0:
                            weakest_state["buy_block_until"] = i + cooldown_days + 1

                        if (
                            daily_records_by_ticker[ticker_to_sell]
                            and daily_records_by_ticker[ticker_to_sell][-1]["date"] == dt
                        ):
                            row = daily_records_by_ticker[ticker_to_sell][-1]
                            row.update(
                                {
                                    "decision": "SELL_REPLACE",
                                    "trade_amount": sell_amount,
                                    "trade_profit": trade_profit,
                                    "trade_pl_pct": hold_ret,
                                    "shares": 0,
                                    "pv": 0,
                                    "avg_cost": 0,
                                    "note": replacement_note,
                                }
                            )

                        # (b) 새 종목 매수 (기준 자산 기반 예산)
                        budget = calculate_buy_budget(
                            cash=cash,
                            current_holdings_value=current_holdings_value,
                            top_n=top_n,
                        )
                        if budget <= 0:
                            continue
                        # 수량/금액 산정
                        req_qty = int(budget // buy_price) if buy_price > 0 else 0
                        if req_qty <= 0:
                            continue
                        buy_amount = req_qty * buy_price

                        # 체결 반영
                        if req_qty > 0 and buy_amount <= cash + 1e-9:
                            new_ticker_state = position_state[replacement_ticker]
                            cash -= buy_amount
                            current_holdings_value += buy_amount
                            new_ticker_state["shares"], new_ticker_state["avg_cost"] = (
                                req_qty,
                                buy_price,
                            )
                            # 매도 쿨다운 제거: 매수 후 바로 매도 가능 (조건 충족 시)

                            # 결과 행 업데이트: 없으면 새로 추가
                            if (
                                daily_records_by_ticker.get(replacement_ticker)
                                and daily_records_by_ticker[replacement_ticker]
                                and daily_records_by_ticker[replacement_ticker][-1]["date"] == dt
                            ):
                                row = daily_records_by_ticker[replacement_ticker][-1]
                                row.update(
                                    {
                                        "decision": "BUY_REPLACE",
                                        "trade_amount": buy_amount,
                                        "shares": req_qty,
                                        "pv": req_qty * buy_price,
                                        "avg_cost": buy_price,
                                        # 추천/리포트와 동일 포맷: 디스플레이명 + 금액 + 대체 정보
                                        "note": f"{DECISION_CONFIG['BUY_REPLACE']['display_name']} "
                                        f"{format_kr_money(buy_amount)} "
                                        f"({ticker_to_sell} 대체)",
                                    }
                                )
                            else:
                                daily_records_by_ticker.setdefault(replacement_ticker, []).append(
                                    {
                                        "date": dt,
                                        "price": buy_price,
                                        "shares": req_qty,
                                        "pv": req_qty * buy_price,
                                        "decision": "BUY_REPLACE",
                                        "avg_cost": buy_price,
                                        "trade_amount": buy_amount,
                                        "trade_profit": 0.0,
                                        "trade_pl_pct": 0.0,
                                        "note": replacement_note,
                                        "signal1": None,
                                        "signal2": None,
                                        "score": None,
                                        "filter": None,
                                    }
                                )
                            # 교체가 성공했으므로, held_stocks_with_scores를 업데이트하여 다음 대기 종목 평가에 반영
                            # 매도된 종목 제거
                            held_stocks_with_scores = [
                                s for s in held_stocks_with_scores if s["ticker"] != ticker_to_sell
                            ]
                            # 새로 매수한 종목 추가
                            held_stocks_with_scores.append(
                                {
                                    "ticker": replacement_ticker,
                                    "score": best_new_score,
                                    "category": wait_stock_category,
                                }
                            )
                            held_stocks_with_scores.sort(key=lambda x: x["score"])  # 다시 정렬
                            # 다음 대기 종목으로 계속 교체 시도 (하루에 여러 교체 가능)
                        else:
                            # 매수 실패 시, 매도만 실행된 상태가 됨. 다음 날 빈 슬롯에 매수 시도.
                            if (
                                daily_records_by_ticker.get(replacement_ticker)
                                and daily_records_by_ticker[replacement_ticker]
                                and daily_records_by_ticker[replacement_ticker][-1]["date"] == dt
                            ):
                                daily_records_by_ticker[replacement_ticker][-1]["note"] = "교체매수 현금부족"
                    else:
                        # 가격 정보가 유효하지 않으면 교체하지 않고 다음 대기 종목으로 넘어감
                        continue  # 다음 buy_ranked_candidate로 넘어감

            # 3. 매수하지 못한 후보에 사유 기록
            # 오늘 매수 또는 교체매수된 종목 목록을 만듭니다.
            bought_tickers_today = {
                ticker_symbol
                for ticker_symbol, records in daily_records_by_ticker.items()
                if records and records[-1]["date"] == dt and records[-1]["decision"] in ("BUY", "BUY_REPLACE")
            }

            held_categories_snapshot = calculate_held_categories(
                position_state, ticker_to_category, valid_core_holdings
            )
            held_categories_normalized = {
                str(cat).strip().upper() for cat in held_categories_snapshot if isinstance(cat, str)
            }
            for _, candidate_ticker in buy_ranked_candidates:
                if candidate_ticker not in bought_tickers_today:
                    if (
                        daily_records_by_ticker[candidate_ticker]
                        and daily_records_by_ticker[candidate_ticker][-1]["date"] == dt
                    ):
                        # RSI 차단이나 카테고리 중복 등 이미 note가 설정된 경우 덮어쓰지 않음
                        current_note = daily_records_by_ticker[candidate_ticker][-1].get("note", "")
                        if not current_note or current_note == "":
                            _apply_wait_note_if_empty(
                                daily_records_by_ticker,
                                candidate_ticker,
                                dt,
                                ticker_to_category,
                                held_categories_snapshot,
                                held_categories_normalized,
                                position_state,
                                score_today,
                                replace_threshold,
                            )

        # --- 당일 최종 라벨 오버라이드 (공용 라벨러) ---
        for tkr, rows in daily_records_by_ticker.items():
            if not rows:
                continue
            last_row = rows[-1]
            current_note = str(last_row.get("note") or "")

            # 리스크 오프 비중 조절 문구가 있으면 덮어쓰지 않음
            if "시장위험회피" in current_note:
                continue

            overrides = compute_net_trade_note(
                tkr=tkr,
                data_by_tkr={
                    tkr: {
                        "shares": last_row.get("shares", 0.0),
                        "price": last_row.get("price", 0.0),
                    }
                },
                buy_trades_today_map=buy_trades_today_map,
                sell_trades_today_map=sell_trades_today_map,
                current_decision=str(last_row.get("decision")),
            )
            if overrides:
                if overrides.get("state") == "SOLD":
                    last_row["decision"] = "SOLD"
                if overrides.get("note") is not None:
                    new_note = overrides["note"]
                    if current_note:
                        new_note = f"{new_note} | {current_note}"
                    last_row["note"] = new_note

        out_cash.append(
            {
                "date": dt,
                "price": 1.0,
                "cash": cash,
                "shares": 0,
                "pv": cash,
                "decision": "HOLD",
                "note": "",  # CASH는 문구 없음
            }
        )

    total_records = sum(len(v) for v in daily_records_by_ticker.values())
    expected_records = len(metrics_by_ticker) * len(union_index)
    if not quiet:
        logger.info(
            f"[백테스트] daily_records_by_ticker: {len(daily_records_by_ticker)}개 종목, "
            f"총 {total_records}개 레코드 (예상: {expected_records}개)"
        )

    result: dict[str, pd.DataFrame] = {}
    for ticker_symbol, records in daily_records_by_ticker.items():
        if records:
            result[ticker_symbol] = pd.DataFrame(records).set_index("date")
    if out_cash:
        result["CASH"] = pd.DataFrame(out_cash).set_index("date")

    return result
