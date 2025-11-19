"""
포트폴리오 백테스트 실행 모듈

전략 중립적인 포트폴리오 백테스트 로직을 제공합니다.
"""

from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import pandas as pd

from config import BACKTEST_SLIPPAGE, CATEGORY_EXCEPTIONS
from utils.indicators import calculate_ma_score
from utils.logger import get_app_logger
from utils.report import format_kr_money
from strategies.maps.labeler import compute_net_trade_note
from logic.common import select_candidates_by_category, calculate_held_categories, is_category_exception
from strategies.maps.constants import DECISION_CONFIG, DECISION_NOTES
from utils.memmap_store import MemmapPriceStore

logger = get_app_logger()


def _format_trend_break_phrase(ma_value: float | None, price_value: float | None, ma_period: Optional[int]) -> str:
    if ma_value is None or pd.isna(ma_value) or price_value is None or pd.isna(price_value):
        threshold = ma_value if (ma_value is not None and not pd.isna(ma_value)) else 0.0
        return f"{DECISION_NOTES['TREND_BREAK']}({threshold:,.0f}원 이하)"

    diff = ma_value - price_value
    direction = "낮습니다" if diff >= 0 else "높습니다"
    period_text = ""
    if ma_period:
        try:
            period_text = f"{int(ma_period)}일 "
        except (TypeError, ValueError):
            period_text = ""
    return f"{DECISION_NOTES['TREND_BREAK']}({period_text}평균 가격 {ma_value:,.0f}원 보다 {abs(diff):,.0f}원 {direction}.)"


def _format_min_score_phrase(score_value: Optional[float], min_buy_score: float) -> str:
    template = DECISION_NOTES.get("MIN_SCORE", "최소 {min_buy_score:.1f}점수 미만")
    try:
        base = template.format(min_buy_score=min_buy_score)
    except Exception:
        base = f"최소 {min_buy_score:.1f}점수 미만"

    if score_value is None or pd.isna(score_value):
        return f"{base} (현재 점수 없음)"
    return f"{base} (현재 {score_value:.1f})"


def _calculate_trade_price(
    current_index: int,
    total_days: int,
    open_values: any,
    close_values: any,
    country_code: str,
    is_buy: bool,
) -> float:
    """
    거래 가격 계산: 다음날 시초가 + 슬리피지

    Args:
        current_index: 현재 인덱스 (i)
        total_days: 전체 거래일 수
        open_values: Open 가격 배열
        close_values: Close 가격 배열
        country_code: 국가 코드
        is_buy: 매수 여부 (True: 매수, False: 매도)

    Returns:
        거래 가격
    """
    # 다음날 시초가 사용
    if current_index + 1 < total_days:
        next_open = open_values[current_index + 1]
        if pd.notna(next_open):
            base_price = float(next_open)
        else:
            # 다음날 시초가가 없으면 당일 종가 사용
            base_price = float(close_values[current_index]) if pd.notna(close_values[current_index]) else 0.0
    else:
        # 마지막 날은 당일 종가 사용
        base_price = float(close_values[current_index]) if pd.notna(close_values[current_index]) else 0.0

    if base_price <= 0:
        return 0.0

    # 슬리피지 적용
    slippage_config = BACKTEST_SLIPPAGE.get(country_code, BACKTEST_SLIPPAGE.get("kor", {}))

    if is_buy:
        # 매수: 시초가보다 높은 가격
        slippage_pct = slippage_config.get("buy_pct", 0.5)
        trade_price = base_price * (1 + slippage_pct / 100)
    else:
        # 매도: 시초가보다 낮은 가격
        slippage_pct = slippage_config.get("sell_pct", 0.5)
        trade_price = base_price * (1 - slippage_pct / 100)

    return trade_price


def _execute_individual_sells(
    position_state: Dict,
    valid_core_holdings: Set[str],
    metrics_by_ticker: Dict,
    today_prices: Dict[str, float],
    score_today: Dict[str, float],
    rsi_score_today: Dict[str, float],
    ticker_to_category: Dict[str, str],
    sell_rsi_categories_today: Set[str],
    sell_trades_today_map: Dict,
    daily_records_by_ticker: Dict,
    i: int,
    total_days: int,
    country_code: str,
    stop_loss_threshold: Optional[float],
    rsi_sell_threshold: float,
    cooldown_days: int,
    cash: float,
    current_holdings_value: float,
    ma_period: int,
    min_buy_score: float,
) -> tuple[float, float]:
    """개별 종목 매도 로직 (손절, RSI, 추세)"""
    for ticker, ticker_metrics in metrics_by_ticker.items():
        ticker_state, price = position_state[ticker], today_prices.get(ticker)

        if ticker_state["shares"] > 0 and pd.notna(price) and metrics_by_ticker[ticker]["available_mask"][i]:
            in_cooldown = i < ticker_state["sell_block_until"]
            decision = None
            hold_ret = (price / ticker_state["avg_cost"] - 1.0) * 100.0 if ticker_state["avg_cost"] > 0 else 0.0
            trend_phrase = DECISION_NOTES["TREND_BREAK"]

            # RSI 과매수 매도 조건 체크
            rsi_score_current = rsi_score_today.get(ticker, 0.0)

            if stop_loss_threshold is not None and hold_ret <= float(stop_loss_threshold):
                decision = "CUT_STOPLOSS"
            elif rsi_score_current >= rsi_sell_threshold:
                decision = "SELL_RSI"
            elif not pd.isna(score_today.get(ticker, float("nan"))) and score_today.get(ticker, 0.0) <= min_buy_score:
                decision = "SELL_TREND"
                ma_val_today = ticker_metrics["ma_values"][i]
                ma_val = float(ma_val_today) if not pd.isna(ma_val_today) else None
                ticker_ma_period = ticker_metrics.get("ma_period", ma_period)
                trend_phrase = _format_trend_break_phrase(ma_val, price, ticker_ma_period)

            # 핵심 보유 종목은 매도 신호 무시
            if decision and ticker in valid_core_holdings:
                decision = None

            if not decision:
                continue

            if in_cooldown and decision != "CUT_STOPLOSS":
                continue

            if decision:
                # 다음날 시초가 + 슬리피지로 매도 가격 계산
                sell_price = _calculate_trade_price(
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
                hold_ret = (sell_price / ticker_state["avg_cost"] - 1.0) * 100.0 if ticker_state["avg_cost"] > 0 else 0.0

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
                if cooldown_days > 0:
                    ticker_state["buy_block_until"] = i + cooldown_days

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
                    note_text = trend_phrase if trend_phrase else DECISION_NOTES["TREND_BREAK"]
                    row["note"] = note_text

    return cash, current_holdings_value


def _rank_buy_candidates(
    tickers_available_today: Set[str],
    position_state: Dict,
    buy_signal_today: Dict[str, int],
    score_today: Dict[str, float],
    i: int,
) -> List[Tuple[float, str]]:
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
    daily_records_by_ticker: Dict,
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
    daily_records_by_ticker: Dict,
    ticker: str,
    dt: pd.Timestamp,
    ticker_to_category: Dict[str, str],
    held_categories: Set[str],
    held_categories_normalized: Set[str],
) -> None:
    """WAIT 상태 종목에 대해 카테고리 중복 여부에 따라 노트를 설정합니다."""

    records = daily_records_by_ticker.get(ticker)
    if not (records and records[-1]["date"] == dt):
        return

    current_note = str(records[-1].get("note") or "").strip()
    if current_note:
        return

    category = ticker_to_category.get(ticker)
    normalized = str(category).strip().upper() if category else ""

    if (
        category
        and not is_category_exception(category)
        and (category in held_categories or (normalized and normalized in held_categories_normalized))
    ):
        records[-1]["note"] = DECISION_NOTES["CATEGORY_DUP"]
    else:
        records[-1]["note"] = DECISION_NOTES["PORTFOLIO_FULL"]


def _execute_new_buys(
    buy_ranked_candidates: List[Tuple[float, str]],
    position_state: Dict,
    valid_core_holdings: Set[str],
    ticker_to_category: Dict[str, str],
    sell_rsi_categories_today: Set[str],
    rsi_score_today: Dict[str, float],
    today_prices: Dict[str, float],
    metrics_by_ticker: Dict,
    daily_records_by_ticker: Dict,
    buy_trades_today_map: Dict,
    cash: float,
    current_holdings_value: float,
    top_n: int,
    rsi_sell_threshold: float,
    cooldown_days: int,
    i: int,
    total_days: int,
    dt: pd.Timestamp,
    country_code: str,
    initial_capital: float = 0.0,
) -> Tuple[float, float, Set[str], Set[str]]:
    """신규 매수 실행

    Returns:
        (cash, current_holdings_value, purchased_today, held_categories)
    """
    from logic.common import calculate_held_count, calculate_held_categories, check_buy_candidate_filters, calculate_buy_budget

    held_count = calculate_held_count(position_state)
    slots_to_fill = max(0, top_n - held_count)
    purchased_today: Set[str] = set()

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
        buy_price = _calculate_trade_price(
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
            if cooldown_days > 0:
                ticker_state["sell_block_until"] = max(ticker_state["sell_block_until"], i + cooldown_days)

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


def process_ticker_data(
    ticker: str,
    df: pd.DataFrame,
    etf_tickers: set,
    etf_ma_period: int,
    stock_ma_period: int,
    precomputed_entry: Optional[Mapping[str, Any]] = None,
    ma_type: str = "SMA",
    *,
    min_buy_score: float,
) -> Optional[Dict]:
    """
    개별 종목의 데이터를 처리하고 지표를 계산합니다.

    Args:
        ticker: 종목 티커
        df: 가격 데이터프레임
        etf_tickers: ETF 티커 집합
        etf_ma_period: ETF 이동평균 기간
        stock_ma_period: 주식 이동평균 기간
        ma_type: 이동평균 타입 (SMA, EMA, WMA, DEMA, TEMA, HMA)

    Returns:
        Dict: 계산된 지표들 또는 None (처리 실패 시)
    """
    if df is None and precomputed_entry is None:
        return None

    working_df = df
    if working_df is None and precomputed_entry:
        # Dummy frame to keep downstream logic consistent
        working_df = pd.DataFrame()

    if working_df is not None and isinstance(working_df.columns, pd.MultiIndex):
        working_df = working_df.copy()
        working_df.columns = working_df.columns.get_level_values(0)
        working_df = working_df.loc[:, ~working_df.columns.duplicated()]

    # 티커 유형에 따라 이동평균 기간 결정
    current_ma_period = etf_ma_period if ticker in etf_tickers else stock_ma_period

    close_prices = None
    open_prices = None
    if isinstance(precomputed_entry, Mapping):
        close_prices = precomputed_entry.get("close")
        open_prices = precomputed_entry.get("open")

    if close_prices is None:
        if working_df is None or len(working_df) < current_ma_period:
            return None

        price_series = None
        if isinstance(working_df.columns, pd.MultiIndex):
            cols = working_df.columns.get_level_values(0)
            working_df = working_df.copy()
            working_df.columns = cols
            working_df = working_df.loc[:, ~working_df.columns.duplicated()]

        if "unadjusted_close" in working_df.columns:
            price_series = working_df["unadjusted_close"]
        else:
            price_series = working_df["Close"]

        if isinstance(price_series, pd.DataFrame):
            price_series = price_series.iloc[:, 0]
        close_prices = price_series.astype(float)

        if len(close_prices) < current_ma_period:
            return None

    if open_prices is None:
        if working_df is not None and "Open" in working_df.columns:
            open_series = working_df["Open"]
            if isinstance(open_series, pd.DataFrame):
                open_series = open_series.iloc[:, 0]
            open_prices = open_series.astype(float)
        else:
            open_prices = close_prices.copy()

    # MAPS 전략 지표 계산
    from utils.moving_averages import calculate_moving_average

    ma_type_key = (ma_type or "SMA").upper()
    ma_key = f"{ma_type_key}_{int(current_ma_period)}"
    moving_average = None
    ma_score = None
    if isinstance(precomputed_entry, Mapping):
        ma_cache = precomputed_entry.get("ma") or {}
        ma_score_cache = precomputed_entry.get("ma_score") or {}
        moving_average = ma_cache.get(ma_key)
        ma_score = ma_score_cache.get(ma_key)

    if moving_average is None:
        moving_average = calculate_moving_average(close_prices, current_ma_period, ma_type)
    if ma_score is None:
        ma_score = calculate_ma_score(close_prices, moving_average)

    # 점수 기반 매수 시그널 지속일 계산
    from logic.common import calculate_consecutive_days

    consecutive_buy_days = calculate_consecutive_days(ma_score, min_buy_score)

    # RSI 전략 지표 계산
    from strategies.rsi.backtest import process_ticker_data_rsi

    rsi_score = None
    if isinstance(precomputed_entry, Mapping):
        rsi_score = precomputed_entry.get("rsi_score")
    if rsi_score is None or isinstance(rsi_score, float):
        rsi_data = process_ticker_data_rsi(close_prices)
        rsi_score = rsi_data.get("rsi_score") if rsi_data else pd.Series(dtype=float)

    return {
        "df": working_df if working_df is not None else df,
        "close": close_prices,
        "open": open_prices,  # 시초가 추가
        "ma": moving_average,
        "ma_score": ma_score,
        "rsi_score": rsi_score,
        "buy_signal_days": consecutive_buy_days,
        "ma_period": current_ma_period,
    }


def run_portfolio_backtest(
    stocks: List[Dict],
    initial_capital: float = 100_000_000.0,
    core_start_date: Optional[pd.Timestamp] = None,
    top_n: int = 10,
    date_range: Optional[List[str]] = None,
    country: str = "kor",
    prefetched_data: Optional[Dict[str, pd.DataFrame]] = None,
    prefetched_metrics: Optional[Mapping[str, Dict[str, Any]]] = None,
    price_store: Optional["MemmapPriceStore"] = None,
    trading_calendar: Optional[Sequence[pd.Timestamp]] = None,
    ma_period: int = 20,
    ma_type: str = "SMA",
    replace_threshold: float = 0.0,
    stop_loss_pct: float = -10.0,
    cooldown_days: int = 5,
    rsi_sell_threshold: float = 10.0,
    core_holdings: Optional[List[str]] = None,
    quiet: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    missing_ticker_sink: Optional[Set[str]] = None,
    *,
    min_buy_score: float,
) -> Dict[str, pd.DataFrame]:
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

    from logic.common import validate_portfolio_topn, validate_core_holdings

    validate_portfolio_topn(top_n)

    # 핵심 보유 종목 (강제 보유, TOPN 포함)

    core_holdings_tickers = set(core_holdings or [])
    universe_tickers_set = {stock["ticker"] for stock in stocks}
    valid_core_holdings = validate_core_holdings(core_holdings_tickers, universe_tickers_set)

    # ETF와 주식을 구분하여 처리
    etf_tickers = {stock["ticker"] for stock in stocks if stock.get("type") == "etf"}

    # 이동평균 계산에 필요한 과거 데이터를 확보하기 위한 추가 조회 범위(웜업)
    # (실제 데이터 요청은 상위 프리패치 단계에서 수행)

    # 시장 레짐 필터 제거됨 (항상 100% 투자)

    # 개별 종목 데이터 로딩 및 지표 계산
    # 티커별 카테고리 매핑 생성 (성능 최적화를 위해 딕셔너리로 변환)
    ticker_to_category = {stock["ticker"]: stock.get("category") for stock in stocks}
    etf_meta = {stock["ticker"]: stock for stock in stocks if stock.get("ticker")}
    metrics_by_ticker = {}
    tickers_to_process = [s["ticker"] for s in stocks]

    for ticker in tickers_to_process:
        df = None
        if prefetched_data and ticker in prefetched_data:
            df = prefetched_data[ticker]
        elif price_store is not None:
            df = price_store.get_frame(ticker)

        if df is None:
            raise RuntimeError(f"[백테스트] '{ticker}' 데이터가 프리패치/메모리맵에 없습니다. 튜닝 프리패치 단계를 확인하세요.")

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
        missing_set = {str(ticker).strip().upper() for ticker in missing_metrics if isinstance(ticker, str) and str(ticker).strip()}
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
        before_filter = len(union_index)
        union_index = union_index[union_index >= core_start_date]
        if not quiet:
            logger.info(
                f"[백테스트] 시작일 필터링: {before_filter}일 → {len(union_index)}일 (core_start_date={core_start_date.strftime('%Y-%m-%d')})"
            )

    if union_index.empty:
        logger.warning(f"[백테스트] union_index가 비어있습니다. core_start_date={core_start_date}, metrics_by_ticker={len(metrics_by_ticker)}")
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

    # 시장 레짐 필터 제거됨 (항상 100% 투자)

    # 시뮬레이션 상태 변수 초기화
    position_state = {
        ticker: {
            "shares": 0,
            "avg_cost": 0.0,
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
        buy_trades_today_map: Dict[str, List[Dict[str, float]]] = {}
        sell_trades_today_map: Dict[str, List[Dict[str, float]]] = {}

        # SELL_RSI로 매도한 카테고리 추적 (같은 날 매수 금지)
        sell_rsi_categories_today: Set[str] = set()

        tickers_available_today: List[str] = []
        today_prices: Dict[str, float] = {}
        score_today: Dict[str, float] = {}
        rsi_score_today: Dict[str, float] = {}
        buy_signal_today: Dict[str, int] = {}

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
                        note = _format_min_score_phrase(score_check, min_buy_score)

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
            cooldown_days=cooldown_days,
            cash=cash,
            current_holdings_value=current_holdings_value,
            ma_period=ma_period,
            min_buy_score=min_buy_score,
        )

        # --- 3-1. 핵심 보유 종목 자동 매수 (최우선) ---
        for core_ticker in valid_core_holdings:
            if position_state[core_ticker]["shares"] == 0:
                # 핵심 보유 종목이 미보유 상태면 자동 매수
                if core_ticker in tickers_available_today:
                    price = today_prices.get(core_ticker)
                    if pd.notna(price) and price > 0 and cash > 0:
                        # 무조건 균등 비중: 초기자본 / TOPN
                        budget = initial_capital / top_n if top_n > 0 else 0

                        budget = min(budget, cash)  # 현금 부족 시 현금만큼만
                        shares_to_buy = budget / price if price > 0 else 0

                        if shares_to_buy > 0 and budget <= cash:
                            trade_amount = shares_to_buy * price
                            cash -= trade_amount
                            position_state[core_ticker]["shares"] = shares_to_buy
                            position_state[core_ticker]["avg_cost"] = price
                            position_state[core_ticker]["buy_block_until"] = i + cooldown_days

                            buy_trades_today_map.setdefault(core_ticker, []).append({"shares": float(shares_to_buy), "price": float(price)})

                            # 레코드 업데이트
                            if daily_records_by_ticker[core_ticker] and daily_records_by_ticker[core_ticker][-1]["date"] == dt:
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
            helper_candidates = [{"tkr": ticker, "score": score} for score, ticker in buy_ranked_candidates if ticker not in purchased_today]

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
                        _update_ticker_note(daily_records_by_ticker, replacement_ticker, dt, DECISION_NOTES["CATEGORY_DUP"])
                        continue  # 다음 buy_ranked_candidate로 넘어감
                elif weakest_held_stock:
                    # 다른 카테고리: 고정 종목 카테고리와 중복 체크 (이미 루프 밖에서 계산됨)
                    # 교체 대상 종목이 고정 종목 카테고리와 중복되면 차단
                    if wait_stock_category and wait_stock_category in core_categories:
                        _update_ticker_note(
                            daily_records_by_ticker,
                            replacement_ticker,
                            dt,
                            DECISION_NOTES["CATEGORY_DUP"],
                        )
                        continue  # 다음 교체 후보로 넘어감

                    # 같은 카테고리 종목이 없는 경우: 가장 약한 종목과 임계값 포함 비교
                    if best_new_score > weakest_held_stock["score"] + replace_threshold:
                        ticker_to_sell = weakest_held_stock["ticker"]
                        replacement_note = f"{ticker_to_sell}(을)를 {replacement_ticker}(으)로 교체 (새 카테고리)"
                    else:
                        # 임계값을 넘지 못하면 교체하지 않고 다음 대기 종목으로 넘어감
                        continue  # 다음 buy_ranked_candidate로 넘어감
                else:
                    # 보유 종목이 없으면 교체할 수 없음
                    continue  # 다음 buy_ranked_candidate로 넘어감

                # 교체할 종목이 결정되었으면 매도/매수 진행
                if ticker_to_sell:
                    # SELL_RSI로 매도한 카테고리는 같은 날 교체 매수 금지
                    replacement_category = ticker_to_category.get(replacement_ticker)
                    if replacement_category and not is_category_exception(replacement_category) and replacement_category in sell_rsi_categories_today:
                        if daily_records_by_ticker[replacement_ticker] and daily_records_by_ticker[replacement_ticker][-1]["date"] == dt:
                            daily_records_by_ticker[replacement_ticker][-1]["note"] = f"RSI 과매수 매도 카테고리 ({replacement_category})"
                        continue  # 다음 교체 후보로 넘어감

                    # RSI 과매수 종목 교체 매수 차단
                    rsi_score_replace_candidate = rsi_score_today.get(replacement_ticker, 0.0)

                    if rsi_score_replace_candidate >= rsi_sell_threshold:
                        # RSI 과매수 종목은 교체 매수하지 않음
                        if daily_records_by_ticker[replacement_ticker] and daily_records_by_ticker[replacement_ticker][-1]["date"] == dt:
                            daily_records_by_ticker[replacement_ticker][-1]["note"] = f"RSI 과매수 (RSI점수: {rsi_score_replace_candidate:.1f})"
                        continue  # 다음 교체 후보로 넘어감

                    sell_price = today_prices.get(ticker_to_sell)
                    buy_price = today_prices.get(replacement_ticker)

                    if pd.notna(sell_price) and sell_price > 0 and pd.notna(buy_price) and buy_price > 0:
                        # (a) 교체 대상 종목 매도
                        weakest_state = position_state[ticker_to_sell]
                        sell_qty = weakest_state["shares"]
                        sell_amount = sell_qty * sell_price
                        hold_ret = (sell_price / weakest_state["avg_cost"] - 1.0) * 100.0 if weakest_state["avg_cost"] > 0 else 0.0
                        trade_profit = (sell_price - weakest_state["avg_cost"]) * sell_qty if weakest_state["avg_cost"] > 0 else 0.0

                        cash += sell_amount
                        current_holdings_value = max(0.0, current_holdings_value - sell_amount)
                        weakest_state["shares"], weakest_state["avg_cost"] = 0, 0.0
                        if cooldown_days > 0:
                            weakest_state["buy_block_until"] = i + cooldown_days

                        if daily_records_by_ticker[ticker_to_sell] and daily_records_by_ticker[ticker_to_sell][-1]["date"] == dt:
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
                            if cooldown_days > 0:
                                new_ticker_state["sell_block_until"] = max(new_ticker_state["sell_block_until"], i + cooldown_days)

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
                            held_stocks_with_scores = [s for s in held_stocks_with_scores if s["ticker"] != ticker_to_sell]
                            # 새로 매수한 종목 추가
                            held_stocks_with_scores.append(
                                {
                                    "ticker": replacement_ticker,
                                    "score": best_new_score,
                                    "category": wait_stock_category,
                                }
                            )
                            held_stocks_with_scores.sort(key=lambda x: x["score"])  # 다시 정렬
                            break  # 하나의 대기 종목으로 하나의 교체만 시도하므로, 다음 날로 넘어감
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

            held_categories_snapshot = calculate_held_categories(position_state, ticker_to_category, valid_core_holdings)
            held_categories_normalized = {str(cat).strip().upper() for cat in held_categories_snapshot if isinstance(cat, str)}
            for _, candidate_ticker in buy_ranked_candidates:
                if candidate_ticker not in bought_tickers_today:
                    if daily_records_by_ticker[candidate_ticker] and daily_records_by_ticker[candidate_ticker][-1]["date"] == dt:
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
            f"[백테스트] daily_records_by_ticker: {len(daily_records_by_ticker)}개 종목, 총 {total_records}개 레코드 (예상: {expected_records}개)"
        )

    result: Dict[str, pd.DataFrame] = {}
    for ticker_symbol, records in daily_records_by_ticker.items():
        if records:
            result[ticker_symbol] = pd.DataFrame(records).set_index("date")
    if out_cash:
        result["CASH"] = pd.DataFrame(out_cash).set_index("date")

    return result
