"""
포트폴리오 백테스트 실행 모듈

전략 중립적인 포트폴리오 백테스트 로직을 제공합니다.
"""

from typing import Callable, Dict, List, Optional, Set, Tuple

import pandas as pd

from config import BACKTEST_SLIPPAGE
from utils.data_loader import fetch_ohlcv
from utils.indicators import calculate_ma_score
from utils.logger import get_app_logger
from utils.report import format_kr_money, format_aud_money
from strategies.maps.labeler import compute_net_trade_note
from logic.common import select_candidates_by_category
from strategies.maps.constants import DECISION_NOTES, DECISION_CONFIG

logger = get_app_logger()


def _load_market_regime_data(
    regime_filter_enabled: bool,
    regime_filter_ticker: str,
    regime_filter_ma_period: int,
    regime_filter_country: str,
    country_code: str,
    fetch_date_range: List[str],
    prefetched_data: Optional[Dict[str, pd.DataFrame]],
) -> Optional[pd.DataFrame]:
    """시장 레짐 필터 데이터를 로딩합니다."""
    if not regime_filter_enabled:
        return None

    regime_country_code = (regime_filter_country or country_code).strip().lower() or country_code
    market_regime_df = None

    if prefetched_data and regime_filter_ticker in prefetched_data:
        market_regime_df = prefetched_data.get(regime_filter_ticker)

    if market_regime_df is None or market_regime_df.empty:
        market_regime_df = fetch_ohlcv(
            regime_filter_ticker,
            country=regime_country_code,
            date_range=fetch_date_range,
            cache_country="common",
            skip_realtime=True,
        )

    if market_regime_df is None or market_regime_df.empty:
        logger.warning(
            "시장 레짐 필터 티커(%s)의 데이터를 가져올 수 없어 필터를 비활성화합니다.",
            regime_filter_ticker,
        )
        return None

    market_regime_df = market_regime_df.sort_index()
    market_regime_df["MA"] = market_regime_df["Close"].rolling(window=regime_filter_ma_period).mean()
    return market_regime_df


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


def _execute_partial_regime_trim(
    position_state: Dict,
    valid_core_holdings: Set[str],
    current_holdings_value: float,
    equity: float,
    risk_off_equity_ratio: float,
    risk_off_equity_ratio_pct: float,
    today_prices: Dict[str, float],
    sell_trades_today_map: Dict,
    daily_records_by_ticker: Dict,
    cash: float,
) -> tuple[float, float]:
    """위험 회피 구간에서 목표 비중 유지 (부분 청산)"""
    desired_holdings_value = equity * risk_off_equity_ratio
    tolerance = max(1e-6 * max(1.0, equity), 1e-6)

    if desired_holdings_value >= current_holdings_value - tolerance or current_holdings_value <= 0:
        return cash, current_holdings_value

    scale_factor = desired_holdings_value / current_holdings_value if current_holdings_value > 0 else 0.0
    scale_factor = max(0.0, min(1.0, scale_factor))

    for held_ticker, held_state in position_state.items():
        # 핵심 보유 종목은 부분 청산에서도 제외
        if held_ticker in valid_core_holdings:
            continue

        shares_before = float(held_state["shares"])
        if shares_before <= 0:
            continue

        price_now = today_prices.get(held_ticker)
        if not pd.notna(price_now) or price_now <= 0:
            continue

        target_shares = shares_before * scale_factor
        sell_qty = shares_before - target_shares
        if sell_qty <= 1e-8:
            continue

        avg_cost_before = float(held_state["avg_cost"])
        trade_amount = sell_qty * price_now
        trade_profit = (price_now - avg_cost_before) * sell_qty if avg_cost_before > 0 else 0.0
        hold_ret = (price_now / avg_cost_before - 1.0) * 100.0 if avg_cost_before > 0 else 0.0

        sell_trades_today_map.setdefault(held_ticker, []).append({"shares": float(sell_qty), "price": float(price_now)})

        remaining_shares = target_shares
        if remaining_shares <= 1e-6:
            remaining_shares = 0.0
            held_state["avg_cost"] = 0.0
        held_state["shares"] = remaining_shares

        cash += trade_amount
        current_holdings_value = max(0.0, current_holdings_value - trade_amount)

        row = daily_records_by_ticker[held_ticker][-1]
        prev_trade_amount = float(row.get("trade_amount") or 0.0)
        prev_trade_profit = float(row.get("trade_profit") or 0.0)
        note_text = f"{DECISION_NOTES['RISK_OFF_TRIM']} (보유목표 {int(risk_off_equity_ratio_pct)}%)"
        row.update(
            {
                "decision": "HOLD",
                "trade_amount": prev_trade_amount + trade_amount,
                "trade_profit": prev_trade_profit + trade_profit,
                "trade_pl_pct": hold_ret,
                "shares": remaining_shares,
                "pv": remaining_shares * price_now,
                "avg_cost": held_state["avg_cost"],
                "note": note_text,
            }
        )

    return cash, current_holdings_value


def _execute_regime_sell(
    position_state: Dict,
    valid_core_holdings: Set[str],
    today_prices: Dict[str, float],
    sell_trades_today_map: Dict,
    daily_records_by_ticker: Dict,
    risk_off_equity_ratio_pct: float,
    cash: float,
    current_holdings_value: float,
) -> tuple[float, float]:
    """시장 레짐 필터에 의한 강제 매도"""
    for held_ticker, held_state in position_state.items():
        # 핵심 보유 종목은 시장 레짐 필터 매도에서도 제외
        if held_ticker in valid_core_holdings:
            continue
        if held_state["shares"] > 0:
            price = today_prices.get(held_ticker)
            if pd.notna(price):
                qty = held_state["shares"]
                trade_amount = qty * price
                hold_ret = (price / held_state["avg_cost"] - 1.0) * 100.0 if held_state["avg_cost"] > 0 else 0.0
                trade_profit = (price - held_state["avg_cost"]) * qty if held_state["avg_cost"] > 0 else 0.0

                sell_trades_today_map.setdefault(held_ticker, []).append({"shares": float(qty), "price": float(price)})

                cash += trade_amount
                current_holdings_value = max(0.0, current_holdings_value - trade_amount)
                held_state["shares"], held_state["avg_cost"] = 0, 0.0

                # 이미 만들어둔 행을 업데이트
                row = daily_records_by_ticker[held_ticker][-1]
                row.update(
                    {
                        "decision": "SOLD",
                        "trade_amount": trade_amount,
                        "trade_profit": trade_profit,
                        "trade_pl_pct": hold_ret,
                        "shares": 0,
                        "pv": 0,
                        "avg_cost": 0,
                        "note": f"{DECISION_NOTES['RISK_OFF_TRIM']} (보유목표 {int(risk_off_equity_ratio_pct)}%)",
                    }
                )
    return cash, current_holdings_value


def _execute_individual_sells(
    position_state: Dict,
    valid_core_holdings: Set[str],
    metrics_by_ticker: Dict,
    today_prices: Dict[str, float],
    ma_today: Dict[str, float],
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
) -> tuple[float, float]:
    """개별 종목 매도 로직 (손절, RSI, 추세)"""
    for ticker, ticker_metrics in metrics_by_ticker.items():
        ticker_state, price = position_state[ticker], today_prices.get(ticker)

        if (
            ticker_state["shares"] > 0
            and pd.notna(price)
            and i >= ticker_state["sell_block_until"]
            and metrics_by_ticker[ticker]["available_mask"][i]
        ):
            decision = None
            hold_ret = (price / ticker_state["avg_cost"] - 1.0) * 100.0 if ticker_state["avg_cost"] > 0 else 0.0

            # RSI 과매수 매도 조건 체크
            rsi_score_current = rsi_score_today.get(ticker, 100.0)

            if stop_loss_threshold is not None and hold_ret <= float(stop_loss_threshold):
                decision = "CUT_STOPLOSS"
            elif rsi_score_current <= rsi_sell_threshold:
                decision = "SELL_RSI"
            elif price < ma_today[ticker]:
                decision = "SELL_TREND"

            # 핵심 보유 종목은 매도 신호 무시
            if decision and ticker in valid_core_holdings:
                decision = None

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
                    if sold_category and sold_category != "TBD":
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


def _execute_replacement_trades(
    replacement_candidates: List[Dict],
    position_state: Dict,
    valid_core_holdings: Set[str],
    held_categories: Set[str],
    sell_rsi_categories_today: Set[str],
    ticker_to_category: Dict[str, str],
    score_today: Dict[str, float],
    rsi_score_today: Dict[str, float],
    today_prices: Dict[str, float],
    replace_threshold: float,
    rsi_sell_threshold: float,
    cooldown_days: int,
    cash: float,
    current_holdings_value: float,
    stocks: List[Dict],
    daily_records_by_ticker: Dict,
    sell_trades_today_map: Dict,
    buy_trades_today_map: Dict,
    dt: pd.Timestamp,
    DECISION_NOTES: Dict,
) -> tuple[float, float]:
    """교체 매매 실행

    Returns:
        (cash, current_holdings_value)
    """
    # 교체 대상이 될 수 있는 보유 종목 목록 생성
    held_stocks_with_scores = []
    for held_ticker, held_position in position_state.items():
        # 핵심 보유 종목은 교체 매매 대상에서 제외
        if held_ticker in valid_core_holdings:
            continue
        if held_position["shares"] > 0:
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

    for candidate in replacement_candidates:
        replacement_ticker = candidate["tkr"]
        wait_stock_category = ticker_to_category.get(replacement_ticker)
        best_new_score_raw = candidate.get("score")
        try:
            best_new_score = float(best_new_score_raw)
        except (TypeError, ValueError):
            best_new_score = float("-inf")

        # 교체 대상 찾기
        held_stock_same_category = next(
            (s for s in held_stocks_with_scores if s["category"] == wait_stock_category),
            None,
        )
        weakest_held_stock = held_stocks_with_scores[0] if held_stocks_with_scores else None

        # 교체 여부 결정
        ticker_to_sell = None
        replacement_note = ""

        if held_stock_same_category:
            # 같은 카테고리: 점수만 비교
            if best_new_score > held_stock_same_category["score"] + replace_threshold:
                ticker_to_sell = held_stock_same_category["ticker"]
                replacement_note = f"{ticker_to_sell}(을)를 {replacement_ticker}(으)로 교체 (동일 카테고리)"
            else:
                if daily_records_by_ticker[replacement_ticker] and daily_records_by_ticker[replacement_ticker][-1]["date"] == dt:
                    stock_info = next((s for s in stocks if s["ticker"] == replacement_ticker), {})
                    stock_name = stock_info.get("name", replacement_ticker)
                    daily_records_by_ticker[replacement_ticker][-1]["note"] = f"{DECISION_NOTES['CATEGORY_DUP']} - {stock_name}({replacement_ticker})"
                continue
        elif weakest_held_stock:
            # 다른 카테고리: 가장 약한 종목과 임계값 비교
            if best_new_score > weakest_held_stock["score"] + replace_threshold:
                ticker_to_sell = weakest_held_stock["ticker"]
                replacement_note = f"{ticker_to_sell}(을)를 {replacement_ticker}(으)로 교체 (새 카테고리)"
            else:
                continue
        else:
            continue

        # 교체 실행
        if ticker_to_sell:
            # 교체 매수 필터링
            from logic.common import check_buy_candidate_filters

            replacement_category = ticker_to_category.get(replacement_ticker)
            rsi_score_replace = rsi_score_today.get(replacement_ticker, 100.0)

            # SELL_RSI 카테고리와 RSI 과매수만 체크 (카테고리 중복은 이미 교체 로직에서 처리됨)
            if replacement_category and replacement_category != "TBD" and replacement_category in sell_rsi_categories_today:
                if daily_records_by_ticker[replacement_ticker] and daily_records_by_ticker[replacement_ticker][-1]["date"] == dt:
                    daily_records_by_ticker[replacement_ticker][-1]["note"] = f"RSI 과매수 매도 카테고리 ({replacement_category})"
                continue

            if rsi_score_replace <= rsi_sell_threshold:
                if daily_records_by_ticker[replacement_ticker] and daily_records_by_ticker[replacement_ticker][-1]["date"] == dt:
                    daily_records_by_ticker[replacement_ticker][-1]["note"] = f"RSI 과매수 (RSI점수: {rsi_score_replace:.1f})"
                continue

            sell_price = today_prices.get(ticker_to_sell)
            buy_price = today_prices.get(replacement_ticker)

            if pd.notna(sell_price) and sell_price > 0 and pd.notna(buy_price) and buy_price > 0:
                # 매도
                weakest_state = position_state[ticker_to_sell]
                sell_qty = weakest_state["shares"]
                sell_amount = sell_qty * sell_price
                hold_ret = (sell_price / weakest_state["avg_cost"] - 1.0) * 100.0 if weakest_state["avg_cost"] > 0 else 0.0
                trade_profit = (sell_price - weakest_state["avg_cost"]) * sell_qty if weakest_state["avg_cost"] > 0 else 0.0

                sell_trades_today_map.setdefault(ticker_to_sell, []).append({"shares": float(sell_qty), "price": float(sell_price)})

                cash += sell_amount
                current_holdings_value = max(0.0, current_holdings_value - sell_amount)
                weakest_state["shares"], weakest_state["avg_cost"] = 0, 0.0
                if cooldown_days > 0:
                    weakest_state["buy_block_until"] = -1  # Will be set by caller

                # 매수
                req_qty = sell_amount / buy_price if buy_price > 0 else 0
                trade_amount = sell_amount

                if req_qty > 0:
                    replacement_state = position_state[replacement_ticker]
                    cash -= trade_amount
                    current_holdings_value += trade_amount
                    replacement_state["shares"] += req_qty
                    replacement_state["avg_cost"] = buy_price
                    if cooldown_days > 0:
                        replacement_state["sell_block_until"] = -1  # Will be set by caller

                    old_category = ticker_to_category.get(ticker_to_sell)
                    if old_category and old_category != "TBD":
                        held_categories.discard(old_category)
                    if replacement_category and replacement_category != "TBD":
                        held_categories.add(replacement_category)

                    buy_trades_today_map.setdefault(replacement_ticker, []).append({"shares": float(req_qty), "price": float(buy_price)})

                    # 레코드 업데이트
                    if daily_records_by_ticker[ticker_to_sell] and daily_records_by_ticker[ticker_to_sell][-1]["date"] == dt:
                        row_sell = daily_records_by_ticker[ticker_to_sell][-1]
                        row_sell.update(
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

                    if daily_records_by_ticker[replacement_ticker] and daily_records_by_ticker[replacement_ticker][-1]["date"] == dt:
                        row_buy = daily_records_by_ticker[replacement_ticker][-1]
                        row_buy.update(
                            {
                                "decision": "BUY_REPLACE",
                                "trade_amount": trade_amount,
                                "shares": replacement_state["shares"],
                                "pv": replacement_state["shares"] * buy_price,
                                "avg_cost": replacement_state["avg_cost"],
                                "note": replacement_note,
                            }
                        )

                    # 교체 성공 후 held_stocks_with_scores 업데이트
                    held_stocks_with_scores = [s for s in held_stocks_with_scores if s["ticker"] != ticker_to_sell]
                    held_stocks_with_scores.append(
                        {
                            "ticker": replacement_ticker,
                            "score": best_new_score,
                            "category": replacement_category,
                        }
                    )
                    held_stocks_with_scores.sort(key=lambda x: x["score"])

    return cash, current_holdings_value


def _process_ticker_data(
    ticker: str,
    df: pd.DataFrame,
    etf_tickers: set,
    etf_ma_period: int,
    stock_ma_period: int,
    ma_type: str = "SMA",
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
    if df is None:
        return None

    # yfinance MultiIndex 컬럼 처리
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        df = df.loc[:, ~df.columns.duplicated()]

    # 티커 유형에 따라 이동평균 기간 결정
    current_ma_period = etf_ma_period if ticker in etf_tickers else stock_ma_period

    if len(df) < current_ma_period:
        return None

    price_series = None
    if "unadjusted_close" in df.columns:
        price_series = df["unadjusted_close"]
    else:
        price_series = df["Close"]

    if isinstance(price_series, pd.DataFrame):
        price_series = price_series.iloc[:, 0]
    close_prices = price_series.astype(float)

    # Open 가격 추출 (시초가 거래용)
    open_series = None
    if "Open" in df.columns:
        open_series = df["Open"]
        if isinstance(open_series, pd.DataFrame):
            open_series = open_series.iloc[:, 0]
        open_prices = open_series.astype(float)
    else:
        open_prices = close_prices.copy()  # Open 데이터 없으면 Close 사용

    # MAPS 전략 지표 계산
    from utils.moving_averages import calculate_moving_average

    moving_average = calculate_moving_average(close_prices, current_ma_period, ma_type)
    ma_score = calculate_ma_score(close_prices, moving_average, normalize=False)

    # 점수 기반 매수 시그널 지속일 계산
    from logic.common import calculate_consecutive_days

    consecutive_buy_days = calculate_consecutive_days(ma_score)

    # RSI 전략 지표 계산
    from strategies.rsi.backtest import process_ticker_data_rsi

    rsi_data = process_ticker_data_rsi(close_prices)
    rsi_score = rsi_data.get("rsi_score") if rsi_data else pd.Series(dtype=float)

    return {
        "df": df,
        "close": close_prices,
        "open": open_prices,  # 시초가 추가
        "ma": moving_average,
        "ma_score": ma_score,
        "rsi_score": rsi_score,
        "buy_signal_days": consecutive_buy_days,
    }


def run_portfolio_backtest(
    stocks: List[Dict],
    initial_capital: float = 100_000_000.0,
    core_start_date: Optional[pd.Timestamp] = None,
    top_n: int = 10,
    date_range: Optional[List[str]] = None,
    country: str = "kor",
    prefetched_data: Optional[Dict[str, pd.DataFrame]] = None,
    ma_period: int = 20,
    ma_type: str = "SMA",
    replace_threshold: float = 0.0,
    regime_filter_enabled: bool = False,
    regime_filter_ticker: str = "^GSPC",
    regime_filter_ma_period: int = 200,
    regime_filter_country: str = "",
    regime_filter_delay_days: int = 0,
    regime_filter_equity_ratio: int = 100,
    regime_behavior: str = "sell_all",
    stop_loss_pct: float = -10.0,
    cooldown_days: int = 5,
    rsi_sell_threshold: float = 10.0,
    core_holdings: Optional[List[str]] = None,
    quiet: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    missing_ticker_sink: Optional[Set[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    이동평균 기반 모멘텀 전략으로 포트폴리오 백테스트를 실행합니다.

    Args:
        stocks: 백테스트할 종목 목록
        initial_capital: 초기 자본금
        core_start_date: 백테스트 시작일
        top_n: 포트폴리오 최대 보유 종목 수
        date_range: 백테스트 기간 [시작일, 종료일]
        country: 시장 국가 코드 (예: kor, aus)
        prefetched_data: 미리 로드된 가격 데이터
        ma_period: 이동평균 기간
        replace_threshold: 종목 교체 임계값
        regime_filter_enabled: 시장 레짐 필터 사용 여부
        regime_filter_ticker: 레짐 필터 지수 티커
        regime_filter_ma_period: 레짐 필터 이동평균 기간
        regime_behavior: 레짐 필터 동작 방식
        regime_filter_delay_days: 레짐 필터 적용 시 참조할 지연 거래일 수
        regime_filter_equity_ratio: 레짐 위험 회피 구간에서 유지할 목표 주식 비중 (0~100)
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

    valid_regime_behaviors = {"sell_all", "hold_block_buy"}
    if regime_behavior not in valid_regime_behaviors:
        raise ValueError("regime_behavior must be one of {'sell_all', 'hold_block_buy'}")

    from logic.common import validate_portfolio_topn, validate_core_holdings

    validate_portfolio_topn(top_n)

    # 핵심 보유 종목 (강제 보유, TOPN 포함)

    core_holdings_tickers = set(core_holdings or [])
    universe_tickers_set = {stock["ticker"] for stock in stocks}
    valid_core_holdings = validate_core_holdings(core_holdings_tickers, universe_tickers_set)

    # ETF와 주식을 구분하여 처리
    etf_tickers = {stock["ticker"] for stock in stocks if stock.get("type") == "etf"}

    # 이동평균 계산에 필요한 과거 데이터를 확보하기 위한 추가 조회 범위(웜업)
    WARMUP_MONTHS = 12
    fetch_date_range = date_range
    if date_range and len(date_range) == 2 and date_range[0] is not None:
        core_start = pd.to_datetime(date_range[0])
        warmup_start = core_start - pd.DateOffset(months=WARMUP_MONTHS)
        fetch_date_range = [warmup_start.strftime("%Y-%m-%d"), date_range[1]]

    # --- 시장 레짐 필터 데이터 로딩 ---
    market_regime_df = _load_market_regime_data(
        regime_filter_enabled=regime_filter_enabled,
        regime_filter_ticker=regime_filter_ticker,
        regime_filter_ma_period=regime_filter_ma_period,
        regime_filter_country=regime_filter_country,
        country_code=country_code,
        fetch_date_range=fetch_date_range,
        prefetched_data=prefetched_data,
    )
    if regime_filter_enabled and market_regime_df is None:
        regime_filter_enabled = False

    # 개별 종목 데이터 로딩 및 지표 계산
    # 티커별 카테고리 매핑 생성 (성능 최적화를 위해 딕셔너리로 변환)
    ticker_to_category = {stock["ticker"]: stock.get("category") for stock in stocks}
    etf_meta = {stock["ticker"]: stock for stock in stocks if stock.get("ticker")}
    metrics_by_ticker = {}
    tickers_to_process = [s["ticker"] for s in stocks]

    for ticker in tickers_to_process:
        # 미리 로드된 데이터가 있으면 사용하고, 없으면 새로 조회
        if prefetched_data and ticker in prefetched_data:
            df = prefetched_data[ticker]
        else:
            # prefetched_data가 없으면 date_range를 사용하여 직접 조회
            df = fetch_ohlcv(ticker, country=country, date_range=fetch_date_range, skip_realtime=True)

        # 공통 함수를 사용하여 데이터 처리 및 지표 계산
        ticker_metrics = _process_ticker_data(ticker, df, etf_tickers, etf_ma_period, stock_ma_period, ma_type)
        if ticker_metrics:
            metrics_by_ticker[ticker] = ticker_metrics

    missing_metrics = [t for t in tickers_to_process if t not in metrics_by_ticker]
    if missing_metrics:
        missing_set = {str(ticker).strip().upper() for ticker in missing_metrics if isinstance(ticker, str) and str(ticker).strip()}
        if missing_ticker_sink is not None:
            missing_ticker_sink.update(missing_set)
        else:
            logger.warning("가격 데이터 부족으로 제외된 종목: %s", ", ".join(sorted(missing_set)))

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

    normalized_union_index = union_index.normalize()

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

    market_close_arr = market_ma_arr = None
    if regime_filter_enabled and market_regime_df is not None and not market_regime_df.empty:
        aligned_regime_df = market_regime_df.copy()
        aligned_regime_df.index = aligned_regime_df.index.normalize()
        aligned_regime_df = aligned_regime_df.reindex(normalized_union_index)
        market_close_arr = aligned_regime_df["Close"].to_numpy()
        market_ma_arr = aligned_regime_df["MA"].to_numpy()
    else:
        market_regime_df = None

    try:
        regime_delay_offset = int(regime_filter_delay_days)
    except (TypeError, ValueError):
        regime_delay_offset = 0
    else:
        if regime_delay_offset < 0:
            regime_delay_offset = 0

    try:
        risk_off_equity_ratio_pct = float(int(regime_filter_equity_ratio))
    except (TypeError, ValueError):
        risk_off_equity_ratio_pct = 100.0
    risk_off_equity_ratio_pct = min(100.0, max(0.0, risk_off_equity_ratio_pct))
    risk_off_equity_ratio = risk_off_equity_ratio_pct / 100.0

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
        ma_today: Dict[str, float] = {}
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

            ma_today[ticker] = float(ma_val) if not pd.isna(ma_val) else float("nan")
            score_today[ticker] = float(score_val) if not pd.isna(score_val) else 0.0
            rsi_score_today[ticker] = float(rsi_score_val) if not pd.isna(rsi_score_val) else 0.0
            buy_signal_today[ticker] = int(buy_signal_val) if not pd.isna(buy_signal_val) else 0

            if available:
                tickers_available_today.append(ticker)

        # RSI 과매수 경고 카테고리도 추적 (쿨다운으로 아직 매도 안 했지만 RSI 낮은 경우)
        for ticker, ticker_state in position_state.items():
            if ticker_state["shares"] > 0:
                rsi_val = rsi_score_today.get(ticker, 100.0)
                if rsi_val <= rsi_sell_threshold:
                    # 쿨다운으로 매도하지 못한 경우에도 카테고리 차단
                    if i < ticker_state["sell_block_until"]:
                        category = ticker_to_category.get(ticker)
                        if category and category != "TBD":
                            sell_rsi_categories_today.add(category)

        # --- 시장 레짐 필터 적용 (리스크 오프 조건 확인) ---
        # RSI 과매수 매도는 항상 활성화됨 (rsi_sell_threshold는 파라미터로 받음)

        is_risk_off = False
        if regime_filter_enabled and market_close_arr is not None:
            market_idx = i - regime_delay_offset
            if market_idx >= 0:
                market_price = market_close_arr[market_idx]
                market_ma = market_ma_arr[market_idx] if market_ma_arr is not None else float("nan")
                if not pd.isna(market_price) and not pd.isna(market_ma) and market_price < market_ma:
                    is_risk_off = True
                market_idx = i - regime_delay_offset
                if market_idx >= 0:
                    market_price = market_close_arr[market_idx]
                    market_ma = market_ma_arr[market_idx] if market_ma_arr is not None else float("nan")
                    if not pd.isna(market_price) and not pd.isna(market_ma) and market_price < market_ma:
                        is_risk_off = True

            risk_off_effective = is_risk_off and risk_off_equity_ratio < 1.0 and regime_behavior == "sell_all"
            full_exit = is_risk_off and regime_behavior == "sell_all" and risk_off_equity_ratio <= 0.0
            partial_regime_active = risk_off_effective and risk_off_equity_ratio > 0.0
            force_regime_sell = full_exit
            allow_individual_sells = True
            # 리스크 오프 상태에서도 신규 매수와 교체 매매 허용 (비중만 제한)
            allow_new_buys = True

            # 현재 총 보유 자산 가치를 계산합니다.
            current_holdings_value = 0
            for held_ticker, held_state in position_state.items():
                if held_state["shares"] > 0:
                    price_h = today_prices.get(held_ticker)
                    if pd.notna(price_h):
                        current_holdings_value += held_state["shares"] * price_h

            # 총 평가금액(현금 + 주식)을 계산합니다.
            equity = cash + current_holdings_value

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
                        note = "매도 쿨다운"
                    elif position_snapshot["shares"] == 0 and i < position_snapshot["buy_block_until"]:
                        note = "매수 쿨다운"

                # 핵심 보유 종목 표시
                if decision_out == "HOLD_CORE" and not note:
                    note = "🔒 핵심 보유"

                ma_value = ma_today.get(ticker, float("nan"))
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

            # --- 1-1. 위험 회피 구간에서 목표 비중 유지 (부분 청산) ---
            if partial_regime_active and current_holdings_value > 0:
                cash, current_holdings_value = _execute_partial_regime_trim(
                    position_state=position_state,
                    valid_core_holdings=valid_core_holdings,
                    current_holdings_value=current_holdings_value,
                    equity=equity,
                    risk_off_equity_ratio=risk_off_equity_ratio,
                    risk_off_equity_ratio_pct=risk_off_equity_ratio_pct,
                    today_prices=today_prices,
                    sell_trades_today_map=sell_trades_today_map,
                    daily_records_by_ticker=daily_records_by_ticker,
                    cash=cash,
                )
                equity = cash + current_holdings_value

                # 부분 청산 이후 slots_to_fill 재계산 (CORE 포함)
                held_count = sum(1 for pos in position_state.values() if pos["shares"] > 0)
                slots_to_fill = max(0, top_n - held_count)

            # --- 2. 매도 로직 ---
            if force_regime_sell:
                cash, current_holdings_value = _execute_regime_sell(
                    position_state=position_state,
                    valid_core_holdings=valid_core_holdings,
                    today_prices=today_prices,
                    sell_trades_today_map=sell_trades_today_map,
                    daily_records_by_ticker=daily_records_by_ticker,
                    risk_off_equity_ratio_pct=risk_off_equity_ratio_pct,
                    cash=cash,
                    current_holdings_value=current_holdings_value,
                )
            elif allow_individual_sells:
                cash, current_holdings_value = _execute_individual_sells(
                    position_state=position_state,
                    valid_core_holdings=valid_core_holdings,
                    metrics_by_ticker=metrics_by_ticker,
                    today_prices=today_prices,
                    ma_today=ma_today,
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
                )

            equity = cash + current_holdings_value

            # --- 3-1. 핵심 보유 종목 자동 매수 (최우선) ---
            for core_ticker in valid_core_holdings:
                if position_state[core_ticker]["shares"] == 0:
                    # 핵심 보유 종목이 미보유 상태면 자동 매수
                    if core_ticker in tickers_available_today:
                        price = today_prices.get(core_ticker)
                        if pd.notna(price) and price > 0 and cash > 0:
                            # 균등 분할 매수 (전체 자산 / TOPN)
                            total_slots = top_n
                            budget = equity / total_slots if total_slots > 0 else 0
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

            # --- 3. 매수 로직 (리스크 온일 때만) ---
            if allow_new_buys:
                # 1. 매수 후보 선정 (종합 점수 기준)
                buy_ranked_candidates = _rank_buy_candidates(
                    tickers_available_today=tickers_available_today,
                    position_state=position_state,
                    buy_signal_today=buy_signal_today,
                    score_today=score_today,
                    i=i,
                )

                # 2. 매수 실행 (신규 또는 교체) (CORE 포함)
                from logic.common import calculate_held_count

                held_count = calculate_held_count(position_state)
                slots_to_fill = max(0, top_n - held_count)

                purchased_today: Set[str] = set()

                if slots_to_fill > 0 and buy_ranked_candidates:
                    # 보유 중인 카테고리 (매수 시 중복 체크용)
                    from logic.common import calculate_held_categories

                    held_categories = calculate_held_categories(position_state, ticker_to_category)

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
                        from logic.common import check_buy_candidate_filters, calculate_buy_budget

                        category = ticker_to_category.get(ticker_to_buy)
                        rsi_score_buy_candidate = rsi_score_today.get(ticker_to_buy, 100.0)

                        can_buy, block_reason = check_buy_candidate_filters(
                            ticker=ticker_to_buy,
                            category=category,
                            held_categories=held_categories,
                            sell_rsi_categories_today=sell_rsi_categories_today,
                            rsi_score=rsi_score_buy_candidate,
                            rsi_sell_threshold=rsi_sell_threshold,
                        )

                        if not can_buy:
                            if daily_records_by_ticker[ticker_to_buy] and daily_records_by_ticker[ticker_to_buy][-1]["date"] == dt:
                                daily_records_by_ticker[ticker_to_buy][-1]["note"] = block_reason
                            continue

                        # 매수 예산 계산
                        budget = calculate_buy_budget(
                            cash=cash,
                            current_holdings_value=current_holdings_value,
                            top_n=top_n,
                            risk_off_effective=risk_off_effective,
                            risk_off_equity_ratio=risk_off_equity_ratio,
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
                            equity = cash + current_holdings_value
                            ticker_state["shares"] += req_qty
                            ticker_state["avg_cost"] = buy_price
                            if cooldown_days > 0:
                                ticker_state["sell_block_until"] = max(ticker_state["sell_block_until"], i + cooldown_days)

                            if category and category != "TBD":
                                held_categories.add(category)

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

                elif slots_to_fill <= 0 and buy_ranked_candidates:
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
                                if daily_records_by_ticker[replacement_ticker] and daily_records_by_ticker[replacement_ticker][-1]["date"] == dt:
                                    stock_info = next((s for s in stocks if s["ticker"] == replacement_ticker), {})
                                    stock_name = stock_info.get("name", replacement_ticker)
                                    daily_records_by_ticker[replacement_ticker][-1][
                                        "note"
                                    ] = f"{DECISION_NOTES['CATEGORY_DUP']} - {stock_name}({replacement_ticker})"
                                continue  # 다음 buy_ranked_candidate로 넘어감
                        elif weakest_held_stock:
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
                            if replacement_category and replacement_category != "TBD" and replacement_category in sell_rsi_categories_today:
                                if daily_records_by_ticker[replacement_ticker] and daily_records_by_ticker[replacement_ticker][-1]["date"] == dt:
                                    daily_records_by_ticker[replacement_ticker][-1]["note"] = f"RSI 과매수 매도 카테고리 ({replacement_category})"
                                continue  # 다음 교체 후보로 넘어감

                            # RSI 과매수 종목 교체 매수 차단
                            rsi_score_replace_candidate = rsi_score_today.get(replacement_ticker, 100.0)

                            if rsi_score_replace_candidate <= rsi_sell_threshold:
                                # RSI 과매수 종목은 교체 매수하지 않음
                                if daily_records_by_ticker[replacement_ticker] and daily_records_by_ticker[replacement_ticker][-1]["date"] == dt:
                                    daily_records_by_ticker[replacement_ticker][-1][
                                        "note"
                                    ] = f"RSI 과매수 (RSI점수: {rsi_score_replace_candidate:.1f})"
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
                                equity_base = cash + current_holdings_value
                                min_val = 1.0 / (top_n * 2.0) * equity_base
                                max_val = 1.0 / top_n * equity_base
                                budget = min(max_val, cash)
                                if budget <= 0 or budget < min_val:
                                    continue
                                if risk_off_effective:
                                    total_equity_now = cash + current_holdings_value
                                    target_holdings_limit = total_equity_now * risk_off_equity_ratio
                                    remaining_capacity = max(0.0, target_holdings_limit - current_holdings_value)
                                    if remaining_capacity <= 0:
                                        continue
                                    budget = min(budget, remaining_capacity)
                                    if budget <= 0:
                                        continue
                                # 수량/금액 산정
                                if country_code == "aus":
                                    req_qty = (budget / buy_price) if buy_price > 0 else 0
                                    buy_amount = budget
                                else:
                                    req_qty = int(budget // buy_price) if buy_price > 0 else 0
                                    buy_amount = req_qty * buy_price
                                    if req_qty <= 0 or buy_amount + 1e-9 < min_val:
                                        continue

                                # 체결 반영
                                if req_qty > 0 and buy_amount <= cash + 1e-9:
                                    new_ticker_state = position_state[replacement_ticker]
                                    cash -= buy_amount
                                    current_holdings_value += buy_amount
                                    equity = cash + current_holdings_value
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
                                                f"{format_aud_money(buy_amount) if country_code == 'aus' else format_kr_money(buy_amount)} "
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

                for _, candidate_ticker in buy_ranked_candidates:
                    if candidate_ticker not in bought_tickers_today:
                        if daily_records_by_ticker[candidate_ticker] and daily_records_by_ticker[candidate_ticker][-1]["date"] == dt:
                            # RSI 차단이나 카테고리 중복 등 이미 note가 설정된 경우 덮어쓰지 않음
                            current_note = daily_records_by_ticker[candidate_ticker][-1].get("note", "")
                            if not current_note or current_note == "":
                                # 포트폴리오 가득 참
                                if slots_to_fill <= 0:
                                    daily_records_by_ticker[candidate_ticker][-1]["note"] = DECISION_NOTES["PORTFOLIO_FULL"]
                                else:
                                    # 매수 시도했지만 실패 (RSI, 카테고리 중복, 리스크 오프 등은 이미 note 설정됨)
                                    # note가 없으면 포트폴리오 가득 참으로 표시
                                    daily_records_by_ticker[candidate_ticker][-1]["note"] = DECISION_NOTES["PORTFOLIO_FULL"]
            else:  # 리스크 오프 상태
                # 매수 후보가 있더라도, 시장이 위험 회피 상태이므로 매수하지 않음
                # 후보들에게 사유 기록
                risk_off_candidates = []
                if cash > 0:
                    for candidate_ticker in tickers_available_today:
                        ticker_state_cand = position_state[candidate_ticker]
                        buy_signal_days_today = buy_signal_today.get(candidate_ticker, 0)
                        if ticker_state_cand["shares"] == 0 and i >= ticker_state_cand["buy_block_until"] and buy_signal_days_today > 0:
                            risk_off_candidates.append(candidate_ticker)

                for candidate_ticker in risk_off_candidates:
                    if daily_records_by_ticker[candidate_ticker] and daily_records_by_ticker[candidate_ticker][-1]["date"] == dt:
                        daily_records_by_ticker[candidate_ticker][-1][
                            "note"
                        ] = f"{DECISION_NOTES['RISK_OFF_TRIM']} (보유목표 {int(risk_off_equity_ratio_pct)}%)"

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
                        last_row["note"] = overrides["note"]

            risk_off_note_for_day = ""
            if is_risk_off:
                risk_off_note_for_day = f"{DECISION_NOTES['RISK_OFF_TRIM']} (보유목표 {int(risk_off_equity_ratio_pct)}%)"
                for rows in daily_records_by_ticker.values():
                    if not rows:
                        continue
                    last_decision = str(rows[-1].get("decision") or "").upper()
                    if last_decision not in {"HOLD", "BUY", "BUY_REPLACE"}:
                        continue
                    record_note = str(rows[-1].get("note") or "")
                    if "시장위험회피" not in record_note:
                        rows[-1]["note"] = f"{record_note} | {risk_off_note_for_day}".strip(" |") if record_note else risk_off_note_for_day

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
