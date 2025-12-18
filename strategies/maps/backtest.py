"""
단일 종목 백테스트 모듈

개별 종목에 대한 백테스트를 수행합니다.
"""

import pandas as pd

from logic.backtest.portfolio import process_ticker_data, run_portfolio_backtest
from utils.data_loader import fetch_ohlcv
from utils.logger import get_app_logger

logger = get_app_logger()


def run_single_ticker_backtest(
    ticker: str,
    stock_type: str = "stock",
    df: pd.DataFrame | None = None,
    initial_capital: float = 1_000_000.0,
    core_start_date: pd.Timestamp | None = None,
    date_range: list[str] | None = None,
    country: str = "kor",
    ma_period: int = 20,
    stop_loss_pct: float = -10.0,
    cooldown_days: int = 5,
) -> pd.DataFrame:
    """
    단일 종목에 대해 이동평균선 교차 전략 백테스트를 실행합니다.

    Args:
        ticker: 종목 티커
        stock_type: 종목 유형 (etf 또는 stock)
        df: 미리 로드된 가격 데이터
        initial_capital: 초기 자본금
        core_start_date: 백테스트 시작일
        date_range: 백테스트 기간
        country: 시장 국가 코드
        ma_period: 이동평균 기간
        stop_loss_pct: 손절 비율
        cooldown_days: 거래 쿨다운 기간

    Returns:
        pd.DataFrame: 백테스트 결과
    """
    country_code = (country or "").strip().lower() or "kor"

    stop_loss_threshold = stop_loss_pct

    # 티커 유형에 따른 이동평균 기간 설정
    current_ma_period = ma_period
    if df is None:
        # df가 제공되지 않으면, date_range를 사용하여 직접 데이터를 조회합니다.
        # date_range가 없으면 기본값(3개월)으로 조회됩니다.
        # CLI 백테스트 실행 시에는 항상 date_range가 전달됩니다.
        df = fetch_ohlcv(ticker, country=country_code, date_range=date_range)

    if df is None or df.empty:
        return pd.DataFrame()

    # 공통 함수를 사용하여 데이터 처리 및 지표 계산
    ticker_metrics = process_ticker_data(
        ticker,
        df,
        ma_period=current_ma_period,
    )

    if not ticker_metrics:
        return pd.DataFrame()

    moving_average = ticker_metrics["ma"]
    consecutive_buy_days = ticker_metrics["buy_signal_days"]

    loop_start_index = 0
    if core_start_date is not None:
        try:
            loop_start_index = df.index.searchsorted(core_start_date, side="left")
        except Exception:
            pass

    available_cash = float(initial_capital)
    held_shares: float = 0.0
    average_cost = 0.0
    buy_cooldown_until = -1
    sell_cooldown_until = -1

    rows = []
    close_prices = ticker_metrics["close"]
    for i in range(loop_start_index, len(df)):
        price_val = close_prices.iloc[i]
        if pd.isna(price_val):
            continue
        current_price = float(price_val)
        decision, trade_amount, trade_profit, trade_pl_pct = None, 0.0, 0.0, 0.0

        ma_today = moving_average.iloc[i]

        if pd.isna(ma_today):
            rows.append(
                {
                    "date": df.index[i],
                    "price": current_price,
                    "cash": available_cash,
                    "shares": held_shares,
                    "pv": available_cash + held_shares * current_price,
                    "decision": "WAIT",
                    "avg_cost": average_cost,
                    "trade_amount": 0.0,
                    "trade_profit": 0.0,
                    "trade_pl_pct": 0.0,
                    "note": "웜업 기간",
                    "signal1": ma_today,
                    "signal2": None,
                    "score": 0.0,
                    "filter": consecutive_buy_days.iloc[i],
                }
            )
            continue

        if held_shares > 0 and i >= sell_cooldown_until:
            hold_return_pct = (current_price / average_cost - 1.0) * 100.0 if average_cost > 0 else 0.0

            if stop_loss_threshold is not None and hold_return_pct <= float(stop_loss_threshold):
                decision = "CUT_STOPLOSS"
            elif current_price < ma_today:
                decision = "SELL_TREND"

            if decision in ("CUT_STOPLOSS", "SELL_TREND"):
                trade_amount = held_shares * current_price
                if average_cost > 0:
                    trade_profit = (current_price - average_cost) * held_shares
                    trade_pl_pct = hold_return_pct
                available_cash += trade_amount
                held_shares, average_cost = 0, 0.0
                if cooldown_days > 0:
                    buy_cooldown_until = i + cooldown_days

        if decision is None and held_shares == 0 and i >= buy_cooldown_until:
            consecutive_buy_days_today = consecutive_buy_days.iloc[i]
            if consecutive_buy_days_today > 0:
                buy_quantity = int(available_cash // current_price)
                if buy_quantity > 0:
                    trade_amount = float(buy_quantity) * current_price
                    available_cash -= trade_amount
                    average_cost, held_shares = current_price, float(buy_quantity)
                    decision = "BUY"
                    if cooldown_days > 0:
                        sell_cooldown_until = i + cooldown_days

        if decision is None:
            decision = "HOLD" if held_shares > 0 else "WAIT"

        ma_score_today = 0.0
        if pd.notna(ma_today) and ma_today > 0:
            ma_score_today = ((current_price / ma_today) - 1.0) * 100.0

        rows.append(
            {
                "date": df.index[i],
                "price": current_price,
                "cash": available_cash,
                "shares": held_shares,
                "pv": available_cash + held_shares * current_price,
                "decision": decision,
                "avg_cost": average_cost,
                "trade_amount": trade_amount,
                "trade_profit": trade_profit,
                "trade_pl_pct": trade_pl_pct,
                "note": "",
                "signal1": ma_today,
                "signal2": None,
                "score": ma_score_today,
                "filter": consecutive_buy_days.iloc[i],
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("date")


__all__ = [
    "run_portfolio_backtest",
    "run_single_ticker_backtest",
]
