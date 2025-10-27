"""실제 거래 내역 기반 성과 계산 모듈"""

from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

from config import BACKTEST_SLIPPAGE
from utils.data_loader import fetch_ohlcv
from utils.db_manager import get_db_connection
from utils.logger import get_app_logger

logger = get_app_logger()


def calculate_actual_performance(
    account_id: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    initial_capital: float,
    country_code: str = "kor",
) -> Optional[Dict[str, Any]]:
    """
    실제 거래 내역 기반으로 수익률을 계산합니다.

    Args:
        account_id: 계정 ID
        start_date: 시작일
        end_date: 종료일
        initial_capital: 초기 자본
        country_code: 국가 코드

    Returns:
        수익률 정보 딕셔너리 또는 None (거래 내역 없음)
    """
    db = get_db_connection()
    if db is None:
        logger.warning("MongoDB 연결 실패")
        return None

    # 거래 내역 조회
    trades = list(db.trades.find({"account": account_id.lower(), "executed_at": {"$gte": start_date, "$lte": end_date}}).sort("executed_at", 1))

    if not trades:
        logger.info(f"[{account_id}] 거래 내역이 없습니다.")
        return None

    logger.info(f"[{account_id}] {len(trades)}개의 거래 내역을 찾았습니다.")

    # 필요한 종목별 가격 데이터 조회
    tickers = list(set(t["ticker"] for t in trades))
    price_cache = {}

    for ticker in tickers:
        df = fetch_ohlcv(ticker, country=country_code, date_range=[start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")])
        if df is not None and not df.empty:
            price_cache[ticker] = df

    # 슬리피지 없음 (벤치마크와 동일 조건으로 비교)
    buy_slippage = 0.0
    sell_slippage = 0.0

    # 포지션 추적
    positions = {}  # {ticker: {"shares": float, "avg_cost": float}}
    cash = initial_capital

    # 날짜별로 거래 그룹화
    from collections import defaultdict

    trades_by_date = defaultdict(list)
    for trade in trades:
        trade_date = pd.to_datetime(trade["executed_at"]).normalize()
        trades_by_date[trade_date].append(trade)

    # 날짜별로 처리
    for trade_date in sorted(trades_by_date.keys()):
        day_trades = trades_by_date[trade_date]

        # 당일 종가 기준 평가액 계산
        current_value = cash
        for ticker, pos in positions.items():
            if pos["shares"] > 0 and ticker in price_cache:
                df = price_cache[ticker]
                if trade_date in df.index:
                    close_price = float(df.loc[trade_date, "Close"])
                else:
                    valid_dates = df.index[df.index <= trade_date]
                    if len(valid_dates) > 0:
                        close_price = float(df.loc[valid_dates[-1], "Close"])
                    else:
                        close_price = pos["avg_cost"]
                current_value += pos["shares"] * close_price

        # 매도 먼저 처리
        for trade in day_trades:
            if trade["action"].upper() != "SELL":
                continue

            ticker = trade["ticker"]
            if ticker not in price_cache:
                continue

            df = price_cache[ticker]
            if trade_date in df.index:
                base_price = float(df.loc[trade_date, "Close"])
            else:
                valid_dates = df.index[df.index <= trade_date]
                if len(valid_dates) == 0:
                    continue
                base_price = float(df.loc[valid_dates[-1], "Close"])

            price = base_price * (1 - sell_slippage)

            if ticker in positions and positions[ticker]["shares"] > 0:
                # 전량 매도
                sell_shares = positions[ticker]["shares"]
                cash += sell_shares * price
                positions[ticker] = {"shares": 0, "avg_cost": 0}

        # 매도 후 보유 종목 수 계산
        current_holdings_after_sell = sum(1 for p in positions.values() if p["shares"] > 0)

        # 매수 처리
        buy_trades = [t for t in day_trades if t["action"].upper() == "BUY"]
        if buy_trades:
            # 매수 후 최종 보유 종목 수
            final_holdings = current_holdings_after_sell + len(buy_trades)

            for trade in buy_trades:
                ticker = trade["ticker"]
                if ticker not in price_cache:
                    continue

                df = price_cache[ticker]
                if trade_date in df.index:
                    base_price = float(df.loc[trade_date, "Close"])
                else:
                    valid_dates = df.index[df.index <= trade_date]
                    if len(valid_dates) == 0:
                        continue
                    base_price = float(df.loc[valid_dates[-1], "Close"])

                price = base_price * (1 + buy_slippage)

                # 최종 보유 종목 수로 균등 분배
                shares = (current_value / final_holdings) / price

                if ticker not in positions:
                    positions[ticker] = {"shares": 0, "avg_cost": 0}

                old_shares = positions[ticker]["shares"]
                old_cost = positions[ticker]["avg_cost"]

                new_shares = old_shares + shares
                new_cost = (old_shares * old_cost + shares * price) / new_shares if new_shares > 0 else price

                positions[ticker] = {"shares": new_shares, "avg_cost": new_cost}
                cash -= shares * price

    # 현재 평가액 계산
    current_value = cash
    for ticker, pos in positions.items():
        if pos["shares"] > 0 and ticker in price_cache:
            df = price_cache[ticker]
            latest_price = float(df["Close"].iloc[-1])
            current_value += pos["shares"] * latest_price

    # 수익률 계산
    cumulative_return_pct = (current_value / initial_capital - 1) * 100 if initial_capital > 0 else 0.0

    return {
        "cumulative_return_pct": cumulative_return_pct,
        "initial_capital": initial_capital,
        "current_value": current_value,
        "cash": cash,
        "positions": positions,
        "trade_count": len(trades),
        "method": "actual_trades",  # 실제 거래 기반
    }


__all__ = ["calculate_actual_performance"]
