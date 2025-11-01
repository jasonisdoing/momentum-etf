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
    portfolio_topn: int = 12,
    rebalance_threshold: float = 0.3,
) -> Optional[Dict[str, Any]]:
    """
    실제 거래 내역 기반으로 수익률을 계산합니다.

    Args:
        account_id: 계정 ID
        start_date: 시작일
        end_date: 종료일
        initial_capital: 초기 자본
        country_code: 국가 코드
        portfolio_topn: 포트폴리오 최대 종목 수
        rebalance_threshold: 리밸런싱 임계값 (비중 편차 %)

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

    # 날짜별 수익률 추적
    daily_records = []
    prev_value = initial_capital

    # 날짜별로 거래 그룹화
    from collections import defaultdict

    trades_by_date = defaultdict(list)
    for trade in trades:
        trade_date = pd.to_datetime(trade["executed_at"]).normalize()
        trades_by_date[trade_date].append(trade)

    # 거래일 목록 생성 (시작일부터 종료일까지)
    from utils.data_loader import get_trading_days

    all_trading_days = get_trading_days(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), country_code)

    # 날짜별로 처리
    for current_date in all_trading_days:
        trade_date = pd.to_datetime(current_date).normalize()
        day_trades = trades_by_date.get(trade_date, [])

        # 당일 종가 기준 평가액 계산 (거래 전)
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

                # 매번 현재 보유 자산 평가액 계산 (백테스트 방식)
                current_holdings_value = 0
                for t, p in positions.items():
                    if p["shares"] > 0 and t in price_cache:
                        df_temp = price_cache[t]
                        if trade_date in df_temp.index:
                            temp_price = float(df_temp.loc[trade_date, "Close"])
                        else:
                            valid_temp_dates = df_temp.index[df_temp.index <= trade_date]
                            if len(valid_temp_dates) > 0:
                                temp_price = float(df_temp.loc[valid_temp_dates[-1], "Close"])
                            else:
                                temp_price = p["avg_cost"]
                        current_holdings_value += p["shares"] * temp_price

                # 현재 equity 기준으로 예산 계산
                equity = cash + current_holdings_value
                target_amount = equity / final_holdings
                budget = min(target_amount, cash)

                # 예산이 너무 작으면 스킵
                min_budget = equity / (final_holdings * 2.0)
                if budget <= 0 or budget < min_budget:
                    continue

                # 한국: 정수 단위만 매수
                if country_code in ("kor", "kr"):
                    shares = int(budget // price) if price > 0 else 0
                    buy_amount = shares * price
                else:
                    # 호주: 소수점 매수 가능
                    shares = budget / price if price > 0 else 0
                    buy_amount = budget

                if shares <= 0 or buy_amount <= 0:
                    continue

                if ticker not in positions:
                    positions[ticker] = {"shares": 0, "avg_cost": 0}

                old_shares = positions[ticker]["shares"]
                old_cost = positions[ticker]["avg_cost"]

                new_shares = old_shares + shares
                new_cost = (old_shares * old_cost + shares * price) / new_shares if new_shares > 0 else price

                positions[ticker] = {"shares": new_shares, "avg_cost": new_cost}
                cash -= buy_amount

        # 매일 리밸런싱 체크 (백테스트와 동일)
        # 보유 종목 수 계산
        held_count = sum(1 for p in positions.values() if p["shares"] > 0)

        if held_count > 0:
            # 현재 보유 자산 평가액 계산
            current_holdings_value = 0
            today_prices = {}
            for t, p in positions.items():
                if p["shares"] > 0 and t in price_cache:
                    df_temp = price_cache[t]
                    if trade_date in df_temp.index:
                        temp_price = float(df_temp.loc[trade_date, "Close"])
                    else:
                        valid_temp_dates = df_temp.index[df_temp.index <= trade_date]
                        if len(valid_temp_dates) > 0:
                            temp_price = float(df_temp.loc[valid_temp_dates[-1], "Close"])
                        else:
                            temp_price = p["avg_cost"]
                    today_prices[t] = temp_price
                    current_holdings_value += p["shares"] * temp_price

            # 총 자산 및 목표 비중 계산
            total_equity = cash + current_holdings_value
            target_weight = 100.0 / portfolio_topn if portfolio_topn > 0 else 0.0
            max_weight_diff = 0.0

            # 각 종목의 비중 편차 계산
            if total_equity > 0:
                for ticker, pos in positions.items():
                    if pos["shares"] > 0 and ticker in today_prices:
                        current_value = pos["shares"] * today_prices[ticker]
                        current_weight = (current_value / total_equity) * 100.0
                        weight_diff = abs(current_weight - target_weight)
                        max_weight_diff = max(max_weight_diff, weight_diff)

            # 리밸런싱 조건: 매수/매도 발생 또는 비중 편차가 임계값 초과
            trades_occurred = bool(day_trades)
            should_rebalance = trades_occurred or max_weight_diff > rebalance_threshold

            if should_rebalance:
                # 균등 비중 리밸런싱 실행
                target_value_per_stock = total_equity / portfolio_topn
                target_cash = total_equity * 0.01  # 목표 현금 1%

                # 최대 5회 반복
                for iteration in range(5):
                    if cash <= target_cash:
                        break

                    # 1단계: 과다 보유 종목 매도
                    for ticker in list(positions.keys()):
                        pos = positions[ticker]
                        if pos["shares"] <= 0:
                            continue

                        price = today_prices.get(ticker)
                        if not price or price <= 0:
                            continue

                        current_value = pos["shares"] * price
                        value_diff = current_value - target_value_per_stock

                        if value_diff > price:  # 과다 보유
                            shares_to_sell = value_diff / price
                            if country_code not in ("aus", "au"):
                                shares_to_sell = int(shares_to_sell)

                            if shares_to_sell > 0:
                                sell_amount = shares_to_sell * price
                                cash += sell_amount
                                pos["shares"] -= shares_to_sell

                    # 2단계: 과소 보유 종목 매수 (비례 배분)
                    underweight_tickers = []
                    for ticker, pos in positions.items():
                        if pos["shares"] <= 0:
                            continue

                        price = today_prices.get(ticker)
                        if not price or price <= 0:
                            continue

                        current_value = pos["shares"] * price
                        value_diff = current_value - target_value_per_stock

                        if value_diff < -price:  # 과소 보유
                            needed_amount = abs(value_diff)
                            underweight_tickers.append(
                                {
                                    "ticker": ticker,
                                    "price": price,
                                    "needed": needed_amount,
                                }
                            )

                    # 총 필요 금액 계산
                    total_needed = sum(item["needed"] for item in underweight_tickers)

                    # 현금을 비례 배분하여 매수
                    if total_needed > 0 and cash > 0:
                        for item in underweight_tickers:
                            ticker = item["ticker"]
                            price = item["price"]
                            needed = item["needed"]

                            # 비례 배분
                            allocated_cash = (needed / total_needed) * cash
                            shares_to_buy = allocated_cash / price

                            if country_code not in ("aus", "au"):
                                shares_to_buy = int(shares_to_buy)

                            if shares_to_buy > 0:
                                buy_amount = shares_to_buy * price

                                if buy_amount <= cash + 1e-9:
                                    pos = positions[ticker]
                                    cash -= buy_amount
                                    pos["shares"] += shares_to_buy

        # 거래 후 최종 평가액 계산
        final_value = cash
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
                final_value += pos["shares"] * close_price

        # 일별 수익률 계산
        daily_return_pct = ((final_value / prev_value) - 1) * 100 if prev_value > 0 else 0.0
        cumulative_return_pct = ((final_value / initial_capital) - 1) * 100 if initial_capital > 0 else 0.0

        # 일별 기록 저장
        daily_records.append(
            {
                "date": trade_date,
                "total_value": final_value,
                "cash": cash,
                "holdings_value": final_value - cash,
                "daily_return_pct": daily_return_pct,
                "cumulative_return_pct": cumulative_return_pct,
                "trade_count": len(day_trades),
            }
        )

        prev_value = final_value

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
        "daily_records": daily_records,  # 일별 수익률 기록
        "method": "actual_trades",  # 실제 거래 기반
    }


__all__ = ["calculate_actual_performance"]
