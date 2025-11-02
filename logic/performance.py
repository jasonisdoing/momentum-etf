"""실제 거래 내역 기반 성과 계산 모듈"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, List, Tuple

import pandas as pd

from utils.data_loader import fetch_ohlcv, get_trading_days
from utils.db_manager import get_db_connection
from utils.logger import get_app_logger
from logic.common import build_weekly_rebalance_cache

logger = get_app_logger()


def _is_weekly_rebalance_day(
    date: pd.Timestamp,
    country_code: str,
    cache: Dict[Tuple[int, int], Optional[pd.Timestamp]],
) -> bool:
    if not isinstance(date, pd.Timestamp):
        return False

    iso_year, iso_week, _ = date.isocalendar()
    key = (iso_year, iso_week)

    cached = cache.get(key)
    if cached is None:
        week_start = date - pd.Timedelta(days=date.weekday())
        week_end = week_start + pd.Timedelta(days=6)
        try:
            trading_days = get_trading_days(
                week_start.strftime("%Y-%m-%d"),
                week_end.strftime("%Y-%m-%d"),
                country_code,
            )
        except Exception:
            trading_days = []

        if trading_days:
            cached = pd.Timestamp(trading_days[-1]).normalize()
        else:
            cached = pd.NaT
        cache[key] = cached

    if pd.isna(cached):
        return False
    return pd.Timestamp(date).normalize() == cached


def _rebalance_positions_weekly(
    positions: Dict[str, Dict[str, float]],
    today_prices: Dict[str, float],
    cash: float,
    country_code: str,
) -> float:
    fractional_allowed = str(country_code).lower() in ("aus", "au")
    active = [ticker for ticker, pos in positions.items() if pos.get("shares", 0) > 0 and today_prices.get(ticker, 0.0) > 0]
    if not active:
        return cash

    total_equity = cash + sum(positions[ticker]["shares"] * today_prices[ticker] for ticker in active)
    if total_equity <= 0:
        return cash

    target_value = total_equity / len(active)

    sell_actions: List[Dict[str, float]] = []
    for ticker in sorted(active, key=lambda t: positions[t]["shares"] * today_prices[t], reverse=True):
        pos = positions[ticker]
        price = today_prices[ticker]
        current_shares = pos["shares"]

        desired_shares = target_value / price
        if not fractional_allowed:
            desired_shares = math.floor(desired_shares)
        desired_shares = max(desired_shares, 0.0)

        if current_shares <= desired_shares + 1e-9:
            continue

        shares_to_sell = current_shares - desired_shares
        if not fractional_allowed:
            shares_to_sell = int(math.floor(shares_to_sell))

        if shares_to_sell <= 0:
            continue

        shares_to_sell = min(current_shares, shares_to_sell)
        sell_amount = shares_to_sell * price

        cash += sell_amount
        pos["shares"] = current_shares - shares_to_sell
        if pos["shares"] <= 0:
            pos["shares"] = 0
            pos["avg_cost"] = 0.0

        sell_actions.append({"ticker": ticker, "shares": shares_to_sell, "price": price, "amount": sell_amount})

    active = [ticker for ticker, pos in positions.items() if pos.get("shares", 0) > 0 and today_prices.get(ticker, 0.0) > 0]
    if not active:
        if sell_actions:
            logger.debug("[PERF] 주간 리밸런싱: 모든 포지션 정리, 잔여 현금 %.0f원", cash)
        return cash

    total_equity = cash + sum(positions[ticker]["shares"] * today_prices[ticker] for ticker in active)
    if total_equity <= 0:
        return cash

    target_value = total_equity / len(active)
    buy_candidates: List[Dict[str, float]] = []
    for ticker in active:
        pos = positions[ticker]
        price = today_prices[ticker]

        desired_shares = target_value / price
        if not fractional_allowed:
            desired_shares = math.floor(desired_shares)
        desired_shares = max(desired_shares, 0.0)

        current_shares = pos["shares"]
        if current_shares >= desired_shares - 1e-9:
            continue

        needed_shares = desired_shares - current_shares
        needed_value = needed_shares * price
        if needed_value <= 0:
            continue

        buy_candidates.append(
            {
                "ticker": ticker,
                "price": price,
                "needed_value": needed_value,
                "needed_shares": needed_shares,
            }
        )

    if not buy_candidates or cash <= 0:
        if sell_actions:
            logger.debug("[PERF] 주간 리밸런싱: 매도 %s, 매수 없음, 잔여 현금 %.0f원", sell_actions, cash)
        return cash

    buy_candidates.sort(key=lambda item: item["needed_value"], reverse=True)
    total_needed_value = sum(item["needed_value"] for item in buy_candidates)
    remaining_cash = cash
    buy_actions: List[Dict[str, float]] = []

    for item in buy_candidates:
        if remaining_cash <= 0 or total_needed_value <= 0:
            break

        allocation_ratio = item["needed_value"] / total_needed_value
        allocated_cash = remaining_cash * allocation_ratio

        price = item["price"]
        if price <= 0 or allocated_cash <= 0:
            total_needed_value -= item["needed_value"]
            continue

        if fractional_allowed:
            shares_to_buy = min(item["needed_shares"], allocated_cash / price)
        else:
            shares_to_buy = int(allocated_cash / price)
            shares_to_buy = min(shares_to_buy, int(math.floor(item["needed_shares"])))

        if shares_to_buy <= 0:
            total_needed_value -= item["needed_value"]
            continue

        buy_amount = shares_to_buy * price
        if buy_amount > remaining_cash:
            buy_amount = remaining_cash
            if fractional_allowed:
                shares_to_buy = buy_amount / price
            else:
                shares_to_buy = int(buy_amount / price)

        if shares_to_buy <= 0:
            total_needed_value -= item["needed_value"]
            continue

        ticker = item["ticker"]
        pos = positions[ticker]
        old_shares = pos["shares"]
        old_cost = pos.get("avg_cost", 0.0)

        pos["shares"] = old_shares + shares_to_buy
        if pos["shares"] > 0:
            pos["avg_cost"] = ((old_shares * old_cost) + buy_amount) / pos["shares"]

        remaining_cash -= buy_amount
        buy_actions.append({"ticker": ticker, "shares": shares_to_buy, "price": price, "amount": buy_amount})
        total_needed_value -= item["needed_value"]

    cash = remaining_cash

    if sell_actions or buy_actions:
        logger.debug("[PERF] 주간 리밸런싱: 매도 %s, 매수 %s, 잔여 현금 %.0f원", sell_actions, buy_actions, cash)

    return cash


def calculate_actual_performance(
    account_id: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    initial_capital: float,
    country_code: str = "kor",
    portfolio_topn: int = 12,
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
    all_trading_days = get_trading_days(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), country_code)
    weekly_rebalance_cache: Dict[Tuple[int, int], Optional[pd.Timestamp]] = build_weekly_rebalance_cache(all_trading_days)

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

        # 주간 리밸런싱 (주 마지막 거래일에만 실행)
        held_count = sum(1 for p in positions.values() if p["shares"] > 0)

        today_prices: Dict[str, float] = {}
        current_holdings_value = 0.0

        if held_count > 0:
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

            if _is_weekly_rebalance_day(trade_date, country_code, weekly_rebalance_cache):
                cash = _rebalance_positions_weekly(positions, today_prices, cash, country_code)

                # 리밸런싱 후 최신 평가액 갱신
                today_prices = {}
                current_holdings_value = 0.0
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
