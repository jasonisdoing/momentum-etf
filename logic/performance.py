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


def _extract_trade_quantity(trade: Dict[str, Any], price: Optional[float] = None) -> Optional[float]:
    """거래 문서에서 수량 정보를 추출합니다."""

    for key in ("shares", "quantity", "qty", "units"):
        value = trade.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue

    amount = trade.get("amount")
    if amount is None:
        amount = trade.get("trade_amount")

    if amount is not None and price and price > 0:
        try:
            return float(amount) / float(price)
        except (TypeError, ValueError, ZeroDivisionError):
            return None

    return None


def _trade_sort_key(trade: Dict[str, Any]) -> Tuple[pd.Timestamp, pd.Timestamp, str]:
    executed_ts = pd.to_datetime(trade.get("executed_at"), errors="coerce")
    created_ts = pd.to_datetime(trade.get("created_at"), errors="coerce")

    if pd.isna(executed_ts):
        executed_ts = created_ts
    if pd.isna(created_ts):
        created_ts = executed_ts

    idx = trade.get("_id")
    idx_str = str(idx) if idx is not None else ""

    return executed_ts, created_ts, idx_str


def _get_price_for_date(df: pd.DataFrame, date: pd.Timestamp) -> float:
    if df is None or df.empty:
        return 0.0
    date_norm = pd.to_datetime(date).normalize()
    if date_norm in df.index:
        return float(df.loc[date_norm, "Close"])
    valid_dates = df.index[df.index <= date_norm]
    if len(valid_dates) > 0:
        return float(df.loc[valid_dates[-1], "Close"])
    valid_dates_future = df.index[df.index > date_norm]
    if len(valid_dates_future) > 0:
        return float(df.loc[valid_dates_future[0], "Close"])
    return 0.0


def _compute_trade_price(
    price_cache: Dict[str, pd.DataFrame],
    ticker: str,
    trade_date: pd.Timestamp,
    buy_slippage: float,
    sell_slippage: float,
    is_buy: bool,
) -> Optional[float]:
    df = price_cache.get(ticker)
    if df is None or df.empty:
        return None

    date_norm = pd.to_datetime(trade_date).normalize()
    if date_norm in df.index:
        base_price = float(df.loc[date_norm, "Close"])
    else:
        valid_dates = df.index[df.index <= date_norm]
        if len(valid_dates) == 0:
            return None
        base_price = float(df.loc[valid_dates[-1], "Close"])

    if base_price <= 0:
        return None

    if is_buy:
        return base_price * (1 + buy_slippage)
    return base_price * (1 - sell_slippage)


def _determine_initial_holdings(seed_trades: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """start_date 이전의 거래를 기반으로 마지막 액션이 BUY인 종목을 반환합니다."""

    holdings_state: Dict[str, Dict[str, Any]] = {}
    for trade in sorted(seed_trades, key=_trade_sort_key):
        ticker = str(trade.get("ticker") or "").upper()
        if not ticker:
            continue
        action = str(trade.get("action") or "").upper()
        holdings_state[ticker] = {"action": action, "trade": trade}

    initial_holdings: Dict[str, Dict[str, Any]] = {}
    for ticker, info in holdings_state.items():
        if info.get("action") == "BUY":
            initial_holdings[ticker] = info.get("trade") or {}
    return initial_holdings


def _seed_positions_from_holdings(
    initial_holdings: Dict[str, Dict[str, Any]],
    price_cache: Dict[str, pd.DataFrame],
    start_date: pd.Timestamp,
    initial_capital: float,
    country_code: str,
    portfolio_topn: int,
) -> Tuple[Dict[str, Dict[str, float]], float]:
    """초기 보유 종목을 균등 비중으로 배분해 포지션과 잔여 현금을 반환합니다."""

    if not initial_holdings:
        return {}, initial_capital

    fractional_allowed = str(country_code).lower() not in {"kor", "kr"}
    tickers = list(initial_holdings.keys())
    target_count = len(tickers)
    if target_count <= 0:
        return {}, initial_capital

    equal_value = initial_capital / target_count if target_count > 0 else 0.0
    positions: Dict[str, Dict[str, float]] = {}
    cash = initial_capital

    for ticker in tickers:
        df = price_cache.get(ticker)
        price = _get_price_for_date(df, start_date) if df is not None else 0.0
        if price <= 0:
            continue

        if fractional_allowed:
            shares = equal_value / price
        else:
            shares = math.floor(equal_value / price)

        if shares <= 0:
            continue

        buy_amount = shares * price
        if buy_amount > cash + 1e-9:
            buy_amount = cash
            shares = buy_amount / price if fractional_allowed else math.floor(buy_amount / price)
            if shares <= 0:
                continue
            buy_amount = shares * price

        positions[ticker] = {"shares": float(shares), "avg_cost": float(price)}
        cash -= buy_amount

    if cash < 0:
        cash = 0.0
    return positions, cash


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

        desired_shares = math.floor(target_value / price)
        desired_shares = max(desired_shares, 0.0)

        if current_shares <= desired_shares + 1e-9:
            continue

        shares_to_sell = int(math.floor(current_shares - desired_shares))

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

        desired_shares = math.floor(target_value / price)
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

        shares_to_buy = int(allocated_cash / price)
        shares_to_buy = min(shares_to_buy, int(math.floor(item["needed_shares"])))

        if shares_to_buy <= 0:
            total_needed_value -= item["needed_value"]
            continue

        buy_amount = shares_to_buy * price
        if buy_amount > remaining_cash:
            shares_to_buy = int(remaining_cash // price)
            if shares_to_buy <= 0:
                total_needed_value -= item["needed_value"]
                continue
            buy_amount = shares_to_buy * price

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
    trades = list(db.trades.find({"account": account_id.lower(), "executed_at": {"$gte": start_date, "$lte": end_date}}))
    trades.sort(key=_trade_sort_key)

    # start_date 이전 거래로 초기 보유 상태 파악
    seed_trades = list(db.trades.find({"account": account_id.lower(), "executed_at": {"$lt": start_date}}))
    seed_trades.sort(key=_trade_sort_key)
    initial_holdings = _determine_initial_holdings(seed_trades)

    if not trades:
        logger.info(f"[{account_id}] 거래 내역이 없습니다.")
        if not initial_holdings:
            return None

    logger.info(f"[{account_id}] {len(trades)}개의 거래 내역을 찾았습니다.")

    # 필요한 종목별 가격 데이터 조회
    tickers = set(t["ticker"] for t in trades)
    tickers.update(initial_holdings.keys())
    tickers.discard(None)
    tickers = list(tickers)
    price_cache = {}

    for ticker in tickers:
        df = fetch_ohlcv(ticker, country=country_code, date_range=[start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")])
        if df is not None and not df.empty:
            price_cache[ticker] = df

    # 슬리피지 없음 (벤치마크와 동일 조건으로 비교)
    buy_slippage = 0.0
    sell_slippage = 0.0

    # 포지션 추적
    positions, cash = _seed_positions_from_holdings(
        initial_holdings=initial_holdings,
        price_cache=price_cache,
        start_date=start_date,
        initial_capital=initial_capital,
        country_code=country_code,
        portfolio_topn=portfolio_topn,
    )

    # 날짜별 수익률 추적
    daily_records = []

    initial_prev_value = cash
    for ticker, pos in positions.items():
        df = price_cache.get(ticker)
        price = _get_price_for_date(df, start_date) if df is not None else 0.0
        if price > 0:
            initial_prev_value += pos["shares"] * price
    prev_value = initial_prev_value if positions else initial_capital

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

        ordered_trades = sorted(day_trades, key=_trade_sort_key)

        from collections import deque

        implicit_indices = deque(
            idx
            for idx, trade in enumerate(ordered_trades)
            if str(trade.get("action", "")).upper() == "BUY"
            and _extract_trade_quantity(trade, None) is None
            and str(trade.get("ticker") or "").upper() in price_cache
        )

        fractional_allowed = str(country_code).lower() not in {"kor", "kr"}

        for idx, trade in enumerate(ordered_trades):
            action = str(trade.get("action", "")).upper()
            ticker = str(trade.get("ticker") or "").upper()
            if not ticker or ticker not in price_cache or action not in {"BUY", "SELL"}:
                continue

            trade_price = _compute_trade_price(
                price_cache=price_cache,
                ticker=ticker,
                trade_date=trade_date,
                buy_slippage=buy_slippage,
                sell_slippage=sell_slippage,
                is_buy=action == "BUY",
            )
            if trade_price is None or trade_price <= 0:
                continue

            if implicit_indices and implicit_indices[0] < idx:
                implicit_indices.popleft()

            if action == "SELL":
                if ticker in positions and positions[ticker]["shares"] > 0:
                    qty = _extract_trade_quantity(trade, trade_price)
                    sell_shares = positions[ticker]["shares"] if qty is None else min(float(qty), positions[ticker]["shares"])
                    if sell_shares <= 0:
                        continue
                    cash += sell_shares * trade_price
                    positions[ticker]["shares"] -= sell_shares
                    if positions[ticker]["shares"] <= 0:
                        positions[ticker] = {"shares": 0, "avg_cost": 0}
                continue

            qty = _extract_trade_quantity(trade, trade_price)
            if qty is not None:
                shares = max(float(qty), 0.0)
                buy_amount = shares * trade_price
                if shares <= 0 or buy_amount > cash + 1e-9:
                    continue
            else:
                remaining_implicits = len(implicit_indices) if implicit_indices else 1
                budget = cash / remaining_implicits if remaining_implicits > 0 else cash
                if fractional_allowed:
                    shares = budget / trade_price
                    buy_amount = shares * trade_price
                else:
                    shares = int(budget // trade_price) if trade_price > 0 else 0
                    if shares <= 0 and budget >= trade_price:
                        shares = 1
                    buy_amount = shares * trade_price
                if implicit_indices and implicit_indices[0] == idx:
                    implicit_indices.popleft()
                if shares <= 0 or buy_amount <= 0 or buy_amount > cash + 1e-9:
                    continue

            if ticker not in positions:
                positions[ticker] = {"shares": 0, "avg_cost": 0}

            old_shares = positions[ticker]["shares"]
            old_cost = positions[ticker]["avg_cost"]
            new_shares = old_shares + shares
            new_cost = ((old_shares * old_cost) + (shares * trade_price)) / new_shares if new_shares > 0 else trade_price
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

        # 거래 상세 정보 정리
        trade_details: List[Dict[str, Any]] = []
        for trade in day_trades:
            detail: Dict[str, Any] = {
                "ticker": trade.get("ticker"),
                "action": str(trade.get("action", "")).upper(),
            }
            shares_raw = trade.get("shares") or trade.get("quantity") or trade.get("qty")
            if shares_raw is not None:
                try:
                    detail["shares"] = float(shares_raw)
                except (TypeError, ValueError):
                    detail["shares"] = shares_raw
            price_raw = trade.get("price") or trade.get("executed_price") or trade.get("execution_price")
            if price_raw is not None:
                try:
                    detail["price"] = float(price_raw)
                except (TypeError, ValueError):
                    detail["price"] = price_raw
            amount_raw = trade.get("amount")
            if amount_raw is None and isinstance(detail.get("shares"), (int, float)) and isinstance(detail.get("price"), (int, float)):
                amount_raw = detail["shares"] * detail["price"]
            if amount_raw is not None:
                try:
                    detail["amount"] = float(amount_raw)
                except (TypeError, ValueError):
                    detail["amount"] = amount_raw
            memo = trade.get("memo")
            if memo:
                detail["memo"] = memo
            detail["raw"] = trade
            trade_details.append(detail)

        # 거래 후 최종 평가액 및 보유 현황 계산
        holdings_snapshot: List[Dict[str, Any]] = []
        holdings_total = 0.0
        for ticker, pos in positions.items():
            if pos["shares"] <= 0:
                continue

            df = price_cache.get(ticker)
            if df is not None:
                if trade_date in df.index:
                    price = float(df.loc[trade_date, "Close"])
                else:
                    valid_dates = df.index[df.index <= trade_date]
                    if len(valid_dates) > 0:
                        price = float(df.loc[valid_dates[-1], "Close"])
                    else:
                        price = float(pos.get("avg_cost", 0.0))
            else:
                price = float(pos.get("avg_cost", 0.0))

            shares = float(pos["shares"])
            value = shares * price
            avg_cost = float(pos.get("avg_cost", 0.0))
            book_value = shares * avg_cost
            profit = value - book_value
            profit_pct = (profit / book_value * 100) if book_value > 0 else 0.0

            holdings_snapshot.append(
                {
                    "ticker": ticker,
                    "shares": shares,
                    "price": price,
                    "value": value,
                    "avg_cost": avg_cost,
                    "book_value": book_value,
                    "profit": profit,
                    "profit_pct": profit_pct,
                }
            )
            holdings_total += value

        final_value = cash + holdings_total
        for item in holdings_snapshot:
            value = item["value"]
            item["weight_pct"] = (value / final_value * 100) if final_value > 0 else 0.0

        # 일별 수익률 계산
        daily_return_pct = ((final_value / prev_value) - 1) * 100 if prev_value > 0 else 0.0
        cumulative_return_pct = ((final_value / initial_capital) - 1) * 100 if initial_capital > 0 else 0.0

        # 일별 기록 저장
        daily_records.append(
            {
                "date": trade_date,
                "total_value": final_value,
                "cash": cash,
                "holdings_value": holdings_total,
                "daily_return_pct": daily_return_pct,
                "cumulative_return_pct": cumulative_return_pct,
                "trade_count": len(day_trades),
                "holdings": holdings_snapshot,
                "trades": trade_details,
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
        "final_value": current_value,
        "current_value": current_value,
        "cash": cash,
        "positions": positions,
        "trade_count": len(trades),
        "daily_records": daily_records,  # 일별 수익률 기록
        "start_date": start_date,
        "end_date": end_date,
        "currency": country_code,
        "method": "actual_trades",  # 실제 거래 기반
    }


__all__ = ["calculate_actual_performance"]
