"""Trade history utilities for signals logic.

Moved from the root signals module to avoid circular imports and duplication.
"""
from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pymongo import DESCENDING

from logic.momentum import COIN_ZERO_THRESHOLD
from utils.db_manager import get_db_connection


def calculate_consecutive_holding_info(
    held_tickers: List[str], country: str, account: str, as_of_date: datetime
) -> Dict[str, Dict]:
    """
    Scan 'trades' collection and compute consecutive holding start date per ticker
    for the given account. Uses a single query to avoid N+1 access.
    """
    holding_info = {tkr: {"buy_date": None} for tkr in held_tickers}
    if not held_tickers:
        return holding_info

    db = get_db_connection()
    if db is None:
        print("-> 경고: DB에 연결할 수 없어 보유일 계산을 건너뜁니다.")
        return holding_info

    if not account:
        raise ValueError("account is required for calculating holding info")

    # include all trades within the same calendar day (till 23:59:59.999999)
    include_until = as_of_date.replace(hour=23, minute=59, second=59, microsecond=999999)

    query = {
        "country": country,
        "account": account,
        "ticker": {"$in": held_tickers},
        "date": {"$lte": include_until},
    }
    all_trades = list(
        db.trades.find(
            query,
            sort=[("date", DESCENDING), ("_id", DESCENDING)],
        )
    )

    from collections import defaultdict

    trades_by_ticker = defaultdict(list)
    for trade in all_trades:
        trades_by_ticker[trade["ticker"]].append(trade)

    for tkr in held_tickers:
        trades = trades_by_ticker.get(tkr)
        if not trades:
            continue

        try:
            current_shares = sum(
                t["shares"] if t["action"] == "BUY" else -t["shares"] for t in trades
            )

            buy_date = None
            for trade in trades:
                if current_shares <= COIN_ZERO_THRESHOLD:
                    break
                buy_date = trade["date"]
                if trade["action"] == "BUY":
                    current_shares -= trade["shares"]
                elif trade["action"] == "SELL":
                    current_shares += trade["shares"]

            if buy_date:
                holding_info[tkr]["buy_date"] = buy_date
        except Exception as e:
            print(f"-> 경고: {tkr} 보유일 계산 중 오류 발생: {e}")

    return holding_info


def calculate_trade_cooldown_info(
    tickers: List[str], country: str, account: str, as_of_date: datetime
) -> Dict[str, Dict[str, Optional[datetime]]]:
    """Compute recent buy/sell dates per ticker for trade cooldown decisions."""
    info: Dict[str, Dict[str, Optional[datetime]]] = {
        tkr: {"last_buy": None, "last_sell": None} for tkr in tickers
    }
    if not tickers:
        return info

    db = get_db_connection()
    if db is None:
        print("-> 경고: DB에 연결할 수 없어 쿨다운 계산을 건너뜁니다.")
        return info

    include_until = as_of_date.replace(hour=23, minute=59, second=59, microsecond=999999)

    query = {
        "country": country,
        "account": account,
        "ticker": {"$in": tickers},
        "date": {"$lte": include_until},
    }

    trades_cursor = db.trades.find(
        query,
        sort=[("date", DESCENDING), ("_id", DESCENDING)],
    )

    for trade in trades_cursor:
        ticker = trade.get("ticker")
        action = (trade.get("action") or "").upper()
        if ticker not in info:
            continue

        if action == "BUY" and info[ticker]["last_buy"] is None:
            info[ticker]["last_buy"] = trade.get("date")
        elif action == "SELL" and info[ticker]["last_sell"] is None:
            info[ticker]["last_sell"] = trade.get("date")

        if info[ticker]["last_buy"] and info[ticker]["last_sell"]:
            continue

    return info


__all__ = [
    "calculate_consecutive_holding_info",
    "calculate_trade_cooldown_info",
]
