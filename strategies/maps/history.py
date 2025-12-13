"""Trade history utilities for signals logic.

Moved from the root signals module to avoid circular imports and duplication.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pymongo import ASCENDING, DESCENDING

from utils.db_manager import get_db_connection, list_open_positions
from utils.logger import get_app_logger
from utils.settings_loader import get_account_settings


def _extract_trade_time(trade: dict) -> datetime | None:
    value = trade.get("date") or trade.get("executed_at")
    if isinstance(value, datetime):
        return value
    return None


def _extract_trade_shares(trade: dict) -> float:
    raw = trade.get("shares")
    if raw is None:
        raw = trade.get("quantity")
    try:
        return float(raw or 0.0)
    except Exception:
        return 0.0


def _generate_ticker_query_keys(ticker: str) -> list[str]:
    keys: set[str] = set()
    raw = str(ticker or "").strip()
    if not raw:
        return []

    upper = raw.upper()
    keys.add(raw)
    keys.add(upper)

    if ":" in upper:
        suffix = upper.split(":")[-1]
        keys.add(suffix)
        keys.add(f"{suffix}.AX")
        keys.add(f"{suffix}.KS")
    if "." in upper:
        keys.add(upper.split(".")[0])

    keys.add(upper.replace("ASX:", ""))
    keys.add(upper.replace("KOR:", ""))

    return [k for k in keys if k]


def _canonical_ticker_key(ticker: str) -> str:
    value = str(ticker or "").strip().upper()
    if not value:
        return ""

    if ":" in value:
        value = value.split(":")[-1]
    if "." in value:
        value = value.split(".")[0]
    value = value.replace("-", "").replace("_", "")
    value = value.replace("ASX", "", 1) if value.startswith("ASX") else value
    value = value.replace("KOR", "", 1) if value.startswith("KOR") else value
    return value.strip()


def calculate_consecutive_holding_info(
    held_tickers: list[str], account_id: str, as_of_date: datetime
) -> dict[str, dict]:
    """
    Scan `trades` collection and compute consecutive holding start date per ticker
    for the given account. Uses a single query to avoid N+1 access.

    Args:
        held_tickers: List of tickers to check
        account_id: 계정 ID (예: 'kor')
        as_of_date: Date to calculate holding info as of

    Returns:
        Dictionary mapping tickers to their holding info
    """
    holding_info = {tkr: {"buy_date": None} for tkr in held_tickers}
    if not held_tickers:
        return holding_info

    logger = get_app_logger()

    db = get_db_connection()
    if db is None:
        logger.warning("DB에 연결할 수 없어 보유일 계산을 건너뜁니다.")
        return holding_info

    include_until = as_of_date.replace(hour=23, minute=59, second=59, microsecond=999999)

    account_norm = (account_id or "").strip().lower()

    query: dict[str, Any] = {
        "account": account_norm,
        "deleted_at": {"$exists": False},
    }

    query_keys: set[str] = set()
    for tkr in held_tickers:
        query_keys.update(_generate_ticker_query_keys(tkr))

    if query_keys:
        query["ticker"] = {"$in": list(query_keys)}

    trades_cursor = db.trades.find(query).sort(
        [
            ("executed_at", ASCENDING),
            ("date", ASCENDING),
            ("_id", ASCENDING),
        ]
    )

    from collections import defaultdict

    trades_by_ticker = defaultdict(list)
    for trade in trades_cursor:
        when = _extract_trade_time(trade)
        if when is None or when > include_until:
            continue
        key = _canonical_ticker_key(trade.get("ticker"))
        if not key:
            continue
        trades_by_ticker[key].append((when, trade))

    threshold = 0.0

    try:
        open_positions = list_open_positions(account_norm)
    except Exception as exc:
        logger.warning("보유 포지션 조회 실패: %s", exc)
        open_positions = []

    fallback_map = {_canonical_ticker_key(pos.get("ticker")): pos for pos in open_positions}

    for tkr in held_tickers:
        key = _canonical_ticker_key(tkr)
        entries = trades_by_ticker.get(key)
        buy_dt: datetime | None = None

        if entries:
            running_shares = 0.0
            consecutive_start: datetime | None = None

            for when, trade in entries:
                action = (trade.get("action") or "").upper()
                qty = _extract_trade_shares(trade)

                if action == "BUY":
                    running_shares += qty
                    if running_shares > threshold and consecutive_start is None:
                        consecutive_start = when
                elif action == "SELL":
                    running_shares -= qty
                    if running_shares <= threshold:
                        consecutive_start = None

            if consecutive_start and running_shares > threshold:
                buy_dt = consecutive_start

        if buy_dt is None:
            fallback_entry = fallback_map.get(key)
            if fallback_entry:
                when = _extract_trade_time(fallback_entry)
                if when and when <= include_until:
                    buy_dt = when

        if buy_dt is not None:
            holding_info[tkr]["buy_date"] = buy_dt

    return holding_info


def calculate_trade_cooldown_info(
    tickers: list[str],
    account_id: str,
    as_of_date: datetime,
    *,
    country_code: str | None = None,
) -> dict[str, dict[str, datetime | None]]:
    """Compute recent buy/sell dates per ticker for trade cooldown decisions.

    Args:
        tickers: List of tickers to check
        account_id: 계정 ID (예: 'kor')
        country_code: 계정이 참조하는 시장 코드 (구 데이터 호환용, 옵션)
        as_of_date: Date to calculate cooldown as of

    Returns:
        Dictionary mapping tickers to their trade cooldown info
    """
    info: dict[str, dict[str, datetime | None]] = {tkr: {"last_buy": None, "last_sell": None} for tkr in tickers}
    if not tickers:
        return info

    logger = get_app_logger()

    db = get_db_connection()
    if db is None:
        logger.warning("DB에 연결할 수 없어 쿨다운 계산을 건너뜁니다.")
        return info

    include_until = as_of_date.replace(hour=23, minute=59, second=59, microsecond=999999)

    account_norm = (account_id or "").strip().lower()
    country_code_normalized = (country_code or "").strip().lower()

    if not country_code_normalized and account_norm:
        try:
            settings = get_account_settings(account_norm)
            country_code_normalized = str(settings.get("country_code") or "").strip().lower()
        except Exception as exc:
            logger.warning("계정 설정에서 국가 코드를 찾지 못했습니다: %s", exc)
            country_code_normalized = ""

    query: dict[str, Any] = {
        "ticker": {"$in": tickers},
        "$or": [{"date": {"$lte": include_until}}, {"executed_at": {"$lte": include_until}}],
    }

    legacy_filters: list[dict[str, Any]] = []
    if account_norm:
        legacy_filters.append({"account": account_norm})
        legacy_filters.append({"country": account_norm})
    if country_code_normalized:
        legacy_filters.append({"country": country_code_normalized})

    if legacy_filters:
        query["$and"] = [{"$or": legacy_filters}]

    trades_cursor = db.trades.find(
        query,
        sort=[("date", DESCENDING), ("executed_at", DESCENDING), ("_id", DESCENDING)],
    )

    for trade in trades_cursor:
        ticker = trade.get("ticker")
        action = (trade.get("action") or "").upper()
        if ticker not in info:
            continue

        trade_date = _extract_trade_time(trade)
        if trade_date is None:
            continue

        if action == "BUY" and info[ticker]["last_buy"] is None:
            info[ticker]["last_buy"] = trade_date
        elif action == "SELL" and info[ticker]["last_sell"] is None:
            info[ticker]["last_sell"] = trade_date

        if info[ticker]["last_buy"] and info[ticker]["last_sell"]:
            continue

    return info


__all__ = [
    "calculate_consecutive_holding_info",
    "calculate_trade_cooldown_info",
]
