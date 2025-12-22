"""Trade history utilities for maps logic.

NOTE: This module previously interacted with MongoDB 'trades' collection.
As of the refactoring to remove legacy trade logic, these functions are stubs
or use empty defaults. Backtest simulation state should be used instead.
"""

from __future__ import annotations

from datetime import datetime


def calculate_consecutive_holding_info(
    held_tickers: list[str], account_id: str, as_of_date: datetime
) -> dict[str, dict]:
    """
    STUB: 'trades' 컬렉션 사용 중단으로 인해 항상 빈 정보를 반환합니다.
    시뮬레이션 기반의 보유 기간 계산을 권장합니다.

    Args:
        held_tickers: List of tickers to check
        account_id: 계정 ID (예: 'kor')
        as_of_date: Date to calculate holding info as of

    Returns:
        Dictionary mapping tickers to their holding info (empty defaults)
    """
    # 기본값: 매수일 정보 없음
    return {tkr: {"buy_date": None} for tkr in held_tickers}


def calculate_trade_cooldown_info(
    tickers: list[str],
    account_id: str,
    as_of_date: datetime,
    *,
    country_code: str | None = None,
) -> dict[str, dict[str, datetime | None]]:
    """
    STUB: 'trades' 컬렉션 사용 중단으로 인해 항상 빈 정보를 반환합니다.

    Args:
        tickers: List of tickers to check
        account_id: 계정 ID (예: 'kor')
        country_code: 계정이 참조하는 시장 코드
        as_of_date: Date to calculate cooldown as of

    Returns:
        Dictionary mapping tickers to their trade cooldown info (empty defaults)
    """
    # 기본값: 최근 매수/매도 없음
    return {tkr: {"last_buy": None, "last_sell": None} for tkr in tickers}


__all__ = [
    "calculate_consecutive_holding_info",
    "calculate_trade_cooldown_info",
]
