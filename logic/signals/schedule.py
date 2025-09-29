"""Schedule and trading-day helpers wrappers for signals logic."""
from __future__ import annotations
import pandas as pd

from signals import (
    is_market_open as _root_is_market_open,
    get_next_trading_day as _root_get_next_trading_day,
    _determine_target_date_for_scheduler as _root_determine_target_date_for_scheduler,
)


def is_market_open(country: str = "kor") -> bool:
    return _root_is_market_open(country)


def get_next_trading_day(country: str, start_date: pd.Timestamp) -> pd.Timestamp:
    return _root_get_next_trading_day(country, start_date)


def determine_target_date_for_scheduler(country: str) -> pd.Timestamp:
    return _root_determine_target_date_for_scheduler(country)


__all__ = [
    "is_market_open",
    "get_next_trading_day",
    "determine_target_date_for_scheduler",
]
