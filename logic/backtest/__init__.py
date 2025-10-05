"""Backtest package: runners and reporting utilities."""

from settings.common import TEST_INITIAL_CAPITAL, TEST_MONTHS_RANGE
from .account_runner import AccountBacktestResult, run_account_backtest
from .reporting import (
    dump_backtest_log,
    format_period_return_with_listing_date,
    print_backtest_summary,
)

__all__ = [
    "AccountBacktestResult",
    "TEST_MONTHS_RANGE",
    "TEST_INITIAL_CAPITAL",
    "run_account_backtest",
    "dump_backtest_log",
    "format_period_return_with_listing_date",
    "print_backtest_summary",
]
