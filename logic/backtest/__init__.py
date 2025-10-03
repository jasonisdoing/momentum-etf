"""Backtest package: runners and reporting utilities."""

from backtest import TEST_MONTHS_RANGE, TEST_INITIAL_CAPITAL
from .country_runner import CountryBacktestResult, run_country_backtest
from .reporting import (
    dump_backtest_log,
    format_period_return_with_listing_date,
    print_backtest_summary,
)

__all__ = [
    "CountryBacktestResult",
    "TEST_MONTHS_RANGE",
    "TEST_INITIAL_CAPITAL",
    "run_country_backtest",
    "dump_backtest_log",
    "format_period_return_with_listing_date",
    "print_backtest_summary",
]
