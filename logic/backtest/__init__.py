"""Backtest package: runners and reporting utilities."""

from utils.settings_loader import (
    get_backtest_initial_capital,
    get_backtest_months_range,
)
from .account_runner import AccountBacktestResult, run_account_backtest
from .reporting import (
    dump_backtest_log,
    format_period_return_with_listing_date,
    print_backtest_summary,
)

__all__ = [
    "AccountBacktestResult",
    "get_backtest_months_range",
    "get_backtest_initial_capital",
    "run_account_backtest",
    "dump_backtest_log",
    "format_period_return_with_listing_date",
    "print_backtest_summary",
]
