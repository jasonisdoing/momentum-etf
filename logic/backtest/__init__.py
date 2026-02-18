"""Backtest package: runners and reporting utilities."""

from .account import AccountBacktestResult, run_account_backtest
from .reporting import (
    dump_backtest_log,
    format_period_return_with_listing_date,
    print_backtest_summary,
)

__all__ = [
    "AccountBacktestResult",
    "run_account_backtest",
    "dump_backtest_log",
    "format_period_return_with_listing_date",
    "print_backtest_summary",
    # From consolidated common logic
    "sort_decisions_by_order_and_score",
    "calculate_buy_budget",
    "calculate_held_count",
    "check_buy_candidate_filters",
    "count_current_holdings",
    "get_hold_states",
    "get_sell_states",
    "validate_portfolio_topn",
    "calculate_consecutive_days",
    "get_buy_signal_streak",
    "has_buy_signal",
    "format_trend_break_phrase",
]

# Re-exporting consolidated logic for backward compatibility (migrated from logic.common)
from .filtering import (
    sort_decisions_by_order_and_score,
)
from .notes import format_trend_break_phrase
from .portfolio import (
    calculate_buy_budget,
    calculate_held_count,
    check_buy_candidate_filters,
    count_current_holdings,
    get_hold_states,
    get_sell_states,
    validate_portfolio_topn,
)
from .signals import (
    calculate_consecutive_days,
    get_buy_signal_streak,
    has_buy_signal,
)
