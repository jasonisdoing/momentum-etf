"""MAPS(Moving Average Position Score) 전략 서브패키지."""

from __future__ import annotations

from logic.backtest.portfolio_runner import run_portfolio_backtest
from .backtest import run_single_ticker_backtest
from .constants import DECISION_CONFIG
from .formatting import load_account_precision, get_header_money_formatter, format_shares
from .history import calculate_consecutive_holding_info, calculate_trade_cooldown_info
from .recommend import (
    generate_daily_recommendations_for_portfolio,
    safe_generate_daily_recommendations_for_portfolio,
)
from .rules import StrategyRules
from .scoring import (
    calculate_maps_score,
)


__all__ = [
    "calculate_consecutive_holding_info",
    "calculate_maps_score",
    "calculate_trade_cooldown_info",
    "DECISION_CONFIG",
    "format_shares",
    "generate_daily_recommendations_for_portfolio",
    "get_header_money_formatter",
    "load_account_precision",
    "run_portfolio_backtest",
    "run_single_ticker_backtest",
    "safe_generate_daily_recommendations_for_portfolio",
    "StrategyRules",
]
