"""MAPS(Moving Average Position Score) 전략 서브패키지."""

from __future__ import annotations

from .backtest import run_portfolio_backtest, run_single_ticker_backtest
from .constants import DECISION_CONFIG
from .recommend import generate_daily_signals_for_portfolio
from .rules import StrategyRules

# MA_PERIOD 상수를 StrategyRules에서 가져옵니다.
MA_PERIOD = StrategyRules.DEFAULT_MA_PERIOD

__all__ = [
    "DECISION_CONFIG",
    "generate_daily_signals_for_portfolio",
    "run_portfolio_backtest",
    "run_single_ticker_backtest",
    "MA_PERIOD",
]
