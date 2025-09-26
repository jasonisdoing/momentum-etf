"""Momentum 전략 엔트리 포인트.

이전 jason 전략 모듈을 대체하며, 외부에서는 이 모듈을 통해 전략 API를 사용합니다.
"""

from __future__ import annotations

from logic.strategies.momentum.backtest import (
    run_portfolio_backtest,
    run_single_ticker_backtest,
)
from logic.strategies.momentum.constants import COIN_ZERO_THRESHOLD, DECISION_CONFIG
from logic.strategies.momentum.signals import generate_daily_signals_for_portfolio

__all__ = [
    "COIN_ZERO_THRESHOLD",
    "DECISION_CONFIG",
    "generate_daily_signals_for_portfolio",
    "run_portfolio_backtest",
    "run_single_ticker_backtest",
]
