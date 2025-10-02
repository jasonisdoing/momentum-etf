"""MAPS(Moving Average Position Score) 전략 서브패키지."""

from __future__ import annotations

from .backtest import run_portfolio_backtest, run_single_ticker_backtest
from .recommend import (
    generate_daily_recommendations_for_portfolio,
    safe_generate_daily_recommendations_for_portfolio,
)
from .rules import StrategyRules

# MA_PERIOD 상수를 StrategyRules에서 가져옵니다.
MA_PERIOD = StrategyRules.DEFAULT_MA_PERIOD


# 전략 실행 인터페이스
def run_strategy_backtest(*args, **kwargs):
    """MAPS 전략 백테스트 실행"""
    return run_portfolio_backtest(*args, **kwargs)


def run_strategy_recommendation(*args, **kwargs):
    """MAPS 전략 추천 실행 (안전한 버전)"""
    return safe_generate_daily_recommendations_for_portfolio(*args, **kwargs)


__all__ = [
    "DECISION_CONFIG",
    "generate_daily_recommendations_for_portfolio",
    "safe_generate_daily_recommendations_for_portfolio",
    "run_portfolio_backtest",
    "run_single_ticker_backtest",
    "run_strategy_backtest",
    "run_strategy_recommendation",
    "MA_PERIOD",
    "StrategyRules",
]
