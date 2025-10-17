"""MAPS(Moving Average Position Score) 전략 서브패키지."""

from __future__ import annotations

from .backtest import run_portfolio_backtest, run_single_ticker_backtest
from .constants import DECISION_CONFIG
from .formatting import load_account_precision, get_header_money_formatter, format_shares
from .history import calculate_consecutive_holding_info, calculate_trade_cooldown_info
from .recommend import (
    generate_daily_recommendations_for_portfolio,
    safe_generate_daily_recommendations_for_portfolio,
)
from .rules import StrategyRules
from .scoring import normalize_ma_score, normalize_ma_score_with_config


# 전략 실행 인터페이스
def run_strategy_backtest(*args, **kwargs):
    """MAPS 전략 백테스트 실행"""
    return run_portfolio_backtest(*args, **kwargs)


def run_strategy_recommendation(*args, **kwargs):
    """MAPS 전략 추천 실행 (안전한 버전)"""
    return safe_generate_daily_recommendations_for_portfolio(*args, **kwargs)


__all__ = [
    "calculate_consecutive_holding_info",
    "calculate_trade_cooldown_info",
    "DECISION_CONFIG",
    "format_shares",
    "generate_daily_recommendations_for_portfolio",
    "get_header_money_formatter",
    "load_account_precision",
    "normalize_ma_score",
    "normalize_ma_score_with_config",
    "run_portfolio_backtest",
    "run_single_ticker_backtest",
    "run_strategy_backtest",
    "run_strategy_recommendation",
    "safe_generate_daily_recommendations_for_portfolio",
    "StrategyRules",
]
