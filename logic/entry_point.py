"""MAPS 전략 컴포넌트를 노출하는 엔트리 포인트."""

from __future__ import annotations

from strategies.maps import backtest as _backtest_mod
from strategies.maps import constants as _constants_mod
from strategies.maps import recommend as _recommend_mod
from strategies.maps import rules as _rules_mod

# 전략 규칙 클래스
StrategyRules = getattr(_rules_mod, "StrategyRules")

# 백테스트 함수들 - portfolio_runner에서 직접 import
from logic.backtest.portfolio_runner import run_portfolio_backtest

run_single_ticker_backtest = getattr(_backtest_mod, "run_single_ticker_backtest")

# 상수
DECISION_CONFIG = getattr(_constants_mod, "DECISION_CONFIG")

# 추천 함수
generate_daily_recommendations_for_portfolio = getattr(_recommend_mod, "generate_daily_recommendations_for_portfolio")

__all__ = [
    "StrategyRules",
    "run_portfolio_backtest",
    "run_single_ticker_backtest",
    "DECISION_CONFIG",
    "generate_daily_recommendations_for_portfolio",
]
