"""MAPS 전략 컴포넌트를 노출하는 엔트리 포인트."""

from __future__ import annotations

from strategies.maps import constants as _constants_mod
from strategies.maps import rules as _rules_mod

# 전략 규칙 클래스
StrategyRules = getattr(_rules_mod, "StrategyRules")

# 백테스트 함수들
from logic.backtest.engine import run_portfolio_backtest
from strategies.maps.backtest import run_single_ticker_backtest

# 상수
DECISION_CONFIG = getattr(_constants_mod, "DECISION_CONFIG")

# 추천 함수 (recommend.py에서 별도 제공 - 백테스트 기반)
# from recommend import generate_recommendation_report
from strategies.maps.evaluator import StrategyEvaluator

# 공유 로직
from strategies.maps.metrics import process_ticker_data

__all__ = [
    "StrategyRules",
    "run_portfolio_backtest",
    "run_single_ticker_backtest",
    "DECISION_CONFIG",
    "process_ticker_data",
    "StrategyEvaluator",
]
