"""선택된 전략을 전략 로더를 통해 노출하는 엔트리 포인트."""

from __future__ import annotations

from logic.strategy_loader import (
    get_backtest_module,
    get_constants_module,
    get_recommend_module,
    get_rules_module,
)

# 모듈 로드
_backtest_mod = get_backtest_module()
_constants_mod = get_constants_module()
_recommend_mod = get_recommend_module()
_rules_mod = get_rules_module()

# 전략 규칙 클래스
StrategyRules = getattr(_rules_mod, "StrategyRules")

# 백테스트 함수들
run_portfolio_backtest = getattr(_backtest_mod, "run_portfolio_backtest")
run_single_ticker_backtest = getattr(_backtest_mod, "run_single_ticker_backtest")

# 상수
DECISION_CONFIG = getattr(_constants_mod, "DECISION_CONFIG")

# 추천 함수
generate_daily_signals_for_portfolio = getattr(
    _recommend_mod, "generate_daily_signals_for_portfolio"
)

# MA_PERIOD 상수 (이전 버전과의 호환성을 위해 유지)
MA_PERIOD = StrategyRules.DEFAULT_MA_PERIOD

__all__ = [
    "StrategyRules",
    "run_portfolio_backtest",
    "run_single_ticker_backtest",
    "DECISION_CONFIG",
    "generate_daily_signals_for_portfolio",
    "MA_PERIOD",
]
