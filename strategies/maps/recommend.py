"""Momentum 전략 추천 생성기 (호환성 레이어)."""

from __future__ import annotations

from typing import Any, Dict, List

# 전략 중립적인 포트폴리오 추천 함수를 logic/recommend/portfolio.py에서 import
from logic.recommend.portfolio import (
    run_portfolio_recommend,
    safe_run_portfolio_recommend,
    generate_daily_recommendations_for_portfolio,  # 호환성
    safe_generate_daily_recommendations_for_portfolio,  # 호환성
)


__all__ = [
    "run_portfolio_recommend",
    "safe_run_portfolio_recommend",
    "generate_daily_recommendations_for_portfolio",  # 호환성
    "safe_generate_daily_recommendations_for_portfolio",  # 호환성
]
