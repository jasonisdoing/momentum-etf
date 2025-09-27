"""Momentum 전략에서 공통으로 사용하는 규칙/검증 헬퍼."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional


@dataclass(frozen=True)
class StrategyRules:
    """Momentum 전략에서 공통으로 사용하는 핵심 파라미터."""

    ma_period: int
    portfolio_topn: int
    replace_threshold: float
    min_buy_score: Optional[float] = None
    coin_min_holding_cost_krw: Optional[float] = None

    @classmethod
    def from_values(
        cls,
        *,
        ma_period: Any,
        portfolio_topn: Any,
        replace_threshold: Any,
        min_buy_score: Any = None,
        coin_min_holding_cost_krw: Any = None,
    ) -> "StrategyRules":
        try:
            ma_period_int = int(ma_period)
        except (TypeError, ValueError):
            raise ValueError("MA_PERIOD는 0보다 큰 정수여야 합니다.") from None
        if ma_period_int <= 0:
            raise ValueError("MA_PERIOD는 0보다 큰 정수여야 합니다.")

        try:
            portfolio_topn_int = int(portfolio_topn)
        except (TypeError, ValueError):
            raise ValueError("PORTFOLIO_TOPN은 0보다 큰 정수여야 합니다.") from None
        if portfolio_topn_int <= 0:
            raise ValueError("PORTFOLIO_TOPN은 0보다 큰 정수여야 합니다.")

        try:
            replace_threshold_float = float(replace_threshold)
        except (TypeError, ValueError):
            raise ValueError("REPLACE_SCORE_THRESHOLD는 숫자여야 합니다.") from None

        min_buy_score_resolved = resolve_min_buy_score(min_buy_score)
        coin_min_resolved = resolve_optional_float(coin_min_holding_cost_krw)

        return cls(
            ma_period=ma_period_int,
            portfolio_topn=portfolio_topn_int,
            replace_threshold=replace_threshold_float,
            min_buy_score=min_buy_score_resolved,
            coin_min_holding_cost_krw=coin_min_resolved,
        )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "StrategyRules":
        return cls.from_values(
            ma_period=mapping.get("MA_PERIOD") or mapping.get("ma_period"),
            portfolio_topn=mapping.get("PORTFOLIO_TOPN") or mapping.get("portfolio_topn"),
            replace_threshold=mapping.get("REPLACE_SCORE_THRESHOLD")
            or mapping.get("replace_threshold"),
            min_buy_score=mapping.get("MIN_BUY_SCORE") or mapping.get("min_buy_score"),
            coin_min_holding_cost_krw=mapping.get("COIN_MIN_HOLDING_COST_KRW")
            or mapping.get("coin_min_holding_cost_krw"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "ma_period": self.ma_period,
            "portfolio_topn": self.portfolio_topn,
            "replace_threshold": self.replace_threshold,
            "min_buy_score": self.min_buy_score,
            "coin_min_holding_cost_krw": self.coin_min_holding_cost_krw,
        }


def resolve_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def resolve_min_buy_score(raw_value: Any) -> Optional[float]:
    """입력값을 부동소수점 최소 점수로 변환합니다 (유효하지 않으면 None)."""

    if raw_value is None:
        return None

    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        return None

    return value


def passes_min_buy_score(score_value: Any, min_buy_score: Optional[float]) -> bool:
    """주어진 점수가 최소 매수 점수 조건을 충족하는지 확인합니다."""

    if min_buy_score is None:
        return True

    try:
        value = float(score_value)
    except (TypeError, ValueError):
        return False

    return value >= min_buy_score


def format_min_buy_shortfall(score_value: Any, min_buy_score: Optional[float]) -> Optional[str]:
    """최소 점수 미달 메시지를 반환합니다 (충족 시 None)."""

    if min_buy_score is None:
        return None

    try:
        value = float(score_value)
    except (TypeError, ValueError):
        return "점수 미달"

    if value >= min_buy_score:
        return None

    return f"점수 미달(최소 {min_buy_score:.2f})"


__all__ = [
    "StrategyRules",
    "format_min_buy_shortfall",
    "passes_min_buy_score",
    "resolve_min_buy_score",
    "resolve_optional_float",
]
