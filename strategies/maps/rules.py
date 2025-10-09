"""Momentum 전략에서 공통으로 사용하는 규칙/검증 헬퍼."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class StrategyRules:
    """Momentum 전략에서 공통으로 사용하는 핵심 파라미터."""

    # 기본 MA 기간 (이동평균 기간)
    DEFAULT_MA_PERIOD = 20

    ma_period: int
    portfolio_topn: int
    replace_threshold: float

    @classmethod
    def from_values(
        cls,
        *,
        ma_period: Any,
        portfolio_topn: Any,
        replace_threshold: Any,
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

        return cls(
            ma_period=ma_period_int,
            portfolio_topn=portfolio_topn_int,
            replace_threshold=replace_threshold_float,
        )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "StrategyRules":
        def _resolve(*keys: str) -> Any:
            sentinel = object()
            for key in keys:
                value = mapping.get(key, sentinel)
                if value is not sentinel:
                    return value
            return None

        return cls.from_values(
            ma_period=_resolve("MA_PERIOD", "ma_period"),
            portfolio_topn=_resolve("PORTFOLIO_TOPN", "portfolio_topn"),
            replace_threshold=_resolve("REPLACE_SCORE_THRESHOLD", "replace_threshold"),
        )

    def to_dict(self) -> dict[str, Any]:
        d = {
            "ma_period": self.ma_period,
            "portfolio_topn": self.portfolio_topn,
            "replace_threshold": self.replace_threshold,
        }
        return d


__all__ = [
    "StrategyRules",
]
