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
    coin_min_holding_cost_krw: Optional[float] = None

    @classmethod
    def from_values(
        cls,
        *,
        ma_period: Any,
        portfolio_topn: Any,
        replace_threshold: Any,
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

        coin_min_resolved = resolve_optional_float(coin_min_holding_cost_krw)

        return cls(
            ma_period=ma_period_int,
            portfolio_topn=portfolio_topn_int,
            replace_threshold=replace_threshold_float,
            coin_min_holding_cost_krw=coin_min_resolved,
        )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "StrategyRules":
        return cls.from_values(
            ma_period=mapping.get("MA_PERIOD") or mapping.get("ma_period"),
            portfolio_topn=mapping.get("PORTFOLIO_TOPN") or mapping.get("portfolio_topn"),
            replace_threshold=mapping.get("REPLACE_SCORE_THRESHOLD")
            or mapping.get("replace_threshold"),
            coin_min_holding_cost_krw=mapping.get("COIN_MIN_HOLDING_COST_KRW")
            or mapping.get("coin_min_holding_cost_krw"),
        )

    def to_dict(self) -> dict[str, Any]:
        d = {
            "ma_period": self.ma_period,
            "portfolio_topn": self.portfolio_topn,
            "replace_threshold": self.replace_threshold,
        }
        if self.coin_min_holding_cost_krw is not None:
            d["coin_min_holding_cost_krw"] = self.coin_min_holding_cost_krw
        return d


def resolve_optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "StrategyRules",
    "resolve_optional_float",
]
