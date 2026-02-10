"""Momentum 전략에서 공통으로 사용하는 규칙/검증 헬퍼."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StrategyRules:
    """Momentum 전략에서 공통으로 사용하는 핵심 파라미터."""

    # 기본 MA 기간 (이동평균 기간)
    DEFAULT_MA_PERIOD = 20
    # 기본 MA 타입 (이동평균 종류)
    DEFAULT_MA_TYPE = "SMA"

    ma_period: int
    portfolio_topn: int
    replace_threshold: float
    ma_type: str = "SMA"
    stop_loss_pct: float | None = None
    enable_data_sufficiency_check: bool = False
    max_per_category: int = 1

    @classmethod
    def from_values(
        cls,
        *,
        ma_period: Any,
        portfolio_topn: Any,
        replace_threshold: Any,
        ma_type: Any = None,
        stop_loss_pct: Any = None,
        enable_data_sufficiency_check: Any = False,
        max_per_category: Any = 1,
    ) -> StrategyRules:
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

        # MA 타입 검증
        ma_type_str = str(ma_type or cls.DEFAULT_MA_TYPE).upper()
        valid_ma_types = {"SMA", "EMA", "WMA", "DEMA", "TEMA", "HMA"}
        if ma_type_str not in valid_ma_types:
            raise ValueError(f"MA_TYPE은 {valid_ma_types} 중 하나여야 합니다. (입력값: {ma_type_str})")

        stop_loss_value: float | None = None
        if stop_loss_pct is not None:
            try:
                stop_loss_value = float(stop_loss_pct)
            except (TypeError, ValueError) as exc:
                raise ValueError("STOP_LOSS_PCT는 숫자여야 합니다.") from exc
            if not (stop_loss_value > 0):
                raise ValueError("STOP_LOSS_PCT는 0보다 커야 합니다.")

        # ENABLE_DATA_SUFFICIENCY_CHECK 검증
        data_sufficiency_check = bool(enable_data_sufficiency_check)

        # MAX_PER_CATEGORY 검증
        try:
            max_per_category_int = int(max_per_category)
        except (TypeError, ValueError):
            max_per_category_int = 1
        if max_per_category_int < 1:
            max_per_category_int = 1

        return cls(
            ma_period=ma_period_int,
            portfolio_topn=portfolio_topn_int,
            replace_threshold=replace_threshold_float,
            ma_type=ma_type_str,
            stop_loss_pct=stop_loss_value,
            enable_data_sufficiency_check=data_sufficiency_check,
            max_per_category=max_per_category_int,
        )

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> StrategyRules:
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
            ma_type=_resolve("MA_TYPE", "ma_type"),
            stop_loss_pct=_resolve("STOP_LOSS_PCT", "stop_loss_pct"),
            # 데이터 충분성 검증 활성화 여부
            # True: 신규 상장 ETF 등 데이터가 부족한 경우 완화된 기준 적용
            # False: 데이터 충분성 검증 비활성화 (모든 종목에 대해 계산 시도)
            enable_data_sufficiency_check=_resolve("ENABLE_DATA_SUFFICIENCY_CHECK", "enable_data_sufficiency_check"),
            max_per_category=_resolve("MAX_PER_CATEGORY", "max_per_category"),
        )

    def to_dict(self) -> dict[str, Any]:
        d = {
            "ma_period": self.ma_period,
            "portfolio_topn": self.portfolio_topn,
            "replace_threshold": self.replace_threshold,
            "ma_type": self.ma_type,
            "stop_loss_pct": self.stop_loss_pct,
            "enable_data_sufficiency_check": self.enable_data_sufficiency_check,
            "max_per_category": self.max_per_category,
        }
        return d


__all__ = [
    "StrategyRules",
]
