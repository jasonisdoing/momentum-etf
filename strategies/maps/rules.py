"""Momentum 전략에서 공통으로 사용하는 규칙/검증 헬퍼."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from config import TRADING_DAYS_PER_MONTH


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return bool(value)


@dataclass(frozen=True)
class StrategyRules:
    """Momentum 전략에서 공통으로 사용하는 핵심 파라미터."""

    ma_days: int
    bucket_topn: int
    ma_type: str
    rebalance_mode: str
    replacement_mode: str
    sell_on_negative_score: bool
    enable_data_sufficiency_check: bool

    @classmethod
    def from_values(
        cls,
        *,
        ma_days: Any = None,
        ma_month: Any = None,
        bucket_topn: Any = None,
        ma_type: Any = None,
        rebalance_mode: Any = None,
        replacement_mode: Any = None,
        sell_on_negative_score: Any = True,
        enable_data_sufficiency_check: Any = False,
    ) -> StrategyRules:
        # MA 기간 결정 (개월 우선)
        final_ma_days = None

        if ma_month is not None:
            try:
                month_val = int(ma_month)
                if month_val > 0:
                    final_ma_days = month_val * TRADING_DAYS_PER_MONTH
            except (TypeError, ValueError):
                pass

        if final_ma_days is None and ma_days is not None:
            try:
                final_ma_days = int(ma_days)
            except (TypeError, ValueError):
                pass

        if final_ma_days is None or final_ma_days <= 0:
            if ma_days is None and ma_month is None:
                raise ValueError("MA_MONTH은 필수입니다.")
            raise ValueError("MA_MONTH은 0보다 큰 정수여야 합니다.")

        # TOPN 처리 (BUCKET_TOPN)
        final_bucket_topn = 1
        topn_source = bucket_topn

        try:
            topn_val = int(topn_source)
            if topn_val > 0:
                final_bucket_topn = topn_val
            else:
                raise ValueError
        except (TypeError, ValueError):
            raise ValueError("BUCKET_TOPN은 0보다 큰 정수여야 합니다.")

        # MA 타입 검증
        if ma_type is None:
            raise ValueError("MA_TYPE은 필수입니다.")
        ma_type_str = str(ma_type).upper()
        valid_ma_types = {"SMA", "EMA", "WMA", "DEMA", "TEMA", "HMA"}
        if ma_type_str not in valid_ma_types:
            raise ValueError(f"MA_TYPE은 {valid_ma_types} 중 하나여야 합니다. (입력값: {ma_type_str})")

        # ENABLE_DATA_SUFFICIENCY_CHECK 검증
        data_sufficiency_check = _coerce_bool(enable_data_sufficiency_check, default=False)
        negative_score_exit = _coerce_bool(sell_on_negative_score, default=True)

        # REBALANCE_MODE 처리
        final_rebalance_mode = str(rebalance_mode).upper() if rebalance_mode else "TWICE_A_MONTH"
        final_replacement_mode = str(replacement_mode).upper() if replacement_mode else "WEEKLY"

        return cls(
            ma_days=final_ma_days,
            bucket_topn=final_bucket_topn,
            ma_type=ma_type_str,
            rebalance_mode=final_rebalance_mode,
            replacement_mode=final_replacement_mode,
            sell_on_negative_score=negative_score_exit,
            enable_data_sufficiency_check=data_sufficiency_check,
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
            ma_month=_resolve("MA_MONTH", "ma_month"),
            ma_days=_resolve("ma_days"),
            bucket_topn=_resolve("BUCKET_TOPN", "bucket_topn"),
            ma_type=_resolve("MA_TYPE", "ma_type"),
            rebalance_mode=_resolve("REBALANCE_MODE", "rebalance_mode"),
            replacement_mode=_resolve("REPLACEMENT_MODE", "replacement_mode"),
            sell_on_negative_score=_resolve("SELL_ON_NEGATIVE_SCORE", "sell_on_negative_score"),
            enable_data_sufficiency_check=_resolve("ENABLE_DATA_SUFFICIENCY_CHECK", "enable_data_sufficiency_check"),
        )

    def to_dict(self) -> dict[str, Any]:
        d = {
            "ma_days": self.ma_days,
            "bucket_topn": self.bucket_topn,
            "ma_type": self.ma_type,
            "rebalance_mode": self.rebalance_mode,
            "replacement_mode": self.replacement_mode,
            "sell_on_negative_score": self.sell_on_negative_score,
            "enable_data_sufficiency_check": self.enable_data_sufficiency_check,
        }
        return d


__all__ = [
    "StrategyRules",
]
