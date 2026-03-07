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

    strategy: str
    ma_days: int
    bucket_topn: int
    ma_type: str
    rebalance_mode: str
    cooldown_days: int
    enable_data_sufficiency_check: bool
    target_weights: dict[str, float] | None = None

    @classmethod
    def from_values(
        cls,
        *,
        strategy: Any = "MAPS",
        ma_days: Any = None,
        ma_month: Any = None,
        topn: Any = None,
        bucket_topn: Any = None,
        ma_type: Any = None,
        rebalance_mode: Any = None,
        cooldown: Any = None,
        cooldown_days: Any = None,
        target_weights: Any = None,
        enable_data_sufficiency_check: Any = False,
    ) -> StrategyRules:
        strategy_str = str(strategy or "MAPS").strip().upper()
        if not strategy_str:
            strategy_str = "MAPS"

        # HR: MA/점수 기반 파라미터 대신 목표 비중(선택)을 사용
        if strategy_str == "HR":
            normalized_weights: dict[str, float] | None = None
            if target_weights is not None:
                if not isinstance(target_weights, Mapping):
                    raise ValueError("TARGET_WEIGHTS는 dict 형태여야 합니다.")
                if not target_weights:
                    raise ValueError("TARGET_WEIGHTS가 비어 있습니다.")
                normalized_weights = {}
                weight_sum = 0.0
                for ticker, weight in target_weights.items():
                    ticker_key = str(ticker or "").strip().upper()
                    if not ticker_key:
                        raise ValueError("TARGET_WEIGHTS의 티커 키가 비어 있습니다.")
                    try:
                        weight_val = float(weight)
                    except (TypeError, ValueError):
                        raise ValueError(f"TARGET_WEIGHTS[{ticker_key}] 값은 숫자여야 합니다.")
                    if weight_val <= 0:
                        raise ValueError(f"TARGET_WEIGHTS[{ticker_key}] 값은 0보다 커야 합니다.")
                    normalized_weights[ticker_key] = weight_val
                    weight_sum += weight_val
                if abs(weight_sum - 1.0) > 1e-3:
                    raise ValueError("TARGET_WEIGHTS의 합계는 1.0이어야 합니다.")

            final_rebalance_mode = str(rebalance_mode).upper() if rebalance_mode else "TWICE_A_MONTH"
            data_sufficiency_check = _coerce_bool(enable_data_sufficiency_check, default=False)
            try:
                hr_topn = int(topn if topn is not None else bucket_topn)
                if hr_topn < 1:
                    raise ValueError
            except (TypeError, ValueError):
                hr_topn = len(normalized_weights) if normalized_weights else 1

            return cls(
                strategy=strategy_str,
                ma_days=1,
                bucket_topn=hr_topn,
                ma_type="SMA",
                rebalance_mode=final_rebalance_mode,
                cooldown_days=1,
                enable_data_sufficiency_check=data_sufficiency_check,
                target_weights=normalized_weights,
            )

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

        # TOPN 처리
        final_bucket_topn = 1
        topn_source = topn if topn is not None else bucket_topn

        try:
            topn_val = int(topn_source)
            if topn_val > 0:
                final_bucket_topn = topn_val
            else:
                raise ValueError
        except (TypeError, ValueError):
            raise ValueError("TOPN은 0보다 큰 정수여야 합니다.")

        # MA 타입 검증
        if ma_type is None:
            raise ValueError("MA_TYPE은 필수입니다.")
        ma_type_str = str(ma_type).upper()
        valid_ma_types = {"SMA", "EMA", "WMA", "DEMA", "TEMA", "HMA"}
        if ma_type_str not in valid_ma_types:
            raise ValueError(f"MA_TYPE은 {valid_ma_types} 중 하나여야 합니다. (입력값: {ma_type_str})")

        # ENABLE_DATA_SUFFICIENCY_CHECK 검증
        data_sufficiency_check = _coerce_bool(enable_data_sufficiency_check, default=False)

        # REBALANCE_MODE 처리
        final_rebalance_mode = str(rebalance_mode).upper() if rebalance_mode else "TWICE_A_MONTH"

        # COOLDOWN 처리
        cooldown_source = cooldown if cooldown is not None else cooldown_days
        if cooldown_source is None:
            raise ValueError("COOLDOWN은 필수입니다.")
        try:
            final_cooldown_days = int(cooldown_source)
        except (TypeError, ValueError):
            raise ValueError("COOLDOWN은 1 이상의 정수여야 합니다.")
        if final_cooldown_days < 1:
            raise ValueError("COOLDOWN은 1 이상의 정수여야 합니다.")

        return cls(
            strategy=strategy_str,
            ma_days=final_ma_days,
            bucket_topn=final_bucket_topn,
            ma_type=ma_type_str,
            rebalance_mode=final_rebalance_mode,
            cooldown_days=final_cooldown_days,
            enable_data_sufficiency_check=data_sufficiency_check,
            target_weights=None,
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
            strategy=_resolve("STRATEGY", "strategy"),
            ma_month=_resolve("MA_MONTH", "ma_month"),
            ma_days=_resolve("ma_days"),
            topn=_resolve("TOPN", "topn"),
            ma_type=_resolve("MA_TYPE", "ma_type"),
            rebalance_mode=_resolve("REBALANCE_MODE", "rebalance_mode"),
            cooldown=_resolve("COOLDOWN", "cooldown", "cooldown_days"),
            target_weights=_resolve("TARGET_WEIGHTS", "target_weights"),
            enable_data_sufficiency_check=_resolve("ENABLE_DATA_SUFFICIENCY_CHECK", "enable_data_sufficiency_check"),
        )

    def to_dict(self) -> dict[str, Any]:
        d = {
            "strategy": self.strategy,
            "ma_days": self.ma_days,
            "topn": self.bucket_topn,
            "bucket_topn": self.bucket_topn,
            "ma_type": self.ma_type,
            "rebalance_mode": self.rebalance_mode,
            "cooldown_days": self.cooldown_days,
            "cooldown": self.cooldown_days,
            "enable_data_sufficiency_check": self.enable_data_sufficiency_check,
        }
        if self.target_weights:
            d["target_weights"] = dict(self.target_weights)
            d["TARGET_WEIGHTS"] = dict(self.target_weights)
        return d

    @property
    def topn(self) -> int:
        return int(self.bucket_topn)


__all__ = [
    "StrategyRules",
]
