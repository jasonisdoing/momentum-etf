"""Momentum 전략에서 공통으로 사용하는 규칙/검증 헬퍼."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from config import TRADING_DAYS_PER_MONTH


@dataclass(frozen=True)
class StrategyRules:
    """Momentum 전략에서 공통으로 사용하는 핵심 파라미터."""

    # 기본 MA 타입 (이동평균 종류)
    DEFAULT_MA_TYPE = "SMA"

    ma_days: int
    bucket_topn: int
    replace_threshold: float
    ma_type: str = "SMA"
    stop_loss_pct: float | None = None
    enable_data_sufficiency_check: bool = False

    # [신규] 리밸런싱 모드 (QUARTERLY, MONTHLY, DAILY)
    rebalance_mode: str = "QUARTERLY"

    # [신규] 음수 점수 허용 여부 (상대 모멘텀)
    allow_negative_score: bool = True

    @classmethod
    def from_values(
        cls,
        *,
        ma_days: Any = None,
        ma_month: Any = None,
        bucket_topn: Any = None,
        replace_threshold: Any,
        ma_type: Any = None,
        stop_loss_pct: Any = None,
        enable_data_sufficiency_check: Any = False,
        rebalance_mode: Any = None,
        allow_negative_score: Any = None,
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

        # Rebalance Mode
        final_rebalance_mode = str(rebalance_mode).upper() if rebalance_mode else "QUARTERLY"

        # Allow Negative Score
        final_allow_negative = bool(allow_negative_score) if allow_negative_score is not None else True

        return cls(
            ma_days=final_ma_days,
            bucket_topn=final_bucket_topn,
            replace_threshold=replace_threshold_float,
            ma_type=ma_type_str,
            stop_loss_pct=stop_loss_value,
            enable_data_sufficiency_check=data_sufficiency_check,
            rebalance_mode=final_rebalance_mode,
            allow_negative_score=final_allow_negative,
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
            replace_threshold=_resolve("REPLACE_SCORE_THRESHOLD", "replace_threshold"),
            ma_type=_resolve("MA_TYPE", "ma_type"),
            stop_loss_pct=_resolve("STOP_LOSS_PCT", "stop_loss_pct"),
            enable_data_sufficiency_check=_resolve("ENABLE_DATA_SUFFICIENCY_CHECK", "enable_data_sufficiency_check"),
            rebalance_mode=_resolve("REBALANCE_MODE", "rebalance_mode"),
            allow_negative_score=_resolve("ALLOW_NEGATIVE_SCORE", "allow_negative_score"),
        )

    def to_dict(self) -> dict[str, Any]:
        d = {
            "ma_days": self.ma_days,
            "bucket_topn": self.bucket_topn,
            "replace_threshold": self.replace_threshold,
            "ma_type": self.ma_type,
            "stop_loss_pct": self.stop_loss_pct,
            "enable_data_sufficiency_check": self.enable_data_sufficiency_check,
            "rebalance_mode": self.rebalance_mode,
            "allow_negative_score": self.allow_negative_score,
        }
        return d


__all__ = [
    "StrategyRules",
]
