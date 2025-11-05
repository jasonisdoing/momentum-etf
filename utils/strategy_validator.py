"""전략 설정 검증 유틸리티."""

from typing import Dict, Any, Optional


def validate_strategy_settings(
    strategy_tuning: Dict[str, Any],
    account_id: Optional[str] = None,
) -> None:
    """
    전략 설정의 모든 필수 항목을 검증합니다.

    이 함수를 한 번만 호출하면 이후 모든 설정 값을 안전하게 사용할 수 있습니다.

    Args:
        strategy_tuning: strategy.tuning 설정
        account_id: 계정 ID (에러 메시지용, 선택사항)

    Raises:
        ValueError: 필수 설정이 누락되었거나 유효하지 않은 경우
    """
    prefix = f"{account_id} 계좌의 " if account_id else ""
    errors = []

    # strategy.tuning 필수 항목 검증
    if "COOLDOWN_DAYS" not in strategy_tuning:
        errors.append("strategy.tuning.COOLDOWN_DAYS")

    if "OVERBOUGHT_SELL_THRESHOLD" not in strategy_tuning:
        errors.append("strategy.tuning.OVERBOUGHT_SELL_THRESHOLD")

    if "REPLACE_SCORE_THRESHOLD" not in strategy_tuning:
        errors.append("strategy.tuning.REPLACE_SCORE_THRESHOLD")

    # 에러가 있으면 한 번에 보고
    if errors:
        missing_fields = ", ".join(errors)
        raise ValueError(f"{prefix}필수 설정이 누락되었습니다: {missing_fields}")

    # 값 범위 검증
    try:
        cooldown_days = int(strategy_tuning["COOLDOWN_DAYS"])
        if cooldown_days < 0:
            raise ValueError(f"{prefix}COOLDOWN_DAYS는 0 이상이어야 합니다.")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{prefix}COOLDOWN_DAYS 값이 유효하지 않습니다.") from exc

    try:
        rsi_threshold = int(strategy_tuning["OVERBOUGHT_SELL_THRESHOLD"])
        if not (0 <= rsi_threshold <= 100):
            raise ValueError(f"{prefix}OVERBOUGHT_SELL_THRESHOLD는 0~100 사이여야 합니다. (현재값: {rsi_threshold})")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{prefix}OVERBOUGHT_SELL_THRESHOLD 값이 유효하지 않습니다.") from exc

    if "STOP_LOSS_PCT" in strategy_tuning and strategy_tuning["STOP_LOSS_PCT"] is not None:
        try:
            stop_loss_pct = float(strategy_tuning["STOP_LOSS_PCT"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{prefix}STOP_LOSS_PCT 값이 유효하지 않습니다.") from exc
        if stop_loss_pct <= 0:
            raise ValueError(f"{prefix}STOP_LOSS_PCT는 0보다 커야 합니다. (현재값: {stop_loss_pct})")


__all__ = ["validate_strategy_settings"]
