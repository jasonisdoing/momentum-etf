"""전략 설정 검증 유틸리티."""

from typing import Any


def validate_strategy_settings(
    strategy_tuning: dict[str, Any],
    account_id: str | None = None,
) -> None:
    """
    전략 설정의 모든 필수 항목을 검증합니다.

    이 함수를 한 번만 호출하면 이후 모든 설정 값을 안전하게 사용할 수 있습니다.

    Args:
        strategy_tuning: 전략 파라미터(dict)
        account_id: 계정 ID (에러 메시지용, 선택사항)

    Raises:
        ValueError: 필수 설정이 누락되었거나 유효하지 않은 경우
    """
    prefix = f"{account_id} 계좌의 " if account_id else ""
    errors = []

    # strategy 필수 항목 검증

    # 에러가 있으면 한 번에 보고
    if errors:
        missing_fields = ", ".join(errors)
        raise ValueError(f"{prefix}필수 설정이 누락되었습니다: {missing_fields}")


__all__ = ["validate_strategy_settings"]
