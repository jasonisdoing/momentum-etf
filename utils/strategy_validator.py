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

    def _require(keys: list[str]) -> list[str]:
        errs: list[str] = []
        for key in keys:
            val = strategy_tuning.get(key)
            if val is None:
                errs.append(key)
                continue
            if isinstance(val, str) and not val.strip():
                errs.append(key)
        return errs

    errors: list[str] = []
    errors.extend(_require(["REBALANCE_MODE"]))
    # 비중은 기본적으로 종목 리스트(weight)에서 읽는다.
    # 설정의 TARGET_WEIGHTS는 하위 호환용 선택값으로만 허용한다.
    weights = strategy_tuning.get("TARGET_WEIGHTS")
    if isinstance(weights, dict):
        if not weights:
            errors.append("TARGET_WEIGHTS(비어있음)")
        total = 0.0
        for ticker, weight in weights.items():
            if not str(ticker).strip():
                errors.append("TARGET_WEIGHTS(티커)")
                continue
            try:
                w = float(weight)
            except (TypeError, ValueError):
                errors.append(f"TARGET_WEIGHTS({ticker}: 숫자)")
                continue
            if w <= 0:
                errors.append(f"TARGET_WEIGHTS({ticker}: 양수)")
                continue
            total += w
        if abs(total - 1.0) > 1e-3:
            errors.append("TARGET_WEIGHTS(합계=1.0)")
    elif weights is not None:
        errors.append("TARGET_WEIGHTS(dict)")

    if errors:
        raise ValueError(f"{prefix}필수 설정이 누락되었습니다: {', '.join(errors)}")


__all__ = ["validate_strategy_settings"]
