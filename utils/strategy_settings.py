"""전략 설정 공용 — 선택지 밖 저장값 보정.

모멘텀·신고가 설정은 서버 상수(선택지)가 화면 셀렉트·검증·튜닝 축을 모두 정한다. 선택지를
바꾸면 저장값이 목록 밖이 될 수 있는데, 화면이 막히지 않게 **첫 선택지로 보정**하고 무엇을
바꿨는지 함께 돌려준다. 화면은 이를 '저장하지 않은 변경'으로 띄워 사용자가 확인·저장하게 한다.
조용한 대체가 아니라 보정 내역을 알리는 것이고, 배치·백테스트는 엄격 검증을 그대로 쓴다.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any


def _label(value: Any) -> str:
    return "없음" if value is None else str(value)


def coerce_to_options(
    settings: dict[str, Any],
    fields: Iterable[tuple[str, str, tuple]],
    validate: Callable[[dict[str, Any]], dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    """``validate`` 를 통과하면 그대로, 실패하면 ``fields`` (키, 라벨, 선택지) 중 목록 밖 값을
    첫 선택지로 바꿔 다시 검증한다. 반환: (정규화된 설정, ["종목 수 7 → 5", ...])."""
    try:
        return validate(settings), []
    except ValueError:
        pass
    coerced: list[str] = []
    merged = dict(settings)
    for key, label, options in fields:
        value = merged.get(key)
        if value not in options:
            coerced.append(f"{label} {_label(value)} → {_label(options[0])}")
            merged[key] = options[0]
    return validate(merged), coerced


def validate_start_date(value: Any) -> str | None:
    """미설정은 유지하고, 입력한 전략 시작일은 ISO 날짜로 엄격히 검증한다."""
    import re
    from datetime import date

    if value is None:
        return None
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError("전략 시작일은 YYYY-MM-DD 형식이어야 합니다.")
    try:
        date.fromisoformat(value)
    except ValueError as error:
        raise ValueError("전략 시작일이 올바른 날짜가 아닙니다.") from error
    return value


def require_start_date(settings: dict[str, Any]) -> str:
    """운용은 전략·종목풀에 저장된 시작일 없이는 계산하지 않는다."""
    value = validate_start_date(settings.get("start_date"))
    if value is None:
        raise ValueError(f"{settings.get('pool', '')} 전략 시작일을 선택하고 저장하세요.")
    return value
