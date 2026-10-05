"""서비스 공통 정규화 유틸리티."""

from __future__ import annotations

import datetime as _dt
import math
from typing import Any


def normalize_exchange_symbol(ticker: str, *, suffix: str) -> str:
    """확인된 상장 시장의 티커를 공통 시세 심볼로 변환한다."""
    base = ticker.strip().rstrip(".").replace(" ", "-")
    if not suffix:
        return base.replace(".", "-").replace("/", "-")
    if suffix == "HK" and base.isdigit():
        base = base.zfill(4)
    return f"{base}.{suffix}"


def normalize_number(value: Any) -> float:
    """숫자로 변환한다. 실패 시 0.0을 반환한다."""
    try:
        val = float(value or 0)
        if math.isnan(val):
            return 0.0
        return val
    except (TypeError, ValueError):
        return 0.0


def normalize_nullable_number(value: Any) -> float | None:
    """숫자로 변환한다. 빈 값이면 None을 반환한다."""
    if value in (None, "", "-"):
        return None
    try:
        val = float(str(value).replace(",", ""))
        if math.isnan(val):
            return None
        return val
    except (TypeError, ValueError):
        return None


def normalize_text(value: Any, fallback: str = "") -> str:
    """문자열로 변환하고 양쪽 공백을 제거한다."""
    text = str(value or "").strip()
    return text or fallback


def to_timestamp_iso(value: Any, *, naive_timezone: _dt.tzinfo) -> str | None:
    """수집 시각을 ISO로 변환하고 시간대 없는 값에는 명시한 원본 시간대를 붙인다."""
    if value is None or value == "":
        return None
    parsed = value if isinstance(value, _dt.datetime) else _dt.datetime.fromisoformat(str(value))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=naive_timezone)
    return parsed.isoformat()


def to_iso_string(value: Any) -> str | None:
    """datetime/date를 ISO 문자열로 변환한다. None이면 None을 반환한다."""
    if value is None:
        return None
    if isinstance(value, _dt.datetime):
        if value.tzinfo is None:
            # Mongo에서 읽힌 naive datetime은 UTC로 간주해 offset을 명시한다.
            value = value.replace(tzinfo=_dt.timezone.utc)
        return value.isoformat()
    if isinstance(value, _dt.date):
        return value.isoformat()
    return str(value)
