"""추천 결과 표시에 사용하는 포맷터."""

from __future__ import annotations

from typing import Any


def format_price(value: Any, country_code: str) -> str:
    """국가별 통화 형식으로 가격을 문자열로 반환한다."""

    if value is None:
        return "-"

    try:
        amount = float(value)
    except (TypeError, ValueError):
        return str(value)

    country_norm = (country_code or "").strip().lower()

    if country_norm in {"kr", "kor"}:
        return f"{amount:,.0f}원"
    if country_norm in {"aus", "au", "australia"}:
        return f"A${amount:,.2f}"
    if country_norm in {"us", "usa", "united states"}:
        return f"${amount:,.2f}"

    return f"{amount:,.2f}"


def format_price_deviation(value: Any) -> str:
    """괴리율 값을 이모지와 함께 문자열로 변환한다."""

    if value is None:
        return "-"
    try:
        deviation = float(value)
    except (TypeError, ValueError):
        return str(value)

    prefix = ""
    if deviation > 2.0:
        prefix = "💀"
    elif deviation < -2.0:
        prefix = "👍"

    return f"{prefix}{deviation:+.2f}%"
