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

    if country_norm in {"us", "usa", "usd"}:
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
    if deviation > 3.0:
        prefix = "💀"
    elif deviation < -2.0:
        prefix = "👍"

    return f"{prefix}{deviation:+.2f}%"


def format_pct_change(value: Any) -> str:
    """수익률(%) 등을 포맷팅한다. 0.00%는 부호 없이 정렬용 공백만 추가."""
    if value is None:
        return "-"
    try:
        val = float(value)
    except (TypeError, ValueError):
        return str(value) if value != "-" else "-"

    if abs(val) < 0.005:
        # " 0.00%" (aligns with "+1.23%")
        return f" {abs(val):.2f}%"

    return f"{val:+.2f}%"
