"""순위/보유 데이터 표시에 사용하는 포맷터."""

from __future__ import annotations

import re
from typing import Any

# 영문 티커 + 한글 풀네임 괄호 패턴 (예: "AMD(어드밴스드 마이크로 디바이시스)" → "AMD").
# 한글 우선주 표기 (예: "현대모비스(우)") 같은 케이스는 영향 받지 않는다 — 영문 시작 패턴만 매치.
_HOLDING_NAME_PAREN_RE = re.compile(r"^([A-Z][A-Z0-9.\-]*)\s*\(.+\)$")


def clean_holding_display_name(name: str | None) -> str:
    """ETF 구성종목 표시 이름에서 영문 티커 뒤 괄호 풀네임을 제거한다.

    예:
        "AMD(어드밴스드 마이크로 디바이시스)" → "AMD"
        "TSMC(타이완반도체제조)" → "TSMC"
        "Kioxia Holdings Corp" → "Kioxia Holdings Corp"  (변경 없음)
        "엔비디아" → "엔비디아"  (변경 없음)
        "현대모비스(우)" → "현대모비스(우)"  (한글 시작이라 변경 없음)
    """
    text = (name or "").strip()
    if not text:
        return text
    match = _HOLDING_NAME_PAREN_RE.match(text)
    if match:
        return match.group(1)
    return text


def format_price(value: Any, country_code: str) -> str:
    """국가별 통화 형식으로 가격을 문자열로 반환한다."""

    if value is None:
        return "-"

    try:
        amount = float(value)
    except (TypeError, ValueError):
        return str(value)

    country_norm = (country_code or "").strip().lower()

    if country_norm in {"kr", "kor", "krw"}:
        return f"{amount:,.0f}원"

    if country_norm in {"us", "usa", "usd"}:
        return f"${amount:,.2f}"

    if country_norm in {"au", "aus", "aud"}:
        return f"A${amount:,.2f}"

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


def format_trading_days(days: Any) -> str:
    """보유일을 '1W 2D' 형식으로 변환한다. (7일 = 1주, 캘린더 일자 기준)"""
    if days is None:
        return "-"

    try:
        val = int(days)
    except (TypeError, ValueError):
        return str(days)

    if val < 0:
        return "-"

    weeks = val // 7
    remaining_days = val % 7

    parts = []
    if weeks > 0:
        parts.append(f"{weeks}W")
    if remaining_days > 0 or val == 0:
        parts.append(f"{remaining_days}D")

    return " ".join(parts)
