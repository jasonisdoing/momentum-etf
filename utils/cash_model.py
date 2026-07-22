"""다통화 현금 잔액 공통 모델.

한 계좌가 KRW/USD/AUD 등 **여러 통화의 현금**을 가질 수 있도록 하는 정규화 계층이다.

두 곳에 데이터가 나뉜다:
- 설정(``account_settings``): ``cash_currencies`` = 활성 통화 목록(상단 입력 박스 표시용).
- 원장(``portfolio_master.accounts[]``): ``cash`` = 통화별 **native 잔액** 맵 ``{"KRW": 3000000, "USD": 320.6}``.

과거 구조(단일 현금)와의 하위호환을 위해, 신 형식이 없으면 레거시 필드
(``cash_balance``/``cash_balance_native``/``cash_currency``)에서 **결정적으로** 합성한다.
임의 보정(환율로 되돌리기 등)은 하지 않는다.
"""

from __future__ import annotations

from typing import Any


def currency_for_country(country_code: str | None) -> str:
    """국가 코드에 해당하는 통화 코드를 반환한다(종목 통화 각인용).

    us→USD, au→AUD, 그 외(kor 등)→KRW. 결정적 매핑이며 임의 보정은 없다.
    """
    code = str(country_code or "").strip().lower()
    if code == "us":
        return "USD"
    if code == "au":
        return "AUD"
    return "KRW"


def default_cash_currencies(currency: str | None, country_code: str | None = None) -> list[str]:
    """활성 통화 목록의 기본값. 계좌 통화 1개로 시작한다.

    호주 계좌는 AUD, 그 외는 계좌 통화(보통 KRW). 마이그레이션·신규 계좌에서 사용.
    """
    curr = str(currency or "").strip().upper()
    if curr:
        return [curr]
    code = str(country_code or "").strip().lower()
    if code == "au":
        return ["AUD"]
    if code == "us":
        return ["USD"]
    return ["KRW"]


def resolve_cash_currencies(settings_doc: dict[str, Any]) -> list[str]:
    """설정 문서에서 활성 통화 목록을 정규화해 반환한다.

    신 형식(``cash_currencies``)이 있으면 그대로, 없으면 계좌 통화 기준 기본값.
    중복 제거 + 대문자화 + 입력 순서 보존.
    """
    raw = settings_doc.get("cash_currencies")
    if isinstance(raw, (list, tuple)) and raw:
        seen: list[str] = []
        for item in raw:
            code = str(item or "").strip().upper()
            if code and code not in seen:
                seen.append(code)
        if seen:
            return seen
    return default_cash_currencies(settings_doc.get("currency"), settings_doc.get("country_code"))


def resolve_cash_native_map(account_doc: dict[str, Any], settings_currency: str | None = None) -> dict[str, float]:
    """원장 계좌 문서에서 통화별 native 현금 잔액 맵을 반환한다.

    우선순위:
    1. 신 형식 ``cash`` 맵이 있으면 그대로 정규화.
    2. 없으면 레거시(``cash_balance``/``cash_balance_native``/``cash_currency``)에서 합성.
       - native 잔액이 있으면 그것을(호주=AUD), 없으면 ``cash_balance``(KRW 계좌는 이게 native).
    """
    raw = account_doc.get("cash")
    if isinstance(raw, dict) and raw:
        result: dict[str, float] = {}
        for key, value in raw.items():
            code = str(key or "").strip().upper()
            if not code:
                continue
            try:
                result[code] = float(value or 0.0)
            except (TypeError, ValueError):
                result[code] = 0.0
        if result:
            return result

    # 레거시 합성
    cash_currency = str(account_doc.get("cash_currency") or settings_currency or "KRW").strip().upper() or "KRW"
    native_raw = account_doc.get("cash_balance_native")
    if native_raw not in (None, ""):
        try:
            native = float(native_raw)
        except (TypeError, ValueError):
            native = 0.0
    else:
        try:
            native = float(account_doc.get("cash_balance") or 0.0)
        except (TypeError, ValueError):
            native = 0.0
    return {cash_currency: native}


def cash_total_krw(cash_map: dict[str, float], rates: dict[str, Any]) -> float:
    """통화별 native 현금 맵을 원화 합계로 환산한다. KRW 배수=1.0."""
    total = 0.0
    for code, native in cash_map.items():
        code_up = str(code or "").strip().upper()
        try:
            amount = float(native or 0.0)
        except (TypeError, ValueError):
            amount = 0.0
        if code_up == "KRW":
            total += amount
        else:
            rate = (rates.get(code_up) or {}).get("rate")
            if rate:
                total += amount * float(rate)
    return total
