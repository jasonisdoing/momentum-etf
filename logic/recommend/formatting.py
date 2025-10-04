"""신호 모듈에서 분리한 표시·정밀도 보조 함수 모음."""
from __future__ import annotations
from typing import Dict, Any, Callable

from utils.report import format_kr_money


def _load_display_precision() -> Dict[str, int]:
    """표시에 사용할 기본 정밀도 설정을 반환한다."""
    return {
        "daily_return_pct": 2,
        "cum_return_pct": 2,
        "weight_pct": 2,
    }


def load_account_precision(country: str) -> Dict[str, Any]:
    """계정 설정에서 통화·정밀도 정보를 불러온다."""
    from utils.account_registry import get_account_settings

    # 계정 설정을 불러온다.
    account_settings = get_account_settings(country)
    if not account_settings:
        raise KeyError(f"Country settings not found for '{country}'")

    # 정밀도 세부 설정을 추출한다.
    precision_settings = account_settings.get("precision", {})
    if not precision_settings:
        raise KeyError(f"Precision settings not found in {country} configuration")

    # 기본값을 구성한다.
    return {
        "header_currency": precision_settings.get("currency", "KRW"),
        "stock_currency": precision_settings.get("currency", "KRW"),
        "stock_qty_precision": int(precision_settings.get("qty_precision", 0)),
        "stock_price_precision": int(precision_settings.get("price_precision", 0)),
        "stock_amt_precision": 0,  # 명시되지 않으면 0으로 간주
    }


def get_header_money_formatter(country: str) -> Callable[[float], str]:
    """헤더 표시용 통화 서식을 반환한다."""
    try:
        account_precision = load_account_precision(country)
        currency = account_precision.get("header_currency", "KRW")

        if currency == "KRW":
            return format_kr_money

        # 다른 통화는 소수 둘째 자리까지 표기한다.
        def _fmt_safe(val: float) -> str:
            try:
                return f"{val:,.2f}"
            except (ValueError, TypeError):
                return str(val)

        return _fmt_safe

    except Exception:
        # 오류가 발생하면 원화 서식을 사용한다.
        return format_kr_money


def format_shares(quantity, country: str) -> str:
    """국가별 정밀도 설정에 맞춰 보유 수량을 문자열로 변환한다."""
    if not isinstance(quantity, (int, float)):
        return str(quantity)

    account_precision = load_account_precision(country)
    qty_precision = int(account_precision.get("stock_qty_precision", 0))

    if qty_precision > 0:
        return f"{quantity:,.{qty_precision}f}".rstrip("0").rstrip(".")
    return f"{int(round(quantity)):,d}"


__all__ = [
    "_load_display_precision",
    "load_account_precision",
    "get_header_money_formatter",
    "format_shares",
]
