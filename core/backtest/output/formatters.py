import math
import numbers

import pandas as pd

from utils.report import format_kr_money
from utils.settings_loader import get_account_precision

BUCKET_NAMES = {
    1: "모멘텀",
    2: "혁신기술",
    3: "시장지수",
    4: "배당방어",
    5: "대체헷지",
}


def _is_finite_number(value: any) -> bool:
    if not isinstance(value, numbers.Number):
        return False
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric)


def _usd_money(value: float) -> str:
    if not _is_finite_number(value):
        return "-"
    is_negative = value < 0
    abs_val = abs(value)
    if abs_val < 0.01 and abs_val != 0:
        formatted_val = f"{abs_val:,.4f}"
    else:
        formatted_val = f"{abs_val:,.2f}"
    if is_negative:
        return f"-${formatted_val}"
    return f"${formatted_val}"


def _format_date_kor(ts: pd.Timestamp) -> str:
    ts = pd.to_datetime(ts)
    weekday_map = {0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토", 6: "일"}
    weekday = weekday_map.get(ts.weekday(), "")
    return f"{ts.strftime('%Y-%m-%d')}({weekday})"


def _format_quantity(amount: float, precision: int) -> str:
    if not _is_finite_number(amount):
        return "-"
    if precision <= 0:
        return f"{int(round(amount)):,}"
    return f"{amount:,.{precision}f}".rstrip("0").rstrip(".")


def _resolve_formatters(account_settings: dict[str, any], account_id: str = ""):
    if not account_id:
        account_id = str(account_settings.get("account") or "").strip().lower()
    try:
        precision = get_account_precision(account_id)
    except Exception:
        precision = {}
    if not isinstance(precision, dict):
        precision = {}
    currency = str(precision.get("currency", "KRW")).strip().upper()
    qty_precision = int(precision.get("qty_precision", 0))
    price_precision = int(precision.get("price_precision", 0))
    digits = max(price_precision, 0)

    def _format_price(value: float) -> str:
        if not _is_finite_number(value):
            return "-"
        return f"{float(value):,.{digits}f}"

    return currency, _usd_money if currency == "USD" else format_kr_money, _format_price, qty_precision, digits
