"""Formatting and precision helpers extracted from signals.py.
Prefer using utils.report/utils.notification where possible.
"""
from __future__ import annotations
from typing import Dict, Any, Callable

from utils.report import format_kr_money


def _load_display_precision() -> Dict[str, int]:
    """Return default display precision values."""
    return {
        "daily_return_pct": 2,
        "cum_return_pct": 2,
        "weight_pct": 2,
    }


def _load_precision_all() -> Dict[str, Any]:
    """Load precision settings from country configuration.

    This is kept for backward compatibility but now returns an empty dict
    since we're using country-specific configuration files.
    """
    return {}


def _load_country_precision(country: str) -> Dict[str, Any]:
    """Load country section precision/currency config from country settings."""
    from utils.account_registry import get_country_settings

    # Load country settings
    country_settings = get_country_settings(country)
    if not country_settings:
        raise KeyError(f"Country settings not found for '{country}'")

    # Get precision settings from country config
    precision_settings = country_settings.get("precision", {})
    if not precision_settings:
        raise KeyError(f"Precision settings not found in {country} configuration")

    # Default values
    return {
        "header_currency": precision_settings.get("currency", "KRW"),
        "stock_currency": precision_settings.get("currency", "KRW"),
        "stock_qty_precision": int(precision_settings.get("qty_precision", 0)),
        "stock_price_precision": int(precision_settings.get("price_precision", 0)),
        "stock_amt_precision": 0,  # Default to 0 if not specified
    }


def _get_header_money_formatter(country: str) -> Callable[[float], str]:
    """Return money formatter for header based on country configuration.

    Args:
        country: Country code (e.g., 'kor', 'aus')

    Returns:
        A function that formats a float value as a currency string
    """
    try:
        cprec = _load_country_precision(country)
        currency = cprec.get("header_currency", "KRW")

        if currency == "KRW":
            return format_kr_money

        # For other currencies, use a simple formatter with 2 decimal places
        def _fmt_safe(val: float) -> str:
            try:
                return f"{val:,.2f}"
            except (ValueError, TypeError):
                return str(val)

        return _fmt_safe

    except Exception as e:
        # Fallback to KRW formatter if there's any error
        return format_kr_money


def format_shares(quantity, country: str) -> str:
    """Format share quantity based on country's precision settings.

    Args:
        quantity: The quantity to format
        country: Country code (e.g., 'kor', 'aus')

    Returns:
        Formatted quantity string
    """
    if not isinstance(quantity, (int, float)):
        return str(quantity)

    cprec = _load_country_precision(country)
    qty_precision = int(cprec.get("stock_qty_precision", 0))

    if qty_precision > 0:
        return f"{quantity:,.{qty_precision}f}".rstrip("0").rstrip(".")
    return f"{int(round(quantity)):,d}"


__all__ = [
    "_load_display_precision",
    "_load_precision_all",
    "_load_country_precision",
    "_get_header_money_formatter",
    "format_shares",
]
