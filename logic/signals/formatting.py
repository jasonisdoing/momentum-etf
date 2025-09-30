"""Formatting and precision helpers extracted from signals.py.
Prefer using utils.report/utils.notification where possible.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Any

from utils.report import format_kr_money


def _load_display_precision() -> Dict[str, int]:
    """Read percent display precision from data/settings/precision.json (common section)."""
    try:
        root = Path(__file__).resolve().parent.parent.parent  # project root
        cfg_path = root / "data" / "settings" / "precision.json"
        if not cfg_path.exists():
            return {
                "daily_return_pct": 2,
                "cum_return_pct": 2,
                "weight_pct": 2,
            }
        import json

        with open(cfg_path, "r", encoding="utf-8") as fp:
            data = json.load(fp) or {}
        prec = data.get("common") or {}
        return {
            "daily_return_pct": int(prec.get("daily_return_pct", 2)),
            "cum_return_pct": int(prec.get("cum_return_pct", 2)),
            "weight_pct": int(prec.get("weight_pct", 2)),
        }
    except Exception:
        return {
            "daily_return_pct": 2,
            "cum_return_pct": 2,
            "weight_pct": 2,
        }


def _load_precision_all() -> Dict[str, Any]:
    """Load full precision.json. Raise if missing or invalid."""
    root = Path(__file__).resolve().parent.parent.parent
    cfg_path = root / "data" / "settings" / "precision.json"
    import json

    if not cfg_path.exists():
        raise FileNotFoundError(f"precision.json not found at {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
        if not isinstance(data, dict):
            raise ValueError("precision.json must be a JSON object at root")
        return data


def _load_country_precision(country: str) -> Dict[str, Any]:
    """Load country section precision/currency config. Raise if required keys missing."""
    data = _load_precision_all()
    countries = data.get("country")
    if not isinstance(countries, dict):
        raise KeyError("precision.json missing 'country' section")
    c = countries.get(country)
    if not isinstance(c, dict):
        raise KeyError(f"precision.json missing country section for '{country}'")

    required_keys = [
        "header_currency",
        "stock_currency",
        "stock_qty_precision",
        "stock_price_precision",
        "stock_amt_precision",
    ]
    missing = [k for k in required_keys if k not in c]
    if missing:
        raise KeyError(f"precision.json country '{country}' missing keys: {', '.join(missing)}")
    return c


def _get_header_money_formatter(country: str):
    """Return money formatter for header based on header_currency in precision.json.
    Strict: raise if required sections missing.
    - If header_currency == 'KRW', keep format_kr_money
    - Else format with thousand separator and precision from currency section
    """
    all_prec = _load_precision_all()
    countries = all_prec.get("country")
    currencies = all_prec.get("currency")
    if not isinstance(countries, dict):
        raise KeyError("precision.json missing 'country' section")
    if not isinstance(currencies, dict):
        raise KeyError("precision.json missing 'currency' section")

    cprec = countries.get(country)
    if not isinstance(cprec, dict):
        raise KeyError(f"precision.json missing country section for '{country}'")

    header_ccy = str(cprec.get("header_currency"))
    if not header_ccy:
        raise KeyError(f"precision.json country '{country}' missing 'header_currency'")
    if header_ccy == "KRW":
        return format_kr_money

    cur = currencies.get(header_ccy)
    if not isinstance(cur, dict) or "precision" not in cur:
        raise KeyError(f"precision.json currency '{header_ccy}' missing 'precision'")
    prec = int(cur["precision"])

    def _fmt_safe(val: float) -> str:
        return f"{float(val):,.{prec}f} {header_ccy}"

    return _fmt_safe


def format_shares(quantity, country: str):
    if not isinstance(quantity, (int, float)):
        return str(quantity)
    # precision.json 기반 수량 정밀도 적용 (모든 국가 공통)
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
