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
    """Load full precision.json."""
    try:
        root = Path(__file__).resolve().parent.parent.parent
        cfg_path = root / "data" / "settings" / "precision.json"
        import json

        with open(cfg_path, "r", encoding="utf-8") as fp:
            return json.load(fp) or {}
    except Exception:
        return {}


def _load_country_precision(country: str) -> Dict[str, Any]:
    """Load country section precision/currency config."""
    data = _load_precision_all()
    c = (data.get("country") or {}).get(country)
    if not isinstance(c, dict):
        return {}
    return c


def _get_header_money_formatter(country: str):
    """Return money formatter for header based on header_currency in precision.json.
    - If header_currency == 'KRW', keep format_kr_money
    - Else format with thousand separator and precision from currency section
    """
    try:
        all_prec = _load_precision_all()
        cprec = (
            (all_prec.get("country") or {}).get(country, {}) if isinstance(all_prec, dict) else {}
        )
        curmap = (all_prec.get("currency") or {}) if isinstance(all_prec, dict) else {}
        header_ccy = str(cprec.get("header_currency", "KRW")) if isinstance(cprec, dict) else "KRW"
        if header_ccy == "KRW":
            return format_kr_money
        prec = int(((curmap.get(header_ccy) or {}).get("precision", 0)))

        def _fmt_safe(val: float) -> str:
            try:
                return f"{float(val):,.{prec}f} {header_ccy}"
            except Exception:
                return f"{val} {header_ccy}"

        return _fmt_safe
    except Exception:
        return format_kr_money


def format_shares(quantity, country: str):
    if not isinstance(quantity, (int, float)):
        return str(quantity)
    if country == "coin":
        return f"{quantity:,.8f}".rstrip("0").rstrip(".")
    return f"{quantity:,.0f}"


__all__ = [
    "_load_display_precision",
    "_load_precision_all",
    "_load_country_precision",
    "_get_header_money_formatter",
    "format_shares",
]
