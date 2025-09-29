"""Message helpers to keep signals and backtests consistent."""

from __future__ import annotations

from utils.report import format_kr_money, format_aud_money
from .constants import DECISION_CONFIG, DECISION_MESSAGES


def money_str(country: str, amount: float) -> str:
    """Return formatted money string by country stock currency.
    - aus -> AUD (A$)
    - others -> KRW
    """
    try:
        val = float(amount or 0.0)
    except Exception:
        val = 0.0
    if country == "aus":
        return format_aud_money(val)
    return format_kr_money(val)


def _normalize_display_name(name: str) -> str:
    # Remove optional leading/trailing angle brackets like "<...>"
    name = str(name or "").strip()
    if name.startswith("<") and name.endswith(">"):
        name = name[1:-1].strip()
    return name


def build_buy_replace_note(country: str, amount: float, ticker_to_sell: str) -> str:
    """Build note for BUY_REPLACE: "🔄 교체매수 금액 (티커 대체)" (no angle brackets)"""
    raw = DECISION_CONFIG["BUY_REPLACE"]["display_name"]
    disp = _normalize_display_name(raw)
    return f"{disp} {money_str(country, amount)} ({ticker_to_sell} 대체)"


def build_sell_replace_note(
    country: str,
    trade_amount: float,
    profit_amount: float,
    pl_pct: float,
    replacement_ticker: str,
) -> str:
    """Build note for SELL_REPLACE using amount, not shares@price.
    Example: "교체매도 2,552만원 수익 45만원 손익률 +1.8% (419430(으)로 교체)"
    """
    amt = money_str(country, trade_amount)
    prof = money_str(country, profit_amount)
    sign_pct = f"{pl_pct:+.1f}%"
    return f"교체매도 {amt} 수익 {prof} 손익률 {sign_pct} ({replacement_ticker}(으)로 교체)"


def build_partial_buy_note(country: str, amount: float) -> str:
    """Build note for partial buy with amount."""
    tmpl = DECISION_MESSAGES["PARTIAL_BUY"]
    return tmpl.format(amount=money_str(country, amount))


def build_partial_sell_note(country: str, amount: float) -> str:
    """Build note for partial sell with amount."""
    tmpl = DECISION_MESSAGES["PARTIAL_SELL"]
    return tmpl.format(amount=money_str(country, amount))


__all__ = [
    "money_str",
    "build_buy_replace_note",
    "build_partial_buy_note",
    "build_partial_sell_note",
    "build_sell_replace_note",
]
