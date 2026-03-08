"""Message helpers to keep signals and backtests consistent."""

from __future__ import annotations

from utils.report import format_money

from .constants import BACKTEST_STATUS_LIST, DECISION_MESSAGES, DECISION_NOTES


def money_str(country: str, amount: float) -> str:
    """Return formatted money string by country stock currency."""
    try:
        val = float(amount or 0.0)
    except Exception:
        val = 0.0
    return format_money(val, country)


def _normalize_display_name(name: str) -> str:
    # Remove optional leading/trailing angle brackets like "<...>"
    name = str(name or "").strip()
    if name.startswith("<") and name.endswith(">"):
        name = name[1:-1].strip()
    return name


def build_buy_replace_note(sell_name: str, sell_ticker: str) -> str:
    """Build note for BUY_REPLACE: "🔄 교체매수 - 종목명(티커) 대체" """
    raw = BACKTEST_STATUS_LIST["BUY_REPLACE"]["display_name"]
    disp = _normalize_display_name(raw)
    return f"{disp} - {sell_name}({sell_ticker}) 대체"


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
    profit_label = "수익" if profit_amount >= 0 else "손실"
    return f"교체매도 {amt} {profit_label} {prof} 손익률 {sign_pct} ({replacement_ticker}(으)로 교체)"


def build_partial_sell_note() -> str:
    """Build note for partial sell with amount."""
    tmpl = DECISION_MESSAGES["SELL"]
    return tmpl


def build_simple_sell_replace_note() -> str:
    """Build note for simple sell replace with no amount."""
    return DECISION_NOTES.get("REPLACE_SELL", "교체 매도")


__all__ = [
    "money_str",
    "build_buy_replace_note",
    "build_partial_sell_note",
    "build_sell_replace_note",
    "build_simple_sell_replace_note",
]
