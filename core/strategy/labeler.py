"""Shared labeler for signal/backtest to keep messages/states consistent."""

from __future__ import annotations

from typing import Any

from .constants import DECISION_MESSAGES, PENDING_ACTION_MESSAGES
from .messages import build_partial_sell_note


def compute_net_trade_note(
    *,
    tkr: str,
    data_by_tkr: dict[str, Any],
    buy_trades_today_map: dict[str, list[dict[str, Any]]],
    sell_trades_today_map: dict[str, list[dict[str, Any]]],
    current_decision: str | None = None,
    current_pending_action: str | None = None,
) -> dict[str, Any]:
    """Compute per-ticker net buy/sell note and state overrides for the day.

    Returns a dict possibly containing keys: state, row4, note.
    """
    trades_buys = buy_trades_today_map.get(tkr, [])
    total_buy_amount = (
        sum(float(tr.get("shares", 0.0) or 0.0) * float(tr.get("price", 0.0) or 0.0) for tr in trades_buys)
        if trades_buys
        else 0.0
    )

    sells = sell_trades_today_map.get(tkr, [])
    total_sold_amount = (
        sum(float(tr.get("shares", 0.0) or 0.0) * float(tr.get("price", 0.0) or 0.0) for tr in sells) if sells else 0.0
    )
    d = data_by_tkr.get(tkr) or {}
    current_shares_now = float(d.get("shares", 0.0) or 0.0)

    is_fully_sold = current_shares_now <= 0.0

    # 거래가 전혀 없으면 아무 것도 변경하지 않음 (WAIT/HOLD 등이 SOLD로 바뀌지 않도록)
    if not trades_buys and not sells:
        return {}

    pending_action_norm = str(current_pending_action or "").upper()

    # 전량 매도 체결일 표시도 SELL로 통일
    if (
        is_fully_sold
        and sells
        and not pending_action_norm.startswith("SELL")
        and (current_decision in (None, "WAIT", "HOLD"))
    ):
        return {"state": "SELL", "row4": "SELL", "note": DECISION_MESSAGES["SELL"]}

    net_amount = total_buy_amount - total_sold_amount

    if net_amount > 0:
        if pending_action_norm.startswith("BUY"):
            pending_note = PENDING_ACTION_MESSAGES.get(pending_action_norm)
            return {"note": pending_note} if pending_note else {}
        return {"note": DECISION_MESSAGES["NEW_BUY"]}
    if net_amount < 0:
        current_decision_norm = str(current_decision or "").upper()
        if pending_action_norm.startswith("SELL"):
            pending_note = PENDING_ACTION_MESSAGES.get(pending_action_norm)
            return {"note": pending_note} if pending_note else {}
        if current_decision_norm == "SELL":
            return {"note": DECISION_MESSAGES["SELL"]}
        note = build_partial_sell_note()
        return {"state": "HOLD", "row4": "HOLD", "note": note}

    # net zero: no override
    return {}


__all__ = ["compute_net_trade_note"]
