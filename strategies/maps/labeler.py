"""Shared labeler for signal/backtest to keep messages/states consistent."""
from __future__ import annotations

from typing import Any, Dict, List

from .messages import build_partial_sell_note
from .constants import DECISION_MESSAGES


def compute_net_trade_note(
    *,
    country: str,
    tkr: str,
    data_by_tkr: Dict[str, Any],
    buy_trades_today_map: Dict[str, List[Dict[str, Any]]],
    sell_trades_today_map: Dict[str, List[Dict[str, Any]]],
    prev_holdings_map: Dict[str, float],
    current_decision: str | None = None,
) -> Dict[str, Any]:
    """Compute per-ticker net buy/sell note and state overrides for the day.

    Returns a dict possibly containing keys: state, row4, note.
    """
    trades_buys = buy_trades_today_map.get(tkr, [])
    total_buy_amount = (
        sum(
            float(tr.get("shares", 0.0) or 0.0) * float(tr.get("price", 0.0) or 0.0)
            for tr in trades_buys
        )
        if trades_buys
        else 0.0
    )

    sells = sell_trades_today_map.get(tkr, [])
    total_sold_amount = (
        sum(
            float(tr.get("shares", 0.0) or 0.0) * float(tr.get("price", 0.0) or 0.0) for tr in sells
        )
        if sells
        else 0.0
    )
    d = data_by_tkr.get(tkr) or {}
    current_shares_now = float(d.get("shares", 0.0) or 0.0)

    is_fully_sold = current_shares_now <= 0.0

    # 거래가 전혀 없으면 아무 것도 변경하지 않음 (WAIT/HOLD 등이 SOLD로 바뀌지 않도록)
    if not trades_buys and not sells:
        return {}

    # SOLD override: 당일 매도가 있었고 현재 보유가 0인 경우에만 SOLD 처리
    if is_fully_sold and sells and (current_decision in (None, "WAIT", "HOLD")):
        return {"state": "SOLD", "row4": "SOLD", "note": DECISION_MESSAGES["SOLD"]}

    net_amount = total_buy_amount - total_sold_amount
    if net_amount > 0:
        # net buy: 부분/신규 구분 없이 동일 메시지 사용
        return {"note": DECISION_MESSAGES["NEW_BUY"]}
    if net_amount < 0:
        # net sell, keep HOLD state
        note = build_partial_sell_note(country, abs(net_amount))
        return {"state": "HOLD", "row4": "HOLD", "note": note}

    # net zero: no override
    return {}


__all__ = ["compute_net_trade_note"]
