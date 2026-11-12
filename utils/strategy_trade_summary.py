"""사고팔기 전략의 계좌별 손익·운용 요약 — 총 자산 요약 슬랙이 쓴다.

회차 원장은 배치가 저장해 둔 것을 읽는다(증권사를 부르지 않는다). 금액은 원화로 환산해 계좌별로 합친다.
"""

from __future__ import annotations

from typing import Any

from utils.strategy_trade_config import ROUND_SIZE_UNIT_KRW
from utils.strategy_trade_market import krw_rate
from utils.strategy_trade_orders import active_order_count
from utils.strategy_trade_service import load_views_where


def _lots(view: dict[str, Any]) -> list[tuple[int, float]]:
    """보유 로트(회차 + 임의 매수) — (수량, 체결가)."""
    regular = [(row["quantity"], row["fill_price"]) for row in view["rounds"] if row["held"]]
    arbitrary = [(lot["quantity"], lot["fill_price"]) for lot in view["arbitrary_lots"]]
    return regular + arbitrary


def _add_view(summary: dict[str, Any], view: dict[str, Any]) -> None:
    rate = krw_rate(view["country"])
    close = float(view["index"]["close"])
    status = view["status"]
    for quantity, fill_price in _lots(view):
        summary["invested"] += quantity * fill_price * rate
        summary["unrealized"] += quantity * (close - fill_price) * rate
    config = view["config"]
    summary["max_capital"] += config["round_size"] * ROUND_SIZE_UNIT_KRW * config["rounds"]
    summary["held_tickers"] += 1 if status["held_quantity"] > 0 else 0
    summary["held_rounds"] += status["held_count"]
    summary["realized_today"] += status["realized_today"] * rate
    summary["realized_recent"] += status["realized_recent"] * rate
    summary["realized_unknown_sells"] += status["realized_unknown_sells"]
    summary["today_buys"] += status["today_buy_count"]
    summary["today_sells"] += status["today_sell_count"]
    summary["manual_orders"] += len(view["manual_orders"])


def load_trade_summaries() -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """→ (계좌 ID → 요약, 전략 ID → 현황을 못 만든 이유). 전략이 없는 계좌는 요약에 없다."""
    loaded = load_views_where(lambda entry: True, fresh=False)
    summaries: dict[str, dict[str, Any]] = {}
    for view in loaded["strategies"]:
        summary = summaries.setdefault(
            view["account_id"],
            {
                "account_name": view["account_name"],
                "strategies": 0,
                "invested": 0.0,
                "max_capital": 0.0,
                "unrealized": 0.0,
                "held_tickers": 0,
                "held_rounds": 0,
                "realized_today": 0.0,
                "realized_recent": 0.0,
                "realized_unknown_sells": 0,
                "today_buys": 0,
                "today_sells": 0,
                "manual_orders": 0,
            },
        )
        summary["strategies"] += 1
        _add_view(summary, view)
    for account_id, summary in summaries.items():
        summary["signal_orders"] = active_order_count(account_id)
    return summaries, loaded["errors"]
