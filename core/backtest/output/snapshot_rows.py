from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from core.backtest.output.formatters import BUCKET_NAMES, _is_finite_number
from strategies.maps.constants import DECISION_MESSAGES

if TYPE_CHECKING:
    from core.backtest.domain import AccountBacktestResult


DISPLAY_DECISION_MAP: dict[tuple[str, str], str] = {
    ("BUY_TOMORROW", "HOLD"): "BUY",
    ("BUY_REPLACE_TOMORROW", "HOLD"): "BUY_REPLACE",
    ("BUY_REBALANCE_TOMORROW", "HOLD"): "BUY_REBALANCE",
    ("SELL_TOMORROW", "WAIT"): "SOLD",
    ("SELL_REPLACE_TOMORROW", "WAIT"): "SELL_REPLACE",
    ("SELL_REBALANCE_TOMORROW", "HOLD"): "SELL_REBALANCE",
}


class SnapshotBuildState:
    def __init__(self) -> None:
        self.buy_date_map: dict[str, pd.Timestamp | None] = {}
        self.holding_days_map: dict[str, int] = {}
        self.prev_rows_cache: dict[str, pd.Series | None] = {}
        self.prev_decisions_map: dict[str, str] = {}


def create_snapshot_build_state() -> SnapshotBuildState:
    return SnapshotBuildState()


def resolve_display_decision(prev_decision: str, current_decision: str) -> str:
    return DISPLAY_DECISION_MAP.get((prev_decision, current_decision), current_decision)


def _iter_tickers_order(ticker_timeseries: dict[str, pd.DataFrame]) -> list[str]:
    tickers_order: list[str] = []
    if "CASH" in ticker_timeseries:
        tickers_order.append("CASH")

    other_tickers = sorted(
        [str(t) for t in ticker_timeseries.keys() if str(t).upper() != "CASH"],
        key=lambda x: str(x).upper(),
    )
    tickers_order.extend(other_tickers)
    return tickers_order


def build_snapshot_rows(
    *,
    result: AccountBacktestResult,
    target_date: pd.Timestamp,
    total_value: float,
    total_cash: float,
    state: SnapshotBuildState,
    price_overrides: dict[str, float] | None = None,
) -> list[dict[str, Any]]:
    entries: list[tuple[tuple[int, int, float, str], dict[str, Any]]] = []

    for ticker in _iter_tickers_order(result.ticker_timeseries):
        ts = result.ticker_timeseries.get(ticker)
        if ts is None or not isinstance(ts, pd.DataFrame) or target_date not in ts.index:
            continue

        row = ts.loc[target_date]
        ticker_key = str(ticker).upper()
        meta = result.ticker_meta.get(ticker_key, {})

        price = float(row.get("price")) if pd.notna(row.get("price")) else 0.0
        if price_overrides and ticker_key in price_overrides:
            price = float(price_overrides[ticker_key])

        shares = float(row.get("shares")) if pd.notna(row.get("shares")) else 0.0
        avg_cost = float(row.get("avg_cost")) if pd.notna(row.get("avg_cost")) else 0.0
        pv = price * shares

        raw_decision = str(row.get("decision", "")).upper()
        prev_decision = state.prev_decisions_map.get(ticker_key, "")
        display_decision = resolve_display_decision(prev_decision, raw_decision)
        score = row.get("score")
        note = str(row.get("note", "") or "")
        is_pending_tomorrow = raw_decision.endswith("_TOMORROW")
        is_cash = ticker_key == "CASH"

        if is_cash:
            price = 1.0
            shares = pv if pv else 1.0

        prev_row = state.prev_rows_cache.get(ticker_key)
        prev_price = (
            float(prev_row.get("price")) if (prev_row is not None and pd.notna(prev_row.get("price"))) else None
        )
        daily_pct = ((price / prev_price) - 1.0) * 100.0 if prev_price else 0.0

        pv_safe = pv if _is_finite_number(pv) else 0.0
        total_value_safe = total_value if _is_finite_number(total_value) and total_value > 0 else 0.0
        weight = (pv_safe / total_value_safe * 100.0) if total_value_safe > 0 else 0.0

        if ticker_key not in state.buy_date_map:
            state.buy_date_map[ticker_key] = None
            state.holding_days_map[ticker_key] = 0

        if not is_cash:
            if shares > 0:
                if is_pending_tomorrow:
                    state.buy_date_map[ticker_key] = None
                    state.holding_days_map[ticker_key] = 0
                elif state.buy_date_map[ticker_key] is None or display_decision.startswith("BUY"):
                    state.buy_date_map[ticker_key] = target_date
                    state.holding_days_map[ticker_key] = 1
                elif prev_decision.startswith("BUY") and raw_decision == "HOLD":
                    state.holding_days_map[ticker_key] = max(state.holding_days_map[ticker_key], 1)
                else:
                    state.holding_days_map[ticker_key] += 1
            else:
                state.buy_date_map[ticker_key] = None
                state.holding_days_map[ticker_key] = 0
        else:
            state.buy_date_map[ticker_key] = None
            state.holding_days_map[ticker_key] = 0

        display_avg_cost = None
        if not is_cash and not is_pending_tomorrow and _is_finite_number(avg_cost) and avg_cost > 0:
            display_avg_cost = avg_cost

        cost_basis = display_avg_cost * shares if display_avg_cost is not None and shares > 0 else 0.0
        evaluation_profit = 0.0 if is_cash or is_pending_tomorrow else (pv - cost_basis)
        evaluation_pct = ((evaluation_profit / cost_basis) * 100.0) if cost_basis > 0 else None

        score_val = float(score) if _is_finite_number(score) else float("-inf")
        bucket_id = meta.get("bucket")
        bucket_display = "-"
        if bucket_id and bucket_id in BUCKET_NAMES:
            bucket_display = f"{bucket_id}. {BUCKET_NAMES[bucket_id]}"
        elif bucket_id:
            bucket_display = str(bucket_id)

        name = str(meta.get("name") or ticker_key)
        if is_cash:
            name = "현금"

        if is_cash and total_value_safe > 0:
            cash_ratio = (total_cash / total_value_safe) if _is_finite_number(total_cash) else 0.0
            weight = cash_ratio * 100.0

        message = note or DECISION_MESSAGES.get(display_decision, "")

        sort_group = 2
        if is_cash:
            sort_group = 0
        elif display_decision in {
            "HOLD",
            "BUY",
            "BUY_REBALANCE",
            "BUY_REPLACE",
            "BUY_TODAY",
            "BUY_TOMORROW",
            "SELL_TOMORROW",
            "BUY_REPLACE_TOMORROW",
            "SELL_REPLACE_TOMORROW",
            "BUY_REBALANCE_TOMORROW",
            "SELL_REBALANCE_TOMORROW",
        }:
            sort_group = 1

        snapshot_row = {
            "ticker": ticker_key,
            "bucket": bucket_id,
            "bucket_display": bucket_display,
            "name": name,
            "display_decision": display_decision or "-",
            "raw_decision": raw_decision or "-",
            "holding_days": state.holding_days_map.get(ticker_key, 0),
            "price": price,
            "avg_cost": display_avg_cost,
            "daily_pct": daily_pct,
            "evaluation_pct": evaluation_pct,
            "shares": shares,
            "pv": pv,
            "evaluation_profit": evaluation_profit if display_avg_cost is not None else None,
            "weight": weight,
            "score": float(score) if _is_finite_number(score) else None,
            "message": message,
            "is_cash": is_cash,
            "is_pending_tomorrow": is_pending_tomorrow,
            "sort_group": sort_group,
        }
        bucket_sort_val = int(bucket_id) if (bucket_id and str(bucket_id).isdigit()) else 99
        sort_key = (sort_group, bucket_sort_val if sort_group == 1 else 0, -score_val, ticker_key)
        entries.append((sort_key, snapshot_row))
        state.prev_decisions_map[ticker_key] = raw_decision

    entries.sort(key=lambda item: item[0])

    sorted_rows: list[dict[str, Any]] = []
    current_idx = 1
    for _, snapshot_row in entries:
        if snapshot_row["ticker"] == "CASH":
            snapshot_row["row_index"] = "0"
            snapshot_row["bucket_display"] = "-"
        else:
            snapshot_row["row_index"] = str(current_idx)
            current_idx += 1
        sorted_rows.append(snapshot_row)

    return sorted_rows


def advance_snapshot_state(
    *,
    result: AccountBacktestResult,
    target_date: pd.Timestamp,
    state: SnapshotBuildState,
) -> None:
    for ticker, ts in result.ticker_timeseries.items():
        if isinstance(ts, pd.DataFrame) and target_date in ts.index:
            state.prev_rows_cache[str(ticker).upper()] = ts.loc[target_date].copy()
