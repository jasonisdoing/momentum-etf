from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

import pandas as pd

from core.backtest.output.formatters import BUCKET_NAMES, _is_finite_number
from core.strategy.constants import DECISION_MESSAGES
from utils.data_loader import get_trading_days

if TYPE_CHECKING:
    from core.backtest.domain import AccountBacktestResult


class SnapshotBuildState:
    def __init__(self) -> None:
        self.buy_date_map: dict[str, pd.Timestamp | None] = {}
        self.holding_days_map: dict[str, int] = {}
        self.prev_rows_cache: dict[str, pd.Series | None] = {}
        self.prev_pending_actions_map: dict[str, str] = {}
        self.prev_pending_reasons_map: dict[str, str] = {}
        self.prev_effective_shares_map: dict[str, float] = {}
        self.prev_effective_avg_cost_map: dict[str, float] = {}
        self.prev_current_weights_map: dict[str, float] = {}
        self.prev_target_weights_map: dict[str, float | None] = {}


def create_snapshot_build_state() -> SnapshotBuildState:
    return SnapshotBuildState()


def _display_from_pending_action(pending_action: str) -> str | None:
    pending_norm = str(pending_action or "").upper()
    if pending_norm == "SELL_REBALANCE":
        return "HOLD"
    return None


def resolve_display_decision(
    prev_pending_action: str,
    prev_pending_reason: str,
    current_decision: str,
    current_pending_action: str,
) -> str:
    prev_pending_norm = str(prev_pending_action or "").upper()
    prev_pending_reason_norm = str(prev_pending_reason or "").strip()
    curr_norm = str(current_decision or "").upper()
    signal_decision = _display_from_pending_action(current_pending_action)
    if prev_pending_norm == "BUY_REPLACE" and curr_norm == "HOLD":
        return "BUY_REPLACE"
    if prev_pending_norm == "BUY" and curr_norm == "HOLD" and prev_pending_reason_norm != "비중 조정":
        return "BUY"
    if prev_pending_norm == "SELL_REPLACE" and curr_norm == "WAIT":
        return "SELL_REPLACE"
    if prev_pending_norm == "SELL" and curr_norm == "WAIT":
        return "SELL"
    return signal_decision or curr_norm


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


def _normalize_note_text(note: str) -> str:
    text = str(note or "").strip()
    if not text:
        return ""
    return re.sub(r"(\d+)일 후", r"\1거래일 후", text)


def _count_trading_days_until(
    portfolio_index: pd.Index,
    target_date: pd.Timestamp,
    execute_on: Any,
) -> int:
    if execute_on is None:
        return 0
    try:
        execute_ts = pd.Timestamp(execute_on)
    except Exception:
        return 0
    if pd.isna(execute_ts):
        return 0

    current_norm = pd.Timestamp(target_date).normalize()
    execute_norm = execute_ts.normalize()
    if execute_norm <= current_norm:
        return 0

    trading_days = [pd.Timestamp(day).normalize() for day in portfolio_index]
    return sum(1 for day in trading_days if current_norm < day <= execute_norm)


def _resolve_rebalance_mode(result: AccountBacktestResult) -> str:
    settings_snapshot = getattr(result, "settings_snapshot", {}) or {}
    if isinstance(settings_snapshot, dict):
        strategy = settings_snapshot.get("strategy") or {}
        if isinstance(strategy, dict):
            mode = strategy.get("rebalance_mode") or strategy.get("REBALANCE_MODE")
            if mode:
                return str(mode).upper()
    mode = getattr(result, "rebalance_mode", None)
    if mode:
        return str(mode).upper()
    return "TWICE_A_MONTH"


def _estimate_next_rebalance_days(result: AccountBacktestResult, target_date: pd.Timestamp) -> int:
    rebalance_mode = _resolve_rebalance_mode(result)
    if rebalance_mode == "DAILY":
        return 1

    try:
        from core.backtest.engine import check_is_rebalance_day
    except Exception:
        return 0

    current_date = pd.Timestamp(target_date).normalize()
    country_code = str(getattr(result, "country_code", "") or "kor").strip().lower()
    search_end = current_date + pd.DateOffset(days=120)

    try:
        trading_calendar = get_trading_days(
            current_date.strftime("%Y-%m-%d"),
            search_end.strftime("%Y-%m-%d"),
            country_code,
        )
    except Exception:
        return 0

    calendar_idx = pd.DatetimeIndex(trading_calendar)
    if calendar_idx.empty:
        return 0

    for idx, day in enumerate(calendar_idx):
        if day.normalize() <= current_date:
            continue
        next_day = calendar_idx[idx + 1] if idx + 1 < len(calendar_idx) else None
        if check_is_rebalance_day(day, next_day, rebalance_mode, trading_calendar=calendar_idx):
            return idx
    return 0


def _extract_replacement_target(reason: str, note: str) -> str:
    for source in (reason, note):
        text = str(source or "").strip()
        if not text:
            continue
        if "교체 -" in text:
            return text.split("교체 -", 1)[1].strip()
    return ""


def _format_weight_change(current_weight: float, target_weight: float | None) -> str:
    current_display = f"{float(current_weight):.1f}%"
    if target_weight is None or not _is_finite_number(target_weight):
        return current_display
    return f"{current_display} => {float(target_weight):.1f}%"


def _format_pending_message(
    *,
    pending_action: str,
    pending_reason: str,
    note: str,
    current_weight: float,
    target_weight: float | None,
    target_name: str,
    days_until_execution: int,
) -> str:
    days_text = f"{max(int(days_until_execution), 1)}거래일 후 "

    if pending_reason == "비중 조정":
        return f"[예정] {days_text}비중조절 - {_format_weight_change(current_weight, target_weight)}"

    if pending_action == "BUY" and pending_reason == "신규 매수":
        return f"[예정] {days_text}신규매수 - {_format_weight_change(current_weight, target_weight)}"

    if pending_action == "BUY_REPLACE":
        source_name = str(pending_reason or "").replace(" 대체", "").strip()
        if source_name:
            return f"[예정] {days_text}교체편입 - {source_name} -> {target_name}"
        return f"[예정] {days_text}교체편입 - {target_name}"

    if pending_action == "SELL_REPLACE":
        replacement_target = _extract_replacement_target(pending_reason, note)
        if replacement_target:
            return f"[예정] {days_text}교체매도 - {target_name} -> {replacement_target}"
        return f"[예정] {days_text}교체매도 - {target_name}"

    return _normalize_note_text(note)


def _format_scheduled_message(
    *,
    current_weight: float,
    target_weight: float | None,
    shares: float,
    days_until_rebalance: int,
) -> str:
    if target_weight is None or not _is_finite_number(target_weight):
        return ""
    days_text = f"{max(int(days_until_rebalance), 1)}거래일 후 "
    if shares <= 0:
        return f"[예정] {days_text}신규매수 - {_format_weight_change(current_weight, target_weight)}"
    return f"[예정] {days_text}비중조절 - {_format_weight_change(current_weight, target_weight)}"


def _format_executed_message(
    *,
    prev_pending_action: str,
    prev_pending_reason: str,
    prev_current_weight: float | None,
    prev_target_weight: float | None,
    target_name: str,
) -> str:
    current_weight = float(prev_current_weight) if prev_current_weight is not None else 0.0

    if prev_pending_reason == "비중 조정":
        return f"비중조절 - {_format_weight_change(current_weight, prev_target_weight)}"

    if prev_pending_action == "BUY" and prev_pending_reason == "신규 매수":
        return f"신규매수 - {_format_weight_change(current_weight, prev_target_weight)}"

    if prev_pending_action == "BUY_REPLACE":
        source_name = str(prev_pending_reason or "").replace(" 대체", "").strip()
        if source_name:
            return f"교체편입 - {source_name} -> {target_name}"
        return f"교체편입 - {target_name}"

    return ""


def build_snapshot_rows(
    *,
    result: AccountBacktestResult,
    target_date: pd.Timestamp,
    total_value: float,
    total_cash: float,
    state: SnapshotBuildState,
    price_overrides: dict[str, float] | None = None,
    bucket_topn: int | None = None,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []

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

        raw_shares = float(row.get("shares")) if pd.notna(row.get("shares")) else 0.0
        avg_cost = float(row.get("avg_cost")) if pd.notna(row.get("avg_cost")) else 0.0
        traded_shares = float(row.get("trade_shares")) if pd.notna(row.get("trade_shares")) else 0.0

        raw_decision = str(row.get("decision", "")).upper()
        pending_action = str(row.get("pending_action", "") or "").upper()
        prev_pending_action = state.prev_pending_actions_map.get(ticker_key, "")
        prev_pending_reason = state.prev_pending_reasons_map.get(ticker_key, "")
        pending_reason = str(row.get("pending_reason", "") or "")
        display_decision = resolve_display_decision(
            prev_pending_action,
            prev_pending_reason,
            raw_decision,
            pending_action,
        )
        score = row.get("score")
        target_weight = row.get("target_weight")
        note = str(row.get("note", "") or "")
        is_pending_tomorrow = bool(pending_action)
        is_cash = ticker_key == "CASH"

        if is_pending_tomorrow and not is_cash:
            if pending_action.startswith("BUY"):
                reconstructed_shares = max(0.0, raw_shares - max(0.0, traded_shares))
                shares = reconstructed_shares
                if shares > 0:
                    avg_cost = state.prev_effective_avg_cost_map.get(ticker_key, avg_cost)
                else:
                    avg_cost = 0.0
            elif pending_action.startswith("SELL"):
                reconstructed_shares = raw_shares + max(0.0, traded_shares)
                shares = (
                    reconstructed_shares
                    if reconstructed_shares > 0
                    else state.prev_effective_shares_map.get(ticker_key, raw_shares)
                )
                avg_cost = state.prev_effective_avg_cost_map.get(ticker_key, avg_cost)
            else:
                shares = raw_shares
        else:
            shares = raw_shares
        pv = price * shares

        if is_cash:
            cash_pv = float(total_cash) if _is_finite_number(total_cash) else 0.0
            if not _is_finite_number(cash_pv) or cash_pv <= 0:
                cash_pv = float(row.get("pv")) if pd.notna(row.get("pv")) else 0.0
            price = 1.0
            shares = cash_pv if _is_finite_number(cash_pv) and cash_pv > 0 else 1.0
            pv = cash_pv if _is_finite_number(cash_pv) else 0.0

        prev_row = state.prev_rows_cache.get(ticker_key)
        prev_price = (
            float(prev_row.get("price")) if (prev_row is not None and pd.notna(prev_row.get("price"))) else None
        )
        daily_pct = ((price / prev_price) - 1.0) * 100.0 if prev_price else 0.0

        pv_safe = pv if _is_finite_number(pv) else 0.0
        total_value_safe = total_value if _is_finite_number(total_value) and total_value > 0 else 0.0
        current_weight = (pv_safe / total_value_safe * 100.0) if total_value_safe > 0 else 0.0

        if ticker_key not in state.buy_date_map:
            state.buy_date_map[ticker_key] = None
            state.holding_days_map[ticker_key] = 0

        # 보유일은 상태/문구가 아니라 실제 보유수량(전일 대비) 기준으로 계산한다.
        # 이렇게 해야 평가(평균매입가 기반)와 보유일 기준이 일치한다.
        if not is_cash:
            prev_effective_shares = float(state.prev_effective_shares_map.get(ticker_key, 0.0) or 0.0)
            has_prev_holding = prev_effective_shares > 0.0
            has_current_holding = shares > 0.0

            if has_current_holding and not has_prev_holding:
                state.buy_date_map[ticker_key] = target_date
                state.holding_days_map[ticker_key] = 1
            elif has_current_holding and has_prev_holding:
                state.holding_days_map[ticker_key] = max(1, int(state.holding_days_map.get(ticker_key, 0)) + 1)
            else:
                state.buy_date_map[ticker_key] = None
                state.holding_days_map[ticker_key] = 0
        else:
            state.buy_date_map[ticker_key] = None
            state.holding_days_map[ticker_key] = 0

        display_avg_cost = None
        if not is_cash and _is_finite_number(avg_cost) and avg_cost > 0:
            display_avg_cost = avg_cost

        cost_basis = display_avg_cost * shares if display_avg_cost is not None and shares > 0 else 0.0
        evaluation_profit = 0.0 if is_cash else (pv - cost_basis)
        evaluation_pct = ((evaluation_profit / cost_basis) * 100.0) if cost_basis > 0 else None

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
            current_weight = cash_ratio * 100.0

        days_until_execution = _count_trading_days_until(
            result.portfolio_timeseries.index,
            target_date,
            row.get("execute_on"),
        )
        message = ""
        if pending_action:
            message = _format_pending_message(
                pending_action=pending_action,
                pending_reason=pending_reason,
                note=note,
                current_weight=current_weight,
                target_weight=float(target_weight) * 100.0 if _is_finite_number(target_weight) else None,
                target_name=name,
                days_until_execution=days_until_execution,
            )
        elif prev_pending_action and not is_cash:
            message = _format_executed_message(
                prev_pending_action=prev_pending_action,
                prev_pending_reason=prev_pending_reason,
                prev_current_weight=state.prev_current_weights_map.get(ticker_key),
                prev_target_weight=state.prev_target_weights_map.get(ticker_key),
                target_name=name,
            )
        elif not is_cash:
            current_target_weight = float(target_weight) * 100.0 if _is_finite_number(target_weight) else None
            if current_target_weight is not None:
                from config import REBALANCE_BUFFER

                weight_gap = abs(current_weight - current_target_weight) / 100.0
                if weight_gap > float(REBALANCE_BUFFER) / 100.0:
                    days_until_rebalance = _estimate_next_rebalance_days(result, target_date)
                    message = _format_scheduled_message(
                        current_weight=current_weight,
                        target_weight=current_target_weight,
                        shares=shares,
                        days_until_rebalance=days_until_rebalance,
                    )

        if not message:
            message = _normalize_note_text(note) or DECISION_MESSAGES.get(display_decision, "")

        is_current_holding = (not is_cash) and shares > 0

        sort_group = 2
        if is_cash:
            sort_group = 0
        elif is_current_holding:
            sort_group = 1

        snapshot_row = {
            "ticker": ticker_key,
            "bucket": bucket_id,
            "bucket_display": bucket_display,
            "name": name,
            "display_decision": "N/A" if is_cash else "HOLD",
            "raw_decision": raw_decision or "-",
            "holding_days": state.holding_days_map.get(ticker_key, 0),
            "price": price,
            "avg_cost": display_avg_cost,
            "daily_pct": daily_pct,
            "evaluation_pct": evaluation_pct,
            "shares": shares,
            "pv": pv,
            "evaluation_profit": evaluation_profit if display_avg_cost is not None else None,
            "weight": current_weight,
            "current_weight": current_weight,
            "target_weight": float(target_weight) * 100.0 if _is_finite_number(target_weight) else None,
            "score": float(score) if _is_finite_number(score) else None,
            "message": message,
            "is_cash": is_cash,
            "is_current_holding": is_current_holding,
            "is_pending_tomorrow": is_pending_tomorrow,
            "pending_action": pending_action or None,
            "sort_group": sort_group,
        }
        entries.append(snapshot_row)
        state.prev_pending_actions_map[ticker_key] = pending_action
        state.prev_pending_reasons_map[ticker_key] = pending_reason
        state.prev_effective_shares_map[ticker_key] = shares
        state.prev_effective_avg_cost_map[ticker_key] = avg_cost
        state.prev_current_weights_map[ticker_key] = current_weight
        state.prev_target_weights_map[ticker_key] = (
            float(target_weight) * 100.0 if _is_finite_number(target_weight) else None
        )

    effective_bucket_topn = bucket_topn
    if effective_bucket_topn is None:
        try:
            effective_bucket_topn = int(getattr(result, "bucket_topn", 0) or 0)
        except (TypeError, ValueError):
            effective_bucket_topn = 0

    if effective_bucket_topn and effective_bucket_topn > 0:
        bucket_counts: dict[int | str, int] = {}
        ranked_entries = sorted(
            entries,
            key=lambda row: (
                0 if row.get("is_current_holding") else 1,
                (int(row.get("bucket")) if row.get("bucket") is not None else 99),
                -(float(row.get("score")) if _is_finite_number(row.get("score")) else float("-inf")),
                str(row.get("ticker", "")),
            ),
        )

        for row in ranked_entries:
            if row.get("is_cash"):
                row["_is_bucket_top"] = False
                continue
            b_idx = row.get("bucket") if row.get("bucket") is not None else 99
            bucket_counts[b_idx] = bucket_counts.get(b_idx, 0) + 1
            row["_is_bucket_top"] = bucket_counts[b_idx] <= effective_bucket_topn
    else:
        for row in entries:
            row["_is_bucket_top"] = False

    def _final_sort_key(row: dict[str, Any]) -> tuple[int, int, float, str]:
        # 백테스트 일별 표는 CASH 고정, 그 다음 보유 여부 -> 버킷 -> 점수 내림차순 정렬
        if row.get("is_cash"):
            return (-1, 0, 0.0, "CASH")

        holding_priority = 0 if row.get("is_current_holding") else 1
        bucket_val = row.get("bucket")
        try:
            bucket_priority = int(bucket_val) if bucket_val is not None else 99
        except (TypeError, ValueError):
            bucket_priority = 99
        score_val = float(row.get("score")) if _is_finite_number(row.get("score")) else float("-inf")
        return (0, holding_priority, bucket_priority, -score_val, str(row.get("ticker", "")))

    entries.sort(key=_final_sort_key)

    sorted_rows: list[dict[str, Any]] = []
    current_idx = 1
    for snapshot_row in entries:
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
