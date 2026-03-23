import math
from typing import Any

import pandas as pd


def _build_sell_trade_mask(df: pd.DataFrame) -> pd.Series:
    """Return mask for realized sell trades compatible with pending_action model."""
    if df.empty:
        return pd.Series(False, index=df.index)

    pending_col = (
        df["pending_action"].astype(str).str.upper()
        if "pending_action" in df.columns
        else pd.Series("", index=df.index, dtype=object)
    )
    trade_shares_col = pd.to_numeric(df.get("trade_shares", 0.0), errors="coerce").fillna(0.0)
    trade_amount_col = pd.to_numeric(df.get("trade_amount", 0.0), errors="coerce").fillna(0.0)

    return pending_col.str.startswith("SELL") & ((trade_shares_col > 0) | (trade_amount_col > 0))


def _build_buy_trade_mask(df: pd.DataFrame) -> pd.Series:
    """매수 체결 행 마스크를 반환합니다."""
    if df.empty:
        return pd.Series(False, index=df.index)

    pending_col = (
        df["pending_action"].astype(str).str.upper()
        if "pending_action" in df.columns
        else pd.Series("", index=df.index, dtype=object)
    )
    trade_shares_col = pd.to_numeric(df.get("trade_shares", 0.0), errors="coerce").fillna(0.0)
    trade_amount_col = pd.to_numeric(df.get("trade_amount", 0.0), errors="coerce").fillna(0.0)

    return pending_col.str.startswith("BUY") & ((trade_shares_col > 0) | (trade_amount_col > 0))


def calculate_realized_trade_stats(df: pd.DataFrame) -> tuple[float, int, int]:
    """매도 체결 행을 기준으로 실현손익/거래횟수/승수를 계산합니다."""
    if df.empty or ("decision" not in df.columns and "pending_action" not in df.columns):
        return 0.0, 0, 0

    sell_mask = _build_sell_trade_mask(df)
    if not sell_mask.any():
        return 0.0, 0, 0

    trades = df[sell_mask].copy()
    if trades.empty:
        return 0.0, 0, 0

    trade_amount_series = pd.to_numeric(trades.get("trade_amount", 0.0), errors="coerce").fillna(0.0)
    trade_shares_series = pd.to_numeric(trades.get("trade_shares", 0.0), errors="coerce").fillna(0.0)
    avg_cost_series = pd.to_numeric(trades.get("avg_cost", 0.0), errors="coerce").fillna(0.0)
    trade_profit_series = pd.to_numeric(trades.get("trade_profit", 0.0), errors="coerce").fillna(0.0)

    # 엔진 기록보다 체결금액-평균단가 기반 복원이 더 안정적이므로 우선 사용합니다.
    recomputed_profit_series = trade_amount_series - (trade_shares_series * avg_cost_series)
    use_recomputed_mask = (trade_amount_series > 0) & (trade_shares_series > 0)
    realized_profit_series = trade_profit_series.where(~use_recomputed_mask, recomputed_profit_series)

    realized_profit = float(realized_profit_series.sum())
    total_trades = int(len(trades))
    winning_trades = int((realized_profit_series > 0).sum())
    return realized_profit, total_trades, winning_trades


def calculate_ticker_profit_components(df: pd.DataFrame) -> dict[str, float]:
    """종목별 기여도 구성요소를 계산합니다."""
    if df.empty:
        return {
            "buy_amount": 0.0,
            "sell_amount": 0.0,
            "final_value": 0.0,
            "remaining_cost_basis": 0.0,
            "realized_profit": 0.0,
            "unrealized_profit": 0.0,
            "total_contribution": 0.0,
        }

    first_row = df.iloc[0]
    bootstrap_shares = float(first_row.get("shares", 0.0) or 0.0)
    bootstrap_avg_cost = float(first_row.get("avg_cost", 0.0) or 0.0)
    bootstrap_buy_amount = (
        bootstrap_shares * bootstrap_avg_cost if bootstrap_shares > 0 and bootstrap_avg_cost > 0 else 0.0
    )

    buy_mask = _build_buy_trade_mask(df)
    sell_mask = _build_sell_trade_mask(df)

    buy_amount = bootstrap_buy_amount
    if buy_mask.any():
        buy_amount += float(pd.to_numeric(df.loc[buy_mask, "trade_amount"], errors="coerce").fillna(0.0).sum())

    sell_amount = 0.0
    if sell_mask.any():
        sell_amount = float(pd.to_numeric(df.loc[sell_mask, "trade_amount"], errors="coerce").fillna(0.0).sum())

    last_row = df.iloc[-1]
    final_shares = float(last_row.get("shares", 0.0) or 0.0)
    final_price = float(last_row.get("price", 0.0) or 0.0)
    final_avg_cost = float(last_row.get("avg_cost", 0.0) or 0.0)

    final_value = final_shares * final_price if final_shares > 0 and final_price > 0 else 0.0
    remaining_cost_basis = final_shares * final_avg_cost if final_shares > 0 and final_avg_cost > 0 else 0.0
    unrealized_profit = final_value - remaining_cost_basis
    total_contribution = final_value + sell_amount - buy_amount
    realized_profit = total_contribution - unrealized_profit

    return {
        "buy_amount": buy_amount,
        "sell_amount": sell_amount,
        "final_value": final_value,
        "remaining_cost_basis": remaining_cost_basis,
        "realized_profit": realized_profit,
        "unrealized_profit": unrealized_profit,
        "total_contribution": total_contribution,
    }


def extract_evaluated_records(ticker_timeseries: dict[str, pd.DataFrame]) -> dict[str, dict[str, Any]]:
    """Extracts the latest status for each ticker."""
    records = {}
    for ticker, df in ticker_timeseries.items():
        if df.empty:
            continue
        last_row = df.iloc[-1]
        rec = last_row.to_dict()
        if isinstance(rec.get("date"), pd.Timestamp):
            rec["date"] = rec["date"].strftime("%Y-%m-%d")
        records[ticker] = rec
    return records


def calculate_weekly_summary(
    portfolio_df: pd.DataFrame, initial_capital: float, universe_count: int
) -> list[dict[str, Any]]:
    """Calculates weekly performance summary."""
    if portfolio_df.empty:
        return []

    pv_series = portfolio_df["total_value"].astype(float)
    weekly_values = pv_series.resample("W-FRI").last().dropna()
    if weekly_values.empty:
        return []

    weekly_return_pct = weekly_values.pct_change().mul(100).fillna(0.0)
    weekly_cum_pct = (
        (weekly_values / initial_capital - 1).mul(100)
        if initial_capital > 0
        else pd.Series(0.0, index=weekly_values.index)
    )

    rows = []
    for dt, value in weekly_values.items():
        if not isinstance(dt, pd.Timestamp):
            continue
        actual_date = dt
        held_count = 0
        if dt in portfolio_df.index:
            held_count = int(portfolio_df.loc[dt].get("held_count", 0))
        else:
            prev_dates = portfolio_df.index[portfolio_df.index < dt]
            if not prev_dates.empty:
                actual_date = prev_dates[-1]
                held_count = int(portfolio_df.loc[actual_date, "held_count"] or 0)

        week_end_label = actual_date.strftime("%Y-%m-%d")
        if actual_date.weekday() != 4:
            weekday_map = {0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토", 6: "일"}
            week_end_label += f"({weekday_map.get(actual_date.weekday(), '')})"

        rows.append(
            {
                "week_end": week_end_label,
                "value": float(value),
                "held_count": held_count,
                "universe_count": universe_count,
                "weekly_return_pct": float(weekly_return_pct.loc[dt] if dt in weekly_return_pct.index else 0.0),
                "cumulative_return_pct": float(weekly_cum_pct.loc[dt] if dt in weekly_cum_pct.index else 0.0),
            }
        )
    return rows


def build_ticker_summaries(
    ticker_timeseries: dict[str, pd.DataFrame], ticker_meta: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Calculates performance summary per ticker."""
    summaries = []

    for ticker, df in ticker_timeseries.items():
        ticker_key = str(ticker).upper()
        if ticker_key == "CASH" or df.empty:
            continue

        realized_profit, total_trades, winning_trades = calculate_realized_trade_stats(df)
        profit_components = calculate_ticker_profit_components(df)
        realized_profit = float(profit_components["realized_profit"])
        unrealized = float(profit_components["unrealized_profit"])
        total_contribution = float(profit_components["total_contribution"])
        final_shares = float(df.iloc[-1].get("shares", 0.0) or 0.0)

        period_return_pct = 0.0
        listing_date = None
        valid_prices = df[df["price"] > 0]
        if not valid_prices.empty:
            first = float(valid_prices["price"].iloc[0])
            last = float(valid_prices["price"].iloc[-1])
            if first > 0:
                period_return_pct = ((last / first) - 1.0) * 100.0
            listing_date = valid_prices.index[0].strftime("%Y-%m-%d")

        if total_trades == 0 and final_shares <= 0 and math.isclose(total_contribution, 0.0, abs_tol=1e-9):
            continue

        summaries.append(
            {
                "ticker": ticker_key,
                "name": ticker_meta.get(ticker_key, {}).get("name") or ticker_key,
                "total_trades": int(total_trades),
                "winning_trades": int(winning_trades),
                "win_rate": (winning_trades / total_trades * 100.0) if total_trades > 0 else 0.0,
                "realized_profit": realized_profit,
                "unrealized_profit": unrealized,
                "total_contribution": total_contribution,
                "period_return_pct": period_return_pct,
                "listing_date": listing_date,
            }
        )

    summaries.sort(key=lambda x: x["total_contribution"], reverse=True)
    return summaries


def build_bucket_summaries(
    ticker_timeseries: dict[str, pd.DataFrame], ticker_meta: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Calculates performance summary per bucket."""
    bucket_data = {}

    for ticker, df in ticker_timeseries.items():
        ticker_key = str(ticker).upper()
        if ticker_key == "CASH" or df.empty:
            continue

        meta = ticker_meta.get(ticker_key, {})
        bucket_id = meta.get("bucket")
        if not bucket_id:
            continue

        profit_components = calculate_ticker_profit_components(df)
        realized_profit = float(profit_components["realized_profit"])
        unrealized = float(profit_components["unrealized_profit"])
        total_contribution = float(profit_components["total_contribution"])

        if bucket_id not in bucket_data:
            bucket_data[bucket_id] = {
                "bucket_id": bucket_id,
                "realized_profit": 0.0,
                "unrealized_profit": 0.0,
                "total_contribution": 0.0,
                "ticker_count": 0,
            }

        bucket_data[bucket_id]["realized_profit"] += realized_profit
        bucket_data[bucket_id]["unrealized_profit"] += unrealized
        bucket_data[bucket_id]["total_contribution"] += total_contribution
        bucket_data[bucket_id]["ticker_count"] += 1

    summaries = list(bucket_data.values())
    summaries.sort(key=lambda x: x["bucket_id"])
    return summaries
