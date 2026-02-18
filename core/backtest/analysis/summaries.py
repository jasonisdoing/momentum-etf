import math
from typing import Any

import pandas as pd


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
    portfolio_df: pd.DataFrame, initial_capital: float, bucket_topn: int
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
                "max_topn": bucket_topn,
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
    sell_decisions = {
        "SELL_MOMENTUM",
        "SELL_TREND",
        "CUT_STOPLOSS",
        "SELL_REPLACE",
        "SELL_TRAILING",
        "SELL_RSI",
        "SELL_REBALANCE",
        "SELL_MACRO",
    }

    for ticker, df in ticker_timeseries.items():
        ticker_key = str(ticker).upper()
        if ticker_key == "CASH" or df.empty:
            continue

        realized_profit = 0.0
        total_trades = 0
        winning_trades = 0

        if "decision" in df.columns:
            sell_mask = df["decision"].isin(sell_decisions) | df["decision"].str.startswith("SELL")
            trades = df[sell_mask]
            realized_profit = trades["trade_profit"].sum() if "trade_profit" in trades.columns else 0.0
            total_trades = len(trades)
            if "trade_profit" in trades.columns:
                winning_trades = (trades["trade_profit"] > 0).sum()

        last_row = df.iloc[-1]
        final_shares = float(last_row.get("shares", 0.0))
        final_price = float(last_row.get("price", 0.0))
        avg_cost = float(last_row.get("avg_cost", 0.0))
        unrealized = (final_price - avg_cost) * final_shares if final_shares > 0 else 0.0
        total_contribution = realized_profit + unrealized

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
    sell_decisions = {
        "SELL_MOMENTUM",
        "SELL_TREND",
        "CUT_STOPLOSS",
        "SELL_REPLACE",
        "SELL_TRAILING",
        "SELL_RSI",
        "SELL_REBALANCE",
        "SELL_MACRO",
    }

    for ticker, df in ticker_timeseries.items():
        ticker_key = str(ticker).upper()
        if ticker_key == "CASH" or df.empty:
            continue

        meta = ticker_meta.get(ticker_key, {})
        bucket_id = meta.get("bucket")
        if not bucket_id:
            continue

        realized_profit = 0.0
        if "decision" in df.columns:
            sell_mask = df["decision"].isin(sell_decisions) | df["decision"].str.startswith("SELL")
            trades = df[sell_mask]
            realized_profit = trades["trade_profit"].sum() if "trade_profit" in trades.columns else 0.0

        last_row = df.iloc[-1]
        final_shares = float(last_row.get("shares", 0.0))
        final_price = float(last_row.get("price", 0.0))
        avg_cost = float(last_row.get("avg_cost", 0.0))
        unrealized = (final_price - avg_cost) * final_shares if final_shares > 0 else 0.0
        total_contribution = realized_profit + unrealized

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
