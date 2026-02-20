from typing import Any

import numpy as np
import pandas as pd


def calculate_benchmark_performance(
    ticker: str,
    name: str,
    country: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    prefetched_data: dict[str, pd.DataFrame] | None = None,
) -> dict[str, Any] | None:
    """Calculates benchmark performance stats."""
    benchmark_df = None
    if prefetched_data:
        for cand in [ticker, ticker.upper(), ticker.lower()]:
            if cand in prefetched_data:
                benchmark_df = prefetched_data[cand]
                break

    if benchmark_df is None or benchmark_df.empty:
        return None

    benchmark_df = benchmark_df.sort_index()
    benchmark_df = benchmark_df.loc[(benchmark_df.index >= start_date) & (benchmark_df.index <= end_date)]
    if benchmark_df.empty or "Close" not in benchmark_df.columns:
        return None

    start_p = float(benchmark_df["Close"].iloc[0])
    end_p = float(benchmark_df["Close"].iloc[-1])
    if start_p <= 0:
        return None

    years = max((end_date - start_date).days / 365.25, 0.0)
    cagr_pct = ((end_p / start_p) ** (1 / years) - 1) * 100.0 if years > 0 else 0.0

    prices = benchmark_df["Close"].astype(float)
    running_max = prices.cummax()
    mdd_pct = float(((running_max - prices) / running_max.replace(0, np.nan)).fillna(0).max()) * 100.0

    daily_rets = prices.pct_change().dropna()
    sharpe = 0.0
    if not daily_rets.empty:
        mean, std = daily_rets.mean(), daily_rets.std()
        if std > 0:
            sharpe = (mean / std) * (252**0.5)

    monthly_vals = prices.resample("ME").last()
    monthly_rets = monthly_vals.pct_change()
    if not monthly_rets.empty:
        first_val = float(monthly_vals.iloc[0])
        if pd.isna(monthly_rets.iloc[0]):
            monthly_rets.iloc[0] = (first_val / start_p) - 1.0

    return {
        "ticker": ticker,
        "name": name,
        "country": country,
        "cumulative_return_pct": ((end_p / start_p) - 1) * 100.0,
        "cagr_pct": cagr_pct,
        "sharpe": sharpe,
        "mdd": mdd_pct,
        "sharpe_to_mdd": (sharpe / mdd_pct) if mdd_pct > 0 else 0.0,
        "monthly_returns": monthly_rets.dropna(),
    }
