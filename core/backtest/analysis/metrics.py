from typing import Any

import numpy as np
import pandas as pd

from utils.data_loader import get_exchange_rate_series


def calculate_performance_summary(
    portfolio_df: pd.DataFrame,
    initial_capital: float,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    currency: str = "KRW",
) -> dict[str, Any]:
    """Calculates CAGR, MDD, Sharpe."""
    if portfolio_df.empty:
        return {}

    pv_series = portfolio_df["total_value"]
    pv_series_krw = pv_series.copy()

    if currency != "KRW":
        try:
            fx = get_exchange_rate_series(start_date, end_date)
            if fx is not None and not fx.empty:
                fx = fx.reindex(pv_series.index).fillna(method="ffill").fillna(method="bfill")
                converted = pv_series * fx
                if not converted.isnull().any() and not (converted == 0).any():
                    pv_series_krw = converted
        except Exception:
            pass

    final_val_local = float(pv_series.iloc[-1]) if not pv_series.empty else 0.0
    final_val_krw = float(pv_series_krw.iloc[-1]) if not pv_series_krw.empty else 0.0

    years = max((end_date - start_date).days / 365.25, 0.0)
    cagr = 0.0
    # Use the first value as the actual initial capital for CAGR calculation
    init_val_krw = float(pv_series_krw.iloc[0]) if not pv_series_krw.empty else 0.0

    if years > 0.05 and init_val_krw > 0:  # Only calc CAGR for periods > ~18 days
        if final_val_krw > 0:
            ratio = final_val_krw / init_val_krw
            try:
                cagr = (ratio ** (1 / years)) - 1
            except (OverflowError, ZeroDivisionError):
                cagr = 0.0
    elif years > 0 and init_val_krw > 0:
        # For very short periods, CAGR is not very meaningful, fallback to simple annualized return
        cagr = (final_val_krw / init_val_krw - 1) / years

    # MDD calculation improvement
    running_max = pv_series_krw.cummax()
    dd = (running_max - pv_series_krw) / running_max.replace(0, np.nan)
    max_dd = float(dd.fillna(0).max())

    # Limit extreme values
    cagr = np.clip(cagr, -0.9999, 1000.0)  # Limit to 100,000% CAGR
    max_dd = np.clip(max_dd, 0.0, 1.0)

    daily_rets = pv_series_krw.pct_change().dropna()
    sharpe = 0.0
    if not daily_rets.empty:
        mean_ret = daily_rets.mean()
        std_ret = daily_rets.std()
        if std_ret > 0:
            sharpe = (mean_ret / std_ret) * (252**0.5)

    return {
        "final_value": final_val_local,
        "final_value_krw": final_val_krw,
        "cagr": float(cagr * 100),
        "mdd": float(max_dd * 100),
        "sharpe": float(sharpe),
    }


def calculate_monthly_returns(portfolio_df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculates monthly returns, cumulative monthly returns, and yearly returns."""
    if portfolio_df.empty:
        return pd.Series(), pd.Series(), pd.Series()

    monthly_vals = portfolio_df["total_value"].resample("M").last()
    monthly_rets = monthly_vals.pct_change().fillna(0.0)
    yearly_vals = portfolio_df["total_value"].resample("Y").last()
    yearly_rets = yearly_vals.pct_change().fillna(0.0)

    return monthly_rets, monthly_vals, yearly_rets
