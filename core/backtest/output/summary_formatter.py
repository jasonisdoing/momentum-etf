from typing import Any

import pandas as pd

from utils.formatters import format_pct_change


def format_period_return_with_listing_date(series_summary: dict[str, Any], core_start_dt: pd.Timestamp) -> str:
    """Format period return percentage, optionally appending listing date."""
    period_return_pct = series_summary.get("period_return_pct", 0.0)
    listing_date = series_summary.get("listing_date")

    if listing_date and core_start_dt:
        listing_dt = pd.to_datetime(listing_date)
        if listing_dt > core_start_dt:
            return f"{format_pct_change(period_return_pct).strip()}({listing_date})"

    return format_pct_change(period_return_pct)
