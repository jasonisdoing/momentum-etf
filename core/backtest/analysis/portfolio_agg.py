import pandas as pd


def build_portfolio_dataframe(
    ticker_timeseries: dict[str, pd.DataFrame], initial_capital: float, bucket_topn: int
) -> pd.DataFrame:
    """Aggregates individual ticker results into a portfolio-level DataFrame."""
    if not ticker_timeseries:
        return pd.DataFrame()

    all_indices = [df.index for df in ticker_timeseries.values() if not df.empty]
    if not all_indices:
        return pd.DataFrame()

    union_index = all_indices[0]
    for idx in all_indices[1:]:
        union_index = union_index.union(idx)
    union_index = union_index.sort_values()

    rows = []
    prev_total = None
    for dt in union_index:
        total_v, total_cash, total_h, held_c = 0.0, 0.0, 0.0, 0
        for ticker, ts in ticker_timeseries.items():
            if dt not in ts.index:
                continue
            row = ts.loc[dt]
            pv = row.get("pv", 0.0)
            if pd.isna(pv):
                pv = 0.0
            total_v += pv
            if ticker == "CASH":
                total_cash += pv
            else:
                total_h += pv
                if row.get("shares", 0) > 0:
                    held_c += 1

        daily_pl, daily_ret = 0.0, 0.0
        if prev_total is not None and prev_total > 0:
            daily_pl = total_v - prev_total
            daily_ret = ((total_v / prev_total) - 1.0) * 100.0
        prev_total = total_v

        rows.append(
            {
                "date": dt,
                "total_value": total_v,
                "total_cash": total_cash,
                "total_holdings": total_h,
                "held_count": held_c,
                "daily_profit_loss": daily_pl,
                "daily_return_pct": daily_ret,
                "cumulative_return_pct": ((total_v / initial_capital) - 1.0) * 100.0 if initial_capital > 0 else 0.0,
                "bucket_topn": bucket_topn,
            }
        )
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()
