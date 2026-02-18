from typing import Any

import pandas as pd

from core.backtest.analysis.benchmark import calculate_benchmark_performance
from core.backtest.analysis.metrics import calculate_monthly_returns, calculate_performance_summary
from core.backtest.analysis.summaries import build_bucket_summaries, calculate_weekly_summary


def build_full_summary(
    portfolio_df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    initial_capital: float,
    initial_capital_krw: float,
    currency: str,
    portfolio_topn: int,
    account_settings: dict[str, Any],
    prefetched_data: dict[str, pd.DataFrame] | None,
    ticker_timeseries: dict[str, pd.DataFrame],
    ticker_meta: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Orchestrates the creation of the full backtest summary."""
    perf = calculate_performance_summary(portfolio_df, initial_capital, start_date, end_date, currency)
    weekly = calculate_weekly_summary(portfolio_df, initial_capital, portfolio_topn)
    bucket_summary = build_bucket_summaries(ticker_timeseries, ticker_meta)
    m_rets, m_cum_rets, y_rets = calculate_monthly_returns(portfolio_df)

    total_trades = 0
    trade_decisions = {
        "SELL_MOMENTUM",
        "SELL_TREND",
        "CUT_STOPLOSS",
        "SELL_REPLACE",
        "SELL_TRAILING",
        "SELL_RSI",
        "SELL_REBALANCE",
        "SELL_MACRO",
        "BUY",
        "BUY_REPLACE",
        "BUY_REBALANCE",
    }
    for df in ticker_timeseries.values():
        if isinstance(df, pd.DataFrame) and "decision" in df.columns:
            total_trades += df["decision"].isin(trade_decisions).sum()

    benchmarks_summary = []
    bench_conf = account_settings.get("benchmark") or (account_settings.get("benchmarks") or [None])[0]
    country = str(account_settings.get("country_code", "kor")).lower()

    if isinstance(bench_conf, dict):
        ticker = str(bench_conf.get("ticker") or "").strip()
        if ticker:
            p = calculate_benchmark_performance(
                ticker,
                str(bench_conf.get("name") or ticker),
                str(bench_conf.get("country") or country),
                start_date,
                end_date,
                prefetched_data,
            )
            if p:
                benchmarks_summary.append(p)

    if not benchmarks_summary:
        p = calculate_benchmark_performance(
            str(account_settings.get("benchmark_ticker") or "^GSPC"),
            str(account_settings.get("benchmark_name") or "S&P 500"),
            country,
            start_date,
            end_date,
            prefetched_data,
        )
        if p:
            benchmarks_summary.append(p)

    final_row = portfolio_df.iloc[-1] if not portfolio_df.empty else {}
    return {
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "initial_capital": initial_capital,
        "initial_capital_local": initial_capital,
        "initial_capital_krw": initial_capital_krw,
        "final_value": perf.get("final_value", 0.0),
        "final_value_local": perf.get("final_value", 0.0),
        "final_value_krw": perf.get("final_value_krw", 0.0),
        "period_return": float(final_row.get("cumulative_return_pct", 0.0)),
        "evaluation_return_pct": 0.0,
        "held_count": int(final_row.get("held_count", 0)),
        "turnover": int(total_trades),
        "cagr": perf.get("cagr", 0.0),
        "mdd": perf.get("mdd", 0.0),
        "sharpe": perf.get("sharpe", 0.0),
        "sharpe_to_mdd": (perf["sharpe"] / perf["mdd"]) if perf.get("mdd", 0) > 0 else 0.0,
        "benchmark_cum_ret_pct": benchmarks_summary[0]["cumulative_return_pct"] if benchmarks_summary else 0.0,
        "benchmark_cagr_pct": benchmarks_summary[0]["cagr_pct"] if benchmarks_summary else 0.0,
        "benchmarks": benchmarks_summary,
        "benchmark_name": benchmarks_summary[0]["name"] if benchmarks_summary else "S&P 500",
        "weekly_summary": weekly,
        "bucket_summary": bucket_summary,
        "monthly_returns": m_rets,
        "monthly_cum_returns": m_cum_rets,
        "yearly_returns": y_rets,
        "benchmark_monthly_returns": {
            (b.get("name") or b.get("ticker")): b.get("monthly_returns")
            for b in benchmarks_summary
            if b.get("monthly_returns") is not None and not b["monthly_returns"].empty
        },
        "currency": currency,
    }
