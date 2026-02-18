import os
import sys

import pandas as pd

# Ensure project root is in path
sys.path.append(os.getcwd())

from core.backtest.engine import run_portfolio_backtest


def verify_strategy():
    print("Starting Verification Backtest...")

    # 1. Setup Dates
    # Q1: Jan-Mar, Q2: Apr-Jun
    trading_days = pd.date_range("2024-01-01", "2024-06-30", freq="B")

    # 2. Setup Tickers (2 Buckets, 2 Tickers each)
    bucket_map = {"A": 1, "B": 1, "C": 2, "D": 2}
    etf_universe = [{"ticker": t} for t in bucket_map]

    # 3. Setup Price Data & Metrics
    prices = {}
    metrics = {}

    start_q2 = pd.Timestamp("2024-04-01")

    for t in bucket_map:
        # Price = 100 flat
        df = pd.DataFrame(index=trading_days)
        df["open"] = 100.0
        df["high"] = 100.0
        df["low"] = 100.0
        df["close"] = 100.0
        df["volume"] = 1000
        # Ensure 'Close' and 'Open' cols exist for prefetch logic
        df["Close"] = 100.0
        df["Open"] = 100.0
        prices[t] = df

        # Determine Score
        # Q1 (Jan-Mar): A=10, B=5, C=10, D=5
        # Q2 (Apr-Jun): A=5, B=10, C=5, D=10
        score_series = pd.Series(index=trading_days, dtype=float)

        q1_mask = trading_days < start_q2
        q2_mask = trading_days >= start_q2

        if t == "A":
            score_series.loc[q1_mask] = 10.0
            score_series.loc[q2_mask] = 5.0
        elif t == "B":
            score_series.loc[q1_mask] = 5.0
            score_series.loc[q2_mask] = 10.0
        elif t == "C":
            score_series.loc[q1_mask] = 10.0
            score_series.loc[q2_mask] = 5.0
        elif t == "D":
            score_series.loc[q1_mask] = 5.0
            score_series.loc[q2_mask] = 10.0

        # Metrics Structure expected by metrics.py
        # ma_key = "SMA_20"
        ma_key = "SMA_20"

        metrics[t] = {
            "close": df["close"],
            "open": df["open"],
            "ma": {ma_key: pd.Series(100.0, index=trading_days)},
            "ma_score": {ma_key: score_series},
            "rsi_score": pd.Series(50.0, index=trading_days),
        }

    # 4. Run Backtest
    print("Running backtest engine...")
    try:
        result = run_portfolio_backtest(
            stocks=etf_universe,
            initial_capital=10000.0,
            core_start_date=trading_days[0],
            top_n=4,  # Legacy
            date_range=["2024-01-01", "2024-06-30"],
            country="kor",
            missing_ticker_sink=set(),
            bucket_map=bucket_map,
            bucket_topn=1,
            rebalance_mode="QUARTERLY",
            allow_negative_score=True,
            # Inject Data
            prefetched_data=prices,
            prefetched_metrics=metrics,
            trading_calendar=trading_days,
            ma_days=20,  # Matches ma_key
        )
    except Exception as e:
        print(f"Backtest Failed: {e}")
        import traceback

        traceback.print_exc()
        return

    # 5. Analyze Results
    print(f"Backtest Finished. Processing {len(result)} tickers.")
    print(f"Backtest Finished. Result Keys: {list(result.keys())}")

    # Check Trades for A and B
    for t in ["A", "B", "C", "D"]:
        if t not in result:
            print(f"Ticker {t} not in results")
            continue

        df_res = pd.DataFrame(result[t])
        if df_res.empty:
            print(f"No records for {t}")
            continue

        print(f"Columns for {t}: {df_res.columns.tolist()}")

        trades = df_res[df_res["trade_amount"] != 0]
        if not trades.empty:
            print(f"\nTrades for {t}:")
            # Filter columns that actually exist
            cols = ["date", "decision", "trade_amount", "shares", "price"]
            exist_cols = [c for c in cols if c in df_res.columns]
            print(trades[exist_cols])
        else:
            print(f"\nNo trades for {t}")

    # Verify Strategy Logic:
    # Q1 End (March 29 approx): Rebalance Triggered.
    # A score 5 (dropping), B score 10 (rising).
    # Expected: Sell A, Buy B.

    # Note: Rebalance logic runs on LAST TRADING DAY of Quarter.
    # March 29 (Fri) is likely last business day.
    # On March 29, Score is still 10 for A?
    # Wait, my mock data:
    # q1_mask = trading_days < start_q2 (April 1).
    # So March 29 is Q1. Score A=10, B=5.
    # Logic:
    # On March 29 Check:
    #   Calculate Score -> Uses today's score (10 vs 5).
    #   Target: A (Top 1 in Bucket 1).
    #   A is kept. B is not bought.
    # So NO Trade on March 29?

    # Wait, when does the score change?
    # April 1.
    # Next Rebalance: June 28.
    # On June 28: Score A=5, B=10.
    # Target: B.
    # Action: Sell A, Buy B.

    # So I expect trades on June 28 (or end of Q2).
    # If I want trade on March 29, I should simulate score change earlier?
    # Or start backtest earlier?
    # But Q1 end rebalance uses Q1 end scores.
    # If scores flip on April 1, then March 29 rebalance sees OLD scores.
    # Correct.

    # So: A bought in Jan. Held through March.
    # April 1 score drops. But NO Rebalance until June!
    # A should be held until June 28?
    # UNLESS Standard Sell Logic triggers (Score < 0).
    # Score A drops to 5 (still positive). So no Sell Trend.
    # So A held until June.
    # June 28: Rebalance. A sold, B bought.

    print("\nVerification Logic:")
    print("- Expecting A/C bought in Jan.")
    print("- Expecting A/C SOLD and B/D BOUGHT in June end (if buckets work).")

    # --- Verify Report Formatting ---
    print("\nVerifying Report Formatting (Bucket Column)...")
    from core.backtest.domain import AccountBacktestResult
    from core.backtest.output import _generate_daily_report_lines

    # Mock Ticker Meta with Buckets
    ticker_meta = {
        "A": {"name": "Stock A", "bucket": 1},
        "B": {"name": "Stock B", "bucket": 1},
        "C": {"name": "Stock C", "bucket": 2},
        "D": {"name": "Stock D", "bucket": 2},
    }

    # Construct Mock Result
    mock_result = AccountBacktestResult(
        account_id="mock_test",
        country_code="kor",
        start_date=trading_days[0],
        end_date=trading_days[-1],
        initial_capital=10000.0,
        initial_capital_krw=10000.0,
        currency="KRW",
        portfolio_topn=4,
        holdings_limit=4,
        summary={},
        portfolio_timeseries=pd.DataFrame(),  # Not used for daily report row generation? Actually it IS used for looping dates.
        ticker_timeseries=result,
        ticker_meta=ticker_meta,
        evaluated_records={},
        monthly_returns=pd.Series(),
        monthly_cum_returns=pd.Series(),
        yearly_returns=pd.Series(),
        ticker_summaries=[],
        settings_snapshot={},
        backtest_start_date="2024-01-01",
        missing_tickers=[],
    )

    # We need portfolio_timeseries to iterate.
    # Let's create a dummy portfolio DF with the last date.
    mock_result.portfolio_timeseries = pd.DataFrame(index=[trading_days[-1]])
    mock_result.portfolio_timeseries["total_value"] = 10000.0
    mock_result.portfolio_timeseries["total_cash"] = 0.0
    mock_result.portfolio_timeseries["total_holdings"] = 10000.0

    mock_settings = {"currency": "KRW", "country_code": "kor"}

    try:
        report_lines = _generate_daily_report_lines(mock_result, mock_settings)
        print("Generated Report Lines (Last Day):")
        # Print lines corresponding to the table (skip some logic if needed)
        # _generate_daily_report_lines returns list of STRINGS (rendered table lines?)
        # No, wait. _generate_daily_report_lines returns list of STRINGS which are LINES of the file.
        # Let's print them.
        for line in report_lines:
            print(line)
    except Exception as e:
        print(f"Report Generation Failed: {e}")
        import traceback

        traceback.print_exc()

    # --- Verify Dual Currency Display ---
    print("\nVerifying Dual Currency Display (USD Account, KRW Initial)...")
    from core.backtest.output import print_backtest_summary

    mock_summary = {
        "final_value": 150_000.0,  # USD
        "final_value_krw": 200_000_000.0,  # KRW
        "initial_capital_local": 100_000.0,  # USD
        "cagr": 50.0,
        "mdd": 10.0,
        "sharpe": 2.0,
        "start_date": "2024-01-01",
        "end_date": "2024-06-30",
        "currency": "USD",
    }

    print_backtest_summary(
        summary=mock_summary,
        account_id="us",
        country_code="US",
        backtest_start_date="2024-01-01",
        initial_capital_krw=130_000_000.0,  # KRW
        portfolio_topn=2,
        ticker_summaries=[],
        core_start_dt=pd.Timestamp("2024-01-01"),
        emit_to_logger=True,
        section_start_index=1,
    )


if __name__ == "__main__":
    verify_strategy()
