#!/usr/bin/env python
"""Inspect tuning debug artifacts and summarize mismatches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.set_index("date")
    return df


def _compare_dataframe(df_prefetch: pd.DataFrame, df_live: pd.DataFrame, tolerance: float) -> List[str]:
    # Align index and columns
    joined = df_prefetch.reindex(df_live.index.union(df_prefetch.index)).join(df_live, how="outer", lsuffix="_prefetch", rsuffix="_live")
    discrepancies: List[str] = []

    for col in df_prefetch.columns:
        col_prefetch = f"{col}_prefetch"
        col_live = f"{col}_live"
        if col_prefetch not in joined.columns or col_live not in joined.columns:
            continue
        series_prefetch = pd.to_numeric(joined[col_prefetch], errors="coerce")
        series_live = pd.to_numeric(joined[col_live], errors="coerce")
        diff = (series_live - series_prefetch).abs()
        mask = diff > tolerance
        if mask.any():
            idx = mask.idxmax()
            value = diff.loc[idx]
            discrepancies.append(f"{col}: first diff {idx.date()} Δ={value:.4f}")

    return discrepancies


def inspect_session(session_dir: Path, tolerance: float) -> None:
    if not session_dir.exists():
        raise FileNotFoundError(session_dir)

    diff_summary_path = session_dir / "diff_summary.json"
    if diff_summary_path.exists():
        diff_rows = json.loads(diff_summary_path.read_text(encoding="utf-8"))
        print("=== Recorded vs Live Summary ===")
        for row in diff_rows:

            def fmt(value: float | None) -> str:
                if value is None or (isinstance(value, float) and (value != value)):
                    return "-"
                return f"{value:.2f}"

            print(
                f"MONTHS={row['months_range']} MA={row['ma_period']} TOPN={row['topn']} TH={row['threshold']:.3f} | "
                f"Recorded={fmt(row['recorded_cagr'])}% Prefetch={fmt(row['prefetch_cagr'])}% Live={fmt(row['live_cagr'])}%"
            )

    for months_dir in sorted(session_dir.glob("months_*")):
        print(f"\n=== {months_dir.name} ===")
        for combo_dir in sorted(months_dir.glob("combo_*")):
            print(f"-- {combo_dir.name} --")
            portfolio_prefetch = _load_csv(combo_dir / "portfolio_prefetch.csv")
            portfolio_live = _load_csv(combo_dir / "portfolio_live.csv")
            discrepancies = _compare_dataframe(portfolio_prefetch, portfolio_live, tolerance)
            if discrepancies:
                print("Portfolio discrepancies:")
                for msg in discrepancies[:5]:
                    print(f"  - {msg}")
            else:
                print("Portfolio matches within tolerance.")

            ticker_prefetch_dir = combo_dir / "ticker_prefetch"
            ticker_live_dir = combo_dir / "ticker_live"
            tickers = sorted({p.stem for p in ticker_live_dir.glob("*.csv")})
            ticker_discrepancies: List[str] = []
            for ticker in tickers:
                pref_path = ticker_prefetch_dir / f"{ticker}.csv"
                live_path = ticker_live_dir / f"{ticker}.csv"
                if not pref_path.exists() or not live_path.exists():
                    continue
                df_prefetch = _load_csv(pref_path)
                df_live = _load_csv(live_path)
                diff = _compare_dataframe(df_prefetch, df_live, tolerance)
                if diff:
                    ticker_discrepancies.append(f"{ticker}: {diff[0]}")
            if ticker_discrepancies:
                print("Ticker discrepancies:")
                for msg in ticker_discrepancies[:10]:
                    print(f"  - {msg}")
            else:
                print("No ticker-level discrepancies beyond tolerance.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect tuning debug session differences.")
    parser.add_argument("session_dir", help="Path to tuning_debug_sessions/<account>_<timestamp>")
    parser.add_argument("--tol", type=float, default=1e-6, help="Absolute tolerance for comparisons")
    args = parser.parse_args()

    inspect_session(Path(args.session_dir), tolerance=args.tol)


if __name__ == "__main__":
    main()
