"""Reporting and export utilities for tuning."""

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
from pandas import DataFrame

from utils.report import render_table_eaw

MAX_TABLE_ROWS = 100


def _safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(num):
        return default
    return num


def _round_float(value: Any, *, digits: int = 6) -> float:
    """Round float-like values with a consistent precision for serialization."""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(num):
        return float("nan")
    return float(round(num, digits))


def _round_float_places(value: Any, digits: int) -> float:
    """Round float with explicit digits, preserving float semantics."""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(num):
        return float("nan")
    return float(round(num, digits))


def _round_up_float_places(value: Any, digits: int) -> float:
    """Round float up (ceiling) to the specified number of decimal places."""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if not math.isfinite(num):
        return float("nan")
    if digits <= 0:
        return float(math.ceil(num))
    factor = 10**digits
    return float(math.ceil(num * factor) / factor)


def _format_table_float(value: Any, *, digits: int = 2) -> str:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(num):
        return "-"
    fmt = f"{{:.{digits}f}}"
    return fmt.format(num)


def _render_tuning_table(
    rows: list[dict[str, Any]],
    *,
    include_samples: bool = False,
    period_str: str | None = None,
) -> list[str]:
    headers = [
        "MA개월",
        "MA타입",
        "TOPN",
        "리밸런스",
        "CAGR(%)",
        "MDD(%)",
    ]
    aligns = [
        "right",
        "center",
        "right",
        "center",
        "right",
        "right",
    ]

    if period_str:
        headers.append(f"{period_str}(%)")
    else:
        headers.append("기간수익률(%)")
    aligns.append("right")

    headers.extend(["Sharpe", "SDR(Sharpe/MDD)", "Trades(거래 수)"])
    aligns.extend(["right", "right", "right"])

    if include_samples:
        headers.append("Samples")
        aligns.append("right")

    table_rows = []
    for row in rows[:MAX_TABLE_ROWS]:
        ma_val = row.get("ma_month") or row.get("ma_days")
        ma_type_val = row.get("ma_type", "SMA")
        topn_val = row.get("bucket_topn")

        rebal_mode_val = row.get("rebalance_mode")
        if not rebal_mode_val and "tuning" in row:
            rebal_mode_val = row["tuning"].get("REBALANCE_MODE")
        if not rebal_mode_val:
            rebal_mode_val = "-"

        row_data = [
            str(int(ma_val)) if isinstance(ma_val, (int, float)) and math.isfinite(float(ma_val)) else "-",
            str(ma_type_val) if ma_type_val else "SMA",
            str(int(topn_val)) if isinstance(topn_val, (int, float)) and math.isfinite(float(topn_val)) else "-",
            str(rebal_mode_val),
            _format_table_float(row.get("cagr")),
            _format_table_float(row.get("mdd")),
            _format_table_float(row.get("period_return")),
            _format_table_float(row.get("sharpe")),
            _format_table_float(row.get("sharpe_to_mdd"), digits=3),
            str(int(row.get("turnover", 0))),
        ]

        if include_samples:
            samples_val = row.get("samples")
            if isinstance(samples_val, (int, float)) and math.isfinite(float(samples_val)):
                row_data.append(str(int(samples_val)))
            else:
                row_data.append("-")

        table_rows.append(row_data)

    lines = render_table_eaw(headers, table_rows, aligns)

    if len(rows) > MAX_TABLE_ROWS:
        lines.append(f"... (총 {len(rows)}개 중 상위 {MAX_TABLE_ROWS}개 표시)")

    return lines


def _save_dataframe_csv(df: DataFrame, path: Path) -> None:
    df_copy = df.copy()
    df_copy.index = df_copy.index.astype(str)
    df_copy.to_csv(path)


def _export_prefetched_data(debug_dir: Path, prefetched_data: Mapping[str, DataFrame]) -> None:
    prefetch_dir = debug_dir / "prefetched_data"
    prefetch_dir.mkdir(parents=True, exist_ok=True)
    for ticker, frame in prefetched_data.items():
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            continue
        _save_dataframe_csv(frame, prefetch_dir / f"{ticker}.csv")


def _extract_summary(result) -> dict[str, Any]:
    summary = result.summary or {}
    return {
        "cagr": float(summary.get("cagr") or 0.0),
        "mdd": float(summary.get("mdd") or 0.0),
        "sharpe": float(summary.get("sharpe") or 0.0),
        "sharpe_to_mdd": float(summary.get("sharpe_to_mdd") or 0.0),
        "period_return": float(summary.get("period_return") or 0.0),
        "start_date": result.start_date.strftime("%Y-%m-%d"),
        "end_date": result.end_date.strftime("%Y-%m-%d"),
        "excluded": list(result.missing_tickers),
        "turnover": int(summary.get("turnover") or 0),
    }


def _export_combo_debug(
    combo_dir: Path,
    *,
    recorded_metrics: dict[str, Any],
    result_prefetch,
    result_live,
) -> dict[str, Any]:
    combo_dir.mkdir(parents=True, exist_ok=True)

    combo_dir.joinpath("recorded_metrics.json").write_text(
        json.dumps(recorded_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    metrics_prefetch = _extract_summary(result_prefetch)
    metrics_live = _extract_summary(result_live)

    combo_dir.joinpath("metrics_prefetch.json").write_text(
        json.dumps(metrics_prefetch, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    combo_dir.joinpath("metrics_live.json").write_text(
        json.dumps(metrics_live, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    diff_payload = {
        "delta_cagr": metrics_live["cagr"] - recorded_metrics.get("cagr", 0.0)
        if recorded_metrics.get("cagr") is not None
        else None,
        "delta_mdd": metrics_live["mdd"] - recorded_metrics.get("mdd", 0.0)
        if recorded_metrics.get("mdd") is not None
        else None,
        "delta_period_return": (
            metrics_live["period_return"] - recorded_metrics.get("period_return", 0.0)
            if recorded_metrics.get("period_return") is not None
            else None
        ),
    }
    combo_dir.joinpath("metrics_diff.json").write_text(
        json.dumps(diff_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    _save_dataframe_csv(result_prefetch.portfolio_timeseries, combo_dir / "portfolio_prefetch.csv")
    _save_dataframe_csv(result_live.portfolio_timeseries, combo_dir / "portfolio_live.csv")

    prefetch_ticker_dir = combo_dir / "ticker_prefetch"
    live_ticker_dir = combo_dir / "ticker_live"
    prefetch_ticker_dir.mkdir(exist_ok=True)
    live_ticker_dir.mkdir(exist_ok=True)

    for ticker, frame in result_prefetch.ticker_timeseries.items():
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            _save_dataframe_csv(frame, prefetch_ticker_dir / f"{ticker}.csv")

    for ticker, frame in result_live.ticker_timeseries.items():
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            _save_dataframe_csv(frame, live_ticker_dir / f"{ticker}.csv")

    combo_dir.joinpath("ticker_meta_prefetch.json").write_text(
        json.dumps(result_prefetch.ticker_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    combo_dir.joinpath("ticker_meta_live.json").write_text(
        json.dumps(result_live.ticker_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return {
        "metrics_prefetch": metrics_prefetch,
        "metrics_live": metrics_live,
        "diff": diff_payload,
    }
