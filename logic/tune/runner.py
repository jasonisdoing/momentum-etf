"""Entry points for running country-level parameter tuning."""

from __future__ import annotations

import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from os import cpu_count
from pathlib import Path
from typing import Any, Collection, Dict, List, Mapping, Optional, Tuple, Set
import tempfile
import shutil

import pandas as pd
from pandas import DataFrame, Timestamp

from logic.backtest.account_runner import run_account_backtest
from logic.entry_point import StrategyRules
from utils.account_registry import get_strategy_rules
from utils.settings_loader import (
    AccountSettingsError,
    get_account_settings,
    get_backtest_months_range,
    load_common_settings,
    get_tune_month_configs,
)
from utils.logger import get_app_logger
from utils.data_loader import (
    prepare_price_data,
    get_latest_trading_day,
    fetch_ohlcv,
)
from utils.stock_list_io import get_etfs
from utils.cache_utils import save_cached_frame

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[2] / "data" / "results"
WORKERS = None  # 병렬 실행 프로세스 수 (None이면 CPU 개수 기반 자동 결정)
MAX_TABLE_ROWS = 20


def _normalize_tuning_values(values: Any, *, dtype, fallback: Any) -> List[Any]:
    if values is None:
        values = []
    if hasattr(values, "tolist"):
        values = values.tolist()
    elif isinstance(values, range):
        values = list(values)
    elif not isinstance(values, (list, tuple, set)):
        values = [values]

    normalized: List[Any] = []
    for item in values:
        if item is None:
            continue
        try:
            normalized.append(dtype(item))
        except (TypeError, ValueError):
            continue

    if not normalized:
        try:
            normalized = [dtype(fallback)]
        except (TypeError, ValueError):
            normalized = []

    return list(dict.fromkeys(normalized))


def _resolve_month_configs(months_range: Optional[int], account_id: str = None) -> List[Dict[str, Any]]:
    if months_range is not None:
        try:
            months = int(months_range)
        except (TypeError, ValueError):
            return []
        if months <= 0:
            return []
        return [
            {
                "months_range": months,
                "weight": 1.0,
                "source": "manual",
            }
        ]

    configs = get_tune_month_configs(account_id=account_id)
    if configs:
        return configs

    fallback = get_backtest_months_range()
    if fallback <= 0:
        return []
    return [
        {
            "months_range": int(fallback),
            "weight": 1.0,
            "source": "fallback",
        }
    ]


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


def _format_threshold(value: Any) -> str:
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(num):
        return "-"
    if abs(num - round(num)) < 1e-9:
        return f"{int(round(num))}"
    return f"{num:.1f}"


def _render_tuning_table(rows: List[Dict[str, Any]], *, include_samples: bool = False, months_range: Optional[int] = None) -> List[str]:
    from utils.report import render_table_eaw

    headers = ["MA", "MA타입", "TOPN", "교체점수", "과매수", "쿨다운", "CAGR(%)", "MDD(%)"]
    if months_range:
        headers.append(f"{months_range}개월(%)")
    else:
        headers.append("기간수익률(%)")
    headers.extend(["Sharpe", "SDR(Sharpe/MDD)"])
    if include_samples:
        headers.append("Samples")

    # 정렬 방향 설정 (right: 오른쪽 정렬, left: 왼쪽 정렬, center: 가운데 정렬)
    aligns = ["right", "center", "right", "right", "right", "right", "right", "right", "right", "right", "right"]
    if include_samples:
        aligns.append("right")

    table_rows = []
    for row in rows[:MAX_TABLE_ROWS]:
        ma_val = row.get("ma_period")
        ma_type_val = row.get("ma_type", "SMA")
        topn_val = row.get("portfolio_topn")
        threshold_val = row.get("replace_threshold")
        rsi_threshold_val = row.get("rsi_sell_threshold")
        cooldown_val = row.get("cooldown_days")

        row_data = [
            str(int(ma_val)) if isinstance(ma_val, (int, float)) and math.isfinite(float(ma_val)) else "-",
            str(ma_type_val) if ma_type_val else "SMA",
            str(int(topn_val)) if isinstance(topn_val, (int, float)) and math.isfinite(float(topn_val)) else "-",
            _format_threshold(threshold_val),
            str(int(rsi_threshold_val)) if isinstance(rsi_threshold_val, (int, float)) and math.isfinite(float(rsi_threshold_val)) else "-",
            str(int(cooldown_val)) if isinstance(cooldown_val, (int, float)) and math.isfinite(float(cooldown_val)) else "-",
            _format_table_float(row.get("cagr")),
            _format_table_float(row.get("mdd")),
            _format_table_float(row.get("period_return")),
            _format_table_float(row.get("sharpe")),
            _format_table_float(row.get("sharpe_to_mdd"), digits=3),
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


def _extract_summary(result) -> Dict[str, Any]:
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
    }


def _export_combo_debug(
    combo_dir: Path,
    *,
    recorded_metrics: Dict[str, Any],
    result_prefetch,
    result_live,
) -> Dict[str, Any]:
    combo_dir.mkdir(parents=True, exist_ok=True)

    combo_dir.joinpath("recorded_metrics.json").write_text(json.dumps(recorded_metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    metrics_prefetch = _extract_summary(result_prefetch)
    metrics_live = _extract_summary(result_live)

    combo_dir.joinpath("metrics_prefetch.json").write_text(json.dumps(metrics_prefetch, ensure_ascii=False, indent=2), encoding="utf-8")
    combo_dir.joinpath("metrics_live.json").write_text(json.dumps(metrics_live, ensure_ascii=False, indent=2), encoding="utf-8")

    diff_payload = {
        "delta_cagr": metrics_live["cagr"] - recorded_metrics.get("cagr", 0.0) if recorded_metrics.get("cagr") is not None else None,
        "delta_mdd": metrics_live["mdd"] - recorded_metrics.get("mdd", 0.0) if recorded_metrics.get("mdd") is not None else None,
        "delta_period_return": (
            metrics_live["period_return"] - recorded_metrics.get("period_return", 0.0) if recorded_metrics.get("period_return") is not None else None
        ),
    }
    combo_dir.joinpath("metrics_diff.json").write_text(json.dumps(diff_payload, ensure_ascii=False, indent=2), encoding="utf-8")

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
        json.dumps(result_prefetch.ticker_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    combo_dir.joinpath("ticker_meta_live.json").write_text(json.dumps(result_live.ticker_meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "metrics_prefetch": metrics_prefetch,
        "metrics_live": metrics_live,
        "diff": diff_payload,
    }


def _export_debug_month(
    debug_dir: Path,
    *,
    account_id: str,
    months_range: int,
    raw_rows: List[Dict[str, Any]],
    prefetched_data: Mapping[str, DataFrame],
    capture_top_n: int,
) -> List[Dict[str, Any]]:
    if capture_top_n <= 0 or not raw_rows:
        return []

    try:
        capture_top_n = int(capture_top_n)
    except (TypeError, ValueError):
        capture_top_n = 1
    capture_top_n = max(1, capture_top_n)

    sorted_rows = sorted(
        raw_rows,
        key=lambda row: float(row.get("CAGR") or float("-inf")),
        reverse=True,
    )

    summary_rows: List[Dict[str, Any]] = []
    month_dir = debug_dir / f"months_{months_range:02d}"

    for idx, row in enumerate(sorted_rows[:capture_top_n], 1):
        tuning = row.get("tuning") or {}
        try:
            ma = int(tuning.get("MA_PERIOD"))
            topn = int(tuning.get("PORTFOLIO_TOPN"))
            threshold = float(tuning.get("REPLACE_SCORE_THRESHOLD"))
        except (TypeError, ValueError):
            continue

        recorded_metrics = {
            "cagr": float(row.get("CAGR")) if row.get("CAGR") is not None else None,
            "mdd": float(row.get("MDD")) if row.get("MDD") is not None else None,
            "period_return": float(row.get("period_return")) if row.get("period_return") is not None else None,
        }

        strategy_rules = StrategyRules.from_values(
            ma_period=ma,
            portfolio_topn=topn,
            replace_threshold=threshold,
        )

        result_prefetch = run_account_backtest(
            account_id,
            months_range=months_range,
            quiet=True,
            prefetched_data=prefetched_data,
            strategy_override=strategy_rules,
        )

        result_live = run_account_backtest(
            account_id,
            months_range=months_range,
            quiet=True,
            strategy_override=strategy_rules,
        )

        combo_dir = month_dir / f"combo_{idx:02d}_MA{ma}_TOPN{topn}_TH{threshold:.3f}"
        combo_metrics = _export_combo_debug(
            combo_dir,
            recorded_metrics=recorded_metrics,
            result_prefetch=result_prefetch,
            result_live=result_live,
        )

        metrics_prefetch = combo_metrics["metrics_prefetch"]
        metrics_live = combo_metrics["metrics_live"]

        summary_rows.append(
            {
                "months_range": months_range,
                "ma_period": ma,
                "topn": topn,
                "threshold": threshold,
                "recorded_cagr": recorded_metrics["cagr"],
                "prefetch_cagr": metrics_prefetch["cagr"],
                "live_cagr": metrics_live["cagr"],
                "prefetch_minus_recorded": (metrics_prefetch["cagr"] - recorded_metrics["cagr"]) if recorded_metrics["cagr"] is not None else None,
                "live_minus_recorded": (metrics_live["cagr"] - recorded_metrics["cagr"]) if recorded_metrics["cagr"] is not None else None,
                "recorded_mdd": recorded_metrics["mdd"],
                "prefetch_mdd": metrics_prefetch["mdd"],
                "live_mdd": metrics_live["mdd"],
                "recorded_period_return": recorded_metrics["period_return"],
                "prefetch_period_return": metrics_prefetch["period_return"],
                "live_period_return": metrics_live["period_return"],
                "current_start_date": metrics_live["start_date"],
                "current_end_date": metrics_live["end_date"],
                "current_excluded_count": len(metrics_live["excluded"]),
                "artifact_path": str(combo_dir.relative_to(debug_dir)),
            }
        )

    return summary_rows


def _evaluate_single_combo(
    payload: Tuple[str, int, Tuple[str, str], int, int, float, int, int, str, Tuple[str, ...], Tuple[str, ...], Mapping[str, DataFrame]]
) -> Tuple[str, Any, List[str]]:
    (
        account_norm,
        months_range,
        date_range,
        ma_int,
        topn_int,
        threshold_float,
        rsi_int,
        cooldown_int,
        ma_type_str,
        excluded_tickers,
        core_holdings_tuple,
        prefetched_data,
    ) = payload

    try:
        override_rules = StrategyRules.from_values(
            ma_period=int(ma_int),
            portfolio_topn=int(topn_int),
            replace_threshold=float(threshold_float),
            ma_type=str(ma_type_str),
            core_holdings=list(core_holdings_tuple) if core_holdings_tuple else [],
        )
    except ValueError as exc:
        return (
            "failure",
            {
                "ma_period": ma_int,
                "portfolio_topn": topn_int,
                "replace_threshold": threshold_float,
                "rsi_sell_threshold": rsi_int,
                "ma_type": ma_type_str,
                "error": str(exc),
            },
            [],
        )

    strategy_overrides: Dict[str, Any] = {
        "OVERBOUGHT_SELL_THRESHOLD": rsi_int,
        "COOLDOWN_DAYS": cooldown_int,
    }

    try:
        bt_result = run_account_backtest(
            account_norm,
            months_range=months_range,
            quiet=True,
            override_settings={
                "start_date": date_range[0],
                "end_date": date_range[1],
                "strategy_overrides": strategy_overrides,
            },
            prefetched_data=prefetched_data,
            strategy_override=override_rules,
            excluded_tickers=set(excluded_tickers) if excluded_tickers else None,
        )
    except Exception as exc:
        return (
            "failure",
            {
                "ma_period": ma_int,
                "portfolio_topn": topn_int,
                "replace_threshold": threshold_float,
                "rsi_sell_threshold": rsi_int,
                "error": str(exc),
            },
            [],
        )

    summary = bt_result.summary or {}
    final_value_local = _safe_float(summary.get("final_value"), 0.0)
    final_value_krw = _safe_float(summary.get("final_value_krw"), final_value_local)

    entry = {
        "ma_period": ma_int,
        "portfolio_topn": topn_int,
        "replace_threshold": float(threshold_float),
        "rsi_sell_threshold": rsi_int,
        "cooldown_days": cooldown_int,
        "ma_type": ma_type_str,
        "cagr": _round_float(_safe_float(summary.get("cagr"), 0.0)),
        "mdd": _round_float(_safe_float(summary.get("mdd"), 0.0)),
        "sharpe": _round_float(_safe_float(summary.get("sharpe"), 0.0)),
        "sharpe_to_mdd": _round_float(_safe_float(summary.get("sharpe_to_mdd"), 0.0)),
        "period_return": _round_float(_safe_float(summary.get("period_return"), 0.0)),
        "final_value_local": final_value_local,
        "final_value": final_value_krw,
    }

    missing = getattr(bt_result, "missing_tickers", []) or []
    return ("success", entry, list(missing))


def _execute_tuning_for_months(
    account_norm: str,
    *,
    months_range: int,
    search_space: Mapping[str, List[Any]],
    end_date: Timestamp,
    excluded_tickers: Optional[Collection[str]],
    prefetched_data: Mapping[str, DataFrame],
    output_path: Optional[Path] = None,
    progress_callback: Optional[callable] = None,
) -> Optional[Dict[str, Any]]:
    logger = get_app_logger()

    ma_candidates = list(search_space.get("MA_PERIOD", []))
    topn_candidates = list(search_space.get("PORTFOLIO_TOPN", []))
    replace_candidates = list(search_space.get("REPLACE_SCORE_THRESHOLD", []))
    rsi_candidates = list(search_space.get("OVERBOUGHT_SELL_THRESHOLD", []))
    cooldown_candidates = list(search_space.get("COOLDOWN_DAYS", []))
    ma_type_candidates = list(search_space.get("MA_TYPE", ["SMA"]))

    if not ma_candidates or not topn_candidates or not replace_candidates or not rsi_candidates or not cooldown_candidates or not ma_type_candidates:
        logger.warning("[튜닝] %s (%d개월) 유효한 탐색 공간이 없습니다.", account_norm.upper(), months_range)
        return None

    combos: List[Tuple[int, int, float, int, int, str]] = [
        (ma, topn, replace, rsi, cooldown, ma_type)
        for ma in ma_candidates
        for topn in topn_candidates
        for replace in replace_candidates
        for rsi in rsi_candidates
        for cooldown in cooldown_candidates
        for ma_type in ma_type_candidates
    ]

    if not combos:
        logger.warning("[튜닝] %s (%d개월) 평가할 조합이 없습니다.", account_norm.upper(), months_range)
        return None

    start_date = end_date - pd.DateOffset(months=months_range)
    date_range = (start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

    logger.info(
        "[튜닝] %s (%d개월) 전수조사 시작 (조합 %d개)",
        account_norm.upper(),
        months_range,
        len(combos),
    )

    workers = WORKERS or (cpu_count() or 1)
    workers = max(1, min(workers, len(combos)))

    success_entries: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []
    encountered_missing: Set[str] = set()
    best_cagr_so_far = float("-inf")

    # search_space에서 core_holdings 가져오기
    core_holdings_from_space = search_space.get("CORE_HOLDINGS", [])

    payloads = [
        (
            account_norm,
            months_range,
            date_range,
            int(ma),
            int(topn),
            float(replace),
            int(rsi),
            int(cooldown),
            str(ma_type),
            tuple(excluded_tickers) if excluded_tickers else tuple(),
            tuple(core_holdings_from_space) if core_holdings_from_space else tuple(),
            prefetched_data,
        )
        for ma, topn, replace, rsi, cooldown, ma_type in combos
    ]

    if workers <= 1:
        for idx, payload in enumerate(payloads, 1):
            status, data, missing = _evaluate_single_combo(payload)
            if status == "success":
                success_entries.append(data)
                encountered_missing.update(missing)
            else:
                failures.append(data)

            if idx % max(1, len(combos) // 100) == 0 or idx == len(combos):
                logger.info(
                    "[튜닝] %s (%d개월) 진행률: %d/%d (%.1f%%)",
                    account_norm.upper(),
                    months_range,
                    idx,
                    len(combos),
                    (idx / len(combos)) * 100,
                )

                # 1%마다 중간 저장 (성공한 조합이 있을 때만)
                if success_entries and output_path and progress_callback:
                    current_best_cagr = max(_safe_float(entry.get("cagr"), float("-inf")) for entry in success_entries)
                    if current_best_cagr > best_cagr_so_far:
                        best_cagr_so_far = current_best_cagr

                    # 중간 결과 저장 콜백 호출
                    progress_callback(
                        success_entries=success_entries,
                        progress_pct=(idx / len(combos)) * 100,
                        completed=idx,
                        total=len(combos),
                    )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_map = {executor.submit(_evaluate_single_combo, payload): payload for payload in payloads}
            for idx, future in enumerate(as_completed(future_map), 1):
                status, data, missing = future.result()
                if status == "success":
                    success_entries.append(data)
                    encountered_missing.update(missing)
                else:
                    failures.append(data)

                if idx % max(1, len(combos) // 100) == 0 or idx == len(combos):
                    logger.info(
                        "[튜닝] %s (%d개월) 진행률: %d/%d (%.1f%%)",
                        account_norm.upper(),
                        months_range,
                        idx,
                        len(combos),
                        (idx / len(combos)) * 100,
                    )

                    # 1%마다 중간 저장 (성공한 조합이 있을 때만)
                    if success_entries and output_path and progress_callback:
                        current_best_cagr = max(_safe_float(entry.get("cagr"), float("-inf")) for entry in success_entries)
                        if current_best_cagr > best_cagr_so_far:
                            best_cagr_so_far = current_best_cagr

                        # 중간 결과 저장 콜백 호출
                        progress_callback(
                            success_entries=success_entries,
                            progress_pct=(idx / len(combos)) * 100,
                            completed=idx,
                            total=len(combos),
                        )

    if not success_entries:
        logger.warning("[튜닝] %s (%d개월) 성공한 조합이 없습니다.", account_norm.upper(), months_range)
        return None

    # 최적화 지표 선택 (config에서 가져오기)
    optimization_metric = search_space.get("OPTIMIZATION_METRIC").upper()

    def _sort_key(entry: Dict[str, Any]) -> float:
        if optimization_metric == "CAGR":
            return _safe_float(entry.get("cagr"), float("-inf"))
        elif optimization_metric == "SHARPE":
            return _safe_float(entry.get("sharpe"), float("-inf"))
        else:  # SDR (default)
            return _safe_float(entry.get("sharpe_to_mdd"), float("-inf"))

    success_entries.sort(key=_sort_key, reverse=True)
    best_entry = success_entries[0]

    raw_data_payload: List[Dict[str, Any]] = []
    for item in success_entries:
        cagr_val = _safe_float(item.get("cagr"), float("nan"))
        mdd_val = _safe_float(item.get("mdd"), float("nan"))
        period_return_val = _safe_float(item.get("period_return"), float("nan"))
        sharpe_val = _safe_float(item.get("sharpe"), float("nan"))
        sharpe_to_mdd_val = _safe_float(item.get("sharpe_to_mdd"), float("nan"))

        raw_data_payload.append(
            {
                "MONTHS_RANGE": months_range,
                "CAGR": _round_float_places(cagr_val, 2) if math.isfinite(cagr_val) else None,
                "MDD": _round_float_places(-mdd_val, 2) if math.isfinite(mdd_val) else None,
                "period_return": _round_float_places(period_return_val, 2) if math.isfinite(period_return_val) else None,
                "sharpe": _round_float_places(sharpe_val, 2) if math.isfinite(sharpe_val) else None,
                "sharpe_to_mdd": _round_float_places(sharpe_to_mdd_val, 3) if math.isfinite(sharpe_to_mdd_val) else None,
                "tuning": {
                    "MA_PERIOD": int(item.get("ma_period", 0)),
                    "MA_TYPE": str(item.get("ma_type", "SMA")),
                    "PORTFOLIO_TOPN": int(item.get("portfolio_topn", 0)),
                    "REPLACE_SCORE_THRESHOLD": _round_up_float_places(item.get("replace_threshold", 0.0), 1),
                    "OVERBOUGHT_SELL_THRESHOLD": int(item.get("rsi_sell_threshold", 10)),
                    "COOLDOWN_DAYS": int(item.get("cooldown_days", 2)),
                },
            }
        )

    return {
        "months_range": months_range,
        "best": best_entry,
        "failures": failures,
        "success_count": len(success_entries),
        "missing_tickers": sorted(encountered_missing),
        "raw_data": raw_data_payload,
    }


def _build_run_entry(
    months_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    param_fields = {
        "MA_PERIOD": ("ma_period", True),
        "PORTFOLIO_TOPN": ("portfolio_topn", True),
        "REPLACE_SCORE_THRESHOLD": ("replace_threshold", False),
        "OVERBOUGHT_SELL_THRESHOLD": ("rsi_sell_threshold", True),
    }

    entry: Dict[str, Any] = {
        "result": {},
    }

    raw_data_payload: List[Dict[str, Any]] = []
    weighted_cagr_sum = 0.0
    weighted_cagr_weight = 0.0
    weighted_mdd_sum = 0.0
    weighted_mdd_weight = 0.0
    cagr_values: List[float] = []
    mdd_values: List[float] = []

    for item in months_results:
        best = item.get("best") or {}
        months = item.get("months_range")
        if not best or months is None:
            continue

        try:
            weight = float(item.get("weight", 0.0))
        except (TypeError, ValueError):
            weight = 0.0

        cagr_val = _safe_float(best.get("cagr"), float("nan"))
        if math.isfinite(cagr_val):
            weighted_cagr_sum += weight * cagr_val
            weighted_cagr_weight += weight
            cagr_values.append(cagr_val)

        mdd_val = _safe_float(best.get("mdd"), float("nan"))
        if math.isfinite(mdd_val):
            weighted_mdd_sum += weight * mdd_val
            weighted_mdd_weight += weight
            mdd_values.append(mdd_val)

        period_return_val = _safe_float(best.get("period_return"), float("nan"))
        period_return_display = _round_float_places(period_return_val, 2) if math.isfinite(period_return_val) else None
        cagr_display = _round_float_places(cagr_val, 2) if math.isfinite(cagr_val) else None
        mdd_display = _round_float_places(-mdd_val, 2) if math.isfinite(mdd_val) else None

        # 추가 지표 추출
        sharpe_val = _safe_float(best.get("sharpe"), float("nan"))
        sharpe_display = _round_float_places(sharpe_val, 2) if math.isfinite(sharpe_val) else None

        sharpe_to_mdd_val = _safe_float(best.get("sharpe_to_mdd"), float("nan"))
        sharpe_to_mdd_display = _round_float_places(sharpe_to_mdd_val, 2) if math.isfinite(sharpe_to_mdd_val) else None

        def _to_int(val: Any) -> Optional[int]:
            try:
                return int(val)
            except (TypeError, ValueError):
                return None

        def _to_float(val: Any) -> Optional[float]:
            try:
                num = float(val)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(num):
                return None
            return num

        tuning_snapshot: Dict[str, Any] = {}
        for field, key in (
            ("MA_PERIOD", "ma_period"),
            ("PORTFOLIO_TOPN", "portfolio_topn"),
            ("REPLACE_SCORE_THRESHOLD", "replace_threshold"),
            ("OVERBOUGHT_SELL_THRESHOLD", "rsi_sell_threshold"),
            ("COOLDOWN_DAYS", "cooldown_days"),
        ):
            value = best.get(key)
            if value is None:
                continue
            if field == "REPLACE_SCORE_THRESHOLD":
                rounded_up = _round_up_float_places(value, 1)
                if math.isfinite(rounded_up):
                    tuning_snapshot[field] = rounded_up
            else:
                converted = _to_int(value)
                if converted is not None:
                    tuning_snapshot[field] = converted

        raw_data_payload.append(
            {
                "MONTHS_RANGE": months,
                "CAGR": cagr_display,
                "MDD": mdd_display,
                "period_return": period_return_display,
                "sharpe": sharpe_display,
                "sharpe_to_mdd": sharpe_to_mdd_display,
                "tuning": tuning_snapshot,
            }
        )

    if weighted_cagr_weight > 0:
        entry["weighted_expected_CAGR"] = _round_float(weighted_cagr_sum / weighted_cagr_weight)
    elif cagr_values:
        entry["weighted_expected_CAGR"] = _round_float(sum(cagr_values) / len(cagr_values))

    if weighted_mdd_weight > 0:
        entry["weighted_expected_MDD"] = _round_float(-(weighted_mdd_sum / weighted_mdd_weight))
    elif mdd_values:
        entry["weighted_expected_MDD"] = _round_float(-(sum(mdd_values) / len(mdd_values)))

    if raw_data_payload:
        entry["raw_data"] = raw_data_payload

    result_values: Dict[str, Any] = entry["result"]

    for field, (key, is_int) in param_fields.items():
        values: List[float] = []
        weights: List[float] = []

        for item in months_results:
            best = item.get("best", {})
            value = best.get(key)
            if value is None:
                continue
            try:
                value_float = float(value)
            except (TypeError, ValueError):
                continue
            values.append(value_float)
            try:
                weights.append(float(item.get("weight", 0.0)))
            except (TypeError, ValueError):
                weights.append(0.0)

        if not values:
            continue

        weight_total = sum(w for w in weights if w > 0)
        if weight_total > 0:
            raw = sum(v * w for v, w in zip(values, weights)) / weight_total
        else:
            raw = sum(values) / len(values)

        final_value = int(round(raw)) if is_int else _round_up_float_places(raw, 1)
        result_values[field] = final_value

    return entry


def _read_existing_results(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return []

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return []

    if isinstance(data, list):
        return [_ensure_entry_schema(item) for item in data]
    return []


def _ensure_entry_schema(entry: Any) -> Dict[str, Any]:
    if not isinstance(entry, dict):
        return {}

    normalized = dict(entry)

    result_map: Dict[str, Any] = {}
    existing_result = normalized.get("result")
    if isinstance(existing_result, dict):
        result_map.update(existing_result)
    legacy_tuning = normalized.pop("tuning", None)
    if isinstance(legacy_tuning, dict):
        result_map.update(legacy_tuning)

    normalized.pop("result", None)

    for field in ("MA_PERIOD", "PORTFOLIO_TOPN", "REPLACE_SCORE_THRESHOLD"):
        normalized.pop(field, None)

    raw_results = normalized.get("raw_data")
    legacy_results = normalized.pop("results", None)
    if not isinstance(raw_results, list):
        raw_results = []
    if isinstance(legacy_results, list):
        raw_results.extend(legacy_results)

    cleaned_results: List[Dict[str, Any]] = []
    cagr_values: List[float] = []
    mdd_positive_values: List[float] = []

    for item in raw_results:
        if not isinstance(item, dict):
            continue

        cleaned: Dict[str, Any] = {
            "MONTHS_RANGE": item.get("MONTHS_RANGE"),
            "tuning": item.get("tuning", {}),
        }

        cagr_val = _safe_float(item.get("CAGR"), float("nan"))
        if math.isfinite(cagr_val):
            cleaned["CAGR"] = _round_float_places(cagr_val, 2)
            cagr_values.append(cagr_val)
        else:
            cleaned["CAGR"] = None

        mdd_source = item.get("MDD")
        if mdd_source is None and item.get("mdd_pct") is not None:
            mdd_source = -_safe_float(item.get("mdd_pct"), float("nan"))
        mdd_val = _safe_float(mdd_source, float("nan"))
        if math.isfinite(mdd_val):
            cleaned["MDD"] = _round_float_places(mdd_val, 2)
            mdd_positive_values.append(abs(mdd_val))
        else:
            cleaned["MDD"] = None

        period_val = _safe_float(item.get("period_return"), float("nan"))
        cleaned["period_return"] = _round_float_places(period_val, 2) if math.isfinite(period_val) else None

        cleaned_results.append(cleaned)

    def _normalize_float(value: Any) -> Optional[float]:
        try:
            num = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(num):
            return None
        return num

    weighted_cagr = _normalize_float(normalized.pop("weighted_expected_cagr", None))
    weighted_cagr = _normalize_float(normalized.pop("weighted_expected_CAGR", weighted_cagr))

    weighted_mdd = _normalize_float(normalized.pop("weighted_expected_MDD", None))

    if weighted_cagr is None and cagr_values:
        weighted_cagr = sum(cagr_values) / len(cagr_values)

    if weighted_mdd is None and mdd_positive_values:
        weighted_mdd = -(sum(mdd_positive_values) / len(mdd_positive_values))

    ordered: Dict[str, Any] = {}
    ordered["run_date"] = normalized.get("run_date")
    ordered["result"] = result_map

    if weighted_cagr is not None:
        ordered["weighted_expected_CAGR"] = _round_float(weighted_cagr)

    if weighted_mdd is not None:
        ordered["weighted_expected_MDD"] = _round_float(weighted_mdd)

    if cleaned_results:
        ordered["raw_data"] = cleaned_results

    if ordered.get("run_date") is None:
        ordered.pop("run_date", None)

    return ordered


def _compose_tuning_report(
    account_id: str,
    *,
    month_results: List[Dict[str, Any]],
    progress_info: Optional[Dict[str, Any]] = None,
    tuning_metadata: Optional[Dict[str, Any]] = None,
) -> List[str]:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: List[str] = [
        f"실행 시각: {timestamp}",
        f"계정: {account_id.upper()}",
    ]

    if progress_info:
        completed = progress_info.get("completed", 0)
        total = progress_info.get("total", 0)
        if total > 0:
            pct = (completed / total) * 100
            lines.append(f"진행률: {completed}/{total} ({pct:.1f}%) - 중간 결과")

    # 튜닝 메타 정보 추가
    if tuning_metadata:
        lines.append("")
        lines.append("=== 튜닝 설정 ===")

        # 탐색 공간
        search_space = tuning_metadata.get("search_space", {})
        if search_space:
            ma_range = search_space.get("MA_RANGE", [])
            ma_type_range = search_space.get("MA_TYPE", [])
            topn_range = search_space.get("PORTFOLIO_TOPN", [])
            threshold_range = search_space.get("REPLACE_SCORE_THRESHOLD", [])
            rsi_range = search_space.get("OVERBOUGHT_SELL_THRESHOLD", [])
            cooldown_range = search_space.get("COOLDOWN_DAYS", [])

            # MA_TYPE이 있으면 포함해서 표시
            if ma_type_range and len(ma_type_range) > 1:
                lines.append(
                    f"탐색 공간: MA {len(ma_range)}개 × MA타입 {len(ma_type_range)}개 × TOPN {len(topn_range)}개 × TH {len(threshold_range)}개 × RSI {len(rsi_range)}개 × COOLDOWN {len(cooldown_range)}개 = {tuning_metadata.get('combo_count', 0)}개 조합"
                )
            else:
                lines.append(
                    f"탐색 공간: MA {len(ma_range)}개 × TOPN {len(topn_range)}개 × TH {len(threshold_range)}개 × RSI {len(rsi_range)}개 × COOLDOWN {len(cooldown_range)}개 = {tuning_metadata.get('combo_count', 0)}개 조합"
                )

            # 각 파라미터 범위 표시
            if ma_range:
                ma_min, ma_max = min(ma_range), max(ma_range)
                lines.append(f"  MA_RANGE: {ma_min}~{ma_max}")
            if ma_type_range:
                lines.append(f"  MA_TYPE: {', '.join(ma_type_range)}")
            if topn_range:
                topn_min, topn_max = min(topn_range), max(topn_range)
                lines.append(f"  PORTFOLIO_TOPN: {topn_min}~{topn_max}")
            if threshold_range:
                th_min, th_max = min(threshold_range), max(threshold_range)
                lines.append(f"  REPLACE_SCORE_THRESHOLD: {th_min}~{th_max}")
            if rsi_range:
                rsi_min, rsi_max = min(rsi_range), max(rsi_range)
                lines.append(f"  OVERBOUGHT_SELL_THRESHOLD: {rsi_min}~{rsi_max}")
            if cooldown_range:
                cd_min, cd_max = min(cooldown_range), max(cooldown_range)
                lines.append(f"  COOLDOWN_DAYS: {cd_min}~{cd_max}")

            # CORE_HOLDINGS 표시 (빈 리스트도 표시)
            core_holdings = search_space.get("CORE_HOLDINGS", [])
            if core_holdings is not None:
                if core_holdings:
                    # 종목명 가져오기
                    from utils.stock_list_io import get_etfs

                    try:
                        # tuning_metadata에서 country_code 추출
                        lookup_country = tuning_metadata.get("country_code", "kor") if tuning_metadata else "kor"
                        etf_list = get_etfs(lookup_country)
                        ticker_to_name = {str(etf.get("ticker")): etf.get("name", "") for etf in etf_list if etf.get("ticker")}

                        core_holdings_display = []
                        for ticker in core_holdings:
                            name = ticker_to_name.get(str(ticker), "")
                            if name:
                                core_holdings_display.append(f"{name}({ticker})")
                            else:
                                core_holdings_display.append(str(ticker))

                        lines.append(f"  CORE_HOLDINGS: {', '.join(core_holdings_display)}")
                    except Exception as e:
                        # 종목명을 가져오지 못하면 티커만 표시
                        logger = get_app_logger()
                        logger.debug(f"[튜닝] CORE_HOLDINGS 종목명 조회 실패: {e}")
                        lines.append(f"  CORE_HOLDINGS: {', '.join(map(str, core_holdings))}")
                else:
                    # 빈 리스트인 경우
                    lines.append("  CORE_HOLDINGS: (없음)")

        # 종목 수
        ticker_count = tuning_metadata.get("ticker_count", 0)
        if ticker_count > 0:
            lines.append(f"대상 종목: {ticker_count}개")

        # 제외된 종목
        excluded_tickers = tuning_metadata.get("excluded_tickers", [])
        if excluded_tickers:
            lines.append(f"제외된 종목: {len(excluded_tickers)}개 ({', '.join(excluded_tickers)})")

        # 테스트 기간
        test_periods = tuning_metadata.get("test_periods", [])
        data_period = tuning_metadata.get("data_period", {})
        if test_periods:
            # 날짜 정보가 있으면 함께 표시
            if data_period:
                start_date = data_period.get("start_date", "")
                end_date = data_period.get("end_date", "")
                period_strs = [f"{start_date} ~ {end_date} ({p}개월)" for p in test_periods]
                lines.append(f"테스트 기간: {', '.join(period_strs)}")
            else:
                period_str = ", ".join([f"{p}개월" for p in test_periods])
                lines.append(f"테스트 기간: {period_str}")

    lines.append("")

    # 최적화 지표 가져오기
    optimization_metric = None
    if tuning_metadata:
        search_space = tuning_metadata.get("search_space", {})
        optimization_metric = search_space.get("OPTIMIZATION_METRIC")

    # OPTIMIZATION_METRIC이 없으면 에러
    if not optimization_metric:
        raise ValueError("OPTIMIZATION_METRIC이 tuning_metadata에 없습니다.")

    # 정렬 키 함수 정의
    def _get_sort_key(row):
        if optimization_metric == "CAGR":
            return _safe_float(row.get("cagr"), float("-inf"))
        elif optimization_metric == "SHARPE":
            return _safe_float(row.get("sharpe"), float("-inf"))
        else:  # SDR
            return _safe_float(row.get("sharpe_to_mdd"), float("-inf"))

    # 지표 이름 매핑
    metric_names = {"CAGR": "CAGR", "SHARPE": "Sharpe", "SDR": "SDR(Sharpe/MDD)"}
    metric_display = metric_names.get(optimization_metric, "SDR(Sharpe/MDD)")

    for item in sorted(month_results, key=lambda x: int(x.get("months_range", 0))):
        months_range = item.get("months_range")
        if months_range is None:
            continue

        raw_rows = item.get("raw_data") or []
        normalized_rows: List[Dict[str, Any]] = []

        for entry in raw_rows:
            tuning = entry.get("tuning") or {}
            ma_val = tuning.get("MA_PERIOD")
            ma_type_val = tuning.get("MA_TYPE", "SMA")
            topn_val = tuning.get("PORTFOLIO_TOPN")
            threshold_val = tuning.get("REPLACE_SCORE_THRESHOLD")
            rsi_val = tuning.get("OVERBOUGHT_SELL_THRESHOLD")
            cooldown_val = tuning.get("COOLDOWN_DAYS")

            cagr_val = entry.get("CAGR")
            mdd_val = entry.get("MDD")
            period_val = entry.get("period_return")
            sharpe_val = entry.get("sharpe")
            sharpe_to_mdd_val = entry.get("sharpe_to_mdd")

            normalized_rows.append(
                {
                    "ma_period": ma_val,
                    "ma_type": ma_type_val,
                    "portfolio_topn": topn_val,
                    "replace_threshold": threshold_val,
                    "rsi_sell_threshold": rsi_val,
                    "cooldown_days": cooldown_val,
                    "cagr": cagr_val,
                    "mdd": mdd_val,
                    "period_return": period_val,
                    "sharpe": sharpe_val,
                    "sharpe_to_mdd": sharpe_to_mdd_val,
                }
            )

        normalized_rows.sort(key=_get_sort_key, reverse=True)
        lines.append(f"=== 최근 {months_range}개월 결과 - 정렬 기준: {metric_display} ===")
        lines.extend(_render_tuning_table(normalized_rows, months_range=months_range))
        lines.append("")

    return lines


def _save_intermediate_results(
    output_path: Path,
    *,
    account_id: str,
    month_results: List[Dict[str, Any]],
    progress_info: Optional[Dict[str, Any]] = None,
    tuning_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """중간 결과를 임시 파일에 쓰고 atomic rename으로 안전하게 저장합니다."""
    try:
        report_lines = _compose_tuning_report(
            account_id,
            month_results=month_results,
            progress_info=progress_info,
            tuning_metadata=tuning_metadata,
        )

        # 임시 파일에 먼저 쓰기
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output_path.parent, delete=False, suffix=".tmp") as tmp_file:
            tmp_file.write("\n".join(report_lines) + "\n")
            tmp_path = Path(tmp_file.name)

        # Atomic rename (기존 파일 덮어쓰기)
        shutil.move(str(tmp_path), str(output_path))
    except Exception as e:
        # 중간 저장 실패 로그 출력
        logger = get_app_logger()
        logger.warning("[튜닝] 중간 결과 저장 실패: %s", e)
        if "tmp_path" in locals() and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def run_account_tuning(
    account_id: str,
    *,
    output_path: Optional[Path | str] = None,
    results_dir: Optional[Path | str] = None,
    tuning_config: Optional[Dict[str, Dict[str, Any]]] = None,
    months_range: Optional[int] = None,
    debug_export_dir: Optional[Path | str] = None,
    debug_capture_top_n: int = 1,
) -> Optional[Path]:
    """Execute parameter tuning for the given account and return the output path."""

    account_norm = (account_id or "").strip().lower()
    logger = get_app_logger()

    try:
        account_settings = get_account_settings(account_norm)
    except AccountSettingsError as exc:
        logger.error("[튜닝] 계정 설정 로딩 실패: %s", exc)
        return None

    config_map = tuning_config or {}
    config = config_map.get(account_norm)
    if not config:
        logger.warning("[튜닝] '%s' 계정에 대한 튜닝 설정이 없습니다.", account_norm.upper())
        return None

    # country_code는 ETF 리스트 조회용으로 필요
    country_code = (account_settings.get("country_code") or account_norm).strip().lower()

    debug_dir: Optional[Path] = None
    capture_top_n = 0
    if debug_export_dir is not None:
        debug_dir = Path(debug_export_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        try:
            capture_top_n = max(1, int(debug_capture_top_n))
        except (TypeError, ValueError):
            capture_top_n = 1
        logger.info("[튜닝] 디버그 아티팩트를 '%s'에 저장합니다.", debug_dir)
    debug_diff_rows: List[Dict[str, Any]] = []
    debug_month_configs: List[Dict[str, Any]] = []

    base_rules = get_strategy_rules(account_norm)
    ma_values = _normalize_tuning_values(config.get("MA_RANGE"), dtype=int, fallback=base_rules.ma_period)
    topn_values = _normalize_tuning_values(config.get("PORTFOLIO_TOPN"), dtype=int, fallback=base_rules.portfolio_topn)
    replace_values = _normalize_tuning_values(
        config.get("REPLACE_SCORE_THRESHOLD"),
        dtype=float,
        fallback=base_rules.replace_threshold,
    )
    rsi_sell_values = _normalize_tuning_values(
        config.get("OVERBOUGHT_SELL_THRESHOLD"),
        dtype=int,
        fallback=10,
    )
    cooldown_values = _normalize_tuning_values(
        config.get("COOLDOWN_DAYS"),
        dtype=int,
        fallback=2,
    )

    # CORE_HOLDINGS 처리: tune.py에서 지정 가능, 없으면 base_rules에서 가져옴
    core_holdings_raw = config.get("CORE_HOLDINGS")
    if core_holdings_raw is None:
        core_holdings = base_rules.core_holdings or []
    elif isinstance(core_holdings_raw, (list, tuple)):
        core_holdings = [str(v).strip() for v in core_holdings_raw if v]
    else:
        core_holdings = []

    # MA_TYPE 처리: 문자열 리스트로 받음
    ma_type_raw = config.get("MA_TYPE")
    if ma_type_raw is None:
        ma_type_values = [base_rules.ma_type]
    elif isinstance(ma_type_raw, (list, tuple)):
        ma_type_values = [str(v).upper() for v in ma_type_raw if v]
    else:
        ma_type_values = [str(ma_type_raw).upper()]

    if not ma_type_values:
        ma_type_values = [base_rules.ma_type]

    if not ma_values or not topn_values or not replace_values or not rsi_sell_values or not cooldown_values or not ma_type_values:
        logger.warning("[튜닝] 유효한 파라미터 조합이 없습니다.")
        return None

    etf_universe = get_etfs(country_code)
    if not etf_universe:
        logger.error("[튜닝] '%s' 종목 데이터를 찾을 수 없습니다.", country_code)
        return None

    tickers = [str(item.get("ticker")) for item in etf_universe if item.get("ticker")]
    if not tickers:
        logger.error("[튜닝] '%s' 유효한 티커가 없습니다.", country_code)
        return None

    combo_count = len(ma_values) * len(topn_values) * len(replace_values) * len(rsi_sell_values) * len(cooldown_values) * len(ma_type_values)
    if combo_count <= 0:
        logger.warning("[튜닝] 조합 생성에 실패했습니다.")
        return None

    # OPTIMIZATION_METRIC 필수 확인
    optimization_metric = config.get("OPTIMIZATION_METRIC")
    if not optimization_metric:
        logger.error("[튜닝] '%s' 계정에 OPTIMIZATION_METRIC 설정이 없습니다. config.py에 추가해주세요.", account_norm.upper())
        return None

    search_space = {
        "MA_PERIOD": ma_values,
        "PORTFOLIO_TOPN": topn_values,
        "REPLACE_SCORE_THRESHOLD": replace_values,
        "OVERBOUGHT_SELL_THRESHOLD": rsi_sell_values,
        "COOLDOWN_DAYS": cooldown_values,
        "MA_TYPE": ma_type_values,
        "CORE_HOLDINGS": core_holdings,
        "OPTIMIZATION_METRIC": optimization_metric,
    }

    ma_count = len(ma_values)
    topn_count = len(topn_values)
    replace_count = len(replace_values)
    rsi_count = len(rsi_sell_values)
    cooldown_count = len(cooldown_values)
    logger.info(
        "[튜닝] 탐색 공간: MA %d개 × TOPN %d개 × TH %d개 × RSI %d개 × COOLDOWN %d개 = %d개 조합",
        ma_count,
        topn_count,
        replace_count,
        rsi_count,
        cooldown_count,
        combo_count,
    )

    try:
        ma_max = max([base_rules.ma_period, *ma_values])
    except ValueError:
        ma_max = base_rules.ma_period

    month_items = _resolve_month_configs(months_range, account_id=account_id)
    if not month_items:
        logger.error("[튜닝] 테스트할 기간 설정이 없습니다.")
        return None

    valid_month_ranges = [
        int(item.get("months_range", 0))
        for item in month_items
        if isinstance(item.get("months_range"), (int, float)) and int(item.get("months_range", 0)) > 0
    ]
    if not valid_month_ranges:
        logger.error("[튜닝] 유효한 기간 정보가 없습니다.")
        return None

    longest_months = max(valid_month_ranges)

    end_date = get_latest_trading_day(country_code)
    if not isinstance(end_date, pd.Timestamp):
        end_date = pd.Timestamp.now().normalize()

    start_date_prefetch = end_date - pd.DateOffset(months=longest_months)
    date_range_prefetch = [
        start_date_prefetch.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
    ]

    warmup_days = int(max(ma_max, base_rules.ma_period) * 1.5)

    logger.info(
        "[튜닝] 데이터 프리패치: 티커 %d개, 기간 %s~%s, 웜업 %d일",
        len(tickers),
        date_range_prefetch[0],
        date_range_prefetch[1],
        warmup_days,
    )

    common_settings = load_common_settings()
    cache_seed_raw = (common_settings or {}).get("CACHE_START_DATE")
    cache_seed_dt = None
    if cache_seed_raw:
        try:
            cache_seed_dt = pd.to_datetime(cache_seed_raw).normalize()
        except Exception:
            cache_seed_dt = None

    if cache_seed_dt is not None and cache_seed_dt < start_date_prefetch:
        start_date_prefetch = cache_seed_dt
        date_range_prefetch = [start_date_prefetch.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")]

    prefetched, missing_prefetch = prepare_price_data(
        tickers=tickers,
        country=country_code,
        start_date=date_range_prefetch[0],
        end_date=date_range_prefetch[1],
        warmup_days=warmup_days,
        skip_realtime=True,
    )
    prefetched_map: Dict[str, DataFrame] = dict(prefetched)

    for ticker, frame in prefetched_map.items():
        save_cached_frame(country_code, ticker, frame)

    excluded_ticker_set: set[str] = {str(ticker).strip().upper() for ticker in missing_prefetch if isinstance(ticker, str) and str(ticker).strip()}
    if excluded_ticker_set:
        logger.warning(
            "[튜닝] %s 데이터 부족으로 제외할 종목 (%d): %s",
            account_norm.upper(),
            len(excluded_ticker_set),
            ", ".join(sorted(excluded_ticker_set)),
        )

    if debug_dir is not None:
        _export_prefetched_data(debug_dir, prefetched_map)

    runtime_missing_registry: Set[str] = set()
    results_per_month: List[Dict[str, Any]] = []

    # 출력 경로 미리 결정 (중간 저장용) - 계정별 폴더
    if results_dir is not None:
        base_dir = Path(results_dir) / account_norm
    else:
        base_dir = DEFAULT_RESULTS_DIR / account_norm

    if output_path is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
        txt_path = base_dir / f"tune_{date_str}.log"
    else:
        txt_path = Path(output_path)
        if txt_path.suffix.lower() not in (".log", ".txt"):
            txt_path = txt_path.with_suffix(".log")
        if not txt_path.is_absolute():
            txt_path = Path.cwd() / txt_path
    txt_path.parent.mkdir(parents=True, exist_ok=True)

    # 튜닝 메타데이터 생성
    tuning_metadata = {
        "combo_count": combo_count,
        "country_code": country_code,
        "search_space": {
            "MA_RANGE": list(ma_values),
            "MA_TYPE": list(ma_type_values),
            "PORTFOLIO_TOPN": list(topn_values),
            "REPLACE_SCORE_THRESHOLD": list(replace_values),
            "OVERBOUGHT_SELL_THRESHOLD": list(rsi_sell_values),
            "COOLDOWN_DAYS": list(cooldown_values),
            "CORE_HOLDINGS": list(core_holdings) if core_holdings else [],
            "OPTIMIZATION_METRIC": optimization_metric,
        },
        "data_period": {
            "start_date": date_range_prefetch[0],
            "end_date": date_range_prefetch[1],
        },
        "ticker_count": len(tickers),
        "excluded_tickers": sorted(excluded_ticker_set),
        "test_periods": valid_month_ranges,
    }

    for item in month_items:
        months_raw = item.get("months_range")
        try:
            months_value = int(months_raw)
        except (TypeError, ValueError):
            logger.warning(
                "[튜닝] %s (%s) 월 범위를 정수로 변환할 수 없습니다. 항목을 건너뜁니다.",
                account_norm.upper(),
                months_raw,
            )
            continue

        if months_value <= 0:
            logger.warning(
                "[튜닝] %s (%s) 유효하지 않은 월 범위입니다. 항목을 건너뜁니다.",
                account_norm.upper(),
                months_raw,
            )
            continue

        sanitized_item = dict(item)
        sanitized_item["months_range"] = months_value
        if debug_dir is not None:
            debug_month_configs.append(
                {
                    "months_range": months_value,
                    "weight": float(item.get("weight", 0.0) or 0.0),
                    "source": item.get("source"),
                }
            )

        # Adjust prefetched window to honor CACHE_START_DATE if provided
        if cache_seed_dt is not None and cache_seed_dt < start_date_prefetch:
            adjusted_start = cache_seed_dt
        else:
            adjusted_start = start_date_prefetch

        adjusted_date_range = [adjusted_start.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")]

        prefetched_adjusted, additional_missing = prepare_price_data(
            tickers=tickers,
            country=country_code,
            start_date=adjusted_date_range[0],
            end_date=adjusted_date_range[1],
            warmup_days=warmup_days,
            skip_realtime=True,
        )
        prefetched_map.update(prefetched_adjusted)
        for ticker, frame in prefetched_adjusted.items():
            save_cached_frame(country_code, ticker, frame)
        missing_prefetch.extend(additional_missing)

        # 최적화 지표에 따른 정렬 함수 정의
        optimization_metric = search_space.get("OPTIMIZATION_METRIC").upper()

        def _sort_key_local(entry):
            """최적화 지표에 따른 정렬 키"""
            if optimization_metric == "CAGR":
                return _safe_float(entry.get("cagr"), float("-inf"))
            elif optimization_metric == "SHARPE":
                return _safe_float(entry.get("sharpe"), float("-inf"))
            else:  # SDR (default)
                return _safe_float(entry.get("sharpe_to_mdd"), float("-inf"))

        # 중간 저장 콜백 함수 정의
        def save_progress_callback(success_entries, progress_pct, completed, total):
            """1%마다 호출되는 중간 저장 콜백"""
            # 현재까지의 결과로 임시 결과 생성
            temp_result = {
                "months_range": months_value,
                "best": success_entries[0] if success_entries else {},
                "weight": item.get("weight", 0.0),
                "source": item.get("source"),
                "raw_data": [
                    {
                        "MONTHS_RANGE": months_value,
                        "CAGR": _round_float_places(entry.get("cagr", 0.0), 2),
                        "MDD": _round_float_places(-entry.get("mdd", 0.0), 2),
                        "period_return": _round_float_places(entry.get("period_return", 0.0), 2),
                        "sharpe": _round_float_places(entry.get("sharpe", 0.0), 2),
                        "sharpe_to_mdd": _round_float_places(entry.get("sharpe_to_mdd", 0.0), 3),
                        "tuning": {
                            "MA_PERIOD": int(entry.get("ma_period", 0)),
                            "MA_TYPE": str(entry.get("ma_type", "SMA")),
                            "PORTFOLIO_TOPN": int(entry.get("portfolio_topn", 0)),
                            "REPLACE_SCORE_THRESHOLD": _round_up_float_places(entry.get("replace_threshold", 0.0), 1),
                            "OVERBOUGHT_SELL_THRESHOLD": int(entry.get("rsi_sell_threshold", 10)),
                            "COOLDOWN_DAYS": int(entry.get("cooldown_days", 2)),
                        },
                    }
                    for entry in sorted(success_entries, key=_sort_key_local, reverse=True)
                ],
            }

            temp_results = results_per_month + [temp_result]
            _save_intermediate_results(
                txt_path,
                account_id=account_norm,
                month_results=temp_results,
                progress_info={
                    "completed": completed,
                    "total": total,
                    "months_range": months_value,
                    "progress_pct": progress_pct,
                },
                tuning_metadata=tuning_metadata,
            )

        single_result = _execute_tuning_for_months(
            account_norm,
            months_range=months_value,
            search_space=search_space,
            end_date=end_date,
            excluded_tickers=excluded_ticker_set,
            prefetched_data=prefetched_map,
            output_path=txt_path,
            progress_callback=save_progress_callback,
        )

        if not single_result:
            continue

        try:
            weight = float(item.get("weight", 0.0))
        except (TypeError, ValueError):
            weight = 0.0

        single_result["weight"] = weight
        single_result["source"] = item.get("source")
        missing_in_result = single_result.get("missing_tickers") or []
        if missing_in_result:
            runtime_missing_registry.update(missing_in_result)
        results_per_month.append(single_result)

        # 각 기간 완료 시마다 중간 결과 저장
        if results_per_month:
            _save_intermediate_results(
                txt_path,
                account_id=account_norm,
                month_results=results_per_month,
                progress_info={
                    "completed": len(results_per_month),
                    "total": len(month_items),
                },
                tuning_metadata=tuning_metadata,
            )
            logger.info(
                "[튜닝] %s 중간 결과 저장 완료 (%d/%d 기간)",
                account_norm.upper(),
                len(results_per_month),
                len(month_items),
            )

        if debug_dir is not None and capture_top_n > 0:
            raw_rows = single_result.get("raw_data") or []
            debug_diff_rows.extend(
                _export_debug_month(
                    debug_dir,
                    account_id=account_norm,
                    months_range=months_value,
                    raw_rows=raw_rows,
                    prefetched_data=prefetched_map,
                    capture_top_n=capture_top_n,
                )
            )

    if not results_per_month:
        logger.warning("[튜닝] 실행 가능한 기간이 없어 결과가 없습니다.")
        return None

    if runtime_missing_registry:
        unseen_missing = sorted(set(runtime_missing_registry) - set(excluded_ticker_set or []))
        if unseen_missing:
            logger.warning(
                "[튜닝] %s 실행 중 데이터가 부족해 제외된 추가 종목 (%d): %s",
                account_norm.upper(),
                len(unseen_missing),
                ", ".join(unseen_missing),
            )

    # 최적화 지표 가져오기
    optimization_metric = search_space.get("OPTIMIZATION_METRIC").upper()

    for item in results_per_month:
        best = item.get("best", {})
        logger.info(
            "[튜닝] %s (%d개월) 최적 조합 (%s 기준): MA=%d / TOPN=%d / TH=%.3f / RSI=%d / COOLDOWN=%d / CAGR=%.2f%% / Sharpe=%.2f / SDR=%.3f",
            account_norm.upper(),
            item.get("months_range"),
            optimization_metric,
            best.get("ma_period", 0),
            best.get("portfolio_topn", 0),
            best.get("replace_threshold", 0.0),
            best.get("rsi_sell_threshold", 10),
            best.get("cooldown_days", 2),
            best.get("cagr_pct", 0.0),
            best.get("sharpe", 0.0),
            best.get("sharpe_to_mdd", 0.0),
        )

    entry = _build_run_entry(months_results=results_per_month)

    if debug_dir is not None:
        meta_payload: Dict[str, Any] = {
            "account": account_norm,
            "country": country_code,
            "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "combo_count": combo_count,
            "search_space": {
                "MA_RANGE": ma_values,
                "PORTFOLIO_TOPN": topn_values,
                "REPLACE_SCORE_THRESHOLD": replace_values,
                "OVERBOUGHT_SELL_THRESHOLD": rsi_sell_values,
            },
            "month_configs": debug_month_configs,
            "excluded_tickers_initial": sorted(excluded_ticker_set),
            "runtime_missing_tickers": sorted(runtime_missing_registry),
            "debug_capture_top_n": capture_top_n,
            "tuning_entry": entry,
        }
        (debug_dir / "meta.json").write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8")

        if debug_diff_rows:
            summary_fields = [
                "months_range",
                "ma_period",
                "topn",
                "threshold",
                "recorded_cagr",
                "prefetch_cagr",
                "live_cagr",
                "prefetch_minus_recorded",
                "live_minus_recorded",
                "recorded_mdd",
                "prefetch_mdd",
                "live_mdd",
                "recorded_period_return",
                "prefetch_period_return",
                "live_period_return",
                "current_start_date",
                "current_end_date",
                "current_excluded_count",
                "artifact_path",
            ]

            (debug_dir / "diff_summary.json").write_text(json.dumps(debug_diff_rows, ensure_ascii=False, indent=2), encoding="utf-8")

            with (debug_dir / "diff_summary.csv").open("w", newline="", encoding="utf-8") as fp:
                writer = csv.DictWriter(fp, fieldnames=summary_fields)
                writer.writeheader()
                writer.writerows(debug_diff_rows)
    report_lines = _compose_tuning_report(
        account_norm,
        month_results=results_per_month,
        tuning_metadata=tuning_metadata,
    )

    # Remove only the exact output file if it exists (preserve backup copies)
    try:
        if txt_path.exists():
            txt_path.unlink()
    except OSError:
        pass

    txt_path.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    for line in report_lines:
        print(line)

    logger.info("튜닝 요약을 '%s'에 기록했습니다.", txt_path)

    return txt_path


__all__ = ["run_account_tuning"]
