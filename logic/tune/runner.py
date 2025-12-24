"""Entry points for running country-level parameter tuning."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
import tempfile
from collections.abc import Collection, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from os import cpu_count
from pathlib import Path
from typing import Any

import pandas as pd
from pandas import DataFrame, Timestamp

from config import TUNING_ENSEMBLE_SIZE
from logic.backtest.account import run_account_backtest
from logic.entry_point import StrategyRules
from utils.account_registry import get_benchmark_tickers, get_strategy_rules
from utils.cache_utils import save_cached_frame
from utils.data_loader import (
    MissingPriceDataError,
    get_latest_trading_day,
    get_trading_days,
    prepare_price_data,
)
from utils.logger import get_app_logger
from utils.settings_loader import (
    ACCOUNT_SETTINGS_DIR,
    AccountSettingsError,
    get_account_settings,
    get_tune_month_configs,
    load_common_settings,
)
from utils.stock_list_io import get_etfs

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[2] / "zaccounts"
WORKERS = None  # 병렬 실행 프로세스 수 (None이면 CPU 개수 기반 자동 결정)
MAX_TABLE_ROWS = 50

# Worker 글로벌 변수 - 프로세스당 한 번만 초기화
_WORKER_PREFETCHED_DATA: Mapping[str, DataFrame] | None = None
_WORKER_PREFETCHED_METRICS: Mapping[str, dict[str, Any]] | None = None
_WORKER_PREFETCHED_UNIVERSE: Sequence[Mapping[str, Any]] | None = None
_WORKER_TRADING_CALENDAR: Sequence[pd.Timestamp] | None = None
_WORKER_PREFETCHED_FX_SERIES: pd.Series | None = None


def _init_worker_prefetch(
    prefetched_data: Mapping[str, DataFrame],
    prefetched_metrics: Mapping[str, dict[str, Any]],
    prefetched_universe: Sequence[Mapping[str, Any]],
    trading_calendar: Sequence[pd.Timestamp],
    fx_series: pd.Series | None = None,
) -> None:
    """ProcessPoolExecutor initializer - 각 worker 프로세스당 한 번만 실행"""
    global \
        _WORKER_PREFETCHED_DATA, \
        _WORKER_PREFETCHED_METRICS, \
        _WORKER_PREFETCHED_UNIVERSE, \
        _WORKER_TRADING_CALENDAR, \
        _WORKER_PREFETCHED_FX_SERIES
    _WORKER_PREFETCHED_DATA = prefetched_data
    _WORKER_PREFETCHED_METRICS = prefetched_metrics
    _WORKER_PREFETCHED_UNIVERSE = prefetched_universe
    _WORKER_TRADING_CALENDAR = trading_calendar
    _WORKER_PREFETCHED_FX_SERIES = fx_series


def _filter_trading_days(
    calendar: Sequence[pd.Timestamp] | None,
    start_str: str,
    end_str: str,
) -> list[pd.Timestamp] | None:
    if not calendar:
        return None
    start_ts = pd.to_datetime(start_str)
    end_ts = pd.to_datetime(end_str)
    filtered = []
    for raw in calendar:
        dt = pd.Timestamp(raw)
        if start_ts <= dt <= end_ts:
            filtered.append(dt)
    return filtered or None


def _extract_price_series_for_prefetch(
    df: pd.DataFrame,
) -> tuple[pd.Series | None, pd.Series | None]:
    if df is None or df.empty:
        return None, None

    working = df
    if isinstance(working.columns, pd.MultiIndex):
        working = working.copy()
        working.columns = working.columns.get_level_values(0)
        working = working.loc[:, ~working.columns.duplicated()]

    close_series = None
    for candidate in ("unadjusted_close", "Close", "close"):
        if candidate in working.columns:
            series = working[candidate]
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            close_series = series.astype(float)
            break
    if close_series is None:
        return None, None

    open_series = None
    if "Open" in working.columns:
        open_col = working["Open"]
        if isinstance(open_col, pd.DataFrame):
            open_col = open_col.iloc[:, 0]
        open_series = open_col.astype(float)

    return close_series, open_series


def _build_prefetched_metric_cache(
    prefetched_data: Mapping[str, pd.DataFrame],
    *,
    ma_periods: Sequence[int],
    ma_types: Sequence[str],
) -> dict[str, dict[str, Any]]:
    if not prefetched_data:
        return {}

    period_pool = sorted({int(p) for p in ma_periods if isinstance(p, (int, float)) and int(p) > 0})
    type_pool = sorted({(t or "SMA").upper() for t in ma_types if isinstance(t, str) and t})
    if not period_pool or not type_pool:
        return {}

    from strategies.rsi.backtest import process_ticker_data_rsi
    from utils.indicators import calculate_ma_score
    from utils.moving_averages import calculate_moving_average

    cache: dict[str, dict[str, Any]] = {}
    for ticker, df in prefetched_data.items():
        if df is None or df.empty:
            continue
        close_series, open_series = _extract_price_series_for_prefetch(df)
        if close_series is None or close_series.empty:
            continue

        entry: dict[str, Any] = {
            "close": close_series,
            "open": open_series if open_series is not None else close_series.copy(),
            "ma": {},
            "ma_score": {},
        }

        rsi_payload = process_ticker_data_rsi(close_series)
        if rsi_payload:
            entry["rsi_score"] = rsi_payload.get("rsi_score")

        for ma_type in type_pool:
            for period in period_pool:
                if len(close_series) < period:
                    continue
                ma_series = calculate_moving_average(close_series, period, ma_type)
                if ma_series is None:
                    continue
                ma_key = f"{ma_type}_{period}"
                entry["ma"][ma_key] = ma_series
                entry["ma_score"][ma_key] = calculate_ma_score(close_series, ma_series)

        cache[ticker] = entry

    return cache


def _normalize_tuning_values(values: Any, *, dtype, fallback: Any) -> list[Any]:
    if values is None:
        values = []
    if hasattr(values, "tolist"):
        values = values.tolist()
    elif isinstance(values, range):
        values = list(values)
    elif not isinstance(values, (list, tuple, set)):
        values = [values]

    normalized: list[Any] = []
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


def _resolve_month_configs(months_range: int | None, account_id: str = None) -> list[dict[str, Any]]:
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

    if account_id:
        try:
            account_settings = get_account_settings(account_id)
            fallback = account_settings.get("strategy", {}).get("MONTHS_RANGE")
            if fallback is not None:
                try:
                    fallback_val = int(fallback)
                    if fallback_val > 0:
                        return [
                            {
                                "months_range": fallback_val,
                                "weight": 1.0,
                                "source": f"account_{account_id}",
                            }
                        ]
                except (TypeError, ValueError):
                    pass
        except Exception:
            pass

    return []


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


def _render_tuning_table(
    rows: list[dict[str, Any]],
    *,
    include_samples: bool = False,
    months_range: int | None = None,
) -> list[str]:
    from utils.report import render_table_eaw

    headers = [
        "MA",
        "MA타입",
        "TOPN",
        "교체점수",
        "손절",
        "과매수",
        "쿨다운",
        "CAGR(%)",
        "MDD(%)",
    ]
    aligns = [
        "right",
        "center",
        "right",
        "right",
        "right",
        "right",
        "right",
        "right",
        "right",
        "right",
    ]

    if months_range:
        headers.append(f"{months_range}개월(%)")
    else:
        headers.append("기간수익률(%)")
    aligns.append("right")

    headers.extend(["Sharpe", "SDR(Sharpe/MDD)", "Turnover(매매회전율)"])
    aligns.extend(["right", "right", "right"])

    if include_samples:
        headers.append("Samples")
        aligns.append("right")

    table_rows = []
    for row in rows[:MAX_TABLE_ROWS]:
        ma_val = row.get("ma_period")
        ma_type_val = row.get("ma_type", "SMA")
        topn_val = row.get("portfolio_topn")
        threshold_val = row.get("replace_threshold")
        stop_loss_val = row.get("stop_loss_pct")
        stop_loss_num = _safe_float(stop_loss_val, float("nan"))
        if math.isfinite(stop_loss_num):
            stop_loss_display = f"{int(stop_loss_num)}%"
        else:
            stop_loss_display = "-"

            stop_loss_display = "-"

        rsi_threshold_val = row.get("rsi_sell_threshold")
        cooldown_val = row.get("cooldown_days")

        row_data = [
            str(int(ma_val)) if isinstance(ma_val, (int, float)) and math.isfinite(float(ma_val)) else "-",
            str(ma_type_val) if ma_type_val else "SMA",
            str(int(topn_val)) if isinstance(topn_val, (int, float)) and math.isfinite(float(topn_val)) else "-",
            str(int(threshold_val))
            if isinstance(threshold_val, (int, float)) and math.isfinite(float(threshold_val))
            else "-",
            stop_loss_display,
            str(int(rsi_threshold_val))
            if isinstance(rsi_threshold_val, (int, float)) and math.isfinite(float(rsi_threshold_val))
            else "-",
            str(int(cooldown_val))
            if isinstance(cooldown_val, (int, float)) and math.isfinite(float(cooldown_val))
            else "-",
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


def _apply_tuning_to_strategy_file(account_id: str, entry: dict[str, Any]) -> None:
    """Persist the final tuning result back into the account strategy file."""

    logger = get_app_logger()
    result_params = entry.get("result")
    if not isinstance(result_params, dict) or not result_params:
        logger.warning("[튜닝] %s 계정 결과에 반영할 파라미터가 없습니다.", account_id.upper())
        return

    settings_path = ACCOUNT_SETTINGS_DIR / account_id / "config.json"
    try:
        raw = settings_path.read_text(encoding="utf-8")
        settings_data = json.loads(raw)
    except Exception as exc:  # pragma: no cover - 파일 접근 오류
        logger.error(
            "[튜닝] %s 계정 설정을 읽지 못해 갱신을 건너뜁니다: %s",
            account_id.upper(),
            exc,
        )
        return

    if not isinstance(settings_data, dict):
        logger.error("[튜닝] %s 계정 설정 형식이 잘못되어 갱신을 건너뜁니다.", account_id.upper())
        return

    strategy_cfg = settings_data.get("strategy")
    if isinstance(strategy_cfg, dict):
        strategy_data = dict(strategy_cfg)
    else:
        strategy_data = {}

    legacy_tuning = strategy_data.pop("tuning", None)
    if isinstance(legacy_tuning, dict):
        strategy_data.update(legacy_tuning)

    integer_keys = {
        "REPLACE_SCORE_THRESHOLD",
        "STOP_LOSS_PCT",
        "COOLDOWN_DAYS",
        "PORTFOLIO_TOPN",
        "OVERBOUGHT_SELL_THRESHOLD",
        "MA_PERIOD",
    }

    for key, value in result_params.items():
        if value is None:
            continue
        strategy_data[key] = value

    # 정수형이어야 하는 필드들 강제 형변환 (튜닝 여부와 무관하게)
    for key in integer_keys:
        if key in strategy_data and strategy_data[key] is not None:
            try:
                strategy_data[key] = int(float(strategy_data[key]))
            except (ValueError, TypeError):
                pass

    weighted_cagr = entry.get("weighted_expected_CAGR")
    if weighted_cagr is not None:
        strategy_data["CAGR"] = _round_float(weighted_cagr)

    weighted_mdd = entry.get("weighted_expected_MDD")
    if weighted_mdd is not None:
        strategy_data["MDD"] = _round_float(weighted_mdd)

    strategy_data["BACKTESTED_DATE"] = datetime.now().strftime("%Y-%m-%d %H:%M")

    # [Key Reordering]
    # 사용자가 요청한 순서대로 키를 정렬하여 저장
    desired_order = [
        "MONTHS_RANGE",
        "BACKTESTED_DATE",
        "CAGR",
        "MDD",
        "PORTFOLIO_TOPN",
        "MA_PERIOD",
        "MA_TYPE",
        "REPLACE_SCORE_THRESHOLD",
        "STOP_LOSS_PCT",
        "OVERBOUGHT_SELL_THRESHOLD",
        "COOLDOWN_DAYS",
        "OPTIMIZATION_METRIC",
    ]

    ordered_strategy = {}

    # 1. Desired keys first
    for key in desired_order:
        if key in strategy_data:
            ordered_strategy[key] = strategy_data[key]

    # 2. Remaining keys
    for key, val in strategy_data.items():
        if key not in ordered_strategy:
            ordered_strategy[key] = val

    strategy_data = ordered_strategy

    settings_data["strategy"] = strategy_data

    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(settings_path.parent),
            delete=False,
            suffix=".tmp",
        ) as tmp_file:
            json.dump(settings_data, tmp_file, ensure_ascii=False, indent=4)
            tmp_file.write("\n")
            tmp_path = Path(tmp_file.name)
        shutil.move(str(tmp_path), str(settings_path))
        get_account_settings.cache_clear()
        logger.info("[튜닝] %s 계정 전략 설정을 최신 결과로 갱신했습니다.", account_id.upper())
    except Exception as exc:  # pragma: no cover - 파일 쓰기 오류
        logger.error("[튜닝] %s 계정 설정을 갱신하지 못했습니다: %s", account_id.upper(), exc)
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


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


def _export_debug_month(
    debug_dir: Path,
    *,
    account_id: str,
    months_range: int,
    raw_rows: list[dict[str, Any]],
    prefetched_data: Mapping[str, DataFrame],
    capture_top_n: int,
    prefetched_etf_universe: Sequence[Mapping[str, Any]],
    prefetched_metrics: Mapping[str, dict[str, Any]] | None,
    trading_calendar: Sequence[pd.Timestamp] | None,
) -> list[dict[str, Any]]:
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

    summary_rows: list[dict[str, Any]] = []
    month_dir = debug_dir / f"months_{months_range:02d}"

    for idx, row in enumerate(sorted_rows[:capture_top_n], 1):
        tuning = row.get("tuning") or {}
        try:
            ma = int(tuning.get("MA_PERIOD"))
            topn = int(tuning.get("PORTFOLIO_TOPN"))
            threshold = float(tuning.get("REPLACE_SCORE_THRESHOLD"))
            stop_loss = tuning.get("STOP_LOSS_PCT")
            stop_loss_value = float(stop_loss) if stop_loss is not None else None
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
            stop_loss_pct=stop_loss_value,
        )

        result_prefetch = run_account_backtest(
            account_id,
            months_range=months_range,
            quiet=True,
            prefetched_data=prefetched_data,
            strategy_override=strategy_rules,
            prefetched_etf_universe=prefetched_etf_universe,
            prefetched_metrics=prefetched_metrics,
            trading_calendar=trading_calendar,
        )

        result_live = run_account_backtest(
            account_id,
            months_range=months_range,
            quiet=True,
            strategy_override=strategy_rules,
            prefetched_etf_universe=prefetched_etf_universe,
            prefetched_metrics=None,
            trading_calendar=trading_calendar,
        )

        stop_loss_dir_part = f"SL{stop_loss_value:.2f}" if stop_loss_value is not None else "SLauto"
        combo_dir = month_dir / f"combo_{idx:02d}_MA{ma}_TOPN{topn}_{stop_loss_dir_part}_TH{threshold:.3f}"
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
                "stop_loss_pct": stop_loss_value,
                "threshold": threshold,
                "recorded_cagr": recorded_metrics["cagr"],
                "prefetch_cagr": metrics_prefetch["cagr"],
                "live_cagr": metrics_live["cagr"],
                "prefetch_minus_recorded": (metrics_prefetch["cagr"] - recorded_metrics["cagr"])
                if recorded_metrics["cagr"] is not None
                else None,
                "live_minus_recorded": (metrics_live["cagr"] - recorded_metrics["cagr"])
                if recorded_metrics["cagr"] is not None
                else None,
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
    payload: tuple[
        str,
        int,
        tuple[str, str],
        int,
        int,
        float,
        float,
        int,
        int,
        str,
        tuple[str, ...],
        float,
    ],
) -> tuple[str, Any, list[str]]:
    (
        account_norm,
        months_range,
        date_range,
        ma_int,
        topn_int,
        stop_loss_float,
        threshold_float,
        rsi_int,
        cooldown_int,
        ma_type_str,
        excluded_tickers,
    ) = payload

    # Worker 글로벌 변수에서 데이터 가져오기 (프로세스당 한 번만 pickle됨)
    data_source = _WORKER_PREFETCHED_DATA
    metrics_source = _WORKER_PREFETCHED_METRICS
    universe_source = _WORKER_PREFETCHED_UNIVERSE
    calendar_source = _WORKER_TRADING_CALENDAR
    fx_series_source = _WORKER_PREFETCHED_FX_SERIES

    try:
        override_rules = StrategyRules.from_values(
            ma_period=int(ma_int),
            portfolio_topn=int(topn_int),
            replace_threshold=float(threshold_float),
            ma_type=str(ma_type_str),
            stop_loss_pct=float(stop_loss_float),
        )
    except ValueError as exc:
        return (
            "failure",
            {
                "ma_period": ma_int,
                "portfolio_topn": topn_int,
                "stop_loss_pct": stop_loss_float,
                "replace_threshold": threshold_float,
                "rsi_sell_threshold": rsi_int,
                "ma_type": ma_type_str,
                "error": str(exc),
            },
            [],
        )

    strategy_overrides: dict[str, Any] = {
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
            prefetched_data=data_source,
            strategy_override=override_rules,
            excluded_tickers=set(excluded_tickers) if excluded_tickers else None,
            prefetched_etf_universe=universe_source,
            prefetched_metrics=metrics_source,
            trading_calendar=calendar_source,
            prefetched_fx_series=fx_series_source,
        )
    except Exception as exc:
        logger = get_app_logger()
        logger.warning(
            "[튜닝] 조합 실행 실패 months=%d MA=%s TOPN=%s STOP=%.2f RSI=%d score=%.2f TS=%.2f error=%s",
            months_range,
            ma_int,
            topn_int,
            stop_loss_float,
            rsi_int,
            rsi_int,
            exc,
        )
        return (
            "failure",
            {
                "ma_period": ma_int,
                "portfolio_topn": topn_int,
                "stop_loss_pct": stop_loss_float,
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
        "stop_loss_pct": float(stop_loss_float),
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
        "turnover": int(summary.get("turnover") or 0),
    }

    missing = getattr(bt_result, "missing_tickers", []) or []
    return ("success", entry, list(missing))


def _execute_tuning_for_months(
    account_norm: str,
    *,
    months_range: int,
    search_space: Mapping[str, list[Any]],
    end_date: Timestamp,
    excluded_tickers: Collection[str] | None,
    prefetched_data: Mapping[str, DataFrame],
    output_path: Path | None = None,
    progress_callback: callable | None = None,
    prefetched_etf_universe: Sequence[Mapping[str, Any]],
    prefetched_metrics: Mapping[str, dict[str, Any]],
    trading_calendar: Sequence[pd.Timestamp] | None,
) -> dict[str, Any] | None:
    logger = get_app_logger()

    ma_candidates = list(search_space.get("MA_PERIOD", []))
    topn_candidates = list(search_space.get("PORTFOLIO_TOPN", []))
    replace_candidates = list(search_space.get("REPLACE_SCORE_THRESHOLD", []))
    stop_loss_candidates = list(search_space.get("STOP_LOSS_PCT", []))
    rsi_candidates = list(search_space.get("OVERBOUGHT_SELL_THRESHOLD", []))
    cooldown_candidates = list(search_space.get("COOLDOWN_DAYS", []))
    ma_type_candidates = list(search_space.get("MA_TYPE", ["SMA"]))

    if (
        not ma_candidates
        or not topn_candidates
        or not replace_candidates
        or not stop_loss_candidates
        or not rsi_candidates
        or not cooldown_candidates
        or not ma_type_candidates
    ):
        logger.warning(
            "[튜닝] %s (%d개월) 유효한 탐색 공간이 없습니다.",
            account_norm.upper(),
            months_range,
        )
        return None

    combos: list[tuple[int, int, float, float, int, int, str]] = [
        (ma, topn, replace, stop_loss, rsi, cooldown, ma_type)
        for ma in ma_candidates
        for topn in topn_candidates
        for replace in replace_candidates
        for stop_loss in stop_loss_candidates
        for rsi in rsi_candidates
        for cooldown in cooldown_candidates
        for ma_type in ma_type_candidates
    ]

    if not combos:
        logger.warning(
            "[튜닝] %s (%d개월) 평가할 조합이 없습니다.",
            account_norm.upper(),
            months_range,
        )
        return None

    start_date = end_date - pd.DateOffset(months=months_range)
    date_range = (start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

    logger.info(
        "[튜닝] %s (%d개월) 전수조사 시작 (조합 %d개)",
        account_norm.upper(),
        months_range,
        len(combos),
    )

    filtered_calendar = _filter_trading_days(trading_calendar, date_range[0], date_range[1])
    if filtered_calendar is None:
        raise RuntimeError(
            f"[튜닝] {account_norm.upper()} ({months_range}개월) 구간의 거래일 정보를 준비하지 못했습니다."
        )

    workers = WORKERS or (cpu_count() or 1)
    workers = max(1, min(workers, len(combos)))

    success_entries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    encountered_missing: set[str] = set()
    best_cagr_so_far = float("-inf")

    # US 계정인 경우 환율 데이터 prefetch (모든 워커가 공유)
    fx_series: pd.Series | None = None
    country_code = account_norm.strip().lower()
    if country_code in {"us"}:
        try:
            from utils.data_loader import get_exchange_rate_series

            fx_series = get_exchange_rate_series(date_range[0], date_range[1])
            if fx_series is not None and not fx_series.empty:
                logger.info(
                    "[튜닝] %s 환율 데이터 prefetch 완료 (%d일)",
                    account_norm.upper(),
                    len(fx_series),
                )
        except Exception as exc:
            logger.warning("[튜닝] %s 환율 데이터 로드 실패, fallback 사용: %s", account_norm.upper(), exc)

    # 각 payload에는 파라미터만 포함 (데이터는 worker 초기화 시 한 번만 전달)
    payloads = [
        (
            account_norm,
            months_range,
            date_range,
            int(ma),
            int(topn),
            float(stop_loss),
            float(replace),
            int(rsi),
            int(cooldown),
            str(ma_type),
            tuple(excluded_tickers) if excluded_tickers else tuple(),
        )
        for ma, topn, replace, stop_loss, rsi, cooldown, ma_type in combos
    ]

    logger.info(
        "[튜닝] %s (%d개월) 백테스트 워커 초기화 중... (조합 %d개, 거래일 %d일)",
        account_norm.upper(),
        months_range,
        len(combos),
        len(filtered_calendar) if filtered_calendar else 0,
    )

    if workers <= 1:
        # 단일 프로세스: 글로벌 변수 직접 설정
        global \
            _WORKER_PREFETCHED_DATA, \
            _WORKER_PREFETCHED_METRICS, \
            _WORKER_PREFETCHED_UNIVERSE, \
            _WORKER_TRADING_CALENDAR, \
            _WORKER_PREFETCHED_FX_SERIES
        _WORKER_PREFETCHED_DATA = prefetched_data
        _WORKER_PREFETCHED_METRICS = prefetched_metrics
        _WORKER_PREFETCHED_UNIVERSE = prefetched_etf_universe
        _WORKER_TRADING_CALENDAR = filtered_calendar
        _WORKER_PREFETCHED_FX_SERIES = fx_series
        iterator = map(_evaluate_single_combo, payloads)
    else:
        # 멀티 프로세스: worker 초기화 시 데이터 한 번만 전달
        init_args = (
            prefetched_data,
            prefetched_metrics,
            prefetched_etf_universe,
            filtered_calendar,
            fx_series,
        )
        # chunksize를 적절히 설정: IPC 오버헤드와 로드 밸런싱 균형
        # 너무 크면 느린 조합으로 인한 대기 발생, 너무 작으면 IPC 오버헤드 증가
        chunksize = max(10, len(payloads) // (workers * 10)) if len(payloads) > workers else 10
        executor = ProcessPoolExecutor(max_workers=workers, initializer=_init_worker_prefetch, initargs=init_args)
        iterator = executor.map(_evaluate_single_combo, payloads, chunksize=chunksize)

    try:
        for idx, result in enumerate(iterator, 1):
            status, data, missing = result
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

                if success_entries and output_path and progress_callback:
                    current_best_cagr = max(_safe_float(entry.get("cagr"), float("-inf")) for entry in success_entries)
                    if current_best_cagr > best_cagr_so_far:
                        best_cagr_so_far = current_best_cagr

                    progress_callback(
                        success_entries=success_entries,
                        progress_pct=(idx / len(combos)) * 100,
                        completed=idx,
                        total=len(combos),
                    )
    finally:
        if workers > 1:
            executor.shutdown(wait=True, cancel_futures=False)

    if not success_entries:
        logger.warning(
            "[튜닝] %s (%d개월) 성공한 조합이 없습니다.",
            account_norm.upper(),
            months_range,
        )
        return None

    # 최적화 지표 선택 (config에서 가져오기)
    optimization_metric_raw = search_space.get("OPTIMIZATION_METRIC")
    if isinstance(optimization_metric_raw, list):
        optimization_metric = optimization_metric_raw[0].upper()
    else:
        optimization_metric = str(optimization_metric_raw).upper()

    def _sort_key(entry: dict[str, Any]) -> float:
        if optimization_metric == "CAGR":
            return _safe_float(entry.get("cagr"), float("-inf"))
        elif optimization_metric == "SHARPE":
            return _safe_float(entry.get("sharpe"), float("-inf"))
        else:  # SDR (default)
            return _safe_float(entry.get("sharpe_to_mdd"), float("-inf"))

    success_entries.sort(key=_sort_key, reverse=True)

    # --- Top N Ensemble Logic ---
    # 상위 N개의 결과를 사용하여 파라미터를 결정합니다.
    # 1. MA_PERIOD: 상위 N개의 평균 (반올림)
    # 2. 나머지: 상위 N개의 최빈값 (Mode)

    # 앙상블 크기 검증 (홀수만 허용)
    if TUNING_ENSEMBLE_SIZE % 2 == 0:
        raise ValueError(f"TUNING_ENSEMBLE_SIZE는 반드시 홀수여야 합니다. (현재값: {TUNING_ENSEMBLE_SIZE})")

    ensemble_size = min(len(success_entries), TUNING_ENSEMBLE_SIZE)
    top_n_entries = success_entries[:ensemble_size]
    best_entry = success_entries[0].copy()  # Top 1의 메트릭(CAGR 등)은 유지하되 파라미터만 덮어씀

    if top_n_entries:
        import statistics
        from collections import Counter

        def _get_mode(values):
            if not values:
                return None
            # 빈도수가 같으면 먼저 나온 것(순위가 높은 것)을 선호
            c = Counter(values)
            return c.most_common(1)[0][0]

        # 1. MA_PERIOD (Average)
        ma_periods = [e.get("ma_period") for e in top_n_entries if e.get("ma_period") is not None]
        if ma_periods:
            best_entry["ma_period"] = int(round(statistics.mean(ma_periods)))

        # 2. Others (Mode)
        param_keys = [
            "ma_type",
            "portfolio_topn",
            "replace_threshold",
            "stop_loss_pct",
            "rsi_sell_threshold",
            "cooldown_days",
        ]

        for key in param_keys:
            values = [e.get(key) for e in top_n_entries if e.get(key) is not None]
            mode_val = _get_mode(values)
            if mode_val is not None:
                best_entry[key] = mode_val

        logger.info(
            "[튜닝] Top %d 앙상블 적용: MA=%s (Avg), Others=Mode",
            ensemble_size,
            best_entry.get("ma_period"),
        )
    # -----------------------------

    raw_data_payload: list[dict[str, Any]] = []
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
                "period_return": _round_float_places(period_return_val, 2)
                if math.isfinite(period_return_val)
                else None,
                "sharpe": _round_float_places(sharpe_val, 2) if math.isfinite(sharpe_val) else None,
                "sharpe_to_mdd": _round_float_places(sharpe_to_mdd_val, 3)
                if math.isfinite(sharpe_to_mdd_val)
                else None,
                "turnover": int(item.get("turnover") or 0),
                "tuning": {
                    "MA_PERIOD": int(item.get("ma_period", 0)),
                    "MA_TYPE": str(item.get("ma_type", "SMA")),
                    "PORTFOLIO_TOPN": int(item.get("portfolio_topn", 0)),
                    "REPLACE_SCORE_THRESHOLD": _round_up_float_places(item.get("replace_threshold", 0.0), 1),
                    "STOP_LOSS_PCT": _round_up_float_places(item.get("stop_loss_pct"), 1)
                    if item.get("stop_loss_pct") is not None
                    else None,
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
    months_results: list[dict[str, Any]],
) -> dict[str, Any]:
    param_fields = {
        "MA_PERIOD": ("ma_period", True),
        "PORTFOLIO_TOPN": ("portfolio_topn", True),
        "REPLACE_SCORE_THRESHOLD": ("replace_threshold", False),
        "STOP_LOSS_PCT": ("stop_loss_pct", False),
        "OVERBOUGHT_SELL_THRESHOLD": ("rsi_sell_threshold", True),
        "COOLDOWN_DAYS": ("cooldown_days", True),
    }

    entry: dict[str, Any] = {
        "result": {},
    }

    raw_data_payload: list[dict[str, Any]] = []
    ma_type_weights: dict[str, float] = {}
    weighted_cagr_sum = 0.0
    weighted_cagr_weight = 0.0
    weighted_mdd_sum = 0.0
    weighted_mdd_weight = 0.0
    cagr_values: list[float] = []
    mdd_values: list[float] = []

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

        def _to_int(val: Any) -> int | None:
            try:
                return int(val)
            except (TypeError, ValueError):
                return None

        def _to_float(val: Any) -> float | None:
            try:
                num = float(val)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(num):
                return None
            return num

        tuning_snapshot: dict[str, Any] = {}
        field_key_pairs = [
            ("MA_PERIOD", "ma_period"),
            ("PORTFOLIO_TOPN", "portfolio_topn"),
            ("REPLACE_SCORE_THRESHOLD", "replace_threshold"),
            ("STOP_LOSS_PCT", "stop_loss_pct"),
            ("OVERBOUGHT_SELL_THRESHOLD", "rsi_sell_threshold"),
            ("COOLDOWN_DAYS", "cooldown_days"),
        ]

        for field, key in field_key_pairs:
            value = best.get(key)
            if value is None:
                continue
            if field == "REPLACE_SCORE_THRESHOLD":
                rounded_up = _round_up_float_places(value, 1)
                if math.isfinite(rounded_up):
                    tuning_snapshot[field] = rounded_up
            elif field == "STOP_LOSS_PCT":
                rounded_up = _round_up_float_places(value, 1)
                if math.isfinite(rounded_up):
                    tuning_snapshot[field] = rounded_up
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
                "turnover": int(best.get("turnover") or 0),
                "tuning": tuning_snapshot,
            }
        )
        ma_type_val = best.get("ma_type")
        if ma_type_val:
            weight_for_type = weight if weight > 0 else 1.0
            key = str(ma_type_val)
            ma_type_weights[key] = ma_type_weights.get(key, 0.0) + weight_for_type

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

    result_values: dict[str, Any] = entry["result"]

    for field, (key, is_int) in param_fields.items():
        values: list[float] = []
        weights: list[float] = []

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

    if ma_type_weights:
        result_values["MA_TYPE"] = max(ma_type_weights.items(), key=lambda item: (item[1], item[0]))[0]

    return entry


def _ensure_entry_schema(entry: Any) -> dict[str, Any]:
    if not isinstance(entry, dict):
        return {}

    normalized = dict(entry)

    result_map: dict[str, Any] = {}
    existing_result = normalized.get("result")
    if isinstance(existing_result, dict):
        result_map.update(existing_result)
    legacy_tuning = normalized.pop("tuning", None)
    if isinstance(legacy_tuning, dict):
        result_map.update(legacy_tuning)

    normalized.pop("result", None)

    for field in (
        "MA_PERIOD",
        "PORTFOLIO_TOPN",
        "REPLACE_SCORE_THRESHOLD",
        "STOP_LOSS_PCT",
        "COOLDOWN_DAYS",
        "MA_TYPE",
    ):
        normalized.pop(field, None)

    raw_results = normalized.get("raw_data")
    legacy_results = normalized.pop("results", None)
    if not isinstance(raw_results, list):
        raw_results = []
    if isinstance(legacy_results, list):
        raw_results.extend(legacy_results)

    cleaned_results: list[dict[str, Any]] = []
    cagr_values: list[float] = []
    mdd_positive_values: list[float] = []

    for item in raw_results:
        if not isinstance(item, dict):
            continue

        cleaned: dict[str, Any] = {
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

    def _normalize_float(value: Any) -> float | None:
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

    ordered: dict[str, Any] = {}
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
    month_results: list[dict[str, Any]],
    progress_info: dict[str, Any] | None = None,
    tuning_metadata: dict[str, Any] | None = None,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
) -> list[str]:
    start_ts = start_time or datetime.now()
    start_str = start_ts.strftime("%Y-%m-%d %H:%M:%S")
    lines: list[str] = [f"실행 시각: {start_str}"]

    if end_time is not None:
        end_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
        lines.append(f"종료 시각: {end_str}")
        elapsed = end_time - start_ts
        elapsed_seconds = max(int(elapsed.total_seconds()), 0)
        hours = elapsed_seconds // 3600
        minutes = (elapsed_seconds % 3600) // 60
        seconds = elapsed_seconds % 60
        lines.append(f"걸린 시간: {hours}시간 {minutes}분 {seconds}초")

    lines.append(f"계정: {account_id.upper()}")

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
            stop_loss_range = search_space.get("STOP_LOSS_PCT", [])
            rsi_range = search_space.get("OVERBOUGHT_SELL_THRESHOLD", [])
            cooldown_range = search_space.get("COOLDOWN_DAYS", [])

            # MA_TYPE이 있으면 포함해서 표시
            if ma_type_range and len(ma_type_range) > 1:
                lines.append(
                    f"탐색 공간: MA {len(ma_range)}개 × MA타입 {len(ma_type_range)}개 × TOPN {len(topn_range)}개 "
                    f"× 교체점수 {len(threshold_range)}개 × 손절 {len(stop_loss_range)}개 "
                    f"× RSI {len(rsi_range)}개 "
                    f"× COOLDOWN {len(cooldown_range)}개 "
                    f"= {tuning_metadata.get('combo_count', 0)}개 조합"
                )
            else:
                lines.append(
                    f"탐색 공간: MA {len(ma_range)}개 × TOPN {len(topn_range)}개 "
                    f"× 교체점수 {len(threshold_range)}개 × 손절 {len(stop_loss_range)}개 "
                    f"× RSI {len(rsi_range)}개 "
                    f"× COOLDOWN {len(cooldown_range)}개 "
                    f"= {tuning_metadata.get('combo_count', 0)}개 조합"
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
            if stop_loss_range:
                sl_min, sl_max = min(stop_loss_range), max(stop_loss_range)
                lines.append(f"  STOP_LOSS_PCT: {sl_min}~{sl_max}")
            if rsi_range:
                rsi_min, rsi_max = min(rsi_range), max(rsi_range)
                lines.append(f"  OVERBOUGHT_SELL_THRESHOLD: {rsi_min}~{rsi_max}")

            if cooldown_range:
                cd_min, cd_max = min(cooldown_range), max(cooldown_range)
                lines.append(f"  COOLDOWN_DAYS: {cd_min}~{cd_max}")

        # 종목 수
        ticker_count = tuning_metadata.get("ticker_count", 0)
        if ticker_count > 0:
            lines.append(f"대상 종목: {ticker_count}개")

        # 제외된 종목
        excluded_tickers = tuning_metadata.get("excluded_tickers", [])
        if excluded_tickers:
            lines.append(f"제외된 종목: {len(excluded_tickers)}개 ({', '.join(excluded_tickers)})")

        # 테스트 기간 및 데이터 범위
        data_period = tuning_metadata.get("data_period", {})
        test_period_ranges = tuning_metadata.get("test_period_ranges", [])
        test_periods = tuning_metadata.get("test_periods", [])

        period_lines: list[str] = []
        for entry in test_period_ranges:
            start_date = entry.get("start_date")
            end_date = entry.get("end_date")
            months = entry.get("months")
            if start_date and end_date and months:
                try:
                    months_int = int(months)
                except (TypeError, ValueError):
                    months_int = months
                period_lines.append(f"{start_date} ~ {end_date} ({months_int}개월)")

        if period_lines:
            lines.append(f"테스트 기간: {', '.join(period_lines)}")
        elif test_periods:
            period_str = ", ".join([f"{p}개월" for p in test_periods])
            lines.append(f"테스트 기간: {period_str}")

        if data_period:
            data_start = data_period.get("start_date")
            data_end = data_period.get("end_date")
            if data_start and data_end:
                lines.append(f"사용 데이터 범위: {data_start} ~ {data_end}")

    lines.append("")

    # 최적화 지표 가져오기
    optimization_metric = None
    if tuning_metadata:
        search_space = tuning_metadata.get("search_space", {})
        optimization_metric_raw = search_space.get("OPTIMIZATION_METRIC")
        if isinstance(optimization_metric_raw, list):
            optimization_metric = optimization_metric_raw[0].upper()
        else:
            optimization_metric = str(optimization_metric_raw).upper()

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
        normalized_rows: list[dict[str, Any]] = []

        for entry in raw_rows:
            tuning = entry.get("tuning") or {}
            ma_val = tuning.get("MA_PERIOD")
            ma_type_val = tuning.get("MA_TYPE", "SMA")
            topn_val = tuning.get("PORTFOLIO_TOPN")
            threshold_val = tuning.get("REPLACE_SCORE_THRESHOLD")
            stop_loss_val = entry.get("stop_loss_pct")
            if stop_loss_val is None:
                stop_loss_val = tuning.get("STOP_LOSS_PCT")
            rsi_val = tuning.get("OVERBOUGHT_SELL_THRESHOLD")
            cooldown_val = tuning.get("COOLDOWN_DAYS")
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
                    "stop_loss_pct": stop_loss_val,
                    "rsi_sell_threshold": rsi_val,
                    "cooldown_days": cooldown_val,
                    "cagr": cagr_val,
                    "mdd": mdd_val,
                    "period_return": period_val,
                    "sharpe": sharpe_val,
                    "sharpe_to_mdd": sharpe_to_mdd_val,
                    "turnover": entry.get("turnover", 0),
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
    month_results: list[dict[str, Any]],
    progress_info: dict[str, Any] | None = None,
    tuning_metadata: dict[str, Any] | None = None,
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
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            delete=False,
            suffix=".tmp",
        ) as tmp_file:
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
    output_path: Path | str | None = None,
    results_dir: Path | str | None = None,
    tuning_config: dict[str, dict[str, Any]] | None = None,
    months_range: int | None = None,
    debug_export_dir: Path | str | None = None,
    debug_capture_top_n: int = 1,
) -> Path | None:
    """Execute parameter tuning for the given account and return the output path."""

    account_norm = (account_id or "").strip().lower()
    logger = get_app_logger()
    tuning_start_ts = datetime.now()

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

    debug_dir: Path | None = None
    capture_top_n = 0
    if debug_export_dir is not None:
        debug_dir = Path(debug_export_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        try:
            capture_top_n = max(1, int(debug_capture_top_n))
        except (TypeError, ValueError):
            capture_top_n = 1
        logger.info("[튜닝] 디버그 아티팩트를 '%s'에 저장합니다.", debug_dir)
    debug_diff_rows: list[dict[str, Any]] = []
    debug_month_configs: list[dict[str, Any]] = []

    base_rules = get_strategy_rules(account_norm)
    ma_values = _normalize_tuning_values(config.get("MA_RANGE"), dtype=int, fallback=base_rules.ma_period)
    topn_values = _normalize_tuning_values(config.get("PORTFOLIO_TOPN"), dtype=int, fallback=base_rules.portfolio_topn)
    replace_values = _normalize_tuning_values(
        config.get("REPLACE_SCORE_THRESHOLD"),
        dtype=float,
        fallback=base_rules.replace_threshold,
    )
    stop_loss_fallback = base_rules.stop_loss_pct if base_rules.stop_loss_pct is not None else base_rules.portfolio_topn
    stop_loss_values = _normalize_tuning_values(
        config.get("STOP_LOSS_PCT"),
        dtype=float,
        fallback=stop_loss_fallback,
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

    if (
        not ma_values
        or not topn_values
        or not replace_values
        or not stop_loss_values
        or not rsi_sell_values
        or not cooldown_values
        or not ma_type_values
    ):
        logger.warning("[튜닝] 유효한 파라미터 조합이 없습니다.")
        return None

    etf_universe = get_etfs(account_norm)
    if not etf_universe:
        logger.error("[튜닝] '%s' 종목 데이터를 찾을 수 없습니다.", account_norm)
        return None

    tickers = [str(item.get("ticker")) for item in etf_universe if item.get("ticker")]
    if not tickers:
        logger.error("[튜닝] '%s' 유효한 티커가 없습니다.", country_code)
        return None

    # 벤치마크 종목도 프리패치에 포함 (백테스트 결과 비교용)
    benchmark_tickers = get_benchmark_tickers(account_settings)
    if benchmark_tickers:
        tickers_set = set(tickers)
        for bench_ticker in benchmark_tickers:
            if bench_ticker and bench_ticker not in tickers_set:
                tickers.append(bench_ticker)
                tickers_set.add(bench_ticker)
        logger.info(
            "[튜닝] 벤치마크 %d개 종목을 프리패치에 추가합니다: %s",
            len(benchmark_tickers),
            ", ".join(benchmark_tickers),
        )

    combo_count = (
        len(ma_values)
        * len(topn_values)
        * len(replace_values)
        * len(stop_loss_values)
        * len(rsi_sell_values)
        * len(cooldown_values)
        * len(ma_type_values)
    )
    if combo_count <= 0:
        logger.warning("[튜닝] 조합 생성에 실패했습니다.")
        return None

    # OPTIMIZATION_METRIC 필수 확인
    optimization_metric = config.get("OPTIMIZATION_METRIC")
    if not optimization_metric:
        logger.error(
            "[튜닝] '%s' 계정에 OPTIMIZATION_METRIC 설정이 없습니다. config.py에 추가해주세요.",
            account_norm.upper(),
        )
        return None

    search_space = {
        "MA_PERIOD": ma_values,
        "PORTFOLIO_TOPN": topn_values,
        "REPLACE_SCORE_THRESHOLD": replace_values,
        "STOP_LOSS_PCT": stop_loss_values,
        "OVERBOUGHT_SELL_THRESHOLD": rsi_sell_values,
        "COOLDOWN_DAYS": cooldown_values,
        "MA_TYPE": ma_type_values,
        "OPTIMIZATION_METRIC": [optimization_metric],
    }

    ma_count = len(ma_values)
    topn_count = len(topn_values)
    replace_count = len(replace_values)
    stop_loss_count = len(stop_loss_values)  # Defined for logger.info
    rsi_sell_count = len(rsi_sell_values)  # Defined for logger.info
    cooldown_count = len(cooldown_values)
    ma_type_count = len(ma_type_values)  # Defined for logger.info
    logger.info(
        "[튜닝] 탐색 공간: MA %d개 × TOPN %d개 × 교체점수 %d개 × 손절 %d개 "
        "× RSI %d개 × COOLDOWN %d개 × MA_TYPE %d개 = %d개 조합",
        ma_count,
        topn_count,
        replace_count,
        stop_loss_count,
        rsi_sell_count,
        cooldown_count,
        ma_type_count,
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

    end_date = get_latest_trading_day(country_code)
    if not isinstance(end_date, pd.Timestamp):
        end_date = pd.Timestamp.now().normalize()
    else:
        end_date = end_date.normalize()

    unique_month_ranges = sorted(set(valid_month_ranges))
    test_period_ranges: list[dict[str, Any]] = []
    for months in unique_month_ranges:
        try:
            months_int = int(months)
        except (TypeError, ValueError):
            continue
        start_dt = (end_date - pd.DateOffset(months=months_int)).normalize()
        test_period_ranges.append(
            {
                "months": months_int,
                "start_date": start_dt.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
            }
        )

    longest_months = max(valid_month_ranges)

    start_date_prefetch = end_date - pd.DateOffset(months=longest_months)
    date_range_prefetch = [
        start_date_prefetch.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
    ]

    # 국가 코드에 따라 웜업 기간 조정
    # US 등 일부 전략은 MA_PERIOD가 '월' 단위일 수 있으므로 충분한 웜업 기간 필요
    country_code = "kor"
    try:
        acct_settings = get_account_settings(account_id)
        country_code = (acct_settings.get("country_code") or account_id).strip().lower()
    except Exception:
        pass

    multiplier = 1.5
    if country_code in ("us", "usa"):
        # 월 단위 MA 가정 (약 32배)
        multiplier = 32.0

    warmup_days = int(max(ma_max, base_rules.ma_period) * multiplier)

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
        date_range_prefetch = [
            start_date_prefetch.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
        ]

    prefetched, missing_prefetch = prepare_price_data(
        tickers=tickers,
        country=country_code,
        start_date=date_range_prefetch[0],
        end_date=date_range_prefetch[1],
        warmup_days=warmup_days,
        account_id=account_id,
    )
    if missing_prefetch:
        raise MissingPriceDataError(
            country=country_code,
            start_date=date_range_prefetch[0],
            end_date=date_range_prefetch[1],
            tickers=missing_prefetch,
        )
    prefetched_map: dict[str, DataFrame] = dict(prefetched)

    if SAVE_CACHE_DURING_TUNE:
        for ticker, frame in prefetched_map.items():
            save_cached_frame(country_code, ticker, frame)

    excluded_ticker_set: set[str] = set()

    if debug_dir is not None:
        _export_prefetched_data(debug_dir, prefetched_map)

    runtime_missing_registry: set[str] = set()
    results_per_month: list[dict[str, Any]] = []

    # 출력 경로 미리 결정 (중간 저장용) - 계정별 폴더
    if results_dir is not None:
        base_dir = Path(results_dir) / account_norm / "results"
    else:
        base_dir = DEFAULT_RESULTS_DIR / account_norm / "results"

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
            "STOP_LOSS_PCT": list(stop_loss_values),
            "OVERBOUGHT_SELL_THRESHOLD": list(rsi_sell_values),
            "COOLDOWN_DAYS": list(cooldown_values),
            "OPTIMIZATION_METRIC": optimization_metric,
        },
        "data_period": {
            "start_date": date_range_prefetch[0],
            "end_date": date_range_prefetch[1],
        },
        "ticker_count": len(tickers),
        "excluded_tickers": sorted(excluded_ticker_set),
        "test_periods": valid_month_ranges,
        "test_period_ranges": test_period_ranges,
    }

    normalized_month_items: list[dict[str, Any]] = []

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

        normalized_month_items.append(sanitized_item)

    prefetched_trading_days = get_trading_days(
        date_range_prefetch[0],
        date_range_prefetch[1],
        country_code,
    )
    if not prefetched_trading_days:
        raise RuntimeError(
            f"[튜닝] {account_norm.upper()} 기간 {date_range_prefetch[0]}~{date_range_prefetch[1]}의 "
            "거래일 정보를 로드하지 못했습니다."
        )

    # 실제 탐색 공간에서 사용되는 MA_PERIOD와 MA_TYPE 조합만 캐시
    ma_period_pool = sorted(set(ma_values))  # 탐색 공간의 MA_PERIOD만 사용
    ma_type_pool = sorted(set(ma_type_values))  # 탐색 공간의 MA_TYPE만 사용
    logger.info(
        "[튜닝] MA 지표 캐시 생성: %d개 MA_PERIOD × %d개 MA_TYPE = %d개 조합",
        len(ma_period_pool),
        len(ma_type_pool),
        len(ma_period_pool) * len(ma_type_pool),
    )
    prefetched_metrics_map = _build_prefetched_metric_cache(
        prefetched_map,
        ma_periods=ma_period_pool,
        ma_types=ma_type_pool,
    )

    if not normalized_month_items:
        logger.warning("[튜닝] 실행 가능한 기간 항목이 없습니다.")
        return None

    # 최적화 지표에 따른 정렬 함수 정의
    optimization_metric_raw = search_space.get("OPTIMIZATION_METRIC")
    if isinstance(optimization_metric_raw, list):
        optimization_metric = optimization_metric_raw[0].upper()
    else:
        optimization_metric = str(optimization_metric_raw).upper()

    def _sort_key_local(entry):
        """최적화 지표에 따른 정렬 키"""
        if optimization_metric == "CAGR":
            return _safe_float(entry.get("cagr"), float("-inf"))
        elif optimization_metric == "SHARPE":
            return _safe_float(entry.get("sharpe"), float("-inf"))
        else:  # SDR (default)
            return _safe_float(entry.get("sharpe_to_mdd"), float("-inf"))

    for idx, item in enumerate(normalized_month_items, 1):
        months_value = item.get("months_range", 0)

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
                        "turnover": int(entry.get("turnover") or 0),
                        "tuning": {
                            "MA_PERIOD": int(entry.get("ma_period", 0)),
                            "MA_TYPE": str(entry.get("ma_type", "SMA")),
                            "PORTFOLIO_TOPN": int(entry.get("portfolio_topn", 0)),
                            "REPLACE_SCORE_THRESHOLD": _round_up_float_places(entry.get("replace_threshold", 0.0), 1),
                            "STOP_LOSS_PCT": _round_up_float_places(entry.get("stop_loss_pct", 0.0), 1),
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
            prefetched_etf_universe=etf_universe,
            prefetched_metrics=prefetched_metrics_map,
            trading_calendar=prefetched_trading_days,
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
                    "total": len(normalized_month_items),
                },
                tuning_metadata=tuning_metadata,
            )
            logger.info(
                "[튜닝] %s 중간 결과 저장 완료 (%d/%d 기간)",
                account_norm.upper(),
                len(results_per_month),
                len(normalized_month_items),
            )

        if debug_dir is not None and capture_top_n > 0:
            raw_rows = single_result.get("raw_data") or []
            month_start = (end_date - pd.DateOffset(months=months_value)).strftime("%Y-%m-%d")
            month_end = end_date.strftime("%Y-%m-%d")
            calendar_for_month = _filter_trading_days(prefetched_trading_days, month_start, month_end)
            if calendar_for_month is None:
                raise RuntimeError(
                    f"[튜닝] {account_norm.upper()} ({months_value}개월) 구간의 거래일 정보를 준비하지 못했습니다."
                )

            debug_diff_rows.extend(
                _export_debug_month(
                    debug_dir,
                    account_id=account_norm,
                    months_range=months_value,
                    raw_rows=raw_rows,
                    prefetched_data=prefetched_map,
                    capture_top_n=capture_top_n,
                    prefetched_etf_universe=etf_universe,
                    prefetched_metrics=prefetched_metrics_map,
                    trading_calendar=calendar_for_month,
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

    # 결과 요약 출력
    optimization_metric_raw = search_space.get("OPTIMIZATION_METRIC")
    if isinstance(optimization_metric_raw, list):
        optimization_metric = optimization_metric_raw[0].upper()
    else:
        optimization_metric = str(optimization_metric_raw).upper()

    for item in results_per_month:
        best = item.get("best", {})
        logger.info(
            "[튜닝] %s (%d개월) 최적 조합 (%s 기준): MA=%d / TOPN=%d / TH=%.3f / RSI=%d / "
            "COOLDOWN=%d / CAGR=%.2f%% / Sharpe=%.2f / SDR=%.3f",
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

    # 튜닝시 사용한 최적화 지표도 config에 저장
    if optimization_metric:
        entry["result"]["OPTIMIZATION_METRIC"] = optimization_metric

    _apply_tuning_to_strategy_file(account_norm, entry)

    if debug_dir is not None:
        meta_payload: dict[str, Any] = {
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

            (debug_dir / "diff_summary.json").write_text(
                json.dumps(debug_diff_rows, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            with (debug_dir / "diff_summary.csv").open("w", newline="", encoding="utf-8") as fp:
                writer = csv.DictWriter(fp, fieldnames=summary_fields)
                writer.writeheader()
                writer.writerows(debug_diff_rows)
    tuning_end_ts = datetime.now()

    report_lines = _compose_tuning_report(
        account_norm,
        month_results=results_per_month,
        tuning_metadata=tuning_metadata,
        start_time=tuning_start_ts,
        end_time=tuning_end_ts,
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
    elapsed = tuning_end_ts - tuning_start_ts
    elapsed_seconds = int(elapsed.total_seconds())
    hours = elapsed_seconds // 3600
    minutes = (elapsed_seconds % 3600) // 60
    seconds = elapsed_seconds % 60
    logger.info("[튜닝] 총 소요 시간: %d시간 %d분 %d초", hours, minutes, seconds)

    return txt_path


__all__ = ["run_account_tuning"]
SAVE_CACHE_DURING_TUNE = os.environ.get("TUNE_SAVE_CACHE", "0").lower() in (
    "1",
    "true",
    "yes",
)
