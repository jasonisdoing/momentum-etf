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
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any

import pandas as pd
from pandas import DataFrame, Timestamp

from config import OPTIMIZATION_METRIC, TRADING_DAYS_PER_MONTH
from core.backtest.runner import run_account_backtest
from core.tune.reporting import (
    _export_combo_debug,
    _export_prefetched_data,
    _render_tuning_table,
    _round_float,
    _round_float_places,
    _round_up_float_places,
    _safe_float,
)
from core.tune.worker import (
    evaluate_single_combo,
    init_worker_prefetch,
)
from strategies.maps.rules import StrategyRules
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
WORKERS = None  # 병렬 실행 프로세스 수 (None이면 CPU 개수 기반 자동 결정)

# Worker 글로벌 변수 - 프로세스당 한 번만 초기화


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
    ma_dayss: Sequence[int],
    ma_types: Sequence[str],
) -> dict[str, dict[str, Any]]:
    if not prefetched_data:
        return {}

    period_pool = sorted({int(p) for p in ma_dayss if isinstance(p, (int, float)) and int(p) > 0})
    type_pool = sorted({(t or "SMA").upper() for t in ma_types if isinstance(t, str) and t})
    if not period_pool or not type_pool:
        return {}

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


def _resolve_month_configs(account_id: str = None) -> list[dict[str, Any]]:
    configs = get_tune_month_configs(account_id=account_id)
    if configs:
        return configs

    if account_id:
        try:
            account_settings = get_account_settings(account_id)
            fallback = account_settings.get("strategy", {}).get("BACKTEST_LAST_MONTHS")
            if fallback is not None:
                import pandas as pd

                months_back = int(fallback)
                start_dt = pd.Timestamp.today().normalize() - pd.DateOffset(months=months_back)
                return [
                    {
                        "backtest_start_date": start_dt.strftime("%Y-%m-%d"),
                        "weight": 1.0,
                        "source": f"account_{account_id}",
                    }
                ]
        except Exception:
            pass

    return []


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

    # Legacy cleanup

    integer_keys = {
        "BUCKET_TOPN",
        "MA_MONTH",
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
        "BACKTEST_LAST_MONTHS",
        "BACKTESTED_DATE",
        "CAGR",
        "MDD",
        "BUCKET_TOPN",
        "MA_MONTH",
        "MA_TYPE",
        "REBALANCE_MODE",
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


def _export_debug_month(
    debug_dir: Path,
    *,
    account_id: str,
    backtest_start_date: str,
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
    month_dir = debug_dir / f"start_{backtest_start_date}"

    for idx, row in enumerate(sorted_rows[:capture_top_n], 1):
        tuning = row.get("tuning") or {}
        try:
            ma_month = tuning.get("MA_MONTH")
            if ma_month is None:
                continue
            topn = int(tuning.get("BUCKET_TOPN"))
        except (TypeError, ValueError):
            continue

        recorded_metrics = {
            "cagr": float(row.get("CAGR")) if row.get("CAGR") is not None else None,
            "mdd": float(row.get("MDD")) if row.get("MDD") is not None else None,
            "period_return": float(row.get("period_return")) if row.get("period_return") is not None else None,
        }

        strategy_rules = StrategyRules.from_values(
            ma_month=int(ma_month),
            bucket_topn=topn,
            ma_type=tuning.get("MA_TYPE", "SMA"),
        )
        result_prefetch = run_account_backtest(
            account_id,
            quiet=True,
            prefetched_data=prefetched_data,
            strategy_override=strategy_rules,
            prefetched_etf_universe=prefetched_etf_universe,
            prefetched_metrics=prefetched_metrics,
            trading_calendar=trading_calendar,
        )

        result_live = run_account_backtest(
            account_id,
            quiet=True,
            strategy_override=strategy_rules,
            prefetched_etf_universe=prefetched_etf_universe,
            prefetched_metrics=None,
            trading_calendar=trading_calendar,
        )

        combo_dir = month_dir / f"combo_{idx:02d}_MA{ma_month}_TOPN{topn}"
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
                "backtest_start_date": backtest_start_date,
                "ma_days": ma_month,
                "topn": topn,
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


def _execute_tuning(
    account_norm: str,
    *,
    backtest_start_date: str,
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

    ma_candidates = list(search_space.get("MA_MONTH", []))
    is_ma_month = True

    topn_candidates = list(search_space.get("BUCKET_TOPN", []))
    ma_type_candidates = list(search_space.get("MA_TYPE", ["SMA"]))
    rebalance_mode_candidates = list(search_space.get("REBALANCE_MODE", []))

    if not rebalance_mode_candidates:
        current_rules = get_strategy_rules(account_norm)
        rebalance_mode_candidates = [current_rules.rebalance_mode]

    if not ma_candidates or not topn_candidates or not ma_type_candidates or not rebalance_mode_candidates:
        logger.warning(
            "[튜닝] %s (%s 시작) 유효한 탐색 공간이 없습니다.",
            account_norm.upper(),
            backtest_start_date,
        )
        return None

    combos: list[tuple[int, int, str, str]] = [
        (ma, topn, ma_type, rebal)
        for ma in ma_candidates
        for topn in topn_candidates
        for ma_type in ma_type_candidates
        for rebal in rebalance_mode_candidates
    ]

    if not combos:
        logger.warning(
            "[튜닝] %s (%s 시작) 평가할 조합이 없습니다.",
            account_norm.upper(),
            backtest_start_date,
        )
        return None

    # backtest_start_date를 직접 start_date로 사용
    try:
        start_date = pd.to_datetime(backtest_start_date)
    except (ValueError, TypeError):
        logger.error(f"[튜닝] 유효하지 않은 backtest_start_date: {backtest_start_date}")
        return None

    date_range = (start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))

    logger.info(
        "[튜닝] %s (%s 시작) 전수조사 시작 (조합 %d개)",
        account_norm.upper(),
        backtest_start_date,
        len(combos),
    )

    filtered_calendar = _filter_trading_days(trading_calendar, date_range[0], date_range[1])
    if filtered_calendar is None:
        raise RuntimeError(
            f"[튜닝] {account_norm.upper()} ({backtest_start_date} 시작) 구간의 거래일 정보를 준비하지 못했습니다."
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

    current_rules = get_strategy_rules(account_norm)

    payloads = [
        (
            account_norm,
            date_range,
            int(ma),
            int(topn),
            str(ma_type),
            str(rebal_mode),
            tuple(excluded_tickers) if excluded_tickers else tuple(),
            is_ma_month,
        )
        for ma, topn, ma_type, rebal_mode in combos
    ]

    logger.info(
        "[튜닝] %s (%s 시작) 백테스트 워커 초기화 중... (조합 %d개, 거래일 %d일)",
        account_norm.upper(),
        backtest_start_date,
        len(combos),
        len(filtered_calendar) if filtered_calendar else 0,
    )

    if workers <= 1:
        # 단일 프로세스: 글로벌 변수 직접 설정
        init_worker_prefetch(
            prefetched_data,
            prefetched_metrics,
            prefetched_etf_universe,
            filtered_calendar,
            fx_series,
        )
        iterator = map(evaluate_single_combo, payloads)
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
        executor = ProcessPoolExecutor(max_workers=workers, initializer=init_worker_prefetch, initargs=init_args)
        iterator = executor.map(evaluate_single_combo, payloads, chunksize=chunksize)

    try:
        for idx, result in enumerate(iterator, 1):
            status, data, missing = result
            if status == "success":
                success_entries.append(data)
                encountered_missing.update(missing)
            else:
                failures.append(data)
                logger.error(f"[튜닝] 오류 발생: {data.get('error', 'Unknown Error')}")

            if idx % max(1, len(combos) // 100) == 0 or idx == len(combos):
                logger.info(
                    "[튜닝] %s (%s 시작) 진행률: %d/%d (%.1f%%)",
                    account_norm.upper(),
                    backtest_start_date,
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
            "[튜닝] %s (%s 시작) 성공한 조합이 없습니다.",
            account_norm.upper(),
            backtest_start_date,
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

    # --- Top 1 Selection Logic ---
    # 앙상블 로직 제거: 항상 최적의 단일 결과를 사용합니다.
    # MA_MONTH 평균화나 Mode 방식은 파라미터 간 불일치(Logical Inconsistency)를 유발할 수 있음.

    best_entry = success_entries[0].copy()

    logger.info(
        "[튜닝] 최적 파라미터 선정: Top 1 (Best CAGR/SDR) 사용",
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
                "BACKTEST_START_DATE": backtest_start_date,
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
                    "MA_MONTH": int(item.get("ma_month", 0)),
                    "MA_TYPE": str(item.get("ma_type", "SMA")),
                    "BUCKET_TOPN": int(item.get("bucket_topn", 0)),
                    "REBALANCE_MODE": str(item.get("rebalance_mode", "MONTHLY")),
                },
            }
        )

    return {
        "backtest_start_date": backtest_start_date,
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
        "MA_MONTH": ("ma_month", True),
        "BUCKET_TOPN": ("bucket_topn", True),
    }

    entry: dict[str, Any] = {
        "result": {},
    }

    raw_data_payload: list[dict[str, Any]] = []
    weighted_cagr_sum = 0.0
    weighted_cagr_weight = 0.0
    weighted_mdd_sum = 0.0
    weighted_mdd_weight = 0.0
    ma_type_weights: dict[str, float] = {}
    rebal_mode_weights: dict[str, float] = {}
    cagr_values: list[float] = []
    mdd_values: list[float] = []

    for item in months_results:
        best = item.get("best") or {}
        backtest_start_date = item.get("backtest_start_date", "")
        if not best or not backtest_start_date:
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
            ("MA_MONTH", "ma_month"),
            ("BUCKET_TOPN", "bucket_topn"),
        ]

        for field, key in field_key_pairs:
            value = best.get(key)
            if value is None:
                continue

            converted = _to_int(value)
            if converted is not None:
                tuning_snapshot[field] = converted

        raw_data_payload.append(
            {
                "BACKTEST_START_DATE": backtest_start_date,
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
            weight_for_cat = weight if weight > 0 else 1.0
            type_key = str(ma_type_val)
            ma_type_weights[type_key] = ma_type_weights.get(type_key, 0.0) + weight_for_cat

        rebal_mode_val = best.get("rebalance_mode")
        if rebal_mode_val:
            weight_for_cat = weight if weight > 0 else 1.0
            rebal_key = str(rebal_mode_val)
            rebal_mode_weights[rebal_key] = rebal_mode_weights.get(rebal_key, 0.0) + weight_for_cat

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

    if rebal_mode_weights:
        result_values["REBALANCE_MODE"] = max(rebal_mode_weights.items(), key=lambda item: (item[1], item[0]))[0]

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
        "MA_MONTH",
        "BUCKET_TOPN",
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
            "BACKTEST_START_DATE": item.get("BACKTEST_START_DATE"),
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
            ma_month_range = search_space.get("MA_MONTH", [])
            ma_type_range = search_space.get("MA_TYPE", [])
            topn_range = search_space.get("BUCKET_TOPN", [])
            # MA_TYPE이 있으면 포함해서 표시
            lines.append(
                f"| 탐색 공간: MA {len(ma_month_range)}개 × MA타입 {len(ma_type_range)}개 × TOPN {len(topn_range)}개 "
                f"× 리밸런스 {len(search_space.get('REBALANCE_MODE', []))}개 "
                f"= {tuning_metadata.get('combo_count', 0)}개 조합"
            )
            # 각 파라미터 범위 표시
            if ma_month_range:
                ma_min, ma_max = min(ma_month_range), max(ma_month_range)
                lines.append(f"|   MA_RANGE: {ma_min}~{ma_max}")
            if ma_type_range:
                lines.append(f"|   MA_TYPE: {', '.join(ma_type_range)}")
            if topn_range:
                topn_min, topn_max = min(topn_range), max(topn_range)
                lines.append(f"|   BUCKET_TOPN: {topn_min}~{topn_max}")
            rebalance_range = search_space.get("REBALANCE_MODE", [])
            if rebalance_range:
                lines.append(f"|   REBALANCE_MODE: {', '.join(rebalance_range)}")

        # 종목 수
        ticker_count = tuning_metadata.get("ticker_count", 0)
        if ticker_count > 0:
            lines.append(f"| 대상 종목: {ticker_count}개")

        # 제외된 종목
        excluded_tickers = tuning_metadata.get("excluded_tickers", [])
        if excluded_tickers:
            lines.append(f"제외된 종목: {len(excluded_tickers)}개 ({', '.join(excluded_tickers)})")

        # 테스트 기간 및 데이터 범위
        data_period = tuning_metadata.get("data_period", {})
        test_period_ranges = tuning_metadata.get("test_period_ranges", [])
        test_periods = tuning_metadata.get("test_periods", [])

        # 기간 표시 (시작일 ~ 종료일 (N년 N개월 N일))
        period_lines: list[str] = []
        for entry in test_period_ranges:
            start_date = entry.get("start_date")
            end_date = entry.get("end_date")
            if start_date and end_date:
                try:
                    start_dt = pd.to_datetime(start_date)
                    end_dt = pd.to_datetime(end_date)
                    delta = end_dt - start_dt
                    total_days = delta.days
                    years = total_days // 365
                    remaining_days = total_days % 365
                    months = remaining_days // 30
                    days = remaining_days % 30
                    if years > 0:
                        period_str = f"{years}년 {months}개월 {days}일"
                    elif months > 0:
                        period_str = f"{months}개월 {days}일"
                    else:
                        period_str = f"{days}일"
                    period_lines.append(f"{start_date} ~ {end_date} ({period_str})")
                except Exception:
                    period_lines.append(f"{start_date} ~ {end_date}")

        if period_lines:
            lines.append(f"| 기간: {', '.join(period_lines)}")
        elif test_periods and data_period:
            # test_periods가 backtest_start_date 리스트일 경우
            data_end = data_period.get("end_date")
            if data_end:
                for start_date in test_periods:
                    try:
                        start_dt = pd.to_datetime(start_date)
                        end_dt = pd.to_datetime(data_end)
                        delta = end_dt - start_dt
                        total_days = delta.days
                        years = total_days // 365
                        remaining_days = total_days % 365
                        months = remaining_days // 30
                        days = remaining_days % 30
                        if years > 0:
                            period_str = f"{years}년 {months}개월 {days}일"
                        elif months > 0:
                            period_str = f"{months}개월 {days}일"
                        else:
                            period_str = f"{days}일"
                        period_lines.append(f"{start_date} ~ {data_end} ({period_str})")
                    except Exception:
                        period_lines.append(f"{start_date}")
                if period_lines:
                    lines.append(f"| 기간: {', '.join(period_lines)}")

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

    # 기간 문자열 계산 (tuning_metadata에서)
    period_display = ""
    if tuning_metadata:
        test_period_ranges = tuning_metadata.get("test_period_ranges", [])
        if test_period_ranges:
            for entry in test_period_ranges:
                start_date = entry.get("start_date")
                end_date = entry.get("end_date")
                if start_date and end_date:
                    try:
                        start_dt = pd.to_datetime(start_date)
                        end_dt = pd.to_datetime(end_date)
                        delta = end_dt - start_dt
                        total_days = delta.days
                        years = total_days // 365
                        remaining_days = total_days % 365
                        months = remaining_days // 30
                        days = remaining_days % 30
                        if years > 0:
                            period_display = f"{years}년 {months}개월 {days}일"
                        elif months > 0:
                            period_display = f"{months}개월 {days}일"
                        else:
                            period_display = f"{days}일"
                    except Exception:
                        pass
                    break  # 첫 번째 범위만 사용

    for item in sorted(month_results, key=lambda x: x.get("backtest_start_date", "")):
        backtest_start_date = item.get("backtest_start_date", "")
        if not backtest_start_date:
            continue

        # period_display가 아직 없으면 fallback
        if not period_display and backtest_start_date:
            try:
                start_dt = pd.to_datetime(backtest_start_date)
                end_dt = pd.Timestamp.now().normalize()
                delta = end_dt - start_dt
                total_days = delta.days
                years = total_days // 365
                remaining_days = total_days % 365
                months = remaining_days // 30
                days = remaining_days % 30
                if years > 0:
                    period_display = f"{years}년 {months}개월 {days}일"
                elif months > 0:
                    period_display = f"{months}개월 {days}일"
                else:
                    period_display = f"{days}일"
            except Exception:
                period_display = str(backtest_start_date)

        raw_rows = item.get("raw_data") or []
        normalized_rows: list[dict[str, Any]] = []

        for entry in raw_rows:
            tuning = entry.get("tuning") or {}
            ma_val = tuning.get("MA_MONTH")
            ma_type_val = tuning.get("MA_TYPE", "SMA")
            topn_val = tuning.get("BUCKET_TOPN")
            rebalance_val = tuning.get("REBALANCE_MODE", "MONTHLY")

            cagr_val = entry.get("CAGR")
            mdd_val = entry.get("MDD")
            period_val = entry.get("period_return")
            sharpe_val = entry.get("sharpe")
            sharpe_to_mdd_val = entry.get("sharpe_to_mdd")

            normalized_rows.append(
                {
                    "ma_days": ma_val,
                    "ma_type": ma_type_val,
                    "bucket_topn": topn_val,
                    "rebalance_mode": rebalance_val,
                    "cagr": cagr_val,
                    "mdd": mdd_val,
                    "period_return": period_val,
                    "sharpe": sharpe_val,
                    "sharpe_to_mdd": sharpe_to_mdd_val,
                    "turnover": entry.get("turnover", 0),
                }
            )

        normalized_rows.sort(key=_get_sort_key, reverse=True)
        lines.append(f"=== 최근 {period_display} 결과 - 정렬 기준: {metric_display} ===")
        lines.extend(_render_tuning_table(normalized_rows, period_str=period_display))
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

    # [Modify] MA_MONTH 우선 처리
    ma_month_values = _normalize_tuning_values(config.get("MA_MONTH"), dtype=int, fallback=None)
    ma_values = []

    if ma_month_values:
        # Month 사용 시 Day 변환 없이 그대로 사용
        pass
    else:
        # 기존 로직
        fallback_ma = base_rules.ma_days
        # rules에서 ma_days가 None(월단위 사용시)일 수 있음
        if fallback_ma is None:
            fallback_ma = 20  # default fallback

        ma_values = _normalize_tuning_values(config.get("MA_MONTH"), dtype=int, fallback=fallback_ma)

    topn_values = _normalize_tuning_values(config.get("BUCKET_TOPN"), dtype=int, fallback=base_rules.bucket_topn)

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

    # REBALANCE_MODE 처리: 문자열 리스트로 받음
    rebalance_raw = config.get("REBALANCE_MODE")
    if rebalance_raw is None:
        rebalance_mode_values = [base_rules.rebalance_mode]
    elif isinstance(rebalance_raw, (list, tuple)):
        rebalance_mode_values = [str(v).upper() for v in rebalance_raw if v]
    else:
        rebalance_mode_values = [str(rebalance_raw).upper()]

    if not rebalance_mode_values:
        rebalance_mode_values = [base_rules.rebalance_mode]

    if (not ma_values and not ma_month_values) or not topn_values or not ma_type_values:
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
        (len(ma_month_values) if ma_month_values else len(ma_values))
        * len(topn_values)
        * len(ma_type_values)
        * len(rebalance_mode_values)
    )
    if combo_count <= 0:
        logger.warning("[튜닝] 조합 생성에 실패했습니다.")
        return None

    # OPTIMIZATION_METRIC은 config.py의 전역 설정을 사용합니다.
    optimization_metric = OPTIMIZATION_METRIC

    search_space = {
        "MA_MONTH": ma_month_values or ma_values,
        "BUCKET_TOPN": topn_values,
        "MA_TYPE": ma_type_values,
        "REBALANCE_MODE": rebalance_mode_values,
        "OPTIMIZATION_METRIC": [optimization_metric],
    }

    ma_count = len(ma_month_values) if ma_month_values else len(ma_values)
    topn_count = len(topn_values)
    ma_type_count = len(ma_type_values)
    rebalance_count = len(rebalance_mode_values)

    logger.info(
        "[튜닝] 탐색 공간: MA %d개 × TOPN %d개 × MA_TYPE %d개 × 리밸런스 %d개 = %d개 조합 (최적화 지표: %s)",
        ma_count,
        topn_count,
        ma_type_count,
        rebalance_count,
        combo_count,
        optimization_metric,
    )

    try:
        base_ma = base_rules.ma_days or 20
        if ma_month_values:
            # Month -> Approx Day for warmup
            ma_max = max(ma_month_values) * 20
        else:
            ma_max = max([base_ma, *ma_values] if ma_values else [base_ma])
    except ValueError:
        ma_max = base_rules.ma_days or 20

    month_items = _resolve_month_configs(account_id=account_id)
    if not month_items:
        logger.error("[튜닝] 테스트할 기간 설정이 없습니다.")
        return None

    # backtest_start_date 기반으로 검증
    valid_start_dates = [
        str(item.get("backtest_start_date", "")) for item in month_items if item.get("backtest_start_date")
    ]
    if not valid_start_dates:
        logger.error("[튜닝] 유효한 기간 정보가 없습니다.")
        return None

    end_date = get_latest_trading_day(country_code)
    if not isinstance(end_date, pd.Timestamp):
        end_date = pd.Timestamp.now().normalize()
    else:
        end_date = end_date.normalize()

    # 시작일 기반으로 테스트 기간 생성
    test_period_ranges: list[dict[str, Any]] = []
    for start_date_str in valid_start_dates:
        try:
            start_dt = pd.to_datetime(start_date_str).normalize()
        except Exception:
            continue
        test_period_ranges.append(
            {
                "backtest_start_date": start_date_str,
                "start_date": start_dt.strftime("%Y-%m-%d"),
                "end_date": end_date.strftime("%Y-%m-%d"),
            }
        )

    if not test_period_ranges:
        logger.error("[튜닝] 유효한 테스트 기간이 없습니다.")
        return None

    # 가장 이른 시작일 찾기
    earliest_start = min(pd.to_datetime(item["start_date"]) for item in test_period_ranges)

    date_range_prefetch = [
        earliest_start.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d"),
    ]

    # 국가 코드에 따라 웜업 기간 조정
    # US 등 일부 전략은 MA_MONTH가 큰 값일 수 있으므로 충분한 웜업 기간 필요
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

    warmup_days = int(max(ma_max, base_rules.ma_days) * multiplier)

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

    if cache_seed_dt is not None and cache_seed_dt < earliest_start:
        earliest_start = cache_seed_dt
        date_range_prefetch = [
            earliest_start.strftime("%Y-%m-%d"),
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
            "MA_MONTH": list(ma_month_values) if ma_month_values else list(ma_values),
            "MA_TYPE": list(ma_type_values),
            "BUCKET_TOPN": list(topn_values),
            "REBALANCE_MODE": list(rebalance_mode_values),
            "OPTIMIZATION_METRIC": optimization_metric,
        },
        "data_period": {
            "start_date": date_range_prefetch[0],
            "end_date": date_range_prefetch[1],
        },
        "ticker_count": len(tickers),
        "excluded_tickers": sorted(excluded_ticker_set),
        "test_periods": valid_start_dates,
        "test_period_ranges": test_period_ranges,
    }

    normalized_month_items: list[dict[str, Any]] = []

    for item in month_items:
        backtest_start_date = item.get("backtest_start_date")
        if not backtest_start_date:
            logger.warning(
                "[튜닝] %s 백테스트 시작일이 없습니다. 항목을 건너뜁니다.",
                account_norm.upper(),
            )
            continue

        try:
            start_dt = pd.to_datetime(backtest_start_date)
        except Exception:
            logger.warning(
                "[튜닝] %s (%s) 날짜를 파싱할 수 없습니다. 항목을 건너뜁니다.",
                account_norm.upper(),
                backtest_start_date,
            )
            continue

        sanitized_item = dict(item)
        sanitized_item["backtest_start_date"] = str(backtest_start_date)
        if debug_dir is not None:
            debug_month_configs.append(
                {
                    "backtest_start_date": str(backtest_start_date),
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

    # 실제 탐색 공간에서 사용되는 MA_MONTH와 MA_TYPE 조합만 캐시
    # Month 단위일 경우 x20 한 값으로 캐시 필요
    if ma_month_values:
        ma_days_pool = sorted(set(m * TRADING_DAYS_PER_MONTH for m in ma_month_values))
    else:
        ma_days_pool = sorted(set(ma_values))

    ma_type_pool = sorted(set(ma_type_values))  # 탐색 공간의 MA_TYPE만 사용
    logger.info(
        "[튜닝] MA 지표 캐시 생성: %d개 MA_MONTH × %d개 MA_TYPE = %d개 조합",
        len(ma_days_pool),
        len(ma_type_pool),
        len(ma_days_pool) * len(ma_type_pool),
    )
    prefetched_metrics_map = _build_prefetched_metric_cache(
        prefetched_map,
        ma_dayss=ma_days_pool,
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
        backtest_start_date = str(item.get("backtest_start_date", ""))

        # 중간 저장 콜백 함수 정의
        def save_progress_callback(success_entries, progress_pct, completed, total):
            """1%마다 호출되는 중간 저장 콜백"""
            # 현재까지의 결과로 임시 결과 생성
            temp_result = {
                "backtest_start_date": backtest_start_date,
                "best": success_entries[0] if success_entries else {},
                "weight": item.get("weight", 0.0),
                "source": item.get("source"),
                "raw_data": [
                    {
                        "BACKTEST_START_DATE": backtest_start_date,
                        "CAGR": _round_float_places(entry.get("cagr", 0.0), 2),
                        "MDD": _round_float_places(-entry.get("mdd", 0.0), 2),
                        "period_return": _round_float_places(entry.get("period_return", 0.0), 2),
                        "sharpe": _round_float_places(entry.get("sharpe", 0.0), 2),
                        "sharpe_to_mdd": _round_float_places(entry.get("sharpe_to_mdd", 0.0), 3),
                        "turnover": int(entry.get("turnover") or 0),
                        "tuning": {
                            "MA_MONTH": int(entry.get("ma_month")) if entry.get("ma_month") is not None else None,
                            "MA_TYPE": str(entry.get("ma_type", "SMA")),
                            "BUCKET_TOPN": int(entry.get("bucket_topn", 0)),
                            "REBALANCE_MODE": str(entry.get("rebalance_mode", "MONTHLY")),
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
                    "progress_pct": progress_pct,
                },
                tuning_metadata=tuning_metadata,
            )

        single_result = _execute_tuning(
            account_norm,
            backtest_start_date=backtest_start_date,
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
            month_start = backtest_start_date
            month_end = end_date.strftime("%Y-%m-%d")
            calendar_for_month = _filter_trading_days(prefetched_trading_days, month_start, month_end)
            if calendar_for_month is None:
                raise RuntimeError(
                    f"[튜닝] {account_norm.upper()} ({backtest_start_date} 시작) 구간의 거래일 정보를 준비하지 못했습니다."
                )

            debug_diff_rows.extend(
                _export_debug_month(
                    debug_dir,
                    account_id=account_norm,
                    backtest_start_date=backtest_start_date,
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

    if results_per_month:
        item = results_per_month[-1]
        best = item.get("best") or {}
        logger.info(
            "[튜닝] %s (%s 시작) 최적 조합 (%s 기준): MA=%d / TOPN=%d / CAGR=%.2f%% / Sharpe=%.2f / SDR=%.3f",
            account_norm.upper(),
            item.get("backtest_start_date"),
            optimization_metric,
            best.get("ma_month", 0),
            best.get("bucket_topn", 0),
            best.get("cagr", 0.0),
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
                "MA_MONTH": ma_values,
                "BUCKET_TOPN": topn_values,
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
                "backtest_start_date",
                "ma_days",
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
