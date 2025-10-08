"""Entry points for running country-level parameter tuning."""

from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from os import cpu_count
from pathlib import Path
from typing import Any, Collection, Dict, List, Mapping, Optional, Tuple, Set

import optuna
import pandas as pd
from pandas import DataFrame, Timestamp

from logic.backtest.account_runner import run_account_backtest
from logic.entry_point import StrategyRules
from utils.account_registry import get_strategy_rules
from utils.settings_loader import (
    AccountSettingsError,
    get_account_settings,
    get_account_strategy,
    get_account_strategy_sections,
    get_backtest_months_range,
    get_tune_month_configs,
)
from utils.logger import get_app_logger
from utils.data_loader import (
    fetch_ohlcv_for_tickers,
    get_latest_trading_day,
    fetch_ohlcv,
)
from utils.stock_list_io import get_etfs

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[2] / "data" / "results"

optuna.logging.set_verbosity(optuna.logging.WARNING)


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


def _resolve_month_configs(months_range: Optional[int]) -> List[Dict[str, Any]]:
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

    configs = get_tune_month_configs()
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


def _execute_tuning_for_months(
    account_norm: str,
    *,
    months_range: int,
    search_space: Mapping[str, List[Any]],
    prefetched_data: Mapping[str, DataFrame],
    excluded_tickers: Optional[Collection[str]],
    end_date: Timestamp,
    combo_count: int,
    n_trials: Optional[int],
    timeout: Optional[float],
    sampler_seed: Optional[int],
    regime_ma_period: int,
) -> Optional[Dict[str, Any]]:
    logger = get_app_logger()

    start_date = end_date - pd.DateOffset(months=months_range)
    date_range = [start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")]
    logger.info("[튜닝] %s (%d개월) Optuna 튜닝 시작", account_norm.upper(), months_range)

    failures: List[Dict[str, Any]] = []
    success_entries: List[Dict[str, Any]] = []
    best_entry: Optional[Dict[str, Any]] = None
    best_key = (float("-inf"), float("inf"))
    evaluated_cache: Dict[Tuple[int, int, float], Dict[str, Any]] = {}
    encountered_missing: Set[str] = set()

    ma_candidates = list(search_space.get("MA_PERIOD", []))
    topn_candidates = list(search_space.get("PORTFOLIO_TOPN", []))
    replace_candidates = list(search_space.get("REPLACE_SCORE_THRESHOLD", []))

    if not ma_candidates or not topn_candidates or not replace_candidates:
        logger.warning("[튜닝] %s (%d개월) 유효한 탐색 공간이 없습니다.", account_norm.upper(), months_range)
        return None

    requested_trials = n_trials if n_trials is not None else combo_count
    effective_trials = min(
        max(requested_trials, 1), combo_count if combo_count > 0 else requested_trials
    )
    progress_interval = max(1, effective_trials // 100)

    sampler = optuna.samplers.TPESampler(multivariate=True, group=True, seed=sampler_seed)
    study = optuna.create_study(directions=["maximize", "minimize"], sampler=sampler)

    evaluated_success = 0

    def _log_progress() -> None:
        if evaluated_success == 0:
            return
        if evaluated_success % progress_interval == 0 or evaluated_success == effective_trials:
            logger.info(
                "[튜닝] %s (%d개월) 진행률: %d/%d (%.1f%%)",
                account_norm.upper(),
                months_range,
                evaluated_success,
                effective_trials,
                (evaluated_success / effective_trials) * 100,
            )

    def objective(trial: optuna.Trial) -> Tuple[float, float]:
        nonlocal best_entry, best_key, evaluated_success

        ma = trial.suggest_categorical("MA_PERIOD", ma_candidates)
        topn = trial.suggest_categorical("PORTFOLIO_TOPN", topn_candidates)
        threshold = trial.suggest_categorical("REPLACE_SCORE_THRESHOLD", replace_candidates)
        try:
            ma_int = int(ma)
            topn_int = int(topn)
            threshold_float = float(threshold)
        except (TypeError, ValueError):
            failures.append(
                {
                    "ma_period": ma,
                    "portfolio_topn": topn,
                    "replace_threshold": threshold,
                    "error": "파라미터 캐스팅 실패",
                }
            )
            raise optuna.TrialPruned("Invalid parameter cast")

        params_key = (ma_int, topn_int, threshold_float)
        cached = evaluated_cache.get(params_key)
        if cached is not None:
            trial.set_user_attr("entry", cached)
            return cached["cagr_pct"], -cached["mdd_pct"]

        if topn_int <= 0:
            failures.append(
                {
                    "ma_period": ma_int,
                    "portfolio_topn": topn_int,
                    "replace_threshold": threshold_float,
                    "error": "PORTFOLIO_TOPN must be > 0",
                }
            )
            raise optuna.TrialPruned("Invalid PORTFOLIO_TOPN")

        try:
            override_rules = StrategyRules.from_values(
                ma_period=ma_int,
                portfolio_topn=topn_int,
                replace_threshold=threshold_float,
            )
        except ValueError as exc:
            failures.append(
                {
                    "ma_period": ma_int,
                    "portfolio_topn": topn_int,
                    "replace_threshold": threshold_float,
                    "error": str(exc),
                }
            )
            raise optuna.TrialPruned(str(exc))

        strategy_overrides = {}
        if regime_ma_period > 0:
            strategy_overrides["MARKET_REGIME_FILTER_MA_PERIOD"] = regime_ma_period

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
                excluded_tickers=excluded_tickers,
            )
        except Exception as exc:  # pragma: no cover - 백테스트 예외 방어
            failures.append(
                {
                    "ma_period": ma_int,
                    "portfolio_topn": topn_int,
                    "replace_threshold": threshold_float,
                    "error": str(exc),
                }
            )
            raise optuna.TrialPruned(str(exc))

        if getattr(bt_result, "missing_tickers", None):
            encountered_missing.update(bt_result.missing_tickers)

        summary = bt_result.summary or {}
        final_value_local = _safe_float(summary.get("final_value"), 0.0)
        final_value_krw = _safe_float(summary.get("final_value_krw"), final_value_local)

        entry = {
            "ma_period": ma_int,
            "portfolio_topn": topn_int,
            "replace_threshold": _round_up_float_places(threshold_float, 1),
            "cagr_pct": _round_float(_safe_float(summary.get("cagr_pct"), 0.0)),
            "mdd_pct": _round_float(_safe_float(summary.get("mdd_pct"), 0.0)),
            "sharpe_ratio": _round_float(_safe_float(summary.get("sharpe_ratio"), 0.0)),
            "sortino_ratio": _round_float(_safe_float(summary.get("sortino_ratio"), 0.0)),
            "calmar_ratio": _round_float(_safe_float(summary.get("calmar_ratio"), 0.0)),
            "cumulative_return_pct": _round_float(
                _safe_float(summary.get("cumulative_return_pct"), 0.0)
            ),
            "final_value_local": final_value_local,
            "final_value": final_value_krw,
            "cui": _round_float(_safe_float(summary.get("cui"), 0.0)),
            "ulcer_index": _round_float(_safe_float(summary.get("ulcer_index"), 0.0)),
        }

        evaluated_cache[params_key] = entry
        success_entries.append(entry)
        evaluated_success = len(success_entries)
        trial.set_user_attr("entry", entry)

        cagr = entry["cagr_pct"]
        mdd = entry["mdd_pct"]
        key = (cagr, -mdd)
        if best_entry is None or key > best_key:
            best_entry = entry
            best_key = key

        _log_progress()
        return cagr, -mdd

    study.optimize(objective, n_trials=effective_trials, timeout=timeout)

    success_count = len(success_entries)

    if best_entry is None:
        logger.warning("[튜닝] %s (%d개월) 성공한 조합이 없습니다.", account_norm.upper(), months_range)
        return None

    logger.info(
        "[튜닝] %s (%d개월) 완료: 성공 %d개 / 실패 %d개 (요청 %d회, 처리 %d회)",
        account_norm.upper(),
        months_range,
        success_count,
        len(failures),
        effective_trials,
        len(study.trials),
    )

    return {
        "months_range": months_range,
        "best": best_entry,
        "failures": failures,
        "success_count": success_count,
        "regime_ma_period": regime_ma_period,
        "missing_tickers": sorted(encountered_missing),
        "study": {
            "trials_requested": effective_trials,
            "trials_completed": len(study.trials),
            "failures": len(failures),
        },
    }


def _build_run_entry(
    *,
    run_date: str,
    months_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    param_fields = {
        "MA_PERIOD": ("ma_period", True),
        "PORTFOLIO_TOPN": ("portfolio_topn", True),
        "REPLACE_SCORE_THRESHOLD": ("replace_threshold", False),
    }

    entry: Dict[str, Any] = {
        "run_date": run_date,
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

        cagr_val = _safe_float(best.get("cagr_pct"), float("nan"))
        if math.isfinite(cagr_val):
            weighted_cagr_sum += weight * cagr_val
            weighted_cagr_weight += weight
            cagr_values.append(cagr_val)

        mdd_val = _safe_float(best.get("mdd_pct"), float("nan"))
        if math.isfinite(mdd_val):
            weighted_mdd_sum += weight * mdd_val
            weighted_mdd_weight += weight
            mdd_values.append(mdd_val)

        period_return_val = _safe_float(best.get("cumulative_return_pct"), float("nan"))
        period_return_display = (
            _round_float_places(period_return_val, 2) if math.isfinite(period_return_val) else None
        )
        cagr_display = _round_float_places(cagr_val, 2) if math.isfinite(cagr_val) else None
        mdd_display = _round_float_places(-mdd_val, 2) if math.isfinite(mdd_val) else None

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

    run_date = normalized.get("run_date")

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
        cleaned["period_return"] = (
            _round_float_places(period_val, 2) if math.isfinite(period_val) else None
        )

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
    if run_date is not None:
        ordered["run_date"] = run_date
    else:
        ordered["run_date"] = normalized.get("run_date")

    ordered["result"] = result_map

    if weighted_cagr is not None:
        ordered["weighted_expected_CAGR"] = _round_float(weighted_cagr)

    if weighted_mdd is not None:
        ordered["weighted_expected_MDD"] = _round_float(weighted_mdd)

    if cleaned_results:
        ordered["raw_data"] = cleaned_results

    return ordered


def run_account_tuning(
    account_id: str,
    *,
    output_path: Optional[Path | str] = None,
    results_dir: Optional[Path | str] = None,
    tuning_config: Optional[Dict[str, Dict[str, Any]]] = None,
    months_range: Optional[int] = None,
    n_trials: Optional[int] = None,
    timeout: Optional[float] = None,
) -> Optional[Path]:
    """Execute parameter tuning for the given account and return the output path."""

    account_norm = (account_id or "").strip().lower()
    logger = get_app_logger()

    try:
        account_settings = get_account_settings(account_norm)
    except AccountSettingsError as exc:
        logger.error("[튜닝] 계정 설정 로딩 실패: %s", exc)
        return None

    country_code = (account_settings.get("country_code") or account_norm).strip().lower()

    config_map = tuning_config or {}
    config = config_map.get(account_norm) or config_map.get(country_code)
    if not config:
        logger.warning("[튜닝] '%s' 계정에 대한 튜닝 설정이 없습니다.", account_norm.upper())
        return None

    base_rules = get_strategy_rules(account_norm)
    strategy_settings = get_account_strategy(account_norm)
    ma_values = _normalize_tuning_values(
        config.get("MA_RANGE"), dtype=int, fallback=base_rules.ma_period
    )
    topn_values = _normalize_tuning_values(
        config.get("PORTFOLIO_TOPN"), dtype=int, fallback=base_rules.portfolio_topn
    )
    replace_values = _normalize_tuning_values(
        config.get("REPLACE_SCORE_THRESHOLD"),
        dtype=float,
        fallback=base_rules.replace_threshold,
    )

    if not ma_values or not topn_values or not replace_values:
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

    combo_count = len(ma_values) * len(topn_values) * len(replace_values)
    if combo_count <= 0:
        logger.warning("[튜닝] 조합 생성에 실패했습니다.")
        return None

    if n_trials is not None:
        try:
            trials_limit = int(n_trials)
        except (TypeError, ValueError):
            trials_limit = combo_count
        else:
            if trials_limit <= 0:
                trials_limit = combo_count
    else:
        trials_limit = None

    search_space = {
        "MA_PERIOD": ma_values,
        "PORTFOLIO_TOPN": topn_values,
        "REPLACE_SCORE_THRESHOLD": replace_values,
    }

    _, static_strategy = get_account_strategy_sections(account_norm)
    regime_ma_raw = None
    if isinstance(static_strategy, dict):
        regime_ma_raw = static_strategy.get("MARKET_REGIME_FILTER_MA_PERIOD")
    if regime_ma_raw is None:
        regime_ma_raw = strategy_settings.get("MARKET_REGIME_FILTER_MA_PERIOD")

    try:
        regime_ma_period = int(regime_ma_raw)
    except (TypeError, ValueError):
        logger.warning(
            "[튜닝] %s 정적 레짐 MA 기간을 확인할 수 없어 기본값(%d)을 사용합니다.",
            account_norm.upper(),
            base_rules.ma_period,
        )
        regime_ma_period = base_rules.ma_period
    else:
        if regime_ma_period <= 0:
            logger.warning(
                "[튜닝] %s 레짐 MA 기간이 0 이하(%d)로 설정되어 기본값(%d)으로 대체합니다.",
                account_norm.upper(),
                regime_ma_period,
                base_rules.ma_period,
            )
            regime_ma_period = base_rules.ma_period

    ma_count = len(ma_values)
    topn_count = len(topn_values)
    replace_count = len(replace_values)
    requested_trials = trials_limit if trials_limit is not None else combo_count
    logger.info(
        "[튜닝] 탐색 공간: MA %d개 × TOPN %d개 × TH %d개 = %d개 조합 (요청 시도 %d회)",
        ma_count,
        topn_count,
        replace_count,
        combo_count,
        requested_trials,
    )

    try:
        ma_max = max([base_rules.ma_period, *ma_values])
    except ValueError:
        ma_max = base_rules.ma_period

    regime_ma_max = max(regime_ma_period, 1)

    month_items = _resolve_month_configs(months_range)
    if not month_items:
        logger.error("[튜닝] 테스트할 기간 설정이 없습니다.")
        return None

    valid_month_ranges = [
        int(item.get("months_range", 0))
        for item in month_items
        if isinstance(item.get("months_range"), (int, float))
        and int(item.get("months_range", 0)) > 0
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

    warmup_days = int(max(ma_max, base_rules.ma_period, regime_ma_max) * 1.5)

    logger.info(
        "[튜닝] 데이터 프리패치: 티커 %d개, 기간 %s~%s, 웜업 %d일",
        len(tickers),
        date_range_prefetch[0],
        date_range_prefetch[1],
        warmup_days,
    )

    prefetched, missing_prefetch = fetch_ohlcv_for_tickers(
        tickers,
        country_code,
        date_range=date_range_prefetch,
        warmup_days=warmup_days,
    )
    prefetched_map: Dict[str, DataFrame] = dict(prefetched)

    regime_ticker = str(strategy_settings.get("MARKET_REGIME_FILTER_TICKER") or "").strip()
    if regime_ticker and regime_ticker not in prefetched_map:
        regime_prefetch = fetch_ohlcv(
            regime_ticker,
            country=country_code,
            date_range=date_range_prefetch,
        )
        if regime_prefetch is not None and not regime_prefetch.empty:
            prefetched_map[regime_ticker] = regime_prefetch
        else:
            missing_prefetch.append(regime_ticker)

    excluded_ticker_set: set[str] = {
        str(ticker).strip().upper()
        for ticker in missing_prefetch
        if isinstance(ticker, str) and str(ticker).strip()
    }
    if excluded_ticker_set:
        logger.warning(
            "[튜닝] %s 데이터 부족으로 제외할 종목 (%d): %s",
            account_norm.upper(),
            len(excluded_ticker_set),
            ", ".join(sorted(excluded_ticker_set)),
        )

    if timeout is not None:
        try:
            timeout_sec = float(timeout)
        except (TypeError, ValueError):
            timeout_sec = None
        else:
            if timeout_sec <= 0:
                timeout_sec = None
    else:
        timeout_sec = None

    results_per_month: List[Dict[str, Any]] = []
    max_workers = min(len(month_items), cpu_count() or 1)
    futures: Dict[Any, Any] = {}
    runtime_missing_registry: Set[str] = set()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, item in enumerate(month_items):
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
            seed = abs(hash((account_norm, months_value, idx))) % (2**32)

            future = executor.submit(
                _execute_tuning_for_months,
                account_norm,
                months_range=months_value,
                search_space=search_space,
                prefetched_data=prefetched_map,
                excluded_tickers=excluded_ticker_set,
                end_date=end_date,
                combo_count=combo_count,
                n_trials=trials_limit,
                timeout=timeout_sec,
                sampler_seed=seed,
                regime_ma_period=regime_ma_period,
            )
            futures[future] = (idx, sanitized_item)

        collected: List[Tuple[int, Dict[str, Any]]] = []
        for future in as_completed(futures):
            idx, item = futures[future]
            try:
                single_result = future.result()
            except Exception as exc:  # pragma: no cover - 병렬 실행 실패 방어
                logger.error(
                    "[튜닝] %s (%s개월) 병렬 실행 실패: %s",
                    account_norm.upper(),
                    item.get("months_range"),
                    exc,
                )
                continue

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
            collected.append((idx, single_result))

    collected.sort(key=lambda entry: entry[0])
    results_per_month = [item for _, item in collected]

    if not results_per_month:
        logger.warning("[튜닝] 실행 가능한 기간이 없어 결과가 없습니다.")
        return None

    run_date = datetime.now().strftime("%Y-%m-%d")
    if runtime_missing_registry:
        unseen_missing = sorted(set(runtime_missing_registry) - set(excluded_ticker_set or []))
        if unseen_missing:
            logger.warning(
                "[튜닝] %s 실행 중 데이터가 부족해 제외된 추가 종목 (%d): %s",
                account_norm.upper(),
                len(unseen_missing),
                ", ".join(unseen_missing),
            )

    for item in results_per_month:
        best = item.get("best", {})
        logger.info(
            "[튜닝] %s (%d개월) 최적 조합: MA=%d / TOPN=%d / TH=%.3f / REGIME_MA=%d / CAGR=%.2f%%",
            account_norm.upper(),
            item.get("months_range"),
            best.get("ma_period", 0),
            best.get("portfolio_topn", 0),
            best.get("replace_threshold", 0.0),
            item.get("regime_ma_period", 0),
            best.get("cagr_pct", 0.0),
        )

    entry = _build_run_entry(run_date=run_date, months_results=results_per_month)

    base_dir = Path(results_dir) if results_dir is not None else DEFAULT_RESULTS_DIR
    if output_path is None:
        output_path = base_dir / f"tune_{account_norm}.json"
    else:
        output_path = Path(output_path)
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing = _read_existing_results(output_path)
    filtered = [item for item in existing if item.get("run_date") != run_date]
    filtered.append(entry)
    filtered.sort(key=lambda data: data.get("run_date", ""), reverse=True)

    output_path.write_text(
        json.dumps(filtered, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )

    return output_path


__all__ = ["run_account_tuning"]
