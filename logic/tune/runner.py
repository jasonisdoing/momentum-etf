"""Entry points for running country-level parameter tuning."""

from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from itertools import product
from os import cpu_count
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pandas as pd
from pandas import DataFrame, Timestamp

from logic.backtest.account_runner import run_account_backtest
from logic.entry_point import StrategyRules
from utils.account_registry import get_strategy_rules
from utils.settings_loader import (
    AccountSettingsError,
    get_account_settings,
    get_account_strategy,
    get_backtest_months_range,
    get_tune_month_configs,
)
from utils.logger import get_app_logger
from utils.data_loader import fetch_ohlcv_for_tickers, get_latest_trading_day, fetch_ohlcv
from utils.stock_list_io import get_etfs

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[2] / "data" / "results"


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


def _execute_tuning_for_months(
    account_norm: str,
    *,
    months_range: int,
    combos: List[Any],
    prefetched_data: Mapping[str, DataFrame],
    end_date: Timestamp,
) -> Optional[Dict[str, Any]]:
    logger = get_app_logger()

    start_date = end_date - pd.DateOffset(months=months_range)
    date_range = [start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")]
    logger.info(
        "[튜닝] %s (%d개월) 튜닝 시작: 조합 %d개 (데이터 재사용)",
        account_norm.upper(),
        months_range,
        len(combos),
    )

    failures: List[Dict[str, Any]] = []
    success_count = 0
    best_entry: Optional[Dict[str, Any]] = None
    best_key = (float("-inf"), float("inf"))
    progress_interval = max(1, len(combos) // 100)

    for idx, (ma, topn, threshold, regime_ma) in enumerate(combos, 1):
        if idx % progress_interval == 0 or idx == len(combos):
            logger.info(
                "[튜닝] %s (%d개월) 진행률: %d/%d (%.1f%%)",
                account_norm.upper(),
                months_range,
                idx,
                len(combos),
                (idx / len(combos)) * 100,
            )

        if topn <= 0:
            failures.append(
                {
                    "ma_period": ma,
                    "portfolio_topn": topn,
                    "replace_threshold": threshold,
                    "regime_ma_period": regime_ma,
                    "error": "PORTFOLIO_TOPN must be > 0",
                }
            )
            continue

        try:
            override_rules = StrategyRules.from_values(
                ma_period=int(ma),
                portfolio_topn=int(topn),
                replace_threshold=float(threshold),
            )
        except ValueError as exc:
            failures.append(
                {
                    "ma_period": ma,
                    "portfolio_topn": topn,
                    "replace_threshold": threshold,
                    "regime_ma_period": regime_ma,
                    "error": str(exc),
                }
            )
            continue

        try:
            bt_result = run_account_backtest(
                account_norm,
                months_range=months_range,
                quiet=True,
                override_settings={
                    "start_date": date_range[0],
                    "end_date": date_range[1],
                    "strategy_overrides": {
                        "MARKET_REGIME_FILTER_MA_PERIOD": int(regime_ma),
                    },
                },
                prefetched_data=prefetched_data,
                strategy_override=override_rules,
            )
        except Exception as exc:  # pragma: no cover - 백테스트 예외 방어
            failures.append(
                {
                    "ma_period": ma,
                    "portfolio_topn": topn,
                    "replace_threshold": threshold,
                    "regime_ma_period": regime_ma,
                    "error": str(exc),
                }
            )
            continue

        summary = bt_result.summary or {}
        final_value_local = _safe_float(summary.get("final_value"), 0.0)
        final_value_krw = _safe_float(summary.get("final_value_krw"), final_value_local)

        entry = {
            "ma_period": int(ma),
            "portfolio_topn": int(topn),
            "replace_threshold": float(threshold),
            "regime_ma_period": int(regime_ma),
            "cagr_pct": _safe_float(summary.get("cagr_pct"), 0.0),
            "mdd_pct": _safe_float(summary.get("mdd_pct"), 0.0),
            "sharpe_ratio": _safe_float(summary.get("sharpe_ratio"), 0.0),
            "sortino_ratio": _safe_float(summary.get("sortino_ratio"), 0.0),
            "calmar_ratio": _safe_float(summary.get("calmar_ratio"), 0.0),
            "cumulative_return_pct": _safe_float(summary.get("cumulative_return_pct"), 0.0),
            "final_value_local": final_value_local,
            "final_value": final_value_krw,
            "cui": _safe_float(summary.get("cui"), 0.0),
            "ulcer_index": _safe_float(summary.get("ulcer_index"), 0.0),
        }

        success_count += 1
        cagr = entry["cagr_pct"]
        mdd = entry["mdd_pct"]
        key = (cagr, -mdd)
        if best_entry is None or key > best_key:
            best_entry = entry
            best_key = key

    if best_entry is None:
        logger.warning("[튜닝] %s (%d개월) 성공한 조합이 없습니다.", account_norm.upper(), months_range)
        return None

    logger.info(
        "[튜닝] %s (%d개월) 완료: 성공 %d개 / 실패 %d개",
        account_norm.upper(),
        months_range,
        success_count,
        len(failures),
    )

    return {
        "months_range": months_range,
        "best": best_entry,
        "failures": failures,
        "success_count": success_count,
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
        "MARKET_REGIME_FILTER_MA_PERIOD": ("regime_ma_period", True),
    }

    entry: Dict[str, Any] = {
        "run_date": run_date,
        "tuning": {},
    }

    results_payload: List[Dict[str, Any]] = []
    for item in months_results:
        best = item.get("best") or {}
        months = item.get("months_range")
        if not best or months is None:
            continue

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
            ("MARKET_REGIME_FILTER_MA_PERIOD", "regime_ma_period"),
        ):
            value = best.get(key)
            if value is None:
                continue
            if field == "REPLACE_SCORE_THRESHOLD":
                converted = _to_float(value)
            else:
                converted = _to_int(value)
            if converted is not None:
                tuning_snapshot[field] = converted

        results_payload.append(
            {
                "MONTHS_RANGE": months,
                "CAGR": round(_safe_float(best.get("cagr_pct"), 0.0), 4),
                "period_return": round(_safe_float(best.get("cumulative_return_pct"), 0.0), 4),
                "tuning": tuning_snapshot,
            }
        )

    if results_payload:
        entry["results"] = results_payload

    tuning_values: Dict[str, Any] = entry["tuning"]
    weighted_cagr_sum = 0.0
    weighted_cagr_weight = 0.0

    for field, (key, is_int) in param_fields.items():
        details: List[Dict[str, Any]] = []
        weighted_sum = 0.0
        weight_total = 0.0

        for item in months_results:
            weight_raw = item.get("weight", 0.0)
            try:
                weight = float(weight_raw)
            except (TypeError, ValueError):
                weight = 0.0

            value = item.get("best", {}).get(key)
            if value is None:
                continue

            best_result = item.get("best", {})
            cagr_value = best_result.get("cagr_pct")
            if cagr_value is not None:
                try:
                    weighted_cagr_sum += weight * float(cagr_value)
                    weighted_cagr_weight += weight
                except (TypeError, ValueError):
                    pass

            weighted_sum += weight * float(value)
            weight_total += weight

            details.append(
                {
                    "period": item.get("months_range"),
                    "value": value,
                    "weight": weight,
                    "weighted_value": round(weight * float(value), 4),
                }
            )

        if not details:
            final_value = None
        else:
            if weight_total <= 0:
                raw = sum(float(detail["value"]) for detail in details) / len(details)
            else:
                raw = weighted_sum / weight_total
            final_value = int(round(raw)) if is_int else round(raw, 3)

        entry[field] = {
            "details": details,
            "final_value": final_value,
        }
        if final_value is not None:
            tuning_values[field] = final_value

    if weighted_cagr_weight > 0:
        entry["weighted_expected_cagr"] = round(weighted_cagr_sum / weighted_cagr_weight, 6)
    else:
        entry["weighted_expected_cagr"] = None

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
        return data
    return []


def run_account_tuning(
    account_id: str,
    *,
    output_path: Optional[Path | str] = None,
    results_dir: Optional[Path | str] = None,
    tuning_config: Optional[Dict[str, Dict[str, Any]]] = None,
    months_range: Optional[int] = None,
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
    regime_ma_values = _normalize_tuning_values(
        config.get("MARKET_REGIME_FILTER_MA_PERIOD"),
        dtype=int,
        fallback=strategy_settings.get("MARKET_REGIME_FILTER_MA_PERIOD", 10),
    )

    if not ma_values or not topn_values or not replace_values or not regime_ma_values:
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

    combos = list(product(ma_values, topn_values, replace_values, regime_ma_values))
    if not combos:
        logger.warning("[튜닝] 조합 생성에 실패했습니다.")
        return None

    try:
        ma_max = max([base_rules.ma_period, *ma_values])
    except ValueError:
        ma_max = base_rules.ma_period

    regime_candidates: List[int] = list(regime_ma_values)
    strategy_regime = strategy_settings.get("MARKET_REGIME_FILTER_MA_PERIOD")
    try:
        regime_candidates.append(int(strategy_regime))
    except (TypeError, ValueError):
        pass

    try:
        regime_ma_max = max(regime_candidates) if regime_candidates else base_rules.ma_period
    except ValueError:
        regime_ma_max = base_rules.ma_period

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

    prefetched = fetch_ohlcv_for_tickers(
        tickers,
        country_code,
        date_range=date_range_prefetch,
        warmup_days=warmup_days,
    )
    if not prefetched:
        logger.error("[튜닝] 데이터 프리패치에 실패했습니다.")
        return None

    prefetched_map: Dict[str, DataFrame] = dict(prefetched)

    regime_ticker = str(strategy_settings.get("MARKET_REGIME_FILTER_TICKER") or "").strip()
    if regime_ticker and regime_ticker not in prefetched_map:
        try:
            regime_prefetch = fetch_ohlcv(
                regime_ticker,
                country=country_code,
                date_range=date_range_prefetch,
            )
        except Exception:  # pragma: no cover - 데이터 로딩 실패 방어
            regime_prefetch = None

        if regime_prefetch is not None and not regime_prefetch.empty:
            prefetched_map[regime_ticker] = regime_prefetch

    results_per_month: List[Dict[str, Any]] = []
    max_workers = min(len(month_items), cpu_count() or 1)
    futures: Dict[Any, Any] = {}

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, item in enumerate(month_items):
            future = executor.submit(
                _execute_tuning_for_months,
                account_norm,
                months_range=item["months_range"],
                combos=combos,
                prefetched_data=prefetched_map,
                end_date=end_date,
            )
            futures[future] = (idx, item)

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
            collected.append((idx, single_result))

    collected.sort(key=lambda entry: entry[0])
    results_per_month = [item for _, item in collected]

    if not results_per_month:
        logger.warning("[튜닝] 실행 가능한 기간이 없어 결과가 없습니다.")
        return None

    run_date = datetime.now().strftime("%Y-%m-%d")
    for item in results_per_month:
        best = item.get("best", {})
        logger.info(
            "[튜닝] %s (%d개월) 최적 조합: MA=%d / TOPN=%d / TH=%.3f / REGIME_MA=%d / CAGR=%.2f%%",
            account_norm.upper(),
            item.get("months_range"),
            best.get("ma_period", 0),
            best.get("portfolio_topn", 0),
            best.get("replace_threshold", 0.0),
            best.get("regime_ma_period", 0),
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
    filtered.sort(key=lambda data: data.get("run_date", ""))

    output_path.write_text(
        json.dumps(filtered, ensure_ascii=False, indent=4),
        encoding="utf-8",
    )

    return output_path


__all__ = ["run_account_tuning"]
