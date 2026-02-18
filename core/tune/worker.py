"""Worker process logic for parameter tuning."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd
from pandas import DataFrame

from core.backtest.runner import run_account_backtest
from core.tune.reporting import (
    _round_float,
    _safe_float,
)
from strategies.maps.rules import StrategyRules

# Worker 글로벌 변수 - 프로세스당 한 번만 초기화
_WORKER_PREFETCHED_DATA: Mapping[str, DataFrame] | None = None
_WORKER_PREFETCHED_METRICS: Mapping[str, dict[str, Any]] | None = None
_WORKER_PREFETCHED_UNIVERSE: Sequence[Mapping[str, Any]] | None = None
_WORKER_TRADING_CALENDAR: Sequence[pd.Timestamp] | None = None
_WORKER_PREFETCHED_FX_SERIES: pd.Series | None = None


def init_worker_prefetch(
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


def evaluate_single_combo(
    payload: tuple[str, tuple[str, str], int, int, str, tuple[str, ...], bool],
) -> tuple[str, dict[str, Any], list[str]]:
    """단일 파라미터 조합 평가 (Worker Process에서 실행)"""
    (
        account_norm,
        date_range,
        ma_int,
        topn_int,
        ma_type_str,
        rebalance_mode_str,
        excluded_tickers,
        is_ma_month,
    ) = payload

    try:
        # Worker 전역 변수 또는 fallback
        if _WORKER_PREFETCHED_DATA is not None:
            prefetched_data = _WORKER_PREFETCHED_DATA
            prefetched_metrics = _WORKER_PREFETCHED_METRICS
            prefetched_universe = _WORKER_PREFETCHED_UNIVERSE
            trading_calendar = _WORKER_TRADING_CALENDAR
            fx_series = _WORKER_PREFETCHED_FX_SERIES
        else:
            # Fallback: 단일 프로세스 실행 시 직접 호출될 경우 (이 경우 호출 측에서 전역 설정 필요)
            # 그러나 안전을 위해 빈 값 처리
            prefetched_data = {}
            prefetched_metrics = {}
            prefetched_universe = []
            trading_calendar = []
            fx_series = None

        # StrategyRules 구성
        if is_ma_month:
            strategy_rules = StrategyRules.from_values(
                ma_month=int(ma_int),
                bucket_topn=int(topn_int),
                ma_type=str(ma_type_str),
                rebalance_mode=rebalance_mode_str,
            )
        else:
            strategy_rules = StrategyRules.from_values(
                ma_days=int(ma_int),
                bucket_topn=int(topn_int),
                ma_type=str(ma_type_str),
                rebalance_mode=rebalance_mode_str,
            )

        override_settings = {
            "START_DATE": date_range[0],
            "END_DATE": date_range[1],
            "EXCLUDED_TICKERS": list(excluded_tickers),
        }

        # 백테스트 실행
        bt_result = run_account_backtest(
            account_norm,
            prefetched_data=prefetched_data,
            override_settings=override_settings,
            strategy_override=strategy_rules,
            quiet=True,
            prefetched_etf_universe=prefetched_universe,
            prefetched_metrics=prefetched_metrics,
            trading_calendar=trading_calendar,
            prefetched_fx_series=fx_series,
        )

    except Exception as exc:
        return (
            "failure",
            {
                "ma_month" if is_ma_month else "ma_days": ma_int,
                "bucket_topn": topn_int,
                "error": str(exc),
            },
            [],
        )

    summary = bt_result.summary or {}
    final_value_local = _safe_float(summary.get("final_value"), 0.0)
    final_value_krw = _safe_float(summary.get("final_value_krw"), final_value_local)

    entry = {
        "ma_month" if is_ma_month else "ma_days": ma_int,
        "bucket_topn": topn_int,
        "rebalance_mode": rebalance_mode_str,
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
