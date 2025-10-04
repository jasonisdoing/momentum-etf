"""Entry points for running country-level parameter tuning."""

from __future__ import annotations

import math
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from logic.backtest.account_runner import run_account_backtest
from constants import TEST_MONTHS_RANGE
from logic.entry_point import StrategyRules
from utils.account_registry import get_strategy_rules
from utils.settings_loader import AccountSettingsError, get_account_settings
from utils.logger import get_app_logger
from utils.data_loader import fetch_ohlcv_for_tickers, get_latest_trading_day
from utils.report import render_table_eaw
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


def _format_float(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(num):
        return "-"
    return f"{num:.{digits}f}"


def _format_pct(value: float | None, digits: int = 2, *, signed: bool = True) -> str:
    if value is None:
        return "-"
    try:
        num = float(value)
    except (TypeError, ValueError):
        return "-"
    if not math.isfinite(num):
        return "-"
    if signed:
        return f"{num:+.{digits}f}%"
    return f"{num:.{digits}f}%"


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

    if months_range is None:
        config_months = config.get("TEST_MONTHS_RANGE") if config else None
        try:
            months_range = int(config_months) if config_months is not None else TEST_MONTHS_RANGE
        except (TypeError, ValueError):
            months_range = TEST_MONTHS_RANGE
    else:
        months_range = int(months_range)

    etf_universe = get_etfs(country_code)
    if not etf_universe:
        logger.error("[튜닝] '%s' 종목 데이터를 찾을 수 없습니다.", country_code)
        return None

    tickers = [str(item.get("ticker")) for item in etf_universe if item.get("ticker")]
    if not tickers:
        logger.error("[튜닝] '%s' 유효한 티커가 없습니다.", country_code)
        return None

    end_date = get_latest_trading_day(country_code)
    if not isinstance(end_date, pd.Timestamp):
        end_date = pd.Timestamp.now().normalize()
    start_date = end_date - pd.DateOffset(months=months_range)
    date_range = [start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")]

    try:
        max_ma = max(ma_values)
    except ValueError:
        max_ma = base_rules.ma_period
    warmup_days = int(max(max_ma, base_rules.ma_period) * 1.5)

    logger.info(
        "[튜닝] 데이터 미리 로딩: 티커 %d개, 기간 %s~%s, 웜업 %d일",
        len(tickers),
        date_range[0],
        date_range[1],
        warmup_days,
    )

    prefetched_data = fetch_ohlcv_for_tickers(
        tickers,
        country_code,
        date_range=date_range,
        warmup_days=warmup_days,
    )
    if not prefetched_data:
        logger.error("[튜닝] 사전 데이터 로딩에 실패했습니다.")
        return None

    combos = list(product(ma_values, topn_values, replace_values))
    total = len(combos)
    logger.info(
        "[튜닝] %s 튜닝 시작: 조합 %d개 (기간 %d개월)",
        account_norm.upper(),
        total,
        months_range,
    )

    results: List[Dict[str, Any]] = []
    failures: List[Dict[str, Any]] = []

    for idx, (ma, topn, threshold) in enumerate(combos, 1):
        logger.debug(
            "[튜닝] (%d/%d) MA=%s, TOPN=%s, REPLACE_THRESHOLD=%s",
            idx,
            total,
            ma,
            topn,
            threshold,
        )
        if topn <= 0:
            failures.append(
                {
                    "ma_period": ma,
                    "portfolio_topn": topn,
                    "replace_threshold": threshold,
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
                },
                prefetched_data=prefetched_data,
                strategy_override=override_rules,
            )
        except Exception as exc:  # pylint: disable=broad-except
            failures.append(
                {
                    "ma_period": ma,
                    "portfolio_topn": topn,
                    "replace_threshold": threshold,
                    "error": str(exc),
                }
            )
            continue

        summary = bt_result.summary or {}
        entry = {
            "ma_period": int(ma),
            "portfolio_topn": int(topn),
            "replace_threshold": float(threshold),
            "cagr_pct": float(summary.get("cagr_pct", 0.0)),
            "mdd_pct": float(summary.get("mdd_pct", 0.0)),
            "sharpe_ratio": float(summary.get("sharpe_ratio", 0.0)),
            "sortino_ratio": float(summary.get("sortino_ratio", 0.0)),
            "calmar_ratio": float(summary.get("calmar_ratio", 0.0)),
            "cumulative_return_pct": float(summary.get("cumulative_return_pct", 0.0)),
            "final_value": float(summary.get("final_value", 0.0)),
            "cui": float(summary.get("cui", 0.0)),
            "ulcer_index": float(summary.get("ulcer_index", 0.0)),
        }
        results.append(entry)

    if not results:
        logger.warning("[튜닝] 성공한 조합이 없습니다.")
        return None

    def _sort_key(item: Dict[str, Any]):
        cagr = item.get("cagr_pct")
        mdd = item.get("mdd_pct")
        sharpe = item.get("sharpe_ratio")
        calmar = item.get("calmar_ratio")
        cum = item.get("cumulative_return_pct")

        def _neg(val, default=0.0):
            if val is None or not math.isfinite(val):
                return -default
            return -val

        def _pos(val, default=float("inf")):
            if val is None or not math.isfinite(val):
                return default
            return val

        return (
            _neg(cagr, 0.0),
            _pos(mdd),
            _neg(sharpe, 0.0),
            _neg(calmar, 0.0),
            _neg(cum, 0.0),
        )

    sorted_results = sorted(results, key=_sort_key)
    best = sorted_results[0]
    df_results = pd.DataFrame(sorted_results)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines: List[str] = []
    lines.append(f"# {account_norm.upper()} 전략 파라미터 튜닝 결과")
    lines.append(f"생성 시각: {timestamp}")
    lines.append(
        f"총 조합: {total}개 | 성공: {len(results)}개 | 실패: {len(failures)}개 | 기간: {months_range}개월"
    )
    lines.append("")

    def _append_metric_table(title: str, sort_key: str, ascending: bool) -> None:
        lines.append(f"# {title}")
        if df_results.empty:
            lines.append("(데이터 없음)")
            lines.append("")
            return

        subset = df_results.sort_values(by=sort_key, ascending=ascending).head(10)
        if subset.empty:
            lines.append("(데이터 없음)")
            lines.append("")
            return

        headers = [
            "순위",
            "MA",
            "TOPN",
            "임계값",
            "CAGR(%)",
            "MDD(%)",
            "누적(%)",
            "CUI (Calmar/Ulcer)",
            "Sharpe",
            "Sortino",
            "Calmar",
            "Ulcer",
        ]
        aligns = ["right"] * len(headers)
        table_rows: List[List[str]] = []
        for idx, row in enumerate(subset.itertuples(), 1):
            table_rows.append(
                [
                    str(idx),
                    str(int(row.ma_period)),
                    str(int(row.portfolio_topn)),
                    f"{row.replace_threshold:.3f}",
                    _format_pct(row.cagr_pct),
                    _format_pct(row.mdd_pct, signed=False),
                    _format_pct(row.cumulative_return_pct),
                    _format_float(getattr(row, "cui", None)),
                    _format_float(row.sharpe_ratio),
                    _format_float(row.sortino_ratio),
                    _format_float(row.calmar_ratio),
                    _format_float(getattr(row, "ulcer_index", None)),
                ]
            )

        lines.extend(render_table_eaw(headers, table_rows, aligns))
        lines.append("")

    _append_metric_table("CAGR 기준 상위 10개", "cagr_pct", ascending=False)
    _append_metric_table("MDD 기준 상위 10개", "mdd_pct", ascending=True)
    _append_metric_table("CUI 기준 상위 10개", "cui", ascending=False)

    if failures:
        lines.append("# 실패한 조합")
        for item in failures:
            lines.append(
                f"- MA={item['ma_period']}, TOPN={item['portfolio_topn']}, TH={item['replace_threshold']}: {item['error']}"
            )
        lines.append("")

    base_dir = Path(results_dir) if results_dir is not None else DEFAULT_RESULTS_DIR
    if output_path is None:
        output_path = base_dir / f"tune_{account_norm}.txt"
    else:
        output_path = Path(output_path)
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")

    logger.info(
        "[튜닝] 최적 조합: MA=%d / TOPN=%d / TH=%.3f",
        best["ma_period"],
        best["portfolio_topn"],
        best["replace_threshold"],
    )
    logger.info("[튜닝] 결과를 '%s'에 저장했습니다.", output_path)
    return output_path


__all__ = ["run_account_tuning"]
