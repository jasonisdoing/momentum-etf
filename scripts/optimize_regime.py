#!/usr/bin/env python
"""
최근 N개월 백테스트 결과를 이용해 시장 레짐 MA 기간을 탐색하는 스크립트.

예시:
    python scripts/optimize_regime.py k1
"""

from __future__ import annotations

import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Tuple

import pandas as pd

# 프로젝트 루트를 Python 경로에 추가
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from unittest.mock import patch

from logic.backtest.account_runner import run_account_backtest
from utils.account_registry import get_account_settings
from utils.data_loader import fetch_ohlcv, prepare_price_data, get_latest_trading_day
from utils.logger import get_app_logger
from utils.settings_loader import load_common_settings
from utils.stock_list_io import get_etfs

# ==== 설정 (필요 시 수정) ====
MONTHS = 12
MA_MIN = 10
MA_MAX = 100
MA_STEP = 10

RATIO_MIN = 0
RATIO_MAX = 100
RATIO_STEP = 10

WORKERS = None  # 병렬 실행 프로세스 수 (None이면 CPU 개수 기반 자동 결정)

SORT_KEYS = [
    ("cagr_pct", "CAGR"),
]

RESULTS_DIR = Path(ROOT_DIR) / "data" / "results"
PREFETCH_WARMUP_MONTHS = 12


def _build_ma_series(start: int, end: int, step: int) -> List[int]:
    if step <= 0:
        raise ValueError("증가 단위(step)는 1 이상의 정수여야 합니다.")
    if end < start:
        raise ValueError("범위 끝 값은 시작 값 이상이어야 합니다.")
    return list(range(start, end + 1, step))


def _format_float(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "-"


def _prefetch_common_data(account_id: str, months_range: int, base_common: MutableMapping[str, Any]) -> None:
    logger = get_app_logger()
    try:
        account_settings = get_account_settings(account_id)
    except Exception as exc:  # pragma: no cover - 설정 로드 실패 방어
        logger.warning("계정 설정을 로드하지 못해 프리패치를 건너뜁니다: %s", exc)
        return

    country_code = str(account_settings.get("country_code") or account_id).lower()
    etf_universe = get_etfs(country_code) or []
    tickers = [str(item.get("ticker") or "").strip().upper() for item in etf_universe if item.get("ticker")]

    if not tickers:
        logger.debug("%s 계정의 프리패치 대상 티커가 없습니다.", account_id.upper())
        return

    try:
        end_date = get_latest_trading_day(country_code)
    except Exception:
        end_date = None

    if not isinstance(end_date, pd.Timestamp):
        end_date = pd.Timestamp.now().normalize()

    start_date = end_date - pd.DateOffset(months=max(months_range, 1))
    date_range = [start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")]
    warmup_days = int(max(PREFETCH_WARMUP_MONTHS, 0) * 30)

    logger.info(
        "프리패치 시작: 티커 %d개, 기간 %s~%s, 웜업 %d일",
        len(tickers),
        date_range[0],
        date_range[1],
        warmup_days,
    )

    prepare_price_data(
        tickers=tickers,
        country=country_code,
        start_date=date_range[0],
        end_date=date_range[1],
        warmup_days=warmup_days,
    )

    regime_ticker = str(base_common.get("MARKET_REGIME_FILTER_TICKER_MAIN") or "").strip()
    if regime_ticker:
        regime_country = str(base_common.get("MARKET_REGIME_FILTER_COUNTRY") or "common").strip().lower() or "common"
        try:
            fetch_ohlcv(
                regime_ticker,
                country=regime_country,
                date_range=date_range,
                cache_country="common",
            )
        except Exception as exc:  # pragma: no cover - 레짐 데이터 조회 실패 방어
            logger.warning("레짐 데이터 프리패치 실패(%s/%s): %s", regime_country, regime_ticker, exc)


def _evaluate_single_combination(payload: Tuple[str, int, MutableMapping[str, Any], int, int]) -> Dict[str, Any]:
    account_id, months_range, base_common, ma_period, ratio = payload
    override_common: MutableMapping[str, Any] = deepcopy(base_common)
    override_common["MARKET_REGIME_FILTER_MA_PERIOD"] = ma_period

    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "logic.backtest.account_runner.get_common_file_settings",
                lambda: override_common,
            )
        )
        stack.enter_context(
            patch(
                "logic.backtest.reporting.load_common_settings",
                lambda: override_common,
            )
        )

        backtest = run_account_backtest(
            account_id,
            months_range=months_range,
            quiet=True,
            override_settings={
                "strategy_overrides": {
                    "MARKET_REGIME_RISK_OFF_EQUITY_RATIO": ratio,
                }
            },
        )

    summary = backtest.summary or {}
    row = {
        "ma_period": ma_period,
        "risk_off_ratio": ratio,
        "cagr_pct": summary.get("cagr_pct"),
        "mdd_pct": summary.get("mdd_pct"),
        "calmar_ratio": summary.get("calmar_ratio"),
        "sharpe_ratio": summary.get("sharpe_ratio"),
        "cui": summary.get("cui"),
        "final_value": summary.get("final_value"),
        "risk_off_periods": summary.get("risk_off_periods"),
    }
    return row


def _evaluate_combinations(
    account_id: str,
    ma_values: Iterable[int],
    ratio_values: Iterable[int],
    *,
    months_range: int,
    base_common: MutableMapping[str, Any],
    workers: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    주어진 MA/비중 조합에 대해 백테스트를 수행하고 핵심 지표를 반환합니다.
    """
    logger = get_app_logger()
    ma_list = list(ma_values)
    ratio_list = list(ratio_values)
    combos: List[Tuple[int, int]] = [(ma, ratio) for ma in ma_list for ratio in ratio_list]
    total_runs = max(1, len(combos))

    results: List[Dict[str, Any]] = []

    if not combos:
        return results

    # Determine worker count
    if workers is None:
        workers = WORKERS

    if workers is None:
        cpu_count = os.cpu_count() or 1
        workers = min(max(1, cpu_count), total_runs)
    else:
        workers = max(1, min(workers, total_runs))

    logger.info("병렬 작업자 수: %d", workers)

    payloads = [(account_id, months_range, base_common, ma_period, ratio) for ma_period, ratio in combos]

    if workers == 1:
        for idx, payload in enumerate(payloads, 1):
            row = _evaluate_single_combination(payload)
            results.append(row)
            logger.info(
                "[진행률 %d/%d (%.1f%%)] MA=%d, 비중=%d%%",
                idx,
                total_runs,
                (idx / total_runs) * 100,
                row["ma_period"],
                row.get("risk_off_ratio", 0),
            )
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_payload = {executor.submit(_evaluate_single_combination, payload): payload for payload in payloads}
            for idx, future in enumerate(as_completed(future_to_payload), 1):
                payload = future_to_payload[future]
                try:
                    row = future.result()
                except Exception as exc:  # pragma: no cover - 병렬 실행 예외 방어
                    logger.error(
                        "MA=%d, 비중=%d%% 조합 평가 실패: %s",
                        payload[3],
                        payload[4],
                        exc,
                    )
                    continue

                results.append(row)
                logger.info(
                    "[진행률 %d/%d (%.1f%%)] MA=%d, 비중=%d%%",
                    idx,
                    total_runs,
                    (idx / total_runs) * 100,
                    row["ma_period"],
                    row.get("risk_off_ratio", 0),
                )

    return results


def _sort_rows(rows: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    if key == "mdd_pct":
        return sorted(
            rows,
            key=lambda row: (row.get("mdd_pct") is None, float(row.get("mdd_pct") or float("inf"))),
        )
    return sorted(
        rows,
        key=lambda row: float(row.get(key) or float("-inf")),
        reverse=True,
    )


def _render_results_table(rows: List[Dict[str, Any]]) -> List[str]:
    headers = [
        "MA",
        "Ratio(%)",
        "CAGR(%)",
        "MDD(%)",
        "Calmar",
        "Sharpe",
        "CUI",
        "Final Value",
    ]
    lines = [" | ".join(headers), "-" * 72]
    for row in rows:
        line = [
            f"{row['ma_period']:>3d}",
            f"{int(row.get('risk_off_ratio', 0)):>9}",
            f"{_format_float(row.get('cagr_pct')):>8}",
            f"{_format_float(row.get('mdd_pct')):>8}",
            f"{_format_float(row.get('calmar_ratio')):>7}",
            f"{_format_float(row.get('cui')):>7}",
            f"{_format_float(row.get('final_value'), digits=0):>12}",
        ]
        lines.append(" | ".join(line))
    return lines


def _save_json(rows: List[Dict[str, Any]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/optimize_regime.py <account_id>")
        raise SystemExit(1)

    account_id = sys.argv[1].strip().lower()
    logger = get_app_logger()

    try:
        ma_values = _build_ma_series(MA_MIN, MA_MAX, MA_STEP)
        ratio_values = _build_ma_series(RATIO_MIN, RATIO_MAX, RATIO_STEP)
    except ValueError as exc:
        raise SystemExit(str(exc))

    logger.info(
        "계정 '%s'에 대해 MA %s, 비중 %s 값을 최근 %d개월 구간에서 평가합니다.",
        account_id,
        ", ".join(map(str, ma_values)),
        ", ".join(map(str, ratio_values)),
        MONTHS,
    )

    base_common = load_common_settings()
    _prefetch_common_data(account_id, MONTHS, base_common)

    rows = _evaluate_combinations(
        account_id,
        ma_values,
        ratio_values,
        months_range=MONTHS,
        base_common=base_common,
        workers=WORKERS,
    )

    if not rows:
        logger.warning("평가 결과가 비어 있습니다.")
        return

    sorted_tables: List[Tuple[str, List[str], Dict[str, Any]]] = []
    for key, label in SORT_KEYS:
        sorted_rows = _sort_rows(rows, key)
        best = sorted_rows[0]
        logger.info(
            "[%s] 최적 조합: MA=%d / RATIO=%d%% (CAGR=%s%%, MDD=%s%%, Calmar=%s, CUI=%s)",
            label,
            best["ma_period"],
            best.get("risk_off_ratio", 0),
            _format_float(best.get("cagr_pct")),
            _format_float(best.get("mdd_pct")),
            _format_float(best.get("calmar_ratio")),
            _format_float(best.get("cui")),
        )
        table_lines = _render_results_table(sorted_rows)
        title = f"정렬 기준: {label}"
        sorted_tables.append((title, table_lines, best))

    for title, table_lines, _ in sorted_tables:
        print(f"\n=== {title} ===")
        for line in table_lines:
            print(line)

    try:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # pragma: no cover - 디렉토리 생성 실패 방어
        logger.warning("결과 디렉토리를 생성하지 못했습니다: %s", exc)
        return

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    result_path = RESULTS_DIR / f"optimize_regime_{account_id}.txt"
    summary_lines = [
        f"실행 시각: {timestamp}",
        f"계정: {account_id}",
        f"기간: 최근 {MONTHS}개월",
        f"MA 범위: {MA_MIN}~{MA_MAX} (step={MA_STEP})",
        f"비중 범위: {RATIO_MIN}~{RATIO_MAX}% (step={RATIO_STEP}%)",
        "",
    ]

    try:
        with result_path.open("w", encoding="utf-8") as fp:
            for line in summary_lines:
                fp.write(line + "\n")
            for title, table_lines, best in sorted_tables:
                fp.write(f"=== {title} ===\n")
                fp.write(
                    f"최적 조합: MA={best['ma_period']} / 비중={best.get('risk_off_ratio', 0)}%"
                    f" / CAGR={_format_float(best.get('cagr_pct'))}%"
                    f" / MDD={_format_float(best.get('mdd_pct'))}%"
                    f" / Calmar={_format_float(best.get('calmar_ratio'))}"
                    f" / CUI={_format_float(best.get('cui'))}\n"
                )
                for line in table_lines:
                    fp.write(line + "\n")
                fp.write("\n")
        logger.info("요약 결과를 %s 에 기록했습니다.", result_path)
    except Exception as exc:  # pragma: no cover - 기록 실패 방어
        logger.warning("결과 파일 기록에 실패했습니다: %s", exc)


if __name__ == "__main__":
    main()
