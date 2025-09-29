"""Tuning runner extracted from tune.py

Provides a callable `run(account: str)` that performs parameter tuning and logs to
`logs/tune_{country}_{account}.log`.
"""
from __future__ import annotations

import itertools
import os
import sys
import warnings
from typing import List

import numpy as np
import pandas as pd

# silence warnings as in the original script
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

# project root on path for local imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.tee import Tee
from utils.data_loader import fetch_ohlcv_for_tickers
from utils.stock_list_io import get_etfs
from utils.account_registry import get_account_info
from test import main as run_backtest  # reuse existing backtest entrypoint

# country-specific tuning ranges (copied from tune.py)
TUNING_CONFIG = {
    "coin": {
        "MA_RANGE": np.arange(2, 6, 1),
        "PORTFOLIO_TOPN": np.arange(1, 6, 1),
        "REPLACE_SCORE_THRESHOLD": [0.5],
        "TEST_MONTHS_RANGE": 12,
    },
    "aus": {
        "MA_RANGE": [11, 12, 13, 14, 15],
        "PORTFOLIO_TOPN": [7],
        "REPLACE_SCORE_THRESHOLD": [0.5],
        "TEST_MONTHS_RANGE": 12,
    },
    "kor": {
        "MA_RANGE": [11],
        "PORTFOLIO_TOPN": [8, 9, 10],
        "REPLACE_SCORE_THRESHOLD": [0.5],
        "TEST_MONTHS_RANGE": 12,
    },
    "us": {
        "MA_RANGE": np.arange(5, 31, 1),
        "PORTFOLIO_TOPN": np.arange(5, 11, 1),
        "REPLACE_SCORE_THRESHOLD": [0.5],
        "TEST_MONTHS_RANGE": 12,
    },
}

PARAM_LABELS = [
    ("MA_RANGE", "ma_period"),
    ("PORTFOLIO_TOPN", "portfolio_topn"),
    ("REPLACE_SCORE_THRESHOLD", "replace_threshold"),
]


def _run_single_backtest(params, prefetched_data, account, param_names, test_months):
    """Execute one backtest with override settings and return summary."""
    param_values = params[1:]
    override_settings = {name: value for name, value in zip(param_names, param_values)}
    override_settings["test_months_range"] = test_months

    try:
        if not account:
            raise RuntimeError("Account context not initialised for tuning worker")
        summary = run_backtest(
            account=account,
            quiet=True,
            override_settings=override_settings,
            prefetched_data=prefetched_data,
        )
        if summary:
            summary["params"] = param_values
            return summary
    except Exception as e:
        print(f"Error during backtest with params {params}: {e}")

    return None


def run(account: str) -> None:
    """Run parameter tuning for the specified account and write logs to file."""
    if not account:
        raise SystemExit("튜닝에 사용할 계좌 코드가 필요합니다.")

    account_info = get_account_info(account)
    if not account_info:
        raise SystemExit(f"등록되지 않은 계좌입니다: {account}")
    country_code = str(account_info.get("country") or "").strip()
    if not country_code:
        raise SystemExit(f"'{account}' 계좌에 국가 정보가 없습니다.")

    # log file
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"tune_{country_code}_{account}.log")
    print(f"튜닝 로그가 다음 파일에 저장됩니다: {log_path}")

    original_stdout = sys.stdout

    with open(log_path, "w", encoding="utf-8") as log_file:
        sys.stdout = Tee(original_stdout, log_file)
        try:
            # config
            config = TUNING_CONFIG.get(country_code)
            if not config:
                print(f"오류: '{country_code}' 국가에 대한 튜닝 설정이 없습니다.")
                return

            MA_RANGE = config.get("MA_RANGE")
            TEST_MONTHS_RANGE = config.get("TEST_MONTHS_RANGE", 12)

            param_ranges: List[List[float]] = []
            param_names: List[str] = []
            for label, override_key in PARAM_LABELS:
                if label not in config:
                    continue
                cfg_values = config[label]
                if isinstance(cfg_values, (list, tuple, np.ndarray)):
                    values = list(cfg_values)
                else:
                    values = [cfg_values]
                if not values:
                    continue
                param_ranges.append(values)
                param_names.append(override_key)

            if not param_ranges:
                print("오류: 튜닝할 파라미터 범위가 설정되지 않았습니다.")
                return

            print("튜닝 대상 파라미터:")
            for label, override_key in zip(
                [lbl for lbl, _ in PARAM_LABELS if lbl in config], param_names
            ):
                values = config[label]
                if isinstance(values, (list, tuple, np.ndarray)) and len(values) > 1:
                    preview = ", ".join(str(v) for v in values[:5])
                    suffix = "..." if len(values) > 5 else ""
                    print(f"  - {override_key}: [{preview}{suffix}]")
                else:
                    print(f"  - {override_key}: {values}")

            # preload data
            print(f"\n튜닝을 위해 {country_code.upper()} 시장의 데이터를 미리 로딩합니다 (account={account})...")
            etfs_from_file = get_etfs(country_code)
            if not etfs_from_file:
                print(f"오류: 'data/{country_code}/' 폴더에서 백테스트에 사용할 티커를 찾을 수 없습니다.")
                return

            tickers = [s["ticker"] for s in etfs_from_file]
            max_ma_period = int(max(MA_RANGE))
            warmup_days = int(max_ma_period * 1.5)

            core_end_dt = pd.Timestamp.now()
            core_start_dt = core_end_dt - pd.DateOffset(months=int(TEST_MONTHS_RANGE))
            test_date_range = [core_start_dt.strftime("%Y-%m-%d"), core_end_dt.strftime("%Y-%m-%d")]

            prefetched_data = fetch_ohlcv_for_tickers(
                tickers, country_code, date_range=test_date_range, warmup_days=warmup_days
            )
            if not prefetched_data:
                print("오류: 튜닝에 사용할 데이터를 로드하지 못했습니다.")
                return
            print(f"총 {len(prefetched_data)}개 종목의 데이터 로딩 완료.")

            param_combinations = list(itertools.product([country_code], *param_ranges))
            print(f"Total combinations to test: {len(param_combinations)}")

            results = []
            for i, params in enumerate(param_combinations):
                result = _run_single_backtest(
                    params,
                    prefetched_data,
                    account,
                    param_names,
                    TEST_MONTHS_RANGE,
                )
                if result:
                    results.append(result)
                print(f"Progress: {i + 1}/{len(param_combinations)}")

            if not results:
                print("No valid backtest results found.")
                return

            df_results = pd.DataFrame(results)

            # top by CAGR
            top_cagr_results = df_results.sort_values(by="cagr_pct", ascending=False).head(5)
            print("\n" + "=" * 50)
            print(">>> 튜닝 결과: CAGR 상위 5개 파라미터 <<<")
            print("=" * 50)
            for i, (_, row) in enumerate(top_cagr_results.iterrows(), 1):
                params = row["params"]
                print(f"\n--- CAGR {i}위 ---")
                for name, value in zip(param_names, params):
                    print(f"  - {name}: {value}")
                print("-" * 20)
                print(f"  - CAGR: {row['cagr_pct']:.2f}%")
                print(f"  - CUI: {row['cui']:.2f}")
                print(f"  - Ulcer Index: {row['ulcer_index']:.2f}")
                print(f"  - Calmar Ratio: {row['calmar_ratio']:.2f}")
                print(f"  - MDD: {-row['mdd_pct']:.2f}%")
                print(f"  - Sharpe Ratio: {row['sharpe_ratio']:.2f}")

            # top by MDD
            top_mdd_results = df_results.sort_values(by="mdd_pct", ascending=True).head(5)
            print("\n" + "=" * 50)
            print(">>> 튜닝 결과: MDD 상위 5개 파라미터 <<<")
            print("=" * 50)
            for i, (_, row) in enumerate(top_mdd_results.iterrows(), 1):
                params = row["params"]
                print(f"\n--- MDD {i}위 ---")
                for name, value in zip(param_names, params):
                    print(f"  - {name}: {value}")
                print("-" * 20)
                print(f"  - CAGR: {row['cagr_pct']:.2f}%")
                print(f"  - CUI: {row['cui']:.2f}")
                print(f"  - Ulcer Index: {row['ulcer_index']:.2f}")
                print(f"  - Calmar Ratio: {row['calmar_ratio']:.2f}")
                print(f"  - MDD: {-row['mdd_pct']:.2f}%")
                print(f"  - Sharpe Ratio: {row['sharpe_ratio']:.2f}")

            # top by CUI
            top_cui_results = df_results.sort_values(by="cui", ascending=False).head(5)
            print("\n" + "=" * 50)
            print(">>> 튜닝 결과: CUI (Calmar/Ulcer) 상위 5개 파라미터 <<<")
            print("=" * 50)
            for i, (_, row) in enumerate(top_cui_results.iterrows(), 1):
                params = row["params"]
                print(f"\n--- CUI (Calmar/Ulcer) {i}위 ---")
                for name, value in zip(param_names, params):
                    print(f"  - {name}: {value}")
                print("-" * 20)
                print(f"  - CAGR: {row['cagr_pct']:.2f}%")
                print(f"  - CUI: {row['cui']:.2f}")
                print(f"  - Ulcer Index: {row['ulcer_index']:.2f}")
                print(f"  - Calmar Ratio: {row['calmar_ratio']:.2f}")
                print(f"  - MDD: {-row['mdd_pct']:.2f}%")
                print(f"  - Sharpe Ratio: {row['sharpe_ratio']:.2f}")

            print(f"\n튜닝이 완료되었습니다. 상세 내용은 {log_path} 파일을 확인하세요.")
        finally:
            sys.stdout = original_stdout


__all__ = ["run"]
