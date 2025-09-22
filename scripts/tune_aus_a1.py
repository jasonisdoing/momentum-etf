"""
'aus' 국가, 'a1' 계좌에 대한 파라미터 튜닝을 실행하는 스크립트입니다.

이 스크립트는 `test.py`를 병렬로 실행하여
지정된 파라미터 범위 내에서 최적의 조합(최고 CAGR 기준)을 찾습니다.

[사용법]
python scripts/tune_aus_a1.py

[튜닝 설정]
스크립트 상단의 `TUNING_PARAMS` 딕셔너리에서 튜닝할 파라미터의 범위를,
`TEST_MONTHS_RANGE`에서 백테스트 기간을 직접 수정할 수 있습니다.
- 튜닝할 값: 리스트나 `np.arange()`로 지정 (예: `np.arange(10, 101, 10)`)
- 고정할 값: 단일 값을 포함한 리스트로 지정 (예: `[10]`)

결과는 콘솔에 출력되고 `logs/tune_aus_a1.log` 파일에도 저장됩니다.
"""

import itertools
import os
import sys

import numpy as np
import pandas as pd

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from test import main as run_backtest
from utils.data_loader import fetch_ohlcv_for_tickers, get_latest_trading_day
from utils.stock_list_io import get_etfs

# --- 튜닝 파라미터 정의 ---
# 튜닝할 값은 리스트나 np.arange()로, 고정할 값은 단일 값 리스트로 지정합니다.
TUNING_PARAMS = {
    "ma_period": np.arange(10, 21, 1),  # 10부터 150까지 10씩 증가
    "portfolio_topn": [5],
    "replace_threshold": [0],
}
# 백테스트 기간 (개월)
TEST_MONTHS_RANGE = 12


class Tee:
    """STDOUT과 파일에 동시에 쓰기 위한 헬퍼 클래스입니다."""

    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()

    def flush(self):
        for f in self.files:
            f.flush()


def run_single_backtest(params: tuple, prefetched_data: dict, account: str):
    """단일 파라미터 조합으로 백테스트를 실행하고 결과를 반환합니다."""
    country_code, test_months, override_settings = params

    # 튜닝 시 고정할 파라미터 추가
    override_settings["replace_weaker_stock"] = True
    override_settings["test_months_range"] = test_months

    param_desc = ", ".join(
        [f"{k}={v}" for k, v in override_settings.items() if k != "test_months_range"]
    )
    print(f"Testing with: {param_desc}")

    try:
        if not account:
            raise RuntimeError("Account context not initialised for tuning worker")

        summary = run_backtest(
            country=country_code,
            quiet=True,
            override_settings=override_settings,
            prefetched_data=prefetched_data,
            account=account,
        )
        if summary:
            params_to_save = override_settings.copy()
            params_to_save.pop("test_months_range", None)
            summary["params"] = params_to_save
            return summary
    except Exception as e:
        print(f"Error during backtest with params {override_settings}: {e}")

    return None


def main():
    """'aus' 국가, 'a1' 계좌에 대한 파라미터 튜닝을 실행합니다."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "tune_aus_a1.log")

    original_stdout = sys.stdout
    with open(log_path, "w", encoding="utf-8") as log_file:
        sys.stdout = Tee(original_stdout, log_file)
        try:
            country_code = "aus"
            account = "a1"

            print(f"'{country_code.upper()}' 국가, '{account}' 계좌에 대한 파라미터 튜닝을 시작합니다.")

            print(f"\n튜닝을 위해 {country_code.upper()} 시장의 데이터를 미리 로딩합니다...")
            etfs_from_file = get_etfs(country_code)
            if not etfs_from_file:
                print(f"오류: 'data/{country_code}/' 폴더에서 티커를 찾을 수 없습니다.")
                return

            tickers = [s["ticker"] for s in etfs_from_file]
            max_ma_period = int(max(TUNING_PARAMS["ma_period"]))
            warmup_days = int(max_ma_period * 1.5)

            core_end_dt = get_latest_trading_day(country_code)
            core_start_dt = core_end_dt - pd.DateOffset(months=int(TEST_MONTHS_RANGE))
            test_date_range = [
                core_start_dt.strftime("%Y-%m-%d"),
                core_end_dt.strftime("%Y-%m-%d"),
            ]

            prefetched_data = fetch_ohlcv_for_tickers(
                tickers, country_code, date_range=test_date_range, warmup_days=warmup_days
            )
            if not prefetched_data:
                print("오류: 튜닝에 사용할 데이터를 로드하지 못했습니다.")
                return
            print(f"총 {len(prefetched_data)}개 종목의 데이터 로딩 완료.")

            param_keys = TUNING_PARAMS.keys()
            param_values = TUNING_PARAMS.values()
            param_combinations = [
                dict(zip(param_keys, v)) for v in itertools.product(*param_values)
            ]
            tasks = [(country_code, TEST_MONTHS_RANGE, combo) for combo in param_combinations]

            print(f"\n총 {len(tasks)}개의 파라미터 조합을 테스트합니다.")
            results = []

            for i, task in enumerate(tasks):
                result = run_single_backtest(task, prefetched_data, account)
                if result:
                    results.append(result)
                print(f"Progress: {i + 1}/{len(tasks)}")

            if not results:
                print("유효한 백테스트 결과가 없습니다.")
                return

            df_results = pd.DataFrame(results)
            top_5_results = df_results.sort_values(by="cagr_pct", ascending=False).head(5)

            print("\n" + "=" * 50 + "\n>>> 튜닝 결과: CAGR 상위 5개 파라미터 <<<\n" + "=" * 50)
            for i, (_, row) in enumerate(top_5_results.iterrows(), 1):
                params = row["params"]
                param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
                print(f"\n--- {i}위 ---\n  - 파라미터: {param_str}\n" + "-" * 20)
                print(f"  - CAGR: {row['cagr_pct']:.2f}%")
                print(f"  - MDD: {-row['mdd_pct']:.2f}%")
                print(f"  - Calmar Ratio: {row['calmar_ratio']:.2f}")
                print(f"  - Sharpe Ratio: {row['sharpe_ratio']:.2f}")

        except Exception as e:
            print(f"\n튜닝 실행 중 오류가 발생했습니다: {e}")
        finally:
            sys.stdout = original_stdout


if __name__ == "__main__":
    main()
