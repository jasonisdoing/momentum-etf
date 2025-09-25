import itertools
import os
import sys
import argparse

import numpy as np
import pandas as pd

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test import main as run_backtest

from utils.tee import Tee
from utils.data_loader import fetch_ohlcv_for_tickers
from utils.db_manager import get_portfolio_settings
from utils.stock_list_io import get_etfs

# --- 국가별 튜닝 파라미터 범위 정의 ---
TUNING_CONFIG = {
    "aus": {
        "MA_RANGE": np.arange(5, 101, 1),
        "TEST_MONTHS_RANGE": 12,
    },
    "kor": {
        "MA_RANGE": np.arange(5, 151, 1),
        "TEST_MONTHS_RANGE": 12,
    },
    "coin": {
        # 코인은 단일 종목 유형으로 간주합니다.
        "MA_RANGE": np.arange(1, 201, 1),
        "TEST_MONTHS_RANGE": 12,
    },
}


def run_single_backtest(params, prefetched_data, account):
    """단일 파라미터 조합으로 백테스트를 실행하고 결과를 반환합니다."""
    country_code, topn, ma_period, replace_threshold, test_months = params

    override_settings = {
        "portfolio_topn": topn,
        "ma_period": ma_period,
        "replace_threshold": replace_threshold,
        "replace_weaker_stock": True,  # 교체매매는 활성화된 상태에서 테스트
        "test_months_range": test_months,
    }

    print(f"Testing with: TopN={topn}, MA={ma_period}, ReplaceThr={replace_threshold}")

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
            # 파라미터에서 국가 코드는 제외하고 저장
            summary["params"] = params[1:]
            return summary
    except Exception as e:
        print(f"Error during backtest with params {params}: {e}")

    return None


def main():
    """파라미터 튜닝을 실행하고 최적 결과를 출력합니다."""
    parser = argparse.ArgumentParser(description="전략 파라미터를 튜닝합니다.")
    parser.add_argument("country", choices=["kor", "aus", "coin"], help="튜닝할 국가 코드")
    parser.add_argument("--account", required=True, help="튜닝에 사용할 계좌 코드")
    args = parser.parse_args()

    country_code = args.country
    account = args.account

    # 로그 파일 설정
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"tune_{country_code}_{account}.log")

    print(f"튜닝 로그가 다음 파일에 저장됩니다: {log_path}")

    original_stdout = sys.stdout
    with open(log_path, "w", encoding="utf-8") as log_file:
        sys.stdout = Tee(original_stdout, log_file)
        try:
            # --- 국가별 튜닝 설정 로드 ---
            config = TUNING_CONFIG.get(country_code)
            if not config:
                print(f"오류: '{country_code}' 국가에 대한 튜닝 설정이 없습니다.")
                return

            MA_RANGE = config["MA_RANGE"]
            TEST_MONTHS_RANGE = config["TEST_MONTHS_RANGE"]

            # --- DB에서 고정 파라미터 로드 ---
            print(f"DB에서 {country_code.upper()} 포트폴리오의 고정 파라미터를 로드합니다 (account={account})...")
            portfolio_settings = get_portfolio_settings(country_code, account=account)
            if not portfolio_settings:
                print(f"오류: '{country_code}' 국가의 설정을 DB에서 찾을 수 없습니다. 웹 앱의 '설정' 탭에서 값을 저장해주세요.")
                return

            try:
                TOPN_FIXED = [int(portfolio_settings["portfolio_topn"])]
                REPLACE_THRESHOLD_FIXED = [float(portfolio_settings["replace_threshold"])]
                print(f"  - 고정값: TopN={TOPN_FIXED[0]}, ReplaceThr={REPLACE_THRESHOLD_FIXED[0]}")
            except (KeyError, ValueError, TypeError) as e:
                print(f"오류: DB에서 고정 파라미터를 로드하는 중 문제가 발생했습니다: {e}")
                return

            # --- 데이터 사전 로딩 ---
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

            param_combinations = list(
                itertools.product(
                    [country_code],
                    TOPN_FIXED,
                    MA_RANGE,
                    REPLACE_THRESHOLD_FIXED,
                    [TEST_MONTHS_RANGE],
                )
            )

            print(f"Total combinations to test: {len(param_combinations)}")

            results = []

            # 순차 처리를 위해 for 루프 사용
            for i, params in enumerate(param_combinations):
                result = run_single_backtest(params, prefetched_data, account)
                if result:
                    results.append(result)
                print(f"Progress: {i + 1}/{len(param_combinations)}")

            if not results:
                print("No valid backtest results found.")
                return

            # --- 결과 분석 ---
            df_results = pd.DataFrame(results)

            # CAGR 기준으로 상위 5개 결과를 찾습니다.
            top_3_results = df_results.sort_values(by="cagr_pct", ascending=False).head(5)

            print("\n" + "=" * 50)
            print(">>> 튜닝 결과: CAGR 상위 5개 파라미터 <<<")
            print("=" * 50)

            for i, (_, row) in enumerate(top_3_results.iterrows(), 1):
                params = row["params"]
                _, ma_period, _, _ = params

                print(f"\n--- {i}위 ---")
                print(f"  - MA_PERIOD: {ma_period}")
                print("-" * 20)
                print(f"  - CAGR: {row['cagr_pct']:.2f}%")
                print(f"  - MDD: {-row['mdd_pct']:.2f}%")
                print(f"  - Calmar Ratio: {row['calmar_ratio']:.2f}")
                print(f"  - Sharpe Ratio: {row['sharpe_ratio']:.2f}")

            if not top_3_results.empty:
                print(f"\n튜닝이 완료되었습니다. 상세 내용은 {log_path} 파일을 확인하세요.")
        finally:
            sys.stdout = original_stdout
            if not top_3_results.empty:
                first_row_params = top_3_results.iloc[0]["params"]
                topn, _, replace_thr, _ = first_row_params
                print("\n" + "=" * 50)
                print(f"(고정 파라미터: TopN={topn}, ReplaceThr={replace_thr})")
