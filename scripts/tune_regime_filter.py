"""
시장 레짐 필터 파라미터 최적화(튜닝) 스크립트입니다.

이 스크립트는 MARKET_REGIME_FILTER_MA_PERIOD 값을 변경해가며
여러 기간에 대해 병렬로 백테스트를 실행하여, 최고의 성과를 내는
최적의 파라미터 값을 찾습니다.
"""

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict

import numpy as np
import pandas as pd

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import fetch_ohlcv_for_tickers
from utils.db_manager import get_app_settings, get_stocks


def run_backtest_worker(params: tuple, prefetched_data: Dict[str, pd.DataFrame]) -> tuple:
    """
    단일 파라미터 조합에 대한 백테스트를 실행하는 워커 함수입니다.
    이 함수는 별도의 프로세스에서 실행됩니다.
    """
    regime_ma_period, months_range, country = params

    # 각 프로세스는 자체적인 모듈 컨텍스트를 가지므로, 여기서 다시 임포트하고 설정합니다.
    from test import main as run_test

    from logic import settings as worker_settings

    worker_settings.MARKET_REGIME_FILTER_MA_PERIOD = int(regime_ma_period)
    worker_settings.TEST_MONTHS_RANGE = int(months_range)

    # 미리 로드된 데이터를 전달하여 API 호출을 최소화합니다.
    result = run_test(country=country, quiet=True, prefetched_data=prefetched_data)

    return regime_ma_period, months_range, result


def tune_regime_filter(country: str):
    """
    MARKET_REGIME_FILTER_MA_PERIOD 파라미터를 튜닝하여 최적의 값을 찾습니다.
    """
    # --- 튜닝할 파라미터 범위 설정 ---
    # 1부터 500까지 1씩 테스트하면 시간이 매우 오래 걸리므로, 10 단위로 테스트합니다.
    regime_ma_periods = np.arange(10, 501, 10)
    test_months_ranges = [3, 6, 12, 24, 36, 48, 60]

    # --- 데이터 사전 로딩 (API 호출 최소화) ---
    print(f"\n튜닝을 위해 {country.upper()} 시장의 데이터를 미리 로딩합니다...")

    # 1. 튜닝에 필요한 최대 기간 계산
    app_settings = get_app_settings(country)
    if (
        not app_settings
        or "ma_period_etf" not in app_settings
        or "ma_period_stock" not in app_settings
    ):
        print(
            f"오류: '{country}' 국가의 전략 파라미터(MA 기간)가 설정되지 않았습니다. 웹 앱의 '설정' 탭에서 값을 지정해주세요."
        )
        return
    try:
        ma_period_etf = int(app_settings["ma_period_etf"])
        ma_period_stock = int(app_settings["ma_period_stock"])
    except (ValueError, TypeError):
        print(f"오류: '{country}' 국가의 MA 기간 설정이 올바르지 않습니다.")
        return

    max_months_range = max(test_months_ranges)
    max_strategy_ma = max(ma_period_etf, ma_period_stock)
    max_regime_ma = max(regime_ma_periods)
    warmup_days = int(max(max_strategy_ma, max_regime_ma) * 1.5)

    # 2. 데이터 로딩을 위한 날짜 범위 설정
    core_end_dt = pd.Timestamp.now()
    core_start_dt = core_end_dt - pd.DateOffset(months=max_months_range)

    # 3. DB에서 티커 목록 읽기
    stocks_from_db = get_stocks(country)
    if not stocks_from_db:
        print(f"오류: '{country}_stocks' 컬렉션에서 튜닝에 사용할 종목을 찾을 수 없습니다.")
        return
    tickers_to_process = [s["ticker"] for s in stocks_from_db]

    # 4. 모든 종목의 시세 데이터를 병렬로 미리 로딩합니다.
    prefetched_data = fetch_ohlcv_for_tickers(
        tickers_to_process,
        country=country,
        date_range=[core_start_dt.strftime("%Y-%m-%d"), core_end_dt.strftime("%Y-%m-%d")],
        warmup_days=warmup_days,
    )

    if not prefetched_data:
        print("오류: 튜닝에 사용할 데이터를 로드하지 못했습니다.")
        return
    print(f"총 {len(prefetched_data)}개 종목의 데이터 로딩 완료.")

    # --- 병렬 백테스트 실행 ---
    param_combinations = []
    for months_range in test_months_ranges:
        for regime_ma in regime_ma_periods:
            param_combinations.append((regime_ma, months_range, country))

    total_combinations = len(param_combinations)
    print(f"\n시장 레짐 필터의 이동평균 기간(MA Period) 튜닝을 시작합니다 ({country.upper()}).")
    print(f"총 {total_combinations}개의 조합을 테스트합니다...")
    print("=" * 60)

    start_time = time.time()
    results_by_month = {months: {} for months in test_months_ranges}

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(run_backtest_worker, params, prefetched_data)
            for params in param_combinations
        ]
        for future in as_completed(futures):
            try:
                regime_ma_period, months_range, result = future.result()
                if result and "cagr_pct" in result:
                    results_by_month[months_range][regime_ma_period] = result["cagr_pct"]
            except Exception as e:
                print(f"  -> 파라미터 테스트 중 오류 발생: {e}")

    end_time = time.time()
    print(f"\n\n파라미터 튜닝 완료! (총 소요 시간: {end_time - start_time:.2f}초)")

    # --- 결과 리포트 ---
    print("\n>>> 기간별 최적의 시장 레짐 필터 MA 기간 (CAGR 기준) <<<")
    summary_data = []
    for months in sorted(results_by_month.keys()):
        results_for_this_month = results_by_month.get(months)
        if not results_for_this_month:
            best_ma, best_cagr = "N/A", "N/A"
        else:
            best_ma = max(results_for_this_month, key=results_for_this_month.get)
            best_cagr = results_for_this_month[best_ma]
        summary_data.append(
            [
                f"{months}개월",
                str(best_ma),
                f"{best_cagr:.2f}%" if isinstance(best_cagr, float) else best_cagr,
            ]
        )

    from utils.report import render_table_eaw

    headers = ["백테스트 기간", "최적 MA 기간", "최고 CAGR"]
    aligns = ["left", "right", "right"]
    print("\n" + "\n".join(render_table_eaw(headers, summary_data, aligns)))
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="시장 레짐 필터의 이동평균 기간(MA Period)을 튜닝합니다."
    )
    parser.add_argument("country", choices=["kor", "aus"], help="튜닝을 진행할 시장 (kor, aus)")
    args = parser.parse_args()
    tune_regime_filter(country=args.country)
