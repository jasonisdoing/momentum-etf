"""
전략 파라미터 최적화(튜닝) 스크립트입니다.

이 스크립트는 여러 파라미터 조합에 대해 병렬로 백테스트를 실행하여,
최고의 성과(CAGR, MDD 등)를 내는 최적의 파라미터 조합을 찾습니다.
"""

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logic import settings
from utils.db_manager import get_stocks, get_app_settings
from utils.data_loader import fetch_ohlcv_for_tickers
from utils.report import format_aud_money
 
 
def run_backtest_worker(params: tuple, prefetched_data: Dict[str, pd.DataFrame]) -> tuple:
    """
    단일 파라미터 조합에 대한 백테스트를 실행하는 워커 함수입니다.
    이 함수는 별도의 프로세스에서 실행됩니다.
    """
    ma_etf, ma_stock, replace_stock, replace_threshold, portfolio_topn, country = params
    # 각 프로세스는 자체적인 모듈 컨텍스트를 가지므로, 여기서 다시 임포트하고 설정합니다.
    from test import main as run_test
    from logic import settings as worker_settings

    worker_settings.MA_PERIOD_FOR_ETF = ma_etf
    worker_settings.MA_PERIOD_FOR_STOCK = ma_stock
    worker_settings.PORTFOLIO_TOPN = portfolio_topn
    worker_settings.REPLACE_WEAKER_STOCK = replace_stock
    worker_settings.REPLACE_SCORE_THRESHOLD = replace_threshold

    # 미리 로드된 데이터를 전달하여 yfinance 호출을 방지합니다.
    result = run_test(country=country, quiet=True, prefetched_data=prefetched_data)
    return ma_etf, ma_stock, replace_stock, replace_threshold, portfolio_topn, result
 
 
class MetricTracker:
    """최적의 성능 지표를 추적하고 관리하는 헬퍼 클래스입니다."""
 
    def __init__(self, name: str, compare_func: Callable[[float, float], bool], money_formatter: Callable, country: str):
        """
        Args:
            name (str): 지표의 이름 (예: "CAGR", "MDD")
            compare_func (Callable): 두 값을 비교하는 함수. (예: a > b 이면 더 큰 값을, a < b 이면 더 작은 값을 선택)
            money_formatter (Callable): 금액 포맷팅 함수
        """
        self.name = name
        self.compare_func = compare_func
        self.money_formatter = money_formatter
        self.country = country
        # 비교 함수에 따라 초기값을 무한대 또는 음의 무한대로 설정
        self.best_value = -float("inf") if compare_func(1, 0) else float("inf")
        self.params = None
        self.result = None
 
    def update(self, value: float, params: Dict, result: Dict):
        """새로운 값이 기존의 최고/최저 값보다 나은 경우, 정보를 업데이트합니다."""
        if self.compare_func(value, self.best_value):
            self.best_value = value
            self.params = params
            self.result = result

            param_str = f"Threshold={params['replace_threshold']:.2f}"
            # 'portfolio_topn'이 params에 있고, 기본값과 다를 경우 (즉, coin 튜닝일 경우)
            if self.country == 'coin':
                param_str = f"TopN={params['portfolio_topn']}, MA={params['ma_stock']}, {param_str}"

            tqdm.write(
                f"  -> 새로운 최적 {self.name} 발견! {self.best_value:.2f} ({param_str})"
            )
 
    def print_report(self, title: str):
        """추적된 최적 지표에 대한 최종 리포트를 출력합니다."""
        if not self.params:
            return
 
        print("\n" + "=" * 60)
        print(f"\n[{title}]")
        print(f"  - MA_PERIOD_FOR_ETF: {self.params['ma_etf']}")
        print(f"  - MA_PERIOD_FOR_STOCK: {self.params['ma_stock']}")
        if 'portfolio_topn' in self.params:
            print(f"  - PORTFOLIO_TOPN: {self.params['portfolio_topn']}")
        print(f"  - REPLACE_WEAKER_STOCK: {self.params['replace_stock']}")
        print(f"  - REPLACE_SCORE_THRESHOLD: {self.params['replace_threshold']:.2f}")
        print(f"\n[{self.name} 기준 성과]")
        print(f"  - {self.name}: {self.best_value:.2f}")
        print(f"  - 기간: {self.result['start_date']} ~ {self.result['end_date']}")
        print(f"  - 최종 자산: {self.money_formatter(self.result['final_value'])}")
        print(f"  - CAGR: {self.result['cagr_pct']:.2f}%")
        print(f"  - MDD: -{self.result['mdd_pct']:.2f}%")
        print(f"  - Calmar Ratio: {self.result.get('calmar_ratio', 0.0):.2f}")
 
 
def tune_parameters(country: str):
    """
    전략의 파라미터를 튜닝하여 최적의 값을 찾습니다.
    """
    # --- 튜닝할 파라미터 범위 설정 ---
    # 다른 파라미터는 settings.py의 기본값으로 고정합니다.
    ma_etf_range = [settings.MA_PERIOD_FOR_ETF]
    ma_stock_range = [settings.MA_PERIOD_FOR_STOCK]
    # 임계값 튜닝은 교체매매가 True일 때만 의미가 있습니다.
    replace_stock_options = [True]
    replace_threshold_range = np.arange(0.5, 5.1, 0.5)  # 0.5부터 5.0까지 0.5 간격으로 테스트합니다.

    if country == "coin":
        # 가상화폐의 경우, 더 넓은 범위의 파라미터를 튜닝합니다.
        # MA_PERIOD_FOR_STOCK: 5부터 200까지 5 간격
        ma_stock_range = np.arange(5, 201, 5)
        # REPLACE_SCORE_THRESHOLD: 0.5부터 10.0까지 0.5 간격
        replace_threshold_range = np.arange(0.5, 10.1, 0.5)
        # PORTFOLIO_TOPN: 1부터 5까지 1 간격
        portfolio_topn_range = np.arange(1, 6, 1)
    else:
        # 한국/호주의 경우, DB에서 PORTFOLIO_TOPN 값을 가져옵니다.
        app_settings_from_db = get_app_settings(country)
        db_topn = app_settings_from_db.get("portfolio_topn") if app_settings_from_db else None
        if not db_topn:
            print(f"오류: {country.upper()} 국가의 PORTFOLIO_TOPN 설정이 DB에 없습니다. 웹 앱의 '설정' 탭에서 값을 지정해주세요.")
            return None, None
        portfolio_topn_range = [int(db_topn)]


    # 국가별 포맷터 설정
    if country == "aus":
        money_formatter = format_aud_money
    else:  # kor, coin
        from utils.report import format_kr_money
        money_formatter = format_kr_money

    # 추적할 성능 지표들을 정의합니다.
    trackers = {
        "cagr": MetricTracker("CAGR", lambda a, b: a > b, money_formatter, country),
        "mdd": MetricTracker("MDD", lambda a, b: a < b, money_formatter, country),
        "calmar": MetricTracker("Calmar Ratio", lambda a, b: a > b, money_formatter, country),
        "sharpe": MetricTracker("Sharpe Ratio", lambda a, b: a > b, money_formatter, country),
        "sortino": MetricTracker("Sortino Ratio", lambda a, b: a > b, money_formatter, country),
    }
 
    # --- 데이터 사전 로딩 (yfinance 호출 최소화) ---
    print(f"\n튜닝을 위해 {country.upper()} 시장의 데이터를 미리 로딩합니다...")
 
    # 1. 튜닝에 필요한 최대 기간 계산
    max_ma_period = max(max(ma_etf_range), max(ma_stock_range))
    atr_period_norm = int(getattr(settings, "ATR_PERIOD_FOR_NORMALIZATION", 20))
    warmup_days = int(max(max_ma_period, atr_period_norm) * 1.5)
 
    # 2. logic 설정에서 백테스트 기간 가져오기
    test_months_range = getattr(settings, "TEST_MONTHS_RANGE", 12)
    core_end_dt = pd.Timestamp.now()
    core_start_dt = core_end_dt - pd.DateOffset(months=test_months_range)
    test_date_range = [core_start_dt.strftime('%Y-%m-%d'), core_end_dt.strftime('%Y-%m-%d')]
 
    stocks_from_db = get_stocks(country)
    if not stocks_from_db:
        print(f"오류: '{country}_stocks' 컬렉션에서 튜닝에 사용할 종목을 찾을 수 없습니다.")
        return
    tickers_to_process = [s['ticker'] for s in stocks_from_db]
 
    # 3. 모든 종목의 시세 데이터를 병렬로 미리 로딩합니다.
    prefetched_data = fetch_ohlcv_for_tickers(
        tickers_to_process, country, date_range=test_date_range, warmup_days=warmup_days
    )
 
    if not prefetched_data:
        print("오류: 튜닝에 사용할 데이터를 로드하지 못했습니다.")
        return
    print(f"총 {len(prefetched_data)}개 종목의 데이터 로딩 완료.")
 
    # 모든 유효한 조합 생성
    param_combinations = []
    for ma_etf in ma_etf_range:
        for ma_stock in ma_stock_range:
            for portfolio_topn in portfolio_topn_range:
                for replace_stock in replace_stock_options:
                    for replace_threshold in replace_threshold_range:
                        param_combinations.append(
                            (ma_etf, ma_stock, replace_stock, replace_threshold, portfolio_topn, country)
                        )
 
    total_combinations = len(param_combinations)
    print(f"전략 파라미터 튜닝을 시작합니다 ({country.upper()}).")
    print(f"총 {total_combinations}개의 조합을 테스트합니다...")
    print("=" * 60)
    print("주의: 테스트에 매우 오랜 시간이 소요될 수 있습니다.")
    print("=" * 60)
 
    start_time = time.time()
 
    # ProcessPoolExecutor를 사용하여 병렬로 백테스트를 실행합니다.
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(run_backtest_worker, params, prefetched_data)
            for params in param_combinations
        ]
 
        for future in tqdm(as_completed(futures), total=total_combinations, desc="튜닝 진행률"):
            try:
                ma_etf, ma_stock, replace_stock, replace_threshold, portfolio_topn, result = future.result()
 
                if result:
                    params = {
                        "ma_etf": ma_etf,
                        "ma_stock": ma_stock,
                        "portfolio_topn": portfolio_topn,
                        "replace_stock": replace_stock,
                        "replace_threshold": replace_threshold,
                    }
                    trackers["cagr"].update(result.get("cagr_pct", -float("inf")), params, result)
                    trackers["mdd"].update(result.get("mdd_pct", float("inf")), params, result)
                    trackers["calmar"].update(
                        result.get("calmar_ratio", -float("inf")), params, result
                    )
                    trackers["sharpe"].update(
                        result.get("sharpe_ratio", -float("inf")), params, result
                    )
                    trackers["sortino"].update(
                        result.get("sortino_ratio", -float("inf")), params, result
                    )
            except Exception as e:
                tqdm.write(f"  -> 파라미터 테스트 중 오류 발생: {e}")
 
    end_time = time.time()
    elapsed_time = end_time - start_time
 
    print("\n\n" + "=" * 60 + "\n파라미터 튜닝 완료!\n" + f"총 소요 시간: {elapsed_time:.2f}초\n" + "=" * 60)
 
    trackers["cagr"].print_report("최고 수익률 (CAGR 기준)")
    trackers["mdd"].print_report("최저 위험 (MDD 기준)")
    trackers["calmar"].print_report("최고 효율 (Calmar Ratio 기준)")
    trackers["sharpe"].print_report("최고 효율 (Sharpe Ratio 기준)")
    trackers["sortino"].print_report("최고 효율 (Sortino Ratio 기준)")
 
    if not trackers["cagr"].params:
        print("\n유효한 결과를 찾지 못했습니다.")
        return None, None

    return trackers["cagr"].best_value, trackers["cagr"].params
 
 
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="전략 파라미터 최적화를 시작합니다.")
    parser.add_argument(
        "country",
        choices=["kor", "aus", "coin"],
        help="튜닝을 진행할 시장 (kor, aus, coin)",
    )
    args = parser.parse_args()
    country = args.country

    print("\n" + "=" * 60)
    print(f">>> {country.upper()} 시장 최적화 시작...")
    print("=" * 60)
    best_cagr, best_params = tune_parameters(country=country)

    print("\n\n" + "=" * 60)
    print(">>> 최종 최적화 결과 요약 (최고 CAGR 기준) <<<")
    print("=" * 60)
    if best_cagr is not None and best_params is not None:
        if country == 'coin':
            print(
                f"  - {country.upper()} 최적 CAGR: {best_cagr:.2f}% "
                f"(TopN: {best_params['portfolio_topn']}, MA: {best_params['ma_stock']}, Threshold: {best_params['replace_threshold']:.2f})"
            )
        else:
            print(
                f"  - {country.upper()} 최적 CAGR: {best_cagr:.2f}% (Threshold: {best_params['replace_threshold']:.2f})"
            )
    else:
        print(f"  - {country.upper()}: 유효한 결과를 찾지 못했습니다.")
    print("=" * 60)