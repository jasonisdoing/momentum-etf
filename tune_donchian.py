import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Dict
 
import numpy as np
import pandas as pd
from tqdm import tqdm
 
# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
 
from logic import settings
from utils.report import format_aud_money
from utils.db_manager import get_stocks
 
 
def run_backtest_worker(params: tuple, prefetched_data: Dict[str, pd.DataFrame]) -> tuple:
    """
    단일 파라미터 조합에 대한 백테스트를 실행하는 워커 함수입니다.
    이 함수는 별도의 프로세스에서 실행됩니다.
    """
    ma_etf, ma_stock, replace_stock, replace_threshold, country = params
    # 각 프로세스는 자체적인 모듈 컨텍스트를 가지므로, 여기서 다시 임포트하고 설정합니다.
    from test import main as run_test
    from logic import settings as worker_settings
 
    worker_settings.MA_PERIOD_FOR_ETF = ma_etf
    worker_settings.MA_PERIOD_FOR_STOCK = ma_stock
    worker_settings.REPLACE_WEAKER_STOCK = replace_stock
    worker_settings.REPLACE_SCORE_THRESHOLD = replace_threshold
 
    # 미리 로드된 데이터를 전달하여 yfinance 호출을 방지합니다.
    result = run_test(country=country, quiet=True, prefetched_data=prefetched_data)
    return ma_etf, ma_stock, replace_stock, replace_threshold, result
 
 
class MetricTracker:
    """최적의 성능 지표를 추적하고 관리하는 헬퍼 클래스입니다."""
 
    def __init__(self, name: str, compare_func: Callable):
        """
        Args:
            name (str): 지표의 이름 (예: "CAGR", "MDD")
            compare_func (Callable): 두 값을 비교하는 함수. (예: a > b 이면 더 큰 값을, a < b 이면 더 작은 값을 선택)
        """
        self.name = name
        self.compare_func = compare_func
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
            tqdm.write(
                f"  -> 새로운 최고 {self.name} 발견! {self.best_value:.2f} (Threshold={params['replace_threshold']:.2f})"
            )
 
    def print_report(self, title: str):
        """추적된 최적 지표에 대한 최종 리포트를 출력합니다."""
        if not self.params:
            return
 
        print("\n" + "=" * 50)
        print(f"\n[{title}]")
        print(f"  - MA_PERIOD_FOR_ETF: {self.params['ma_etf']}")
        print(f"  - MA_PERIOD_FOR_STOCK: {self.params['ma_stock']}")
        print(f"  - REPLACE_WEAKER_STOCK: {self.params['replace_stock']}")
        print(f"  - REPLACE_SCORE_THRESHOLD: {self.params['replace_threshold']:.2f}")
        print(f"\n[{self.name} 기준 성과]")
        print(f"  - {self.name}: {self.best_value:.2f}")
        print(f"  - 기간: {self.result['start_date']} ~ {self.result['end_date']}")
        print(f"  - 최종 자산: {format_aud_money(self.result['final_value'])}")
        print(f"  - CAGR: {self.result['cagr_pct']:.2f}%")
        print(f"  - MDD: -{self.result['mdd_pct']:.2f}%")
        print(f"  - Calmar Ratio: {self.result.get('calmar_ratio', 0.0):.2f}")
 
 
def tune_parameters(country: str):
    """
    전략의 파라미터를 튜닝하여 최적의 값을 찾습니다.
    """
    # 튜닝할 파라미터 범위 설정
    # 다른 파라미터는 settings.py의 기본값으로 고정합니다.
    ma_etf_range = [settings.MA_PERIOD_FOR_ETF]
    ma_stock_range = [settings.MA_PERIOD_FOR_STOCK]
    # 임계값 튜닝은 교체매매가 True일 때만 의미가 있습니다.
    replace_stock_options = [True]
    replace_threshold_range = np.arange(0.5, 5.1, 0.5)  # 0.5부터 5.0까지 0.5 간격으로 테스트
 
    # 추적할 성능 지표들을 정의합니다.
    trackers = {
        "cagr": MetricTracker("CAGR", lambda a, b: a > b),
        "mdd": MetricTracker("MDD", lambda a, b: a < b),
        "calmar": MetricTracker("Calmar Ratio", lambda a, b: a > b),
        "sharpe": MetricTracker("Sharpe Ratio", lambda a, b: a > b),
        "sortino": MetricTracker("Sortino Ratio", lambda a, b: a > b),
    }
 
    # --- 데이터 사전 로딩 (yfinance 호출 최소화) ---
    print(f"튜닝을 위해 {country.upper()} 시장의 데이터를 미리 로딩합니다...")
 
    # 1. 튜닝에 필요한 최대 기간 계산
    max_ma_period = max(max(ma_etf_range), max(ma_stock_range))
    warmup_days = int(max_ma_period * 1.5)
 
    # 2. logic 설정에서 백테스트 기간 가져오기
    test_date_range = getattr(settings, "TEST_DATE_RANGE", None)
    if not test_date_range or len(test_date_range) != 2:
        print("오류: settings.py에 TEST_DATE_RANGE가 올바르게 설정되지 않았습니다.")
        return
 
    core_start = pd.to_datetime(test_date_range[0])
    warmup_start = core_start - pd.DateOffset(days=warmup_days)
    adjusted_date_range = [warmup_start.strftime("%Y-%m-%d"), test_date_range[1]]
 
    # 3. DB에서 티커 목록 읽기
    stocks_from_db = get_stocks(country)
    if not stocks_from_db:
        print(f"오류: '{country}_stocks' 컬렉션에서 튜닝에 사용할 종목을 찾을 수 없습니다.")
        return
    tickers_to_process = [s['ticker'] for s in stocks_from_db]
 
    # 4. 데이터 로딩
    prefetched_data = {}
    for tkr in tqdm(tickers_to_process, desc="데이터 로딩"):
        df = fetch_ohlcv(tkr, country=country, date_range=adjusted_date_range)
        if df is not None and not df.empty:
            prefetched_data[tkr] = df
 
    if not prefetched_data:
        print("오류: 튜닝에 사용할 데이터를 로드하지 못했습니다.")
        return
    print(f"총 {len(prefetched_data)}개 종목의 데이터 로딩 완료.")
    # --- 데이터 사전 로딩 완료 ---
 
    # 모든 유효한 조합 생성
    param_combinations = []
    for ma_etf in ma_etf_range:
        for ma_stock in ma_stock_range:
            for replace_stock in replace_stock_options:
                for replace_threshold in replace_threshold_range:
                    param_combinations.append(
                        (ma_etf, ma_stock, replace_stock, replace_threshold, country)
                    )
 
    total_combinations = len(param_combinations)
    print(f"전략의 교체 임계값(Threshold) 튜닝을 시작합니다 ({country.upper()}).")
    print(f"총 {total_combinations}개의 조합을 테스트합니다.")
    print("=" * 50)
    print("주의: 테스트에 매우 오랜 시간이 소요될 수 있습니다.")
    print("=" * 50)
 
    start_time = time.time()
 
    # ProcessPoolExecutor를 사용하여 병렬로 백테스트를 실행합니다.
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(run_backtest_worker, params, prefetched_data)
            for params in param_combinations
        ]
 
        for future in tqdm(as_completed(futures), total=total_combinations, desc="튜닝 진행률"):
            try:
                ma_etf, ma_stock, replace_stock, replace_threshold, result = future.result()
 
                if result:
                    params = {
                        "ma_etf": ma_etf,
                        "ma_stock": ma_stock,
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
 
    print("\n\n" + "=" * 50 + "\n파라미터 튜닝 완료!\n" + f"총 소요 시간: {elapsed_time:.2f}초\n" + "=" * 50)
 
    trackers["cagr"].print_report("최고 수익률 (CAGR 기준)")
    trackers["mdd"].print_report("최저 위험 (MDD 기준)")
    trackers["calmar"].print_report("최고 효율 (Calmar Ratio 기준)")
    trackers["sharpe"].print_report("최고 효율 (Sharpe Ratio 기준)")
    trackers["sortino"].print_report("최고 효율 (Sortino Ratio 기준)")
 
    if not trackers["cagr"].params:
        print("\n유효한 결과를 찾지 못했습니다.")
        return None, None

    return trackers["cagr"].best_value, trackers["cagr"].params["replace_threshold"]
 
 
if __name__ == "__main__":
    print("전략 파라미터 최적화를 시작합니다.")

    print("\n" + "=" * 50)
    print(">>> 한국(KOR) 시장 최적화 시작...")
    print("=" * 50)
    kor_best_cagr, kor_best_threshold = tune_parameters(country="kor")

    print("\n" + "=" * 50)
    print(">>> 호주(AUS) 시장 최적화 시작...")
    print("=" * 50)
    aus_best_cagr, aus_best_threshold = tune_parameters(country="aus")

    print("\n\n" + "=" * 50)
    print(">>> 최종 최적화 결과 요약 (최고 CAGR 기준) <<<")
    print("=" * 50)
    if kor_best_cagr is not None and kor_best_threshold is not None:
        print(
            f"  - 한국 (KOR) 최적 CAGR: {kor_best_cagr:.2f}% (Threshold: {kor_best_threshold:.2f})"
        )
    else:
        print("  - 한국 (KOR): 유효한 결과를 찾지 못했습니다.")
    if aus_best_cagr is not None and aus_best_threshold is not None:
        print(
            f"  - 호주 (AUS) 최적 CAGR: {aus_best_cagr:.2f}% (Threshold: {aus_best_threshold:.2f})"
        )
    else:
        print("  - 호주 (AUS): 유효한 결과를 찾지 못했습니다.")
    print("=" * 50)