"""
코인(coin) 전용 전략 파라미터 튜닝 스크립트.

이 파일 상단의 설정을 수정해 바로 튜닝 범위/기간을 제어할 수 있습니다.

Usage:
  python scripts/tune_coin.py
"""

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Dict

import numpy as np
import pandas as pd

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- 로컬 튜닝 설정 (필요 시 수정) -----------------------------------------

# 백테스트 기간(개월)
TEST_MONTHS_RANGE = 12

# ATR 정규화 기간(워밍업 산정용)
ATR_PERIOD_FOR_NORMALIZATION = 14

# 파라미터 범위 (coin)
# MA_ETF_FIXED = 14  # 코인에는 ETF 없음. 더미 고정값
# MA_STOCK_RANGE = np.arange(5, 201, 5)          # 5 ~ 200 (step 5)
# REPLACE_THRESHOLD_RANGE = np.arange(0.5, 10.1, 0.5)  # 0.5 ~ 10.0 (step 0.5)
# PORTFOLIO_TOPN_RANGE = np.arange(1, 6, 1)      # 1 ~ 5
# REPLACE_WEAKER_STOCK = True


MA_ETF_FIXED = 14  # 코인에는 ETF 없음. 더미 고정값
MA_STOCK_RANGE = np.arange(1, 201, 1)
REPLACE_THRESHOLD_RANGE = np.arange(0.5, 10.1, 0.5)
PORTFOLIO_TOPN_RANGE = np.arange(1, 6, 1)      # 1 ~ 5
REPLACE_WEAKER_STOCK = True

# MA_ETF_FIXED = 14  # 코인에는 ETF 없음. 더미 고정값
# MA_STOCK_RANGE = [14]
# REPLACE_THRESHOLD_RANGE = [1.5]
# PORTFOLIO_TOPN_RANGE = [3]
# REPLACE_WEAKER_STOCK = True

# 튜닝 대상 티커 오버라이드
# 빈 리스트([])이면 DB의 coin_stocks 컬렉션에서 티커를 로드합니다.
# 예시: ["BTC", "ETH", "SOL"]
# TICKERS_OVERRIDE: list[str] = ["BTC"]
# TICKERS_OVERRIDE: list[str] = ["BTC", "ETH", "XRP", "BNB", "SOL"]
TICKERS_OVERRIDE: list[str] = ["BTC", "ETH", "XRP", "BNB", "SOL", "ADA", "DOGE", "TRX"]

# 12개월 - 5종목
# 210.15% (TopN: 3, MA: 3, Threshold: 1.50)
# 12개월 - 8종목
# 210.20% (TopN: 3, MA: 3, Threshold: 1.50)

# 현금 부족 규칙이 있는 경우 + 정수 구매
# 5종목 테스트 누적 수익률: +102.36%
# 8종목 테스트 누적 수익률: +179.62%

# 현금 부족 규칙 없고 + 코인 호주는 소수 구매
# 5종목 테스트 누적 수익률: +140.95%
# 8종목 테스트 누적 수익률: +182.65%


# 5개 종목 튜닝
# [수익률 기준 최적 조합]
#   - MA_PERIOD_FOR_STOCK: 3
#   - PORTFOLIO_TOPN: 3
#   - REPLACE_WEAKER_STOCK: True
#   - REPLACE_SCORE_THRESHOLD: 2.50

# [성과]
#   - 기간: 2024-09-14 ~ 2025-09-13
#   - 누적 수익률: +479.60%
#   - CAGR: +483.11%
# 8개 종목 튜닝
# [수익률 기준 최적 조합]
#   - MA_PERIOD_FOR_STOCK: 3
#   - PORTFOLIO_TOPN: 3
#   - REPLACE_WEAKER_STOCK: True
#   - REPLACE_SCORE_THRESHOLD: 2.50

# [성과]
#   - 기간: 2024-09-14 ~ 2025-09-13
#   - 누적 수익률: +481.08%
#   - CAGR: +484.60%

#  5종목, 8종목 큰 차이 없고 3 - 3 - 2.5가 최적
#   - MA_PERIOD_FOR_STOCK: 3
#   - PORTFOLIO_TOPN: 3
#   - REPLACE_SCORE_THRESHOLD: 2.50


from utils.db_manager import get_stocks
from utils.data_loader import fetch_ohlcv_for_tickers


# --- 워커 프로세스 전역 데이터 (초기화 1회) -----------------------------------
PREFETCHED_DATA: Dict[str, pd.DataFrame] | None = None

def _init_worker(prefetched: Dict[str, pd.DataFrame]):
    """Initializer for each worker process: store prefetched data once."""
    global PREFETCHED_DATA
    PREFETCHED_DATA = prefetched


def run_backtest_worker(params: tuple) -> tuple:
    """단일 파라미터 조합에 대한 백테스트(별도 프로세스)."""
    ma_etf, ma_stock, replace_stock, replace_threshold, portfolio_topn = params
    from test import main as run_test
    from logic import settings as worker_settings



    overrides = {
        "ma_etf": ma_etf,
        "ma_stock": ma_stock,
        "portfolio_topn": portfolio_topn,
        "replace_weaker_stock": replace_stock,
        "replace_threshold": replace_threshold,
        "test_months_range": TEST_MONTHS_RANGE,
    }
    # Use process-local prefetched data to avoid re-pickling per task
    result = run_test(country='coin', quiet=True, prefetched_data=PREFETCHED_DATA, override_settings=overrides)
    return ma_etf, ma_stock, replace_stock, replace_threshold, portfolio_topn, result


class MetricTracker:
    def __init__(self, name: str, compare_func: Callable[[float, float], bool], country: str):
        self.name = name
        self.compare_func = compare_func
        self.country = country
        self.best_value = -float("inf") if compare_func(1, 0) else float("inf")
        self.params = None
        self.result = None

    def update(self, value: float, params: Dict, result: Dict):
        if self.compare_func(value, self.best_value):
            self.best_value = value
            self.params = params
            self.result = result
            param_str = f"TopN={params['portfolio_topn']}, MA={params['ma_stock']}, Threshold={params['replace_threshold']:.2f}"
            print(f"  -> 새로운 최적 {self.name} 발견! {self.best_value:.2f} ({param_str})")

    def print_report(self, title: str):
        if not self.params:
            return
        from utils.report import format_kr_money
        print("\n" + "=" * 60)
        print(f"\n[{title}]")
        print(f"  - MA_PERIOD_FOR_STOCK: {self.params['ma_stock']}")
        print(f"  - PORTFOLIO_TOPN: {self.params['portfolio_topn']}")
        print(f"  - REPLACE_WEAKER_STOCK: {self.params['replace_stock']}")
        print(f"  - REPLACE_SCORE_THRESHOLD: {self.params['replace_threshold']:.2f}")
        print(f"\n[{self.name} 기준 성과]")
        print(f"  - {self.name}: {self.best_value:.2f}")
        print(f"  - 기간: {self.result['start_date']} ~ {self.result['end_date']}")
        print(f"  - 최종 자산: {format_kr_money(self.result['final_value'])}")
        print(f"  - CAGR: {self.result['cagr_pct']:.2f}%")
        print(f"  - MDD: -{self.result['mdd_pct']:.2f}%")
        print(f"  - Calmar Ratio: {self.result.get('calmar_ratio', 0.0):.2f}")


def tune_coin() -> tuple | None:
    # --- 데이터 사전 로딩 ---
    print("\n튜닝을 위해 COIN 시장의 데이터를 미리 로딩합니다...")

    max_ma_period = int(max(MA_STOCK_RANGE))
    atr_period_norm = int(ATR_PERIOD_FOR_NORMALIZATION)
    warmup_days = int(max(max_ma_period, atr_period_norm) * 1.5)

    core_end_dt = pd.Timestamp.now()
    core_start_dt = core_end_dt - pd.DateOffset(months=int(TEST_MONTHS_RANGE))
    test_date_range = [core_start_dt.strftime('%Y-%m-%d'), core_end_dt.strftime('%Y-%m-%d')]

    # 티커 소스: 오버라이드 > DB
    tickers_to_process: list[str]
    if TICKERS_OVERRIDE:
        tickers_to_process = [str(t).upper() for t in TICKERS_OVERRIDE if str(t).strip()]
        print(f"[INFO] Using override tickers: {', '.join(tickers_to_process)}")
    else:
        stocks_from_db = get_stocks('coin')
        if not stocks_from_db:
            print("오류: 'coin_stocks' 컬렉션에서 종목을 찾을 수 없습니다. 또는 상단 TICKERS_OVERRIDE를 설정하세요.")
            return None, None
        tickers_to_process = [s['ticker'] for s in stocks_from_db]

    prefetched_data = fetch_ohlcv_for_tickers(
        tickers_to_process, 'coin', date_range=test_date_range, warmup_days=warmup_days
    )
    if not prefetched_data:
        print("오류: 튜닝에 사용할 데이터를 로드하지 못했습니다.")
        return None, None
    print(f"총 {len(prefetched_data)}개 종목의 데이터 로딩 완료.")

    # 조합 생성
    param_combinations = []
    for ma_stock in MA_STOCK_RANGE:
        for portfolio_topn in PORTFOLIO_TOPN_RANGE:
            for replace_threshold in REPLACE_THRESHOLD_RANGE:
                param_combinations.append(
                    (MA_ETF_FIXED, int(ma_stock), REPLACE_WEAKER_STOCK, float(replace_threshold), int(portfolio_topn))
                )

    total = len(param_combinations)
    print(f"전략 파라미터 튜닝을 시작합니다 (COIN). 총 {total} 조합...")
    print("=" * 60)
    print("주의: 테스트에 시간이 오래 걸릴 수 있습니다.")
    print("=" * 60)

    start_t = time.time()

    # 수익률(누적 수익률 %) 기준 베스트 추적
    best_total_ret = -float("inf")
    best_params = None
    best_result = None

    def _fmt_time(sec: float) -> str:
        sec = max(0, int(sec))
        m, s = divmod(sec, 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    completed = 0
    next_report = 0
    report_step = max(1, total // 100)  # ~1% step

    # 워커/청크 설정 (환경변수로 오버라이드 가능)
    env_max = os.environ.get("TUNE_MAX_WORKERS")
    try:
        max_workers = int(env_max) if env_max else None
    except Exception:
        max_workers = None
    env_chunk = os.environ.get("TUNE_CHUNKSIZE")
    try:
        chunksize = int(env_chunk) if env_chunk else (max(1, total // ((os.cpu_count() or 2) * 4)) if total > 200 else 1)
    except Exception:
        chunksize = 1

    # 안내 로그: 초기 시딩에는 시간이 걸릴 수 있음
    print(f"[INFO] Starting pool: workers={max_workers or (os.cpu_count() or 1)}, tickers={len(tickers_to_process)}, combos={total}")
    print("[INFO] Seeding worker processes with prefetched data... (first result may take a while)")

    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_worker, initargs=(prefetched_data,)) as executor:
        futures = [executor.submit(run_backtest_worker, params) for params in param_combinations]
        print("[INFO] Dispatched all parameter jobs. Waiting for first completion...", flush=True)
        for fut in as_completed(futures):
            try:
                ma_etf, ma_stock, replace_stock, replace_threshold, portfolio_topn, result = fut.result()
                if result:
                    params = {
                        "ma_stock": ma_stock,
                        "portfolio_topn": portfolio_topn,
                        "replace_stock": replace_stock,
                        "replace_threshold": replace_threshold,
                    }
                    total_ret = float(result.get("cumulative_return_pct", -1e18))
                    if total_ret > best_total_ret:
                        best_total_ret = total_ret
                        best_params = params
                        best_result = result
            except Exception as e:
                print(f"  -> 파라미터 테스트 중 오류: {e}")
            finally:
                completed += 1
                if completed >= next_report or completed == total:
                    elapsed = time.time() - start_t
                    pct = (completed / total) * 100.0
                    # naive ETA
                    eta = (elapsed / completed) * (total - completed) if completed > 0 else 0
                    print(f"진행률: {completed}/{total} ({pct:5.1f}%) | 경과 {_fmt_time(elapsed)} | 예상잔여 {_fmt_time(eta)}", flush=True)
                    next_report = completed + report_step

    elapsed = time.time() - start_t
    print("\n\n" + "=" * 60 + "\n파라미터 튜닝 완료!\n" + f"총 소요 시간: {elapsed:.2f}초\n" + "=" * 60)

    if not best_params or not best_result:
        print("\n유효한 결과를 찾지 못했습니다.")
        return None, None

    # 수익률 기준 최적 결과 요약 출력
    start_date = best_result.get('start_date')
    end_date = best_result.get('end_date')
    cagr_pct = float(best_result.get('cagr_pct', 0.0))
    cum_ret_pct = float(best_result.get('cumulative_return_pct', 0.0))
    print("\n" + "=" * 60)
    print("[수익률 기준 최적 조합]")
    print(f"  - MA_PERIOD_FOR_STOCK: {best_params['ma_stock']}")
    print(f"  - PORTFOLIO_TOPN: {best_params['portfolio_topn']}")
    print(f"  - REPLACE_WEAKER_STOCK: {best_params['replace_stock']}")
    print(f"  - REPLACE_SCORE_THRESHOLD: {best_params['replace_threshold']:.2f}")
    print("\n[성과]")
    print(f"  - 기간: {start_date} ~ {end_date}")
    print(f"  - 누적 수익률: {cum_ret_pct:+.2f}%")
    print(f"  - CAGR: {cagr_pct:+.2f}%")

    return cum_ret_pct, best_params


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(">>> COIN 시장 최적화 시작...")
    print("=" * 60)
    best_ret, best_params = tune_coin()

    print("\n\n" + "=" * 60)
    print(">>> 최종 최적화 결과 요약 (최고 CAGR 기준) <<<")
    print("=" * 60)
    if best_ret is not None and best_params is not None:
        print(
            f"  - COIN 최적 누적수익률: {best_ret:.2f}% "
            f"(TopN: {best_params['portfolio_topn']}, MA: {best_params['ma_stock']}, Threshold: {best_params['replace_threshold']:.2f})"
        )
    else:
        print("  - COIN: 유효한 결과를 찾지 못했습니다.")
    print("=" * 60)
