import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Dict

from tqdm import tqdm

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from logics.seykota import settings as seykota_settings
from utils.report import format_kr_money


def run_backtest_worker(params: tuple) -> tuple:
    """
    단일 파라미터 조합에 대한 백테스트를 실행하는 워커 함수입니다.
    이 함수는 별도의 프로세스에서 실행됩니다.
    """
    fast_ma, slow_ma = params
    # 각 프로세스는 자체적인 모듈 컨텍스트를 가지므로, 여기서 다시 임포트하고 설정합니다.
    from test import main as run_test

    seykota_settings.SEYKOTA_FAST_MA = fast_ma
    seykota_settings.SEYKOTA_SLOW_MA = slow_ma

    result = run_test(strategy_name="seykota", country="kor", quiet=True)
    return fast_ma, slow_ma, result


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
                f"  -> 새로운 최고 {self.name} 발견! {self.best_value:.2f} (FAST={params['fast_ma']}, SLOW={params['slow_ma']})"
            )

    def print_report(self, title: str):
        """추적된 최적 지표에 대한 최종 리포트를 출력합니다."""
        if not self.params:
            return

        print("\n" + "=" * 50)
        print(f"\n[{title}]")
        print(f"  - SEYKOTA_FAST_MA: {self.params['fast_ma']}")
        print(f"  - SEYKOTA_SLOW_MA: {self.params['slow_ma']}")
        print(f"\n[{self.name} 기준 성과]")
        print(f"  - {self.name}: {self.best_value:.2f}")
        print(f"  - 기간: {self.result['start_date']} ~ {self.result['end_date']}")
        print(f"  - 최종 자산: {format_kr_money(self.result['final_value'])}")
        print(f"  - CAGR: {self.result['cagr_pct']:.2f}%")
        print(f"  - MDD: -{self.result['mdd_pct']:.2f}%")
        print(f"  - Calmar Ratio: {self.result.get('calmar_ratio', 0.0):.2f}")


def tune_parameters():
    """
    'seykota' 전략의 이동평균선 기간을 튜닝하여 최적의 값을 찾습니다.
    """
    # 튜닝할 파라미터 범위 설정
    fast_ma_range = range(1, 2)
    slow_ma_range = range(2, 50)

    # # 빠른 테스트를 위한 예시 범위 (필요시 아래 주석을 해제하고 위 두 줄을 주석 처리)
    # fast_ma_range = range(5, 51, 5)   # 5부터 50까지 5씩 증가
    # slow_ma_range = range(10, 201, 10) # 10부터 200까지 10씩 증가

    # 추적할 성능 지표들을 정의합니다.
    trackers = {
        "cagr": MetricTracker("CAGR", lambda a, b: a > b),
        "mdd": MetricTracker("MDD", lambda a, b: a < b),
        "calmar": MetricTracker("Calmar Ratio", lambda a, b: a > b),
        "sharpe": MetricTracker("Sharpe Ratio", lambda a, b: a > b),
        "sortino": MetricTracker("Sortino Ratio", lambda a, b: a > b),
    }

    # 모든 유효한 조합 생성
    param_combinations = []
    for fast_ma in fast_ma_range:
        for slow_ma in slow_ma_range:
            if slow_ma > fast_ma:
                param_combinations.append((fast_ma, slow_ma))

    total_combinations = len(param_combinations)
    print(f"'{'seykota'}' 전략 파라미터 튜닝을 시작합니다.")
    print(f"총 {total_combinations}개의 조합을 테스트합니다.")
    print("=" * 50)
    print("주의: 테스트에 매우 오랜 시간이 소요될 수 있습니다.")
    print("=" * 50)

    start_time = time.time()

    # ProcessPoolExecutor를 사용하여 병렬로 백테스트를 실행합니다.
    with ProcessPoolExecutor() as executor:
        # 모든 작업을 풀에 제출합니다.
        futures = [executor.submit(run_backtest_worker, params) for params in param_combinations]

        # 작업이 완료되는 대로 결과를 처리하며, tqdm으로 진행률을 표시합니다.
        for future in tqdm(as_completed(futures), total=total_combinations, desc="튜닝 진행률"):
            try:
                fast_ma, slow_ma, result = future.result()

                if result:
                    params = {"fast_ma": fast_ma, "slow_ma": slow_ma}
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

    print("\n\n" + "=" * 50)
    print("파라미터 튜닝 완료!")
    print(f"총 소요 시간: {elapsed_time:.2f}초")
    print("=" * 50)

    # 각 지표별 최적 결과를 출력합니다.
    trackers["cagr"].print_report("최고 수익률 (CAGR 기준)")
    trackers["mdd"].print_report("최저 위험 (MDD 기준)")
    trackers["calmar"].print_report("최고 효율 (Calmar Ratio 기준)")
    trackers["sharpe"].print_report("최고 효율 (Sharpe Ratio 기준)")
    trackers["sortino"].print_report("최고 효율 (Sortino Ratio 기준)")

    if not trackers["cagr"].params:
        print("\n유효한 결과를 찾지 못했습니다.")


if __name__ == "__main__":
    tune_parameters()
