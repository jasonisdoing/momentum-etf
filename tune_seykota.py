import time
import sys
import os
from concurrent.futures import ProcessPoolExecutor, as_completed  
from tqdm import tqdm

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.report import format_kr_money

def run_backtest_worker(params: tuple) -> tuple:
    """
    단일 파라미터 조합에 대한 백테스트를 실행하는 워커 함수입니다.
    이 함수는 별도의 프로세스에서 실행됩니다.
    """
    fast_ma, slow_ma = params
    # 각 프로세스는 자체적인 모듈 컨텍스트를 가지므로, 여기서 다시 임포트하고 설정합니다.
    from test import main as run_test
    from logics.seykota import settings as seykota_settings

    seykota_settings.SEYKOTA_FAST_MA = fast_ma
    seykota_settings.SEYKOTA_SLOW_MA = slow_ma

    result = run_test(strategy_name='seykota', quiet=True)
    return fast_ma, slow_ma, result

def tune_parameters():
    """
    'seykota' 전략의 이동평균선 기간을 튜닝하여 최적의 값을 찾습니다.
    """
    # 튜닝할 파라미터 범위 설정
    fast_ma_range = range(1, 101)
    slow_ma_range = range(2, 201)

    # # 빠른 테스트를 위한 예시 범위 (필요시 아래 주석을 해제하고 위 두 줄을 주석 처리)
    # fast_ma_range = range(5, 51, 5)   # 5부터 50까지 5씩 증가
    # slow_ma_range = range(10, 201, 10) # 10부터 200까지 10씩 증가

    best_params = None
    best_cagr = -float('inf')
    best_result = None

    best_mdd_params = None
    best_mdd = float('inf')
    best_mdd_result = None

    best_calmar_params = None
    best_calmar = -float('inf')
    best_calmar_result = None

    # 모든 유효한 조합 생성
    param_combinations = []
    for fast_ma in fast_ma_range:
        for slow_ma in slow_ma_range:
            if slow_ma > fast_ma:
                param_combinations.append((fast_ma, slow_ma))

    total_combinations = len(param_combinations)
    print(f"'{'seykota'}' 전략 파라미터 튜닝을 시작합니다.")
    print(f"총 {total_combinations}개의 조합을 테스트합니다.")
    print("="*50)
    print("주의: 테스트에 매우 오랜 시간이 소요될 수 있습니다.")
    print("="*50)

    start_time = time.time()

    # ProcessPoolExecutor를 사용하여 병렬로 백테스트를 실행합니다.
    with ProcessPoolExecutor() as executor:
        # 모든 작업을 풀에 제출합니다.
        futures = [executor.submit(run_backtest_worker, params) for params in param_combinations]
        
        # 작업이 완료되는 대로 결과를 처리하며, tqdm으로 진행률을 표시합니다.
        for future in tqdm(as_completed(futures), total=total_combinations, desc="튜닝 진행률"):
            try:
                fast_ma, slow_ma, result = future.result()
                
                if result and 'cagr_pct' in result and 'mdd_pct' in result:
                    cagr = result['cagr_pct']
                    mdd = result['mdd_pct']
                    calmar_ratio = cagr / mdd if mdd > 0 else 0
                    
                    # 최고 CAGR 업데이트
                    if cagr > best_cagr:
                        best_cagr = cagr
                        best_params = {'fast_ma': fast_ma, 'slow_ma': slow_ma}
                        best_result = result
                        # tqdm.write는 진행률 표시줄을 방해하지 않고 메시지를 출력합니다.
                        tqdm.write(f"  -> 새로운 최고 CAGR 발견! {cagr:.2f}% (FAST={fast_ma}, SLOW={slow_ma})")
                    
                    # 최저 MDD 업데이트
                    if mdd < best_mdd:
                        best_mdd = mdd
                        best_mdd_params = {'fast_ma': fast_ma, 'slow_ma': slow_ma}
                        best_mdd_result = result
                        tqdm.write(f"  -> 새로운 최저 MDD 발견! -{mdd:.2f}% (FAST={fast_ma}, SLOW={slow_ma})")

                    # 최고 Calmar Ratio 업데이트
                    if calmar_ratio > best_calmar:
                        best_calmar = calmar_ratio
                        best_calmar_params = {'fast_ma': fast_ma, 'slow_ma': slow_ma}
                        best_calmar_result = result
                        tqdm.write(f"  -> 새로운 최고 Calmar Ratio 발견! {calmar_ratio:.2f} (FAST={fast_ma}, SLOW={slow_ma})")

            except Exception as e:
                tqdm.write(f"  -> 파라미터 테스트 중 오류 발생: {e}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print("\n\n" + "="*50)
    print("파라미터 튜닝 완료!")
    print(f"총 소요 시간: {elapsed_time:.2f}초")
    print("="*50)

    if best_params:
        print("\n[최고 수익률 (CAGR 기준)]")
        print(f"  - SEYKOTA_FAST_MA: {best_params['fast_ma']}")
        print(f"  - SEYKOTA_SLOW_MA: {best_params['slow_ma']}")
        
        print("\n[최고 CAGR 성과]")
        print(f"  - 기간: {best_result['start_date']} ~ {best_result['end_date']}")
        print(f"  - 최종 자산: {format_kr_money(best_result['final_value'])}")
        print(f"  - 누적 수익률: {best_result['cumulative_return_pct']:.2f}%")
        print(f"  - CAGR: {best_result['cagr_pct']:.2f}%")
        print(f"  - MDD: -{best_result['mdd_pct']:.2f}%")
        calmar = best_result['cagr_pct'] / best_result['mdd_pct'] if best_result['mdd_pct'] > 0 else 0
        print(f"  - Calmar Ratio: {calmar:.2f}")

    if best_mdd_params:
        print("\n" + "="*50)
        print("\n[최저 위험 (MDD 기준)]")
        print(f"  - SEYKOTA_FAST_MA: {best_mdd_params['fast_ma']}")
        print(f"  - SEYKOTA_SLOW_MA: {best_mdd_params['slow_ma']}")
        
        print("\n[최저 MDD 성과]")
        print(f"  - 기간: {best_mdd_result['start_date']} ~ {best_mdd_result['end_date']}")
        print(f"  - 최종 자산: {format_kr_money(best_mdd_result['final_value'])}")
        print(f"  - 누적 수익률: {best_mdd_result['cumulative_return_pct']:.2f}%")
        print(f"  - CAGR: {best_mdd_result['cagr_pct']:.2f}%")
        print(f"  - MDD: -{best_mdd_result['mdd_pct']:.2f}%")
        calmar = best_mdd_result['cagr_pct'] / best_mdd_result['mdd_pct'] if best_mdd_result['mdd_pct'] > 0 else 0
        print(f"  - Calmar Ratio: {calmar:.2f}")

    if best_calmar_params:
        print("\n" + "="*50)
        print("\n[최고 효율 (Calmar Ratio 기준)]")
        print(f"  - SEYKOTA_FAST_MA: {best_calmar_params['fast_ma']}")
        print(f"  - SEYKOTA_SLOW_MA: {best_calmar_params['slow_ma']}")
        
        print("\n[최고 Calmar Ratio 성과]")
        print(f"  - Calmar Ratio (CAGR/MDD): {best_calmar:.2f}")
        print(f"  - 기간: {best_calmar_result['start_date']} ~ {best_calmar_result['end_date']}")
        print(f"  - 최종 자산: {format_kr_money(best_calmar_result['final_value'])}")
        print(f"  - 누적 수익률: {best_calmar_result['cumulative_return_pct']:.2f}%")
        print(f"  - CAGR: {best_calmar_result['cagr_pct']:.2f}%")
        print(f"  - MDD: -{best_calmar_result['mdd_pct']:.2f}%")

    if not best_params and not best_mdd_params and not best_calmar_params:
        print("\n유효한 결과를 찾지 못했습니다.")

if __name__ == '__main__':
    tune_parameters()