"""
MomentumPilot 프로젝트 메인 실행 파일.
"""
import argparse
import sys
import os
import warnings

# Add project root to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

def _fmt_money_kr(v: float) -> str:
    """금액을 '억'과 '만원' 단위의 한글 문자열로 포맷합니다."""
    if v is None:
        return "-"
    man = int(round(v / 10_000))
    if man >= 10_000:
        uk = man // 10_000
        rem = man % 10_000
        return f"{uk}억 {rem:,}만원" if rem > 0 else f"{uk}억"
    return f"{man:,}만원"


def main():
    """CLI 인자를 파싱하여 해당 모듈을 실행합니다."""
    parser = argparse.ArgumentParser(description="MomentumPilot 트레이딩 엔진")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--test', nargs='?', const='__COMPARE__', default=None, help="백테스터(test.py)를 실행합니다. 전략 이름을 지정하면 해당 전략만, 없으면 'jason'과 'seykota'의 요약 비교를 실행합니다.")
    group.add_argument('--today', action='store_true', help='오늘의 액션 플랜(today.py)을 실행합니다')
    parser.add_argument('--portfolio', type=str, default=None, help='포트폴리오 스냅샷 파일 경로. 미지정 시 최신 파일 사용. (예: data/portfolio_2024-01-01.json)')

    args = parser.parse_args()

    if args.test is not None:
        from test import main as run_test
        from utils.report import render_table_eaw
        import pandas as pd
        import settings

        if args.test == '__COMPARE__':
            # `python main.py --test`
            print("전략 비교 백테스트를 실행합니다: 'jason' vs 'seykota'")
            jason_results = run_test(strategy_name='jason', portfolio_path=args.portfolio, quiet=True)
            print("'jason' 전략 백테스트 완료.")
            seykota_results = run_test(strategy_name='seykota', portfolio_path=args.portfolio, quiet=True)
            print("'seykota' 전략 백테스트 완료.")

            all_results = [res for res in [jason_results, seykota_results] if res]
            if not all_results:
                print("\n비교할 결과가 없습니다.")
                return

            headers = ["전략", "기간", "CAGR (연간 복리 성장률)", "MDD (최대 낙폭)", "누적수익률", "최종자산"]
            rows = [[r['strategy'], f"{r['start_date']}~{r['end_date']}", f"{r['cagr_pct']:.2f}%", f"-{r['mdd_pct']:.2f}%", f"{r['cumulative_return_pct']:.2f}%", _fmt_money_kr(r['final_value'])] for r in all_results]
            aligns = ['left', 'left', 'right', 'right', 'right', 'right']
            table_lines = render_table_eaw(headers, rows, aligns)
            
            initial_capital = all_results[0]['initial_capital']

            print("\n" + "="*30 + "\n" + " 전략 비교 결과 요약 ".center(30, "=") + "\n" + "="*30)
            print(f"(초기 자본: {_fmt_money_kr(initial_capital)})")
            print("\n".join(table_lines))

            # 월별 성과 비교
            all_monthly_returns = {
                res['strategy']: res['monthly_returns']
                for res in all_results if 'monthly_returns' in res and not res['monthly_returns'].empty
            }
            all_yearly_returns = {
                res['strategy']: res['yearly_returns']
                for res in all_results if 'yearly_returns' in res and not res['yearly_returns'].empty
            }
            all_monthly_cum_returns = {
                res['strategy']: res['monthly_cum_returns']
                for res in all_results if 'monthly_cum_returns' in res and not res['monthly_cum_returns'].empty
            }

            if all_monthly_returns:
                monthly_df_check = pd.DataFrame(all_monthly_returns)
                if monthly_df_check.empty:
                    print("\n월별 수익률 데이터가 없어 비교를 건너뜁니다.")
                else:
                    print("\n" + "="*30 + "\n" + " 월별 성과 비교 ".center(30, "=") + "\n" + "="*30)
                    print("(괄호 안은 누적 수익률)")

                    monthly_df = monthly_df_check.mul(100)
                    yearly_df = pd.DataFrame(all_yearly_returns).mul(100)
                    monthly_cum_df = pd.DataFrame(all_monthly_cum_returns).mul(100)
                    strategy_names = list(all_monthly_returns.keys())

                    all_years_set = set(monthly_df.index.year)
                    if not yearly_df.empty:
                        all_years_set.update(yearly_df.index.year)
                    
                    all_years = sorted(list(all_years_set))

                    for year in all_years:
                        print(f"\n--- {year}년 ---")
                        
                        year_monthly_df = monthly_df[monthly_df.index.year == year]
                        
                        headers = ["월"] + strategy_names
                        rows_data = []
                        
                        for month in range(1, 13):
                            month_data = [f"{month}월"]
                            month_end_dt = pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)
                            
                            if month_end_dt in year_monthly_df.index:
                                month_ret_row = year_monthly_df.loc[month_end_dt]
                                cum_ret_row = None
                                if not monthly_cum_df.empty and month_end_dt in monthly_cum_df.index:
                                    cum_ret_row = monthly_cum_df.loc[month_end_dt]
                                for strategy in strategy_names:
                                    val = month_ret_row.get(strategy)
                                    cum_val = cum_ret_row.get(strategy) if cum_ret_row is not None else None
                                    cell_str = f"{val:+.2f}%" if pd.notna(val) else "-"
                                    if pd.notna(val) and pd.notna(cum_val):
                                        cell_str += f" ({cum_val:+.2f}%)"
                                    month_data.append(cell_str)
                            else:
                                month_data.extend(["-"] * len(strategy_names))
                            rows_data.append(month_data)
                            
                        yearly_data = ["연간"]
                        year_end_dt = pd.Timestamp(year, 12, 31) + pd.offsets.YearEnd(0)
                        if not yearly_df.empty and year_end_dt in yearly_df.index:
                            year_row = yearly_df.loc[year_end_dt]
                            for strategy in strategy_names:
                                val = year_row.get(strategy)
                                yearly_data.append(f"{val:+.2f}%" if pd.notna(val) else "-")
                        else:
                            yearly_data.extend(["-"] * len(strategy_names))
                        rows_data.append(yearly_data)

                        aligns = ['left'] + ['right'] * len(strategy_names)
                        print("\n".join(render_table_eaw(headers, rows_data, aligns)))
        else:
            # `python main.py --test <strategy_name>`
            strategy_name = args.test
            print(f"'{strategy_name}' 전략에 대한 상세 백테스트를 실행합니다...")
            run_test(strategy_name=strategy_name, portfolio_path=args.portfolio, quiet=False)

    elif args.today:
        from today import main as run_today
        run_today(portfolio_path=args.portfolio)

if __name__ == '__main__':
    main()