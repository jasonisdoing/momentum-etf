"""
MomentumPilot 프로젝트의 한국 시장용 메인 실행 파일입니다.
"""

import argparse
import os
import sys
import warnings

# 프로젝트 루트를 Python 경로에 추가합니다.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")


def main():
    """CLI 인자를 파싱하여 해당 모듈을 실행합니다."""
    parser = argparse.ArgumentParser(description="MomentumPilot 트레이딩 엔진 (KOR)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--test",
        nargs="?",
        const="__COMPARE__",
        default=None,
        help="백테스터(test.py)를 실행합니다. 전략 이름을 지정하면 해당 전략만, 없으면 'jason', 'seykota', 'donchian'의 요약 비교를 실행합니다.",
    )
    group.add_argument(
        "--status",
        type=str,
        help="지정된 전략으로 오늘의 현황(status.py)을 실행합니다. (예: jason, seykota, donchian)",
    )
    parser.add_argument(
        "--portfolio",
        type=str,
        default=None,
        help="포트폴리오 스냅샷 파일 경로. 미지정 시 최신 파일 사용. (예: data/kor/portfolio_2024-01-01.json)",
    )

    args = parser.parse_args()
    country = "kor"

    if args.test is not None:
        from test import main as run_test

        import pandas as pd

        import settings
        from utils.report import generate_strategy_comparison_report

        if args.test == "__COMPARE__":
            # `python kor.py --test`
            strategies_to_compare = ["jason", "seykota", "donchian"]
            print(
                f"전략 비교 백테스트를 실행합니다: {', '.join([f'{s}' for s in strategies_to_compare])}"
            )

            all_results = []
            for strategy in strategies_to_compare:
                results = run_test(strategy_name=strategy, country=country, quiet=True)
                if results:
                    all_results.append(results)
                print(f"'{strategy}' 전략 백테스트 완료.")

            if not all_results:
                print("\n오류: 비교할 백테스트 결과가 없습니다.")
                # Check for a common reason: future date range
                try:
                    test_date_range = getattr(settings, "TEST_DATE_RANGE", None)
                    if test_date_range and len(test_date_range) == 2:
                        end_date = pd.to_datetime(test_date_range[1])
                        if end_date > pd.Timestamp.now():
                            print(
                                f"      원인: settings.py의 TEST_DATE_RANGE 종료일({end_date.strftime('%Y-%m-%d')})이 미래로 설정되어 데이터를 가져올 수 없습니다."
                            )
                except Exception:
                    pass  # Ignore parsing errors, just provide the generic message
                return
            report = generate_strategy_comparison_report(all_results, country=country)
            print(report)
        else:
            # `python kor.py --test <strategy_name>`
            strategy_name = args.test
            print(f"'{strategy_name}' 전략에 대한 상세 백테스트를 실행합니다...")
            run_test(strategy_name=strategy_name, country=country, quiet=False)

    elif args.status:
        strategy_name = args.status
        from status import main as run_status

        run_status(strategy_name=strategy_name, country=country, portfolio_path=args.portfolio)


if __name__ == "__main__":
    main()