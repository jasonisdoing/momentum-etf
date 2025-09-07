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

def main():
    """CLI 인자를 파싱하여 해당 모듈을 실행합니다."""
    parser = argparse.ArgumentParser(description="MomentumPilot 트레이딩 엔진")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--test', action='store_true', help='백테스터(test.py)를 실행합니다')
    group.add_argument('--today', action='store_true', help='오늘의 액션 플랜(today.py)을 실행합니다')
    parser.add_argument('--strategy', type=str, default='jason', help='사용할 전략의 이름을 지정합니다 (기본값: jason).')
    parser.add_argument('--portfolio', type=str, default=None, help='포트폴리오 스냅샷 파일 경로. 미지정 시 최신 파일 사용. (예: data/portfolio_2024-01-01.json)')

    args = parser.parse_args()

    if args.test:
        from test import main as run_test
        run_test(strategy_name=args.strategy, portfolio_path=args.portfolio)
    elif args.today:
        from today import main as run_today
        run_today(portfolio_path=args.portfolio)

if __name__ == '__main__':
    main()