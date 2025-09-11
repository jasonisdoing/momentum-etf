"""
MomentumPilot 프로젝트의 메인 실행 파일입니다.
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
    parser = argparse.ArgumentParser(description="MomentumPilot Trading Engine")
    parser.add_argument("country", choices=["kor", "aus"], help="실행할 포트폴리오 국가 (kor, aus)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--test",
        action="store_true",
        help="백테스터(test.py)를 실행합니다.",
    )
    group.add_argument(
        "--status",
        action="store_true",
        help="오늘의 현황(status.py)을 실행합니다.",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="조회할 포트폴리오 스냅샷의 날짜. (예: 2024-01-01). 미지정 시 최신 날짜 사용.",
    )

    args = parser.parse_args()
    country = args.country

    if args.test:
        from test import main as run_test
        prefetched_data = None

        # 호주 시장의 경우, yfinance API 호출을 최소화하기 위해 데이터를 미리 로딩합니다.
        if country == "aus":
            print("백테스트 속도 향상을 위해 데이터를 미리 로딩합니다...")
            from logic import settings
            from utils.data_loader import format_aus_ticker_for_yfinance, fetch_ohlcv_for_tickers
            from utils.db_manager import get_stocks
            import pandas as pd

            stocks_from_db = get_stocks(country)
            if not stocks_from_db:
                print("오류: 백테스트에 사용할 티커를 찾을 수 없습니다.")
                return

            tickers = [s['ticker'] for s in stocks_from_db]
            
            test_date_range = getattr(settings, "TEST_DATE_RANGE", None)
            max_ma_period = max(getattr(settings, "MA_PERIOD_FOR_ETF", 0), getattr(settings, "MA_PERIOD_FOR_STOCK", 0))
            warmup_days = int(max_ma_period * 1.5)
            
            prefetched_data = fetch_ohlcv_for_tickers(tickers, country, date_range=test_date_range, warmup_days=warmup_days)
            print(f"총 {len(prefetched_data)}개 종목의 데이터 로딩 완료.")

        print("전략에 대한 상세 백테스트를 실행합니다...")
        run_test(country=country, quiet=False, prefetched_data=prefetched_data)

    elif args.status:
        from status import main as run_status

        print("전략으로 오늘의 현황을 조회합니다...")
        run_status(country=country, date_str=args.date)


if __name__ == "__main__":
    main()