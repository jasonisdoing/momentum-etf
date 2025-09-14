"""
MomentumPilot 프로젝트의 메인 실행 파일입니다.

이 스크립트는 CLI(명령줄 인터페이스)를 통해 프로젝트의 주요 기능인
백테스트(`test.py`)와 현황 조회(`status.py`)를 실행하는 통합 진입점 역할을 합니다.

[사용법]
1. 현황 조회: python run.py <국가코드> --status
   - 예: python run.py kor --status

2. 백테스트 실행: python run.py <국가코드> --test
   - 예: python run.py aus --test
"""

import argparse
import os
import sys
import warnings

# 프로젝트 루트를 Python 경로에 추가합니다.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
# Suppress pmc discontinued break warnings globally for CLI flows
warnings.filterwarnings(
    "ignore",
    message=r"\['break_start', 'break_end'\] are discontinued",
    category=UserWarning,
    module=r"^pandas_market_calendars\."
)


def main():
    """CLI 인자를 파싱하여 해당 모듈을 실행합니다."""
    parser = argparse.ArgumentParser(description="MomentumPilot Trading Engine")
    parser.add_argument("country", choices=["kor", "aus", "coin"], help="실행할 포트폴리오 국가 (kor, aus, coin)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--test",
        action="store_true",
        help="백테스터(test.py)를 실행합니다.",
    )
    group.add_argument(
        "--tune-regime",
        action="store_true",
        help="시장 레짐 필터의 이동평균 기간을 튜닝합니다 (scripts/tune_regime_filter.py).",
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
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="테스트에 사용할 티커 리스트 (쉼표구분, 예: BTC,ETH,SOL). 미지정 시 DB 목록 사용",
    )

    args = parser.parse_args()
    country = args.country

    if args.test:
        from test import main as run_test
        prefetched_data = None
        override_settings = {}

        # 티커 오버라이드 파싱 (모든 국가 공통, 특히 coin 용)
        tickers_override = None
        if args.tickers:
            tickers_override = [t.strip().upper() for t in args.tickers.split(',') if t.strip()]
            if tickers_override:
                override_settings["tickers_override"] = tickers_override

        # 호주 시장의 경우, yfinance API 호출을 최소화하기 위해 데이터를 미리 로딩합니다.
        if country == "aus":
            print("백테스트 속도 향상을 위해 데이터를 미리 로딩합니다...")
            from logic import settings
            from utils.data_loader import fetch_ohlcv_for_tickers
            from utils.db_manager import get_stocks
            import pandas as pd

            stocks_from_db = get_stocks(country)
            if not stocks_from_db:
                print("오류: 백테스트에 사용할 티커를 찾을 수 없습니다.")
                return

            tickers = [s['ticker'] for s in stocks_from_db]
            if tickers_override:
                tickers = [t for t in tickers if t.upper() in set(tickers_override)]
                if not tickers:
                    print("오류: 지정한 --tickers 가 DB 목록과 일치하지 않습니다.")
                    return
            
            try:
                test_months_range = settings.TEST_MONTHS_RANGE
                ma_etf = int(settings.MA_PERIOD_FOR_ETF)
                ma_stock = int(settings.MA_PERIOD_FOR_STOCK)
            except AttributeError:
                print("오류: TEST_MONTHS_RANGE, MA_PERIOD_FOR_ETF, MA_PERIOD_FOR_STOCK 설정이 logic/settings.py 에 정의되어야 합니다.")
                return
            core_end_dt = pd.Timestamp.now()
            core_start_dt = core_end_dt - pd.DateOffset(months=test_months_range)
            test_date_range = [core_start_dt.strftime('%Y-%m-%d'), core_end_dt.strftime('%Y-%m-%d')]
            max_ma_period = max(ma_etf, ma_stock)
            warmup_days = int(max_ma_period * 1.5)
            
            prefetched_data = fetch_ohlcv_for_tickers(tickers, country, date_range=test_date_range, warmup_days=warmup_days)
            print(f"총 {len(prefetched_data)}개 종목의 데이터 로딩 완료.")

        print("전략에 대한 상세 백테스트를 실행합니다...")
        run_test(country=country, quiet=False, prefetched_data=prefetched_data, override_settings=override_settings or None)

    elif args.tune_regime:
        from scripts.tune_regime_filter import tune_regime_filter

        print("시장 레짐 필터 파라미터 최적화를 시작합니다...")
        tune_regime_filter(country=country)

    elif args.status:
        from status import main as run_status

        print("전략으로 오늘의 현황을 조회합니다...")
        run_status(country=country, date_str=args.date)


if __name__ == "__main__":
    main()
