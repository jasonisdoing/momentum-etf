"""
MomentumEtf 프로젝트의 CLI(명령줄 인터페이스) 실행 파일입니다.

이 스크립트는 백테스트, 시그널 조회, 파라미터 튜닝 등
웹 UI 외부에서 실행되는 주요 기능들의 통합 진입점 역할을 합니다.
[사용법]
1. 시그널 조회: python cli.py <국가코드> --signal
   - 예: python cli.py kor --signal

2. 백테스트 실행: python cli.py <국가코드> --test
   - 예: python cli.py aus --test

3. 파라미터 튜닝: python cli.py <국가코드> --tune
   - 예: python cli.py kor --tune
"""

"""
[실행 예시]
아래는 'data/accounts.json'에 등록된 계좌를 기반으로 생성된 실행 명령어 예시입니다.
이 목록을 복사하여 터미널에서 바로 사용할 수 있습니다.

# --- 계좌별 기본 명령어 (signal, test, tune) ---

# 한국 (KOR) / m1 계좌
python cli.py kor --signal --account m3 --date 2025-09-22
python cli.py kor --test --account m1
python cli.py kor --tune --account m1

# 호주 (AUS) / a1 계좌
python cli.py aus --signal --account a1
python cli.py aus --test --account a1
python cli.py aus --tune --account a1

# 가상화폐 (COIN) / b1 계좌
python cli.py coin --signal --account b1
python cli.py coin --test --account b1
python cli.py coin --tune --account b1

# --- 특수 목적 명령어 ---

# 시장 레짐 필터 튜닝 (모든 계좌에 공통 적용되는 설정을 튜닝합니다)
python cli.py kor --tune-regime --account m1

"""

import argparse
import os
import time
import sys
from typing import Optional

from test import TEST_MONTHS_RANGE

# 프로젝트 루트를 Python 경로에 추가합니다.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

from utils.account_registry import get_accounts_by_country, load_accounts


def _resolve_account(country: str, explicit: Optional[str]) -> str:
    if explicit:
        return explicit

    load_accounts(force_reload=False)
    entries = get_accounts_by_country(country) or []
    for entry in entries:
        code = entry.get("account")
        if code:
            return str(code)
    raise SystemExit(f"'{country}' 국가에 등록된 계좌가 없습니다. data/accounts.json을 확인하세요.")


def main():
    """CLI 인자를 파싱하여 해당 모듈을 실행합니다."""
    parser = argparse.ArgumentParser(description="MomentumEtf Trading Engine CLI")
    parser.add_argument(
        "country", choices=["kor", "aus", "coin"], help="실행할 포트폴리오 국가 (kor, aus, coin)"
    )
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
        "--tune",
        action="store_true",
        help="전략 파라미터를 튜닝합니다 (tune.py).",
    )
    group.add_argument(
        "--signal",
        action="store_true",
        help="오늘의 매매 신호(signal.py)를 실행합니다.",
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
    parser.add_argument(
        "--account",
        type=str,
        default=None,
        help="국가 내 특정 계좌 코드 (예: m2, a1). 미지정 시 첫 번째 활성 계좌 사용",
    )

    args = parser.parse_args()
    country = args.country
    account = _resolve_account(country, args.account)

    if args.test:
        from test import main as run_test

        prefetched_data = None
        override_settings = {}

        # 티커 오버라이드 파싱 (모든 국가 공통, 특히 coin 용)
        tickers_override = None
        if args.tickers:
            tickers_override = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
            if tickers_override:
                override_settings["tickers_override"] = tickers_override

        # 호주 시장의 경우, yfinance API 호출을 최소화하기 위해 데이터를 미리 로딩합니다.
        if country == "aus":
            print("백테스트 속도 향상을 위해 데이터를 미리 로딩합니다...")
            import pandas as pd

            from utils.data_loader import fetch_ohlcv_for_tickers
            from utils.db_manager import get_portfolio_settings
            from utils.stock_list_io import get_etfs

            etfs_from_file = get_etfs(country)
            if not etfs_from_file:
                print("오류: 'data/aus/' 폴더에서 백테스트에 사용할 티커를 찾을 수 없습니다.")
                return

            tickers = [s["ticker"] for s in etfs_from_file]
            if tickers_override:
                tickers = [t for t in tickers if t.upper() in set(tickers_override)]
                if not tickers:
                    print("오류: 지정한 --tickers 가 DB 목록과 일치하지 않습니다.")
                    return

            portfolio_settings = get_portfolio_settings(country, account=account)
            if not portfolio_settings:
                print(f"오류: '{country}' 국가의 설정을 DB에서 찾을 수 없습니다. 웹 앱의 '설정' 탭에서 값을 지정해주세요.")
                return

            try:
                test_months_range = TEST_MONTHS_RANGE
                # test.py의 하드코딩된 값 대신 DB에서 실제 MA 기간을 가져옵니다.
                ma_etf = int(portfolio_settings["ma_period"])
            except (KeyError, ValueError, TypeError):
                print("오류: DB의 MA 기간 설정이 올바르지 않습니다.")
                return
            core_end_dt = pd.Timestamp.now()
            core_start_dt = core_end_dt - pd.DateOffset(months=test_months_range)
            test_date_range = [core_start_dt.strftime("%Y-%m-%d"), core_end_dt.strftime("%Y-%m-%d")]
            max_ma_period = ma_etf
            warmup_days = int(max_ma_period * 1.5)

            prefetched_data = fetch_ohlcv_for_tickers(
                tickers, country, date_range=test_date_range, warmup_days=warmup_days
            )
            print(f"총 {len(prefetched_data)}개 종목의 데이터 로딩 완료.")

        print("전략에 대한 상세 백테스트를 실행합니다...")
        run_test(
            country=country,
            quiet=False,
            prefetched_data=prefetched_data,
            override_settings=override_settings or None,
            account=account,
        )

    elif args.tune_regime:
        from scripts.tune_regime_filter import tune_regime_filter

        print("시장 레짐 필터 파라미터 최적화를 시작합니다...")
        tune_regime_filter(country=country, account=account)

    elif args.tune:
        from tune import main as run_tune

        print(
            f"{country.upper()} 포트폴리오의 전략 파라미터 튜닝을 시작합니다"
            + (f" (계좌: {account})" if account else "")
            + "..."
        )
        run_tune(country_code=country, account=account)

    elif args.signal:
        from signals import main as run_signal, send_summary_notification
        from utils.db_manager import get_portfolio_snapshot

        print("전략으로 오늘의 매매 신호를 조회합니다...")
        start_time = time.time()

        # 알림에 사용할 이전 평가금액을 미리 가져옵니다.
        old_snapshot = get_portfolio_snapshot(country, account=account)
        old_equity = float(old_snapshot.get("total_equity", 0.0)) if old_snapshot else 0.0

        try:
            report_date = run_signal(country=country, date_str=args.date, account=account)
        except Exception as e:
            print(f"\n오류: 시그널 생성 중 오류가 발생했습니다: {e}")
            return
        print(11111112222222)
        if report_date:
            duration = time.time() - start_time
            send_summary_notification(country, account, report_date, duration, old_equity)


if __name__ == "__main__":
    main()
