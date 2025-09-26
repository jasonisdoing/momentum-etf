"""
MomentumEtf 프로젝트의 CLI(명령줄 인터페이스) 실행 파일입니다.

이 스크립트는 백테스트, 시그널 조회, 파라미터 튜닝 등
웹 UI 외부에서 실행되는 주요 기능들의 통합 진입점 역할을 합니다.
[사용법]
1. 시그널 조회: python cli.py <계좌코드> --signal
   - 예: python cli.py k1 --signal

2. 백테스트 실행: python cli.py <계좌코드> --test
   - 예: python cli.py a1 --test

3. 파라미터 튜닝: python cli.py <계좌코드> --tune
   - 예: python cli.py k1 --tune
"""

"""
[실행 예시]
아래는 'data/accounts/country_mapping.json'에 등록된 계좌를 기반으로 생성된 실행 명령어 예시입니다.
이 목록을 복사하여 터미널에서 바로 사용할 수 있습니다.

# --- 계좌별 기본 명령어 (signal, test, tune) ---

# 한국 (KOR) / m1 계좌
python cli.py m1 --signal --date 2025-09-23
python cli.py m1 --test
python cli.py m1 --tune

# 호주 (AUS) / a1 계좌
python cli.py a1 --signal
python cli.py a1 --test
python cli.py a1 --tune

# 가상화폐 (COIN) / b1 계좌
python cli.py b1 --signal
python cli.py b1 --test
python cli.py b1 --tune

# --- 특수 목적 명령어 ---

# 시장 레짐 필터 튜닝 (모든 계좌에 공통 적용되는 설정을 튜닝합니다)
python cli.py kor --tune-regime --account m1

"""

import argparse
import os
import subprocess
import sys
import time
from typing import List, Optional, Set

from test import TEST_MONTHS_RANGE

# 프로젝트 루트를 Python 경로에 추가합니다.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

from utils.account_registry import (
    get_account_info,
    get_accounts_by_country,
    get_strategy_rules_for_account,
    load_accounts,
)


def _resolve_account(country: str, explicit: Optional[str]) -> str:
    if explicit:
        return explicit

    load_accounts(force_reload=False)
    entries = get_accounts_by_country(country) or []
    for entry in entries:
        code = entry.get("account")
        if code:
            return str(code)
    raise SystemExit(f"'{country}' 국가에 등록된 계좌가 없습니다. data/accounts/country_mapping.json을 확인하세요.")


def main():
    """CLI 인자를 파싱하여 해당 모듈을 실행합니다."""
    parser = argparse.ArgumentParser(description="MomentumEtf Trading Engine CLI")
    parser.add_argument(
        "account",
        nargs="?",
        default=None,
        help="실행할 계좌 코드 (예: k1, a1, b1)",
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
        "--start-date",
        type=str,
        default=None,
        help="여러 날짜를 재계산할 때 사용할 시작일(포함). (예: 2024-01-01)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="여러 날짜를 재계산할 때 사용할 종료일(포함). 지정하지 않으면 시작일과 동일",
    )
    parser.add_argument(
        "--tickers",
        type=str,
        default=None,
        help="테스트에 사용할 티커 리스트 (쉼표구분, 예: BTC,ETH,SOL). 미지정 시 DB 목록 사용",
    )
    parser.add_argument(
        "--accounts",
        type=str,
        default=None,
        help="콤마로 구분된 여러 계좌 코드 (예: k1,k2).",
    )
    parser.add_argument(
        "--country",
        type=str,
        choices=["kor", "aus", "coin"],
        help="특정 국가에 속한 모든 활성 계좌를 실행합니다.",
    )

    args = parser.parse_args()

    requested_accounts: List[str] = []
    if args.account:
        requested_accounts.append(args.account)

    if args.accounts:
        requested_accounts.extend([acc.strip() for acc in args.accounts.split(",") if acc.strip()])

    accounts_to_run: List[str] = []
    if requested_accounts:
        accounts_to_run = requested_accounts
    else:
        load_accounts(force_reload=False)
        if args.country:
            candidates = get_accounts_by_country(args.country)
        else:
            candidates = load_accounts()
        for entry in candidates or []:
            if entry.get("is_active", True):
                code = entry.get("account")
                if code:
                    accounts_to_run.append(str(code).strip())

    if not accounts_to_run:
        raise SystemExit("실행할 활성 계좌를 찾을 수 없습니다. country_mapping.json을 확인하세요.")

    # 각 계좌에 대해 요청된 작업을 실행합니다.
    seen_accounts: Set[str] = set()
    for account in accounts_to_run:
        if account in seen_accounts:
            continue
        seen_accounts.add(account)

        account_info = get_account_info(account)
        if not account_info:
            print(f"경고: 등록되지 않은 계좌를 건너뜁니다: {account}")
            continue
        country = str(account_info.get("country") or "").strip()

        print(f"\n{'=' * 20} [{country.upper()}/{account}] 계좌 작업 시작 {'=' * 20}")

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

            # 백테스트 속도 향상을 위해 모든 국가에 대해 데이터를 미리 로딩합니다.
            print("백테스트 속도 향상을 위해 데이터를 미리 로딩합니다...")
            import pandas as pd

            from utils.data_loader import fetch_ohlcv_for_tickers
            from utils.stock_list_io import get_etfs

            etfs_from_file = get_etfs(country)
            if not etfs_from_file:
                print(f"오류: 'data/{country}/' 폴더에서 백테스트에 사용할 티커를 찾을 수 없습니다.")
                return

            tickers = [s["ticker"] for s in etfs_from_file]
            if tickers_override:
                tickers = [t for t in tickers if t.upper() in set(tickers_override)]
                if not tickers:
                    print("오류: 지정한 --tickers 가 DB 목록과 일치하지 않습니다.")
                    return

            strategy_rules = get_strategy_rules_for_account(account)

            test_months_range = TEST_MONTHS_RANGE
            ma_etf = int(strategy_rules.ma_period)
            core_end_dt = pd.Timestamp.now()
            core_start_dt = core_end_dt - pd.DateOffset(months=test_months_range)
            test_date_range = [
                core_start_dt.strftime("%Y-%m-%d"),
                core_end_dt.strftime("%Y-%m-%d"),
            ]
            max_ma_period = ma_etf
            warmup_days = int(max_ma_period * 1.5)

            prefetched_data = fetch_ohlcv_for_tickers(
                tickers, country, date_range=test_date_range, warmup_days=warmup_days
            )
            print(f"총 {len(prefetched_data)}개 종목의 데이터 로딩 완료.")

            print("전략에 대한 상세 백테스트를 실행합니다...")
            run_test(
                account=account,
                quiet=False,
                prefetched_data=prefetched_data,
                override_settings=override_settings or None,
            )

        elif args.tune_regime:
            from scripts.tune_regime_filter import tune_regime_filter

            print("시장 레짐 필터 파라미터 최적화를 시작합니다...")
            tune_regime_filter(country=country, account=account)

        elif args.tune:
            print(f"{country.upper()} 포트폴리오의 전략 파라미터 튜닝을 시작합니다" + f" (계좌: {account})...")
            # tune.py를 별도 프로세스로 실행하여 파일 로깅이 정상적으로 동작하도록 합니다.
            command = [sys.executable, "tune.py", account]
            subprocess.run(command, check=True)

        elif args.signal:
            from signals import main as run_signal
            from utils.db_manager import get_portfolio_snapshot
            from utils.notification import (
                send_summary_notification,
                send_detailed_signal_notification,
            )

            print("전략으로 오늘의 매매 신호를 조회합니다...")
            if args.start_date and args.date:
                print("오류: --date 와 --start-date 는 동시에 사용할 수 없습니다.")
                return

            if args.end_date and not args.start_date:
                print("오류: --end-date 를 사용하려면 --start-date 도 지정해야 합니다.")
                return

            date_inputs: List[Optional[str]] = []
            if args.start_date:
                try:
                    import pandas as pd

                    start_dt = pd.to_datetime(args.start_date).normalize()
                    end_dt = (
                        pd.to_datetime(args.end_date).normalize() if args.end_date else start_dt
                    )
                except Exception:
                    print("오류: 날짜 형식이 올바르지 않습니다. 예) 2024-01-01")
                    return

                if end_dt < start_dt:
                    print("오류: --end-date 는 --start-date 이후여야 합니다.")
                    return

                date_inputs = [dt.strftime("%Y-%m-%d") for dt in pd.date_range(start_dt, end_dt)]
            else:
                date_inputs = [args.date]  # 단일 실행 (args.date 가 None 이면 최신 기준)

            for idx, date_str in enumerate(date_inputs, start=1):
                run_label = date_str if date_str else "최신 기준일"
                print(f"-> ({idx}/{len(date_inputs)}) {run_label} 데이터 계산 중...")

                start_time = time.time()

                # 알림에 사용할 이전 평가금액을 미리 가져옵니다.
                if date_str:
                    old_snapshot = get_portfolio_snapshot(
                        country, account=account, date_str=date_str
                    )
                else:
                    old_snapshot = get_portfolio_snapshot(country, account=account)
                old_equity = float(old_snapshot.get("total_equity", 0.0)) if old_snapshot else 0.0

                try:
                    signal_result = run_signal(
                        account=account,
                        date_str=date_str,
                    )
                except Exception as e:
                    print(f"\n오류: {run_label} 시그널 생성 중 오류가 발생했습니다: {e}")
                    continue

                if signal_result:
                    duration = time.time() - start_time
                    send_summary_notification(
                        country,
                        account,
                        signal_result.report_date,
                        duration,
                        old_equity,
                        summary_data=signal_result.summary_data,
                        header_line=signal_result.header_line,
                        force_send=True,
                    )

                    time.sleep(2)
                    send_detailed_signal_notification(
                        country,
                        account,
                        signal_result.header_line,
                        signal_result.detail_headers,
                        signal_result.detail_rows,
                        decision_config=signal_result.decision_config,
                        extra_lines=signal_result.detail_extra_lines,
                        force_send=True,
                    )

        print(f"==================== [{country.upper()}/{account}] 계좌 작업 완료 ====================")


if __name__ == "__main__":
    main()
