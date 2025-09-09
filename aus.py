"""
MomentumPilot 프로젝트의 호주 시장용 메인 실행 파일입니다.
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
    parser = argparse.ArgumentParser(description="MomentumPilot 트레이딩 엔진 (AUS)")
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
        help="포트폴리오 스냅샷 파일 경로. 미지정 시 최신 파일 사용. (예: data/aus/portfolio_2024-01-01.json)",
    )

    args = parser.parse_args()
    country = "aus"

    if args.test is not None:
        from test import main as run_test

        import pandas as pd

        import settings
        from utils.data_loader import format_aus_ticker_for_yfinance, read_tickers_file
        from utils.report import generate_strategy_comparison_report

        # --- 데이터 사전 로딩 (yfinance 호출 최소화) ---
        prefetched_data = None
        if country == "aus":
            print("백테스트 속도 향상을 위해 데이터를 미리 로딩합니다...")
            import importlib

            try:
                import yfinance as yf
            except ImportError:
                print("오류: yfinance 라이브러리가 설치되지 않았습니다. 'pip install yfinance'로 설치해주세요.")
                return

            # 1. 티커 목록 읽기
            etf_pairs = read_tickers_file(f"data/{country}/tickers_etf.txt", country=country)
            stock_pairs = read_tickers_file(f"data/{country}/tickers_stock.txt", country=country)
            pairs = etf_pairs + stock_pairs

            if not pairs:
                print("오류: 백테스트에 사용할 티커를 찾을 수 없습니다.")
                return

            # 2. 웜업 기간 계산
            strategies_for_warmup = (
                ["jason", "seykota", "donchian"]
                if args.test == "__COMPARE__"
                else [args.test]
            )
            max_warmup_days = 0
            for strategy in strategies_for_warmup:
                try:
                    s_settings = importlib.import_module(f"logics.{strategy}.settings")
                    warmup_days = 0
                    if strategy == "donchian":
                        ma_etf = int(s_settings.DONCHIAN_MA_PERIOD_FOR_ETF)
                        ma_stock = int(s_settings.DONCHIAN_MA_PERIOD_FOR_STOCK)
                        warmup_days = int(max(ma_etf, ma_stock) * 1.5)
                    elif strategy == "seykota":
                        warmup_days = int(int(s_settings.SEYKOTA_SLOW_MA) * 1.5)
                    elif strategy == "jason":
                        st_p = int(getattr(s_settings, "ST_ATR_PERIOD", 14))
                        warmup_days = int((st_p + 10) * 1.5)
                    max_warmup_days = max(max_warmup_days, warmup_days)
                except (ImportError, AttributeError):
                    max_warmup_days = max(max_warmup_days, 250)  # 안전한 기본값

            # 3. 전체 조회 기간 계산
            test_date_range = getattr(settings, "TEST_DATE_RANGE", None)
            start_dt, end_dt = None, None
            if test_date_range and len(test_date_range) == 2:
                core_start = pd.to_datetime(test_date_range[0])
                start_dt = (core_start - pd.DateOffset(days=max_warmup_days)).strftime("%Y-%m-%d")
                end_dt = (pd.to_datetime(test_date_range[1]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

            # 4. 데이터 로딩
            tickers_yf = [format_aus_ticker_for_yfinance(p[0]) for p in pairs]
            data_batch = yf.download(tickers_yf, start=start_dt, end=end_dt, progress=True, threads=True, auto_adjust=True)

            if not data_batch.empty:
                prefetched_data = {}
                if len(pairs) == 1:
                    prefetched_data[pairs[0][0]] = data_batch.dropna(how="all")
                elif isinstance(data_batch.columns, pd.MultiIndex):
                    for tkr, _ in pairs:
                        df_single = data_batch.loc[:, (slice(None), format_aus_ticker_for_yfinance(tkr))]
                        if not df_single.dropna().empty:
                            df_single.columns = df_single.columns.droplevel(1)
                            prefetched_data[tkr] = df_single.dropna(how="all")
            print(f"총 {len(prefetched_data)}개 종목의 데이터 로딩 완료.")

        if args.test == "__COMPARE__":
            # `python aus.py --test`
            strategies_to_compare = ["jason", "seykota", "donchian"]
            print(
                f"전략 비교 백테스트를 실행합니다: {', '.join([f'{s}' for s in strategies_to_compare])}"
            )

            all_results = []
            for strategy in strategies_to_compare:
                results = run_test(
                    strategy_name=strategy, country=country, quiet=True, prefetched_data=prefetched_data
                )
                if results:
                    all_results.append(results)
                print(f"'{strategy}' 전략 백테스트 완료.")

            if not all_results:
                print("\n오류: 비교할 백테스트 결과가 없습니다.")
                try:
                    test_date_range = getattr(settings, "TEST_DATE_RANGE", None)
                    if test_date_range and len(test_date_range) == 2:
                        end_date = pd.to_datetime(test_date_range[1])
                        if end_date > pd.Timestamp.now():
                            print(
                                f"      원인: settings.py의 TEST_DATE_RANGE 종료일({end_date.strftime('%Y-%m-%d')})이 미래로 설정되어 데이터를 가져올 수 없습니다."
                            )
                except Exception:
                    pass
                return
            report = generate_strategy_comparison_report(all_results, country=country)
            print(report)
        else:
            # `python aus.py --test <strategy_name>`
            strategy_name = args.test
            print(f"'{strategy_name}' 전략에 대한 상세 백테스트를 실행합니다...")
            run_test(
                strategy_name=strategy_name, country=country, quiet=False, prefetched_data=prefetched_data
            )

    elif args.status:
        strategy_name = args.status
        from status import main as run_status

        run_status(strategy_name=strategy_name, country=country, portfolio_path=args.portfolio)


if __name__ == "__main__":
    main()