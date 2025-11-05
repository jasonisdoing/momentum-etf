"""국가별 백테스트 실행 스크립트."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from utils.account_registry import (
    get_account_settings,
    get_strategy_rules,
    list_available_accounts,
)
from logic.backtest.reporting import dump_backtest_log, print_backtest_summary
from logic.recommend.output import print_run_header
from utils.logger import get_app_logger
from utils.stock_list_io import get_etfs
from utils.data_loader import prepare_price_data, get_latest_trading_day
from utils.settings_loader import load_common_settings

RESULTS_DIR = Path(__file__).resolve().parent / "data" / "results"


def _available_account_choices() -> list[str]:
    choices = list_available_accounts()
    if not choices:
        raise SystemExit("계정 설정(JSON)이 존재하지 않습니다. data/settings/account/*.json 파일을 확인하세요.")
    return choices


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MomentumEtf 계정 백테스트 실행기",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("account", choices=_available_account_choices(), help="실행할 계정 ID")
    return parser


def main() -> None:
    logger = get_app_logger()

    # 파서 생성 및 인자 파싱
    parser = build_parser()
    args = parser.parse_args()

    account_id = args.account.lower()

    try:
        account_settings = get_account_settings(account_id)
        strategy_rules = get_strategy_rules(account_id)
    except Exception as exc:  # pragma: no cover - 잘못된 입력 방어 전용 처리
        parser.error(f"계정 설정을 로드하는 중 오류가 발생했습니다: {exc}")

    country_code = (account_settings.get("country_code") or account_id).strip().lower()
    backtest_cfg = account_settings.get("backtest", {}) or {}
    months_range = backtest_cfg.get("months_range")
    if months_range is None:
        months_range = account_settings.get("strategy", {}).get("MONTHS_RANGE")
    if months_range is None:
        parser.error("계정 설정에 'backtest.months_range' 또는 'strategy.MONTHS_RANGE' 값을 지정해야 합니다.")
    try:
        months_range = int(months_range)
    except (TypeError, ValueError):
        parser.error("MONTHS_RANGE 설정이 올바른 숫자여야 합니다.")
    end_date = get_latest_trading_day(country_code)
    if not isinstance(end_date, pd.Timestamp):
        end_date = pd.Timestamp.now().normalize()
    start_date = end_date - pd.DateOffset(months=months_range)

    # 웜업 기간을 전략의 MA_PERIOD로 설정
    warmup_days = strategy_rules.ma_period

    tickers = [etf["ticker"] for etf in get_etfs(country_code)]
    common_settings = load_common_settings()
    cache_seed_raw = (common_settings or {}).get("CACHE_START_DATE")
    cache_seed_dt = None
    if cache_seed_raw:
        try:
            cache_seed_dt = pd.to_datetime(cache_seed_raw).normalize()
        except Exception:
            cache_seed_dt = None

    prefetch_start = start_date - pd.DateOffset(days=warmup_days)
    if cache_seed_dt is not None and cache_seed_dt < prefetch_start:
        prefetch_start = cache_seed_dt
    date_range_prefetch = [prefetch_start.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")]

    prefetched_map, missing = prepare_price_data(
        tickers=tickers,
        country=country_code,
        start_date=date_range_prefetch[0],
        end_date=date_range_prefetch[1],
        warmup_days=0,
    )
    if missing:
        logger.warning("데이터가 부족한 종목 (%d): %s", len(missing), ", ".join(missing))

    print_run_header(account_id, date_str=None)

    from logic.backtest.account_runner import run_account_backtest

    excluded = set(missing)
    result = run_account_backtest(
        account_id,
        months_range=months_range,
        prefetched_data=prefetched_map,
        excluded_tickers=excluded if excluded else None,
    )

    # dump_backtest_log가 계정별 폴더 구조로 저장
    log_path = dump_backtest_log(
        result,
        account_settings,
        results_dir=RESULTS_DIR,
    )

    print_backtest_summary(
        summary=result.summary,
        account_id=account_id,
        country_code=result.country_code,
        test_months_range=months_range,
        initial_capital_krw=result.initial_capital_krw,
        portfolio_topn=result.portfolio_topn,
        ticker_summaries=getattr(result, "ticker_summaries", []),
        core_start_dt=result.start_date,
    )
    logger.info("✅ 백테스트 로그를 '%s'에 저장했습니다.", log_path)


if __name__ == "__main__":
    main()
