"""국가별 백테스트 실행 스크립트."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

from core.entry_point import dump_backtest_log, print_backtest_summary, run_account_backtest
from utils.account_registry import (
    get_account_settings,
    get_benchmark_tickers,
    get_strategy_rules,
    list_available_accounts,
)
from utils.data_loader import MissingPriceDataError, get_latest_trading_day, prepare_price_data
from utils.logger import get_app_logger
from utils.settings_loader import load_common_settings
from utils.stock_list_io import get_etfs

RESULTS_DIR = Path(__file__).resolve().parent / "zaccounts"

logger = get_app_logger()


def print_run_header(account_id: str, *, date_str: str | None) -> None:
    """백테스트 실행 헤더를 출력합니다."""
    banner = f"=== {account_id.upper()} 백테스트 ==="
    logger.info("%s", banner)
    logger.info("기준일: %s", date_str or "auto (latest trading day)")


def _available_account_choices() -> list[str]:
    choices = list_available_accounts()
    if not choices:
        raise SystemExit("계정 설정(JSON)이 존재하지 않습니다. zaccounts/account/*.json 파일을 확인하세요.")
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
    run_start = datetime.now()

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
    strategy_cfg = account_settings.get("strategy", {}) or {}
    backtest_start_date_str = strategy_cfg.get("BACKTEST_START_DATE")
    if backtest_start_date_str is None:
        parser.error("계정 설정에 'strategy.BACKTEST_START_DATE' 값을 지정해야 합니다.")
    try:
        start_date = pd.to_datetime(backtest_start_date_str)
    except Exception:
        parser.error("BACKTEST_START_DATE 설정이 올바른 날짜 형식이어야 합니다.")
    end_date = get_latest_trading_day(country_code)
    if not isinstance(end_date, pd.Timestamp):
        end_date = pd.Timestamp.now().normalize()

    # 웜업 기간을 전략의 MA_MONTH 기반으로 설정
    warmup_days = strategy_rules.ma_days

    universe_tickers = [etf["ticker"] for etf in get_etfs(account_id) if etf.get("ticker")]
    benchmark_tickers = get_benchmark_tickers(account_settings)
    tickers = sorted({*(str(t).strip().upper() for t in universe_tickers if t), *benchmark_tickers})
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
        account_id=account_id,
    )
    if missing:
        raise MissingPriceDataError(
            country=country_code,
            start_date=date_range_prefetch[0],
            end_date=date_range_prefetch[1],
            tickers=missing,
        )

    print_run_header(account_id, date_str=None)

    result = run_account_backtest(
        account_id,
        prefetched_data=prefetched_map,
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
        backtest_start_date=str(backtest_start_date_str),
        initial_capital_krw=result.initial_capital_krw,
        bucket_topn=result.bucket_topn,
        ticker_summaries=getattr(result, "ticker_summaries", []),
        core_start_dt=result.start_date,
    )
    logger.info("✅ 백테스트 로그를 '%s'에 저장했습니다.", log_path)
    elapsed = datetime.now() - run_start
    total_seconds = int(elapsed.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    parts = []
    if hours:
        parts.append(f"{hours}시간")
    if minutes:
        parts.append(f"{minutes}분")
    if seconds or not parts:
        parts.append(f"{seconds}초")
    logger.info(
        "[%s] 총 소요 시간: %s",
        account_id.upper(),
        " ".join(parts),
    )


if __name__ == "__main__":
    try:
        main()
    except MissingPriceDataError as exc:
        logger = get_app_logger()
        logger.error(str(exc))
        raise SystemExit(1)
