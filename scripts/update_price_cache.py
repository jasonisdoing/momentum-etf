#!/usr/bin/env python
"""전체 OHLCV 캐시를 초기화한 뒤 설정된 시작일 이후 데이터를 다시 받아옵니다."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import fetch_ohlcv
from utils.stock_list_io import get_etfs
from utils.env import load_env_if_present
from utils.logger import get_app_logger
from utils.settings_loader import load_common_settings


def get_cache_file_path(country: str, ticker: str) -> Path:
    """주어진 티커의 캐시 파일 경로를 구성합니다."""
    project_root = Path(__file__).resolve().parent.parent
    return project_root / "data" / "stocks" / "cache" / country / f"{ticker}.pkl"


def _determine_start_date() -> str:
    settings = load_common_settings() or {}
    start = settings.get("CACHE_START_DATE")
    if start:
        return str(start)


def refresh_all_caches(countries: list[str], start_date: str):
    """지정된 국가의 모든 종목에 대한 가격 데이터 캐시를 새로 고칩니다."""
    logger = get_app_logger()
    logger.info("캐시 갱신 시작 (국가: %s, 시작일: %s)", ", ".join(countries), start_date)

    for country in countries:
        logger.info("[%s] 국가의 캐시를 갱신합니다...", country.upper())
        all_etfs_from_file = get_etfs(country)

        tickers = [etf["ticker"] for etf in all_etfs_from_file]
        total_tickers = len(tickers)
        for i, ticker in enumerate(tickers, 1):
            logger.debug("  -> 처리 중: %d/%d (%s)", i, total_tickers, ticker)

            cache_file = get_cache_file_path(country, ticker)
            if cache_file.exists():
                try:
                    cache_file.unlink()
                except OSError as e:
                    logger.warning("캐시 파일 삭제 실패 %s: %s", cache_file, e)

            try:
                # fetch_ohlcv는 캐시가 없으면 자동으로 데이터를 조회하고 저장합니다.
                # date_range의 두 번째 인자를 None으로 주면 오늘까지 조회합니다.
                fetch_ohlcv(ticker, country, date_range=[start_date, None], update_listing_meta=True)
            except Exception as e:
                logger.warning("%s 데이터 처리 중 오류 발생: %s", ticker, e)
        logger.info("-> %s 국가의 캐시 갱신 완료.", country.upper())


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OHLCV 캐시 갱신 스크립트",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--countries",
        nargs="+",
        help="캐시를 갱신할 국가 코드 목록 (예: kor aus us). 지정하지 않으면 기본 목록 사용.",
    )
    parser.add_argument(
        "--start",
        help="데이터 조회 시작일 (YYYY-MM-DD). 지정하지 않으면 공통 설정",
    )
    return parser


def main():
    """CLI 진입점"""
    logger = get_app_logger()

    load_env_if_present()

    parser = _build_parser()
    args = parser.parse_args()

    start_date = args.start or _determine_start_date()
    if args.countries:
        countries = [country.strip().lower() for country in args.countries if country.strip()]
    else:
        countries = ["kor", "aus"]

    if not countries:
        parser.error("갱신할 국가를 최소 하나 이상 지정해야 합니다.")

    logger.info("입력 파라미터: countries=%s, start=%s", countries, start_date)
    refresh_all_caches(countries, start_date)


if __name__ == "__main__":
    main()
