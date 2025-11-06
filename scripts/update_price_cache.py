#!/usr/bin/env python
"""전체 OHLCV 캐시를 초기화한 뒤 설정된 시작일 이후 데이터를 다시 받아옵니다."""

from __future__ import annotations

import argparse
import os
import sys
import time
import uuid
from typing import Optional

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.cache_utils import drop_cache_collection, swap_cache_collection
from utils.data_loader import fetch_ohlcv
from utils.stock_list_io import get_etfs
from utils.env import load_env_if_present
from utils.logger import get_app_logger
from utils.settings_loader import load_common_settings


def _determine_start_date() -> str:
    settings = load_common_settings() or {}
    start = settings.get("CACHE_START_DATE")
    if start:
        return str(start)


def refresh_all_caches(countries: list[str], start_date: Optional[str]):
    """지정된 국가의 모든 종목에 대한 가격 데이터 캐시를 새로 고칩니다."""
    logger = get_app_logger()
    logger.info("캐시 갱신 시작 (국가: %s, 시작일: %s)", ", ".join(countries), start_date)

    for country in countries:
        logger.info("[%s] 국가의 캐시를 갱신합니다...", country.upper())
        all_etfs_from_file = get_etfs(country)

        suffix = f"{os.getpid()}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        temp_token = f"{country}_tmp_{suffix}"
        drop_cache_collection(temp_token)

        tickers = [etf["ticker"] for etf in all_etfs_from_file]
        total_tickers = len(tickers)
        try:
            for i, ticker in enumerate(tickers, 1):
                logger.debug("  -> 처리 중: %d/%d (%s)", i, total_tickers, ticker)

                try:
                    range_start = start_date or "1990-01-01"
                    fetch_ohlcv(
                        ticker,
                        country,
                        date_range=[range_start, None],
                        update_listing_meta=True,
                        force_refresh=True,
                        cache_country=temp_token,
                    )
                except Exception as e:
                    logger.warning("%s 데이터 처리 중 오류 발생: %s", ticker, e)

            swap_cache_collection(country, temp_token)
            logger.info("-> %s 국가의 캐시 갱신 완료.", country.upper())
        except Exception as exc:
            logger.error("%s 국가 캐시 갱신 실패: %s", country.upper(), exc)
            drop_cache_collection(temp_token)
            raise
        finally:
            drop_cache_collection(temp_token)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OHLCV 캐시 갱신 스크립트",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--countries",
        nargs="+",
        help="캐시를 갱신할 국가 코드 목록 (예: kor). 지정하지 않으면 기본 목록 사용.",
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
        countries = ["kor"]

    if not countries:
        parser.error("갱신할 국가를 최소 하나 이상 지정해야 합니다.")

    logger.info("입력 파라미터: countries=%s, start=%s", countries, start_date)
    refresh_all_caches(countries, start_date)


if __name__ == "__main__":
    main()
