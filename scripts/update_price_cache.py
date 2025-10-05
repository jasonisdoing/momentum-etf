"""
모든 국가의 모든 종목에 대해 가격 데이터(OHLCV) 캐시를 생성하거나 업데이트합니다.

[사용법]
1. 모든 국가 캐시 업데이트 (기본값: 2020-01-01부터)
   python scripts/update_price_cache.py

2. 특정 국가, 특정 시작일부터 업데이트
   python scripts/update_price_cache.py --country kor --start 2021-01-01

3. 캐시 강제 재빌드 (기존 캐시 삭제 후 전체 다시 다운로드)
   python scripts/update_price_cache.py --rebuild
"""

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


def get_cache_file_path(country: str, ticker: str) -> Path:
    """주어진 티커의 캐시 파일 경로를 구성합니다."""
    project_root = Path(__file__).resolve().parent.parent
    return project_root / "data" / "stocks" / "cache" / country / f"{ticker}.pkl"


def refresh_all_caches(countries: list[str], start_date: str, rebuild: bool = False):
    """지정된 국가의 모든 종목에 대한 가격 데이터 캐시를 새로 고칩니다."""
    logger = get_app_logger()
    logger.info(
        "캐시 갱신 시작 (국가: %s, 시작일: %s, 강제 재빌드: %s)",
        ", ".join(countries),
        start_date,
        rebuild,
    )

    for country in countries:
        logger.info("[%s] 국가의 캐시를 갱신합니다...", country.upper())
        all_etfs_from_file = get_etfs(country)

        tickers = [etf["ticker"] for etf in all_etfs_from_file]
        total_tickers = len(tickers)
        for i, ticker in enumerate(tickers, 1):
            logger.debug("  -> 처리 중: %d/%d (%s)", i, total_tickers, ticker)

            if rebuild:
                cache_file = get_cache_file_path(country, ticker)
                if cache_file.exists():
                    try:
                        cache_file.unlink()
                    except OSError as e:
                        logger.warning("캐시 파일 삭제 실패 %s: %s", cache_file, e)

            try:
                # fetch_ohlcv는 캐시가 없으면 자동으로 데이터를 조회하고 저장합니다.
                # date_range의 두 번째 인자를 None으로 주면 오늘까지 조회합니다.
                fetch_ohlcv(ticker, country, date_range=[start_date, None])
            except Exception as e:
                logger.warning("%s 데이터 처리 중 오류 발생: %s", ticker, e)
        logger.info("-> %s 국가의 캐시 갱신 완료.", country.upper())


def main():
    """CLI 진입점"""
    logger = get_app_logger()

    parser = argparse.ArgumentParser(description="OHLCV 데이터 캐시를 업데이트합니다.")
    parser.add_argument("--country", type=str, default="all", help="국가 코드 (kor, aus 또는 all)")
    parser.add_argument("--start", type=str, default="2020-01-01", help="시작 날짜 (YYYY-MM-DD)")
    parser.add_argument("--rebuild", action="store_true", help="기존 캐시를 강제로 삭제하고 다시 다운로드합니다.")
    args = parser.parse_args()

    load_env_if_present()

    countries = ["kor", "aus"] if args.country.lower() == "all" else [args.country]
    logger.debug("입력 파라미터: countries=%s, start=%s, rebuild=%s", countries, args.start, args.rebuild)
    refresh_all_caches(countries, args.start, args.rebuild)


if __name__ == "__main__":
    main()
