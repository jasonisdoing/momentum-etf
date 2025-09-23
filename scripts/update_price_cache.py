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


def get_cache_file_path(country: str, ticker: str) -> Path:
    """주어진 티커의 캐시 파일 경로를 구성합니다."""
    project_root = Path(__file__).resolve().parent.parent
    return project_root / "data" / "cache" / country / f"{ticker}.pkl"


def refresh_all_caches(countries: list[str], start_date: str, rebuild: bool = False):
    """지정된 국가의 모든 종목에 대한 가격 데이터 캐시를 새로 고칩니다."""
    print(f"캐시 갱신 시작 (국가: {', '.join(countries)}, 시작일: {start_date}, 강제 재빌드: {rebuild})")

    for country in countries:
        print(f"\n[{country.upper()}] 국가의 캐시를 갱신합니다...")
        all_etfs_from_file = get_etfs(country)
        # is_active 필드가 없는 종목이 있는지 확인합니다.
        for etf in all_etfs_from_file:
            if "is_active" not in etf:
                raise ValueError(
                    f"종목 마스터 파일의 '{etf.get('ticker')}' 종목에 'is_active' 필드가 없습니다. 파일을 확인해주세요."
                )
        etfs = [etf for etf in all_etfs_from_file if etf["is_active"] is not False]
        if not etfs:
            print(f"-> {country.upper()} 국가에 등록된 종목이 없습니다.")
            continue

        tickers = [etf["ticker"] for etf in etfs]
        total_tickers = len(tickers)
        for i, ticker in enumerate(tickers, 1):
            print(f"\r  -> 처리 중: {i}/{total_tickers} ({ticker})", end="", flush=True)

            if rebuild:
                cache_file = get_cache_file_path(country, ticker)
                if cache_file.exists():
                    try:
                        cache_file.unlink()
                    except OSError as e:
                        print(f"\n경고: 캐시 파일 삭제 실패 {cache_file}: {e}")

            try:
                # fetch_ohlcv는 캐시가 없으면 자동으로 데이터를 조회하고 저장합니다.
                # date_range의 두 번째 인자를 None으로 주면 오늘까지 조회합니다.
                fetch_ohlcv(ticker, country, date_range=[start_date, None])
            except Exception as e:
                print(f"\n경고: {ticker} 데이터 처리 중 오류 발생: {e}")
        print(f"\n-> {country.upper()} 국가의 캐시 갱신 완료.")


def main():
    """CLI 진입점"""
    parser = argparse.ArgumentParser(description="OHLCV 데이터 캐시를 업데이트합니다.")
    parser.add_argument("--country", type=str, default="all", help="국가 코드 (kor, aus, coin, 또는 all)")
    parser.add_argument("--start", type=str, default="2020-01-01", help="시작 날짜 (YYYY-MM-DD)")
    parser.add_argument("--rebuild", action="store_true", help="기존 캐시를 강제로 삭제하고 다시 다운로드합니다.")
    args = parser.parse_args()

    load_env_if_present()

    countries = ["kor", "aus", "coin"] if args.country.lower() == "all" else [args.country]
    refresh_all_caches(countries, args.start, args.rebuild)


if __name__ == "__main__":
    main()
