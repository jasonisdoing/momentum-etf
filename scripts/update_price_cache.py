"""가격 데이터 캐시를 구축/갱신하는 스크립트."""

import argparse
import os
import sys
from datetime import datetime
from typing import Iterable, List, Optional


import warnings

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

from dotenv import load_dotenv

load_dotenv()

# 프로젝트 루트를 Python 경로에 추가
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.data_loader import fetch_ohlcv
from utils.cache_utils import load_cached_frame
from utils.stock_list_io import get_etfs

DEFAULT_START_DATE = "2020-01-01"


def _list_etf_tickers(country: str) -> List[str]:
    data = get_etfs(country) or []
    tickers: List[str] = []
    for block in data:
        if isinstance(block, dict) and "tickers" in block:
            t_list = block.get("tickers") or []
            for item in t_list:
                tkr = item.get("ticker") if isinstance(item, dict) else None
                if tkr:
                    tickers.append(tkr)
        elif isinstance(block, dict) and "ticker" in block:
            tickers.append(block["ticker"])
    return sorted(list({(t or "").upper() for t in tickers if t}))


def ensure_cache_for_tickers(
    country: str,
    tickers: Iterable[str],
    start_date: str,
    end_date: str,
) -> None:
    total = 0
    success = 0
    for ticker in tickers:
        total += 1
        try:
            before_df = load_cached_frame(country, ticker)
            before_count = 0 if before_df is None else before_df.shape[0]

            _ = fetch_ohlcv(
                ticker=ticker,
                country=country,
                date_range=[start_date, end_date],
            )

            after_df = load_cached_frame(country, ticker)
            after_count = 0 if after_df is None else after_df.shape[0]
            added_count = max(0, after_count - before_count)

            if after_count > 0:
                success += 1
                if added_count > 0:
                    print(
                        f"[{country.upper()}] {ticker}: {after_count} rows cached (+{added_count})"
                    )
                else:
                    print(f"[{country.upper()}] {ticker}: {after_count} rows cached (unchanged)")
            else:
                print(f"[{country.upper()}] {ticker}: 데이터 없음")
        except Exception as exc:  # noqa: BLE001
            print(f"오류: {country}/{ticker} 캐시 갱신 실패: {exc}")
    print(f"[{country.upper()}] 완료 - 총 {total}개 중 {success}개 캐시 반영")


def refresh_all_caches(
    countries: Iterable[str] = ("kor", "aus", "coin"),
    start_date: str = DEFAULT_START_DATE,
    end_date: Optional[str] = None,
) -> None:
    final_end = end_date or datetime.now().strftime("%Y-%m-%d")
    for country in countries:
        ticker_list = _list_etf_tickers(country)
        if not ticker_list:
            print(f"[{country.upper()}] 캐시 대상 티커가 없습니다.")
            continue
        ensure_cache_for_tickers(country, ticker_list, start_date, final_end)


def main():
    parser = argparse.ArgumentParser(description="OHLCV 캐시 업데이트")
    parser.add_argument(
        "--country",
        choices=["kor", "aus", "coin", "all"],
        default="all",
        help="갱신할 국가 (기본 all)",
    )
    parser.add_argument(
        "--start",
        default=DEFAULT_START_DATE,
        help="조회 시작일 (YYYY-MM-DD) - 기본 2020-01-01",
    )
    parser.add_argument(
        "--end",
        default=None,
        help="조회 종료일 (YYYY-MM-DD) - 기본 오늘",
    )
    parser.add_argument(
        "--tickers",
        default=None,
        help="쉼표로 구분된 티커 목록 (지정 시 해당 티커만 갱신)",
    )
    args = parser.parse_args()

    end_date = args.end or datetime.now().strftime("%Y-%m-%d")
    countries = ["kor", "aus", "coin"] if args.country == "all" else [args.country]

    if args.tickers:
        manual_tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
        for country in countries:
            ensure_cache_for_tickers(country, manual_tickers, args.start, end_date)
    else:
        refresh_all_caches(countries=countries, start_date=args.start, end_date=end_date)


if __name__ == "__main__":
    main()
