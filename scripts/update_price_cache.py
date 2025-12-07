#!/usr/bin/env python
"""전체 OHLCV 캐시를 초기화한 뒤 설정된 시작일 이후 데이터를 다시 받아옵니다."""

from __future__ import annotations

import argparse
import os
import sys
import time
import uuid

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.account_registry import get_account_settings, list_available_accounts
from utils.cache_utils import (
    clean_temp_cache_collections,
    drop_cache_collection,
    swap_cache_collection,
)
from utils.data_loader import fetch_ohlcv
from utils.env import load_env_if_present
from utils.logger import get_app_logger
from utils.settings_loader import load_common_settings
from utils.stock_list_io import get_all_etfs


def _determine_start_date() -> str:
    settings = load_common_settings() or {}
    start = settings.get("CACHE_START_DATE")
    if start:
        return str(start)


def _collect_benchmark_tickers(target_country: str | None = None) -> list[str]:
    tickers = set()
    target_country_norm = target_country.strip().lower() if target_country else None

    for account in list_available_accounts():
        try:
            settings = get_account_settings(account)
        except Exception:
            continue

        # 국가 코드가 지정된 경우 필터링
        if target_country_norm:
            acct_country = str(settings.get("country_code") or "").strip().lower()
            if acct_country != target_country_norm:
                continue

        for bm in settings.get("benchmarks", []) or []:
            ticker = str(bm.get("ticker") or "").strip().upper()
            if ticker:
                tickers.add(ticker)
    return sorted(tickers)


def refresh_all_caches(countries: list[str], start_date: str | None):
    """지정된 국가의 모든 종목에 대한 가격 데이터 캐시를 새로 고칩니다."""
    logger = get_app_logger()
    logger.info("캐시 갱신 시작 (국가: %s, 시작일: %s)", ", ".join(countries), start_date)

    for country in countries:
        benchmark_tickers = _collect_benchmark_tickers(country)
        logger.info("[%s] 국가의 캐시를 갱신합니다...", country.upper())
        removed = clean_temp_cache_collections(country, max_age_seconds=3600)
        if removed:
            logger.info(
                "[%s] 기존 임시 컬렉션 %d개를 삭제했습니다. (1시간 이상 경과분)",
                country.upper(),
                removed,
            )
        all_etfs_from_file = get_all_etfs(country)
        all_map = {
            str(item.get("ticker") or "").strip().upper(): item for item in all_etfs_from_file if item.get("ticker")
        }
        for bench in benchmark_tickers:
            norm = str(bench or "").strip().upper()
            if not norm or norm in all_map:
                continue
            all_map[norm] = {
                "ticker": norm,
                "name": norm,
                "category": "BENCHMARK",
                "type": "etf",
            }
        all_etfs_from_file = list(all_map.values())

        suffix = f"{os.getpid()}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        temp_token = f"{country}_tmp_{suffix}"
        drop_cache_collection(temp_token)

        total_tickers = len(all_etfs_from_file)
        try:
            for i, etf in enumerate(all_etfs_from_file, 1):
                ticker = etf.get("ticker")
                name = etf.get("name") or "-"
                logger.info(" -> 처리 중: %d/%d - %s(%s)", i, total_tickers, name, ticker)

                try:
                    range_start = start_date or "1990-01-01"
                    fetch_ohlcv(
                        ticker,
                        country,
                        date_range=[range_start, None],
                        update_listing_meta=False,  # listing_date는 stock_meta_updater에서만 업데이트
                        force_refresh=True,
                        cache_country=temp_token,
                    )
                except Exception as e:
                    logger.error("%s 데이터 처리 중 오류 발생: %s", ticker, e)
                    raise

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

    parser = argparse.ArgumentParser(description="OHLCV 캐시 갱신 스크립트")
    parser.add_argument("target", nargs="?", help="Account ID or Country Code")
    parser.add_argument("--start", help="데이터 조회 시작일 (YYYY-MM-DD)")
    args = parser.parse_args()

    target = (args.target or "").strip().lower()
    start_date = args.start or _determine_start_date()

    countries_to_update = []

    if not target:
        # No argument -> Update all default (kor, us)
        countries_to_update = ["kor", "us"]
    else:
        if target in ["kor", "us"]:
            countries_to_update = [target]
        else:
            # Assume account ID
            try:
                settings = get_account_settings(target)
                country = settings.get("country_code", "kor").lower()
                countries_to_update = [country]
            except Exception:
                logger.warning(f"Account '{target}' not found. Treating as country code.")
                countries_to_update = [target]

    if not countries_to_update:
        parser.error("갱신할 국가를 결정할 수 없습니다.")

    logger.info("입력 파라미터: countries=%s, start=%s", countries_to_update, start_date)
    refresh_all_caches(countries_to_update, start_date)


if __name__ == "__main__":
    main()
