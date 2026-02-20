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
from utils.stock_list_io import get_all_etfs_including_deleted


def _determine_start_date() -> str:
    settings = load_common_settings() or {}
    start = settings.get("CACHE_START_DATE")
    if start:
        return str(start)


from collections.abc import Callable


def refresh_cache_for_target(
    account_id: str,
    start_date: str | None,
    progress_callback: Callable[[int, int, str], None] | None = None,
):
    """지정된 계정(account_id)에 대한 가격 데이터 캐시를 새로 고칩니다."""
    logger = get_app_logger()

    try:
        settings = get_account_settings(account_id)
        country_code = settings.get("country_code", "kor").lower()
    except Exception:
        logger.warning(f"계정 설정을 불러올 수 없어 기본 국가코드(kor)를 사용합니다: {account_id}")
        country_code = "kor"

    logger.info("[%s] 계정 캐시 갱신 시작 (국가설정: %s, 시작일: %s)", account_id.upper(), country_code, start_date)

    # 임시 컬렉션 정리
    removed = clean_temp_cache_collections(account_id, max_age_seconds=3600)
    if removed:
        logger.info(
            ("[%s] 기존 임시 컬렉션 %d개를 삭제했습니다. (1시간 이상 경과분)"),
            account_id,
            removed,
        )

    # 종목 리스트 로드 (계정 전용)
    try:
        all_etfs_from_file = get_all_etfs_including_deleted(account_id)
    except Exception:
        all_etfs_from_file = []

    if not all_etfs_from_file:
        logger.warning("[%s] 갱신할 종목이 없습니다 (종목 파일 비어있음/없음).", account_id.upper())
        return

    all_map = {str(item.get("ticker") or "").strip().upper(): item for item in all_etfs_from_file if item.get("ticker")}

    # 벤치마크 추가
    benchmark_tickers = _collect_benchmark_tickers(account_id)
    for bench in benchmark_tickers:
        norm = str(bench or "").strip().upper()
        if not norm or norm in all_map:
            continue
        all_map[norm] = {
            "ticker": norm,
            "name": norm,
            "type": "etf",
        }

    target_items = list(all_map.values())

    suffix = f"{os.getpid()}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    # 임시 컬렉션 토큰 생성: account_id + _tmp_ + suffix
    temp_token = f"{account_id}_tmp_{suffix}"
    drop_cache_collection(temp_token)

    total_tickers = len(target_items)
    try:
        for i, etf in enumerate(target_items, 1):
            ticker = etf.get("ticker")
            name = etf.get("name") or "-"
            logger.info(" -> 가격 캐시 갱신 중: %d/%d - %s(%s)", i, total_tickers, name, ticker)

            if progress_callback:
                progress_callback(i, total_tickers, f"{name}({ticker})")

            try:
                range_start = start_date or "1990-01-01"
                fetch_ohlcv(
                    ticker,
                    country=country_code,  # 마켓 캘린더/API 설정용
                    date_range=[range_start, None],
                    update_listing_meta=False,
                    force_refresh=True,
                    account_id=temp_token,  # 임시 캐시에 저장
                )
            except Exception as e:
                logger.error("%s 데이터 처리 중 오류 발생: %s", ticker, e)

        swap_cache_collection(account_id, temp_token)
        logger.info("-> [%s] 계정 캐시 갱신 완료.", account_id.upper())
    except Exception as exc:
        logger.error("[%s] 캐시 갱신 실패: %s", account_id.upper(), exc)
        drop_cache_collection(temp_token)
        raise
    finally:
        drop_cache_collection(temp_token)


def _collect_benchmark_tickers(target_id: str) -> list[str]:
    """해당 계정 설정에 정의된 벤치마크 티커들을 수집합니다."""
    tickers = set()

    try:
        settings = get_account_settings(target_id)

        # 'benchmark' (dict, single) 처리
        single_bm = settings.get("benchmark")
        if single_bm and isinstance(single_bm, dict):
            ticker = str(single_bm.get("ticker") or "").strip().upper()
            if ticker:
                tickers.add(ticker)

        return sorted(tickers)
    except Exception:
        pass

    return sorted(tickers)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OHLCV 캐시 갱신 스크립트",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("target", nargs="?", help="Account ID")
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

    target = (args.target or "").strip().lower()
    start_date = args.start or _determine_start_date()

    targets_to_update: list[str] = []

    if not target:
        # Update All Available Accounts
        targets_to_update = list_available_accounts()
    else:
        # Check if target is account
        if target in list_available_accounts():
            targets_to_update = [target]
        else:
            logger.error(f"Target '{target}' is not a valid account ID.")
            return

    if not targets_to_update:
        logger.warning("갱신할 대상이 없습니다.")
        return

    logger.info("입력 파라미터: targets=%s, start=%s", targets_to_update, start_date)

    for t_id in targets_to_update:
        refresh_cache_for_target(t_id, start_date)


if __name__ == "__main__":
    main()
