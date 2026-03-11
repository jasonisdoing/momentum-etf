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

from utils.cache_utils import (
    clean_temp_cache_collections,
    drop_cache_collection,
    swap_cache_collection,
)
from utils.data_loader import fetch_ohlcv, repair_recent_trading_day_gaps
from utils.env import load_env_if_present
from utils.identifier_guard import ensure_account_pool_id_separation
from utils.logger import get_app_logger
from utils.pool_registry import get_pool_country_code, list_available_pools
from utils.portfolio_io import load_portfolio_master
from utils.settings_loader import get_account_settings, list_available_accounts, load_common_settings
from utils.stock_list_io import get_all_etfs_including_deleted


def _determine_start_date() -> str:
    settings = load_common_settings() or {}
    start = settings.get("CACHE_START_DATE")
    if start:
        return str(start)


from collections.abc import Callable


def refresh_cache_for_target(
    target_id: str,
    start_date: str | None,
    progress_callback: Callable[[int, int, str], None] | None = None,
):
    """지정된 계정/종목풀(target_id)에 대한 가격 데이터 캐시를 새로 고칩니다."""
    logger = get_app_logger()
    target_norm = (target_id or "").strip().lower()

    try:
        if target_norm in list_available_accounts():
            settings = get_account_settings(target_norm)
            country_code = settings.get("country_code", "kor").lower()
        elif target_norm in list_available_pools():
            country_code = get_pool_country_code(target_norm, default="kor")
        else:
            country_code = "kor"
    except Exception:
        logger.warning(f"대상 설정을 불러올 수 없어 기본 국가코드(kor)를 사용합니다: {target_norm}")
        country_code = "kor"

    logger.info("[%s] 캐시 갱신 시작 (국가설정: %s, 시작일: %s)", target_norm.upper(), country_code, start_date)

    # 임시 컬렉션 정리
    removed = clean_temp_cache_collections(target_norm, max_age_seconds=3600)
    if removed:
        logger.info(
            ("[%s] 기존 임시 컬렉션 %d개를 삭제했습니다. (1시간 이상 경과분)"),
            target_norm,
            removed,
        )

    # 종목 리스트 로드
    try:
        all_etfs_from_file = get_all_etfs_including_deleted(target_norm)
    except Exception:
        all_etfs_from_file = []

    all_map = {str(item.get("ticker") or "").strip().upper(): item for item in all_etfs_from_file if item.get("ticker")}

    # 계좌 실행 시 portfolio_master 보유 종목도 함께 반영한다.
    if target_norm in list_available_accounts():
        holdings = _collect_portfolio_master_holdings(target_norm)
        for item in holdings:
            ticker = str(item.get("ticker") or "").strip().upper()
            if not ticker or ticker in all_map:
                continue
            all_map[ticker] = item

    # 벤치마크 추가
    benchmark_tickers = _collect_benchmark_tickers(target_norm)
    for bench in benchmark_tickers:
        norm = str(bench or "").strip().upper()
        if not norm or norm in all_map:
            continue
        all_map[norm] = {
            "ticker": norm,
            "name": norm,
            "type": "etf",
        }

    if not all_map:
        logger.warning("[%s] 갱신할 종목이 없습니다 (stock_meta/portfolio_master 모두 비어있음).", target_norm.upper())
        return

    target_items = list(all_map.values())

    suffix = f"{os.getpid()}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    # 임시 컬렉션 토큰 생성: account_id + _tmp_ + suffix
    temp_token = f"{target_norm}_tmp_{suffix}"
    drop_cache_collection(temp_token)

    total_tickers = len(target_items)
    try:
        for i, etf in enumerate(target_items, 1):
            ticker = etf.get("ticker")
            name = etf.get("name") or "-"

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
                unresolved_days = repair_recent_trading_day_gaps(
                    ticker,
                    country_code,
                    account_id=temp_token,
                    lookback_days=15,
                )
                if unresolved_days:
                    unresolved_text = ", ".join(day.strftime("%Y-%m-%d") for day in unresolved_days)
                    logger.warning(
                        " -> 가격 캐시 갱신 중: %d/%d - %s(%s) - 최근 거래일 누락 유지: %s",
                        i,
                        total_tickers,
                        name,
                        ticker,
                        unresolved_text,
                    )
                else:
                    logger.info(" -> 가격 캐시 갱신 중: %d/%d - %s(%s)", i, total_tickers, name, ticker)
            except Exception as e:
                logger.error("%s 데이터 처리 중 오류 발생: %s", ticker, e)

        swap_cache_collection(target_norm, temp_token)
        logger.info("-> [%s] 캐시 갱신 완료.", target_norm.upper())
    except Exception as exc:
        logger.error("[%s] 캐시 갱신 실패: %s", target_norm.upper(), exc)
        drop_cache_collection(temp_token)
        raise
    finally:
        drop_cache_collection(temp_token)


def _collect_benchmark_tickers(target_id: str) -> list[str]:
    """해당 계정 설정에 정의된 벤치마크 티커들을 수집합니다."""
    tickers = set()

    try:
        if target_id not in list_available_accounts():
            return []
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


def _collect_portfolio_master_holdings(target_id: str) -> list[dict[str, str]]:
    """portfolio_master의 현재 보유 종목을 캐시 갱신 대상에 추가한다."""
    snapshot = load_portfolio_master(target_id)
    if not snapshot:
        return []

    holdings = snapshot.get("holdings")
    if not isinstance(holdings, list):
        return []

    results: list[dict[str, str]] = []
    for item in holdings:
        if not isinstance(item, dict):
            continue
        ticker = str(item.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        results.append(
            {
                "ticker": ticker,
                "name": str(item.get("name") or ticker).strip() or ticker,
                "type": "etf",
            }
        )
    return results


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
    try:
        ensure_account_pool_id_separation()
    except Exception as exc:
        logger.error("%s", exc)
        return

    parser = _build_parser()
    args = parser.parse_args()

    target = (args.target or "").strip().lower()
    start_date = args.start or _determine_start_date()

    targets_to_update: list[str] = []
    available_accounts = list_available_accounts()
    available_pools = list_available_pools()
    available_targets = sorted({*available_accounts, *available_pools})

    if not target:
        # Update all accounts + pools
        targets_to_update = available_targets
    else:
        if target in available_targets:
            targets_to_update = [target]
        else:
            logger.error(f"Target '{target}' is not a valid ID (account/pool).")
            return

    if not targets_to_update:
        logger.warning("갱신할 대상이 없습니다.")
        return

    logger.info("입력 파라미터: targets=%s, start=%s", targets_to_update, start_date)

    for t_id in targets_to_update:
        refresh_cache_for_target(t_id, start_date)


if __name__ == "__main__":
    main()
