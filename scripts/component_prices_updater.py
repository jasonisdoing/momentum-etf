"""ETF 구성종목의 실시간 가격을 일괄 조회하여 stock_cache_meta 의 holdings_cache 에 저장.

목적:
    /ticker, /compare 화면이 ticker_detail endpoint 호출 시 구성종목 가격을 실시간으로
    fetch 하면 외부 API 응답 누적으로 30초 timeout 이 자주 발생한다. 이를 회피하기 위해
    가격 조회를 본 배치로 분리하고, ticker_detail 은 캐시된 값만 읽도록 한다.

동작 (핵심: 외부 API 호출 횟수 최소화):
    1. stock_cache_meta 의 모든 ETF (holdings_cache.items 존재) 조회 + 각 ETF 의 base_date 결정
    2. ★ 모든 ETF 의 구성종목을 1개 리스트로 합본 → build_component_price_snapshot() 1회 호출
       → 통화별 unique 종목 set 단위로 외부 API 1번씩만 호출 (KOR / US / AU / worldstock / yahoo)
    3. ★ base_date 별로 baseline 종목 unique set 을 만들어 KOR/yahoo baseline 도 1번씩만 fetch
    4. ETF 별 루프 — enrich_component_prices(external_fetch_enabled=False, snapshot+baseline 주입)
       → 외부 호출 0, mongodb update 만 수행

호출 주기:
    월~토 24시간 매 30분 KST (infra/cron/crontab 참조)

성능:
    - 199 ETF × 평균 50종목 = 약 1만 호출 → unique 종목 수만큼(수백) + base_date 그룹 수만큼으로 축소
    - 예상 소요: 2~5분 (KOR sleep 0.3s × unique KOR 종목 수가 가장 큰 비중)
"""

from __future__ import annotations

import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

# 프로젝트 루트를 Python 경로에 추가 (scripts/ 안의 다른 스크립트와 동일한 패턴)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.component_price_service import (
    _is_cash_like_holding,
    _is_korean_six_digit_holding,
    _is_us_yahoo_symbol,
    _normalize_upper,
    _safe_fetch_cached_baseline_prices,
    _safe_fetch_yahoo_baseline_prices,
    build_component_price_snapshot,
    enrich_component_prices,
)
from services.portfolio_change_service import determine_portfolio_change_base_date
from services.stock_cache_service import refresh_stock_holdings_cache
from utils.db_manager import get_db_connection
from utils.env import load_env_if_present
from utils.logger import get_app_logger

logger = get_app_logger()


def _iter_etfs_with_holdings() -> list[dict[str, Any]]:
    """stock_cache_meta 컬렉션에서 holdings_cache.items 가 있는 모든 ETF 문서 반환."""
    db = get_db_connection()
    if db is None:
        raise RuntimeError("DB 연결 실패 — stock_cache_meta 조회 불가")
    cursor = db.stock_cache_meta.find(
        {"holdings_cache.items": {"$exists": True, "$ne": []}},
        {"_id": 0, "ticker_type": 1, "ticker": 1, "country_code": 1, "name": 1, "holdings_cache": 1},
    )
    return list(cursor)


def update_component_prices() -> dict[str, int]:
    """모든 ETF 의 구성종목 가격을 최신화한다.

    Returns: {"etfs_processed": N, "etfs_updated": N, "failures": N}
    """
    load_env_if_present()
    started = time.perf_counter()
    etfs = _iter_etfs_with_holdings()
    logger.info("[component_prices_updater] 대상 ETF: %d개", len(etfs))

    # ─────────────────────────────────────────────────────────────
    # 1) ETF 메타 정리 + 모든 holdings 합본 + base_date 별 baseline 그룹 구성
    # ─────────────────────────────────────────────────────────────
    etf_meta_list: list[tuple[str, str, str, str, str | None, list[dict[str, Any]], dict[str, Any]]] = []
    all_items: list[dict[str, Any]] = []
    # base_date -> {"kor": set[ticker], "yahoo": set[yahoo_symbol], "prev_day": set[yahoo_symbol]}
    base_date_groups: dict[str, dict[str, set[str]]] = defaultdict(
        lambda: {"kor": set(), "yahoo": set(), "prev_day": set()}
    )

    for etf in etfs:
        ticker_type = str(etf.get("ticker_type") or "").strip()
        ticker = str(etf.get("ticker") or "").strip()
        if not ticker_type or not ticker:
            continue
        country_code = str(etf.get("country_code") or "").strip().lower() or "kor"
        name = str(etf.get("name") or "").strip() or ticker
        holdings_cache = dict(etf.get("holdings_cache") or {})
        holdings_items = list(holdings_cache.get("items") or [])
        if not holdings_items:
            continue

        try:
            base_date = determine_portfolio_change_base_date(ticker_type, ticker)
        except Exception as exc:
            logger.warning(
                "[component_prices_updater] %s/%s base_date 결정 실패: %s",
                ticker_type.upper(),
                ticker,
                exc,
            )
            base_date = None

        etf_meta_list.append(
            (ticker_type, ticker, country_code, name, base_date, holdings_items, holdings_cache)
        )
        all_items.extend(holdings_items)

        if base_date:
            group = base_date_groups[base_date]
            for item in holdings_items:
                if _is_cash_like_holding(item):
                    continue
                component_ticker = _normalize_upper(item.get("ticker"))
                if _is_korean_six_digit_holding(item):
                    if component_ticker:
                        group["kor"].add(component_ticker)
                else:
                    yahoo_symbol = _normalize_upper(item.get("yahoo_symbol")) or component_ticker
                    if yahoo_symbol:
                        group["yahoo"].add(yahoo_symbol)
                        if _is_us_yahoo_symbol(yahoo_symbol):
                            group["prev_day"].add(yahoo_symbol)

    if not etf_meta_list:
        logger.info("[component_prices_updater] 처리할 ETF 없음")
        return {"etfs_processed": 0, "etfs_updated": 0, "failures": 0}

    logger.info(
        "[component_prices_updater] 합본 holdings: %d개 / base_date 그룹: %d개",
        len(all_items),
        len(base_date_groups),
    )

    # ─────────────────────────────────────────────────────────────
    # 2) ★ 통화별 unique 종목 단위로 현재가 snapshot 1회 fetch (외부 API 호출 핵심 구간)
    # ─────────────────────────────────────────────────────────────
    snapshot_started = time.perf_counter()
    logger.info("[component_prices_updater] 현재가 snapshot 시작")
    snapshot = build_component_price_snapshot(all_items)
    logger.info(
        "[component_prices_updater] 현재가 snapshot 완료: %d개 / %.1fs",
        len(snapshot),
        time.perf_counter() - snapshot_started,
    )

    # ─────────────────────────────────────────────────────────────
    # 3) ★ base_date 별로 KOR/Yahoo baseline 도 unique 종목 set 단위로 1회 fetch
    # ─────────────────────────────────────────────────────────────
    # base_date -> (kor_baseline_map, yahoo_baseline_map)
    baseline_by_date: dict[str, tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]] = {}
    for bd, groups in base_date_groups.items():
        bl_started = time.perf_counter()
        kor_baseline_map = _safe_fetch_cached_baseline_prices("kor", sorted(groups["kor"]), bd)
        yahoo_baseline_map = _safe_fetch_yahoo_baseline_prices(
            sorted(groups["yahoo"]), bd, groups["prev_day"]
        )
        baseline_by_date[bd] = (kor_baseline_map, yahoo_baseline_map)
        logger.info(
            "[component_prices_updater] baseline %s: KOR %d / Yahoo %d / %.1fs",
            bd,
            len(kor_baseline_map),
            len(yahoo_baseline_map),
            time.perf_counter() - bl_started,
        )

    # ─────────────────────────────────────────────────────────────
    # 4) ETF 별 매핑 — enrich 호출은 외부 API 호출 차단(snapshot + baseline 만 사용)
    # ─────────────────────────────────────────────────────────────
    map_started = time.perf_counter()
    processed = 0
    updated = 0
    failures = 0
    total = len(etf_meta_list)

    for ticker_type, ticker, country_code, name, base_date, holdings_items, holdings_cache in etf_meta_list:
        processed += 1
        try:
            kor_baseline_map, yahoo_baseline_map = baseline_by_date.get(base_date or "", ({}, {}))
            priced_items, price_as_of_date = enrich_component_prices(
                holdings_items,
                price_fetch_limit=None,
                cumulative_base_date=base_date,
                component_price_snapshot=snapshot,
                external_fetch_enabled=False,  # ★ 외부 API 호출 완전 차단
                korean_baseline_price_map=kor_baseline_map,
                yahoo_baseline_price_map=yahoo_baseline_map,
            )
            holdings_cache["items"] = priced_items
            holdings_cache["price_updated_at"] = datetime.now(ZoneInfo("Asia/Seoul")).isoformat()
            if price_as_of_date:
                holdings_cache["price_as_of_date"] = price_as_of_date
            if base_date:
                holdings_cache["base_date"] = base_date
            refresh_stock_holdings_cache(
                ticker_type,
                ticker,
                country_code=country_code,
                name=name,
                holdings_cache=holdings_cache,
            )
            updated += 1
        except Exception as exc:
            failures += 1
            logger.warning(
                "[component_prices_updater] %s/%s 구성종목 가격 갱신 실패: %s",
                ticker_type.upper(),
                ticker,
                exc,
            )

        # 50개마다 또는 마지막에 진행률 출력
        if processed % 50 == 0 or processed == total:
            logger.info(
                "[component_prices_updater] 매핑 진행: %d / %d (%.1fs)",
                processed,
                total,
                time.perf_counter() - map_started,
            )

    elapsed = time.perf_counter() - started
    logger.info(
        "[component_prices_updater] 완료: 처리 %d / 갱신 %d / 실패 %d / 소요 %.1fs",
        processed,
        updated,
        failures,
        elapsed,
    )
    return {"etfs_processed": processed, "etfs_updated": updated, "failures": failures}


def main() -> None:
    update_component_prices()


if __name__ == "__main__":
    main()
