"""ETF 구성종목의 실시간 가격을 일괄 조회하여 stock_cache_meta 의 holdings_cache 에 저장.

목적:
    /ticker, /compare 화면이 ticker_detail endpoint 호출 시 구성종목 가격을 실시간으로
    fetch 하면 외부 API 응답 누적으로 30초 timeout 이 자주 발생한다. 이를 회피하기 위해
    가격 조회를 본 배치로 분리하고, ticker_detail 은 캐시된 값만 읽도록 한다.

동작:
    1. stock_cache_meta 의 모든 ETF (holdings_cache.items 존재) 조회
    2. 구성종목의 unique set 을 통화별로 분류 → 한 번에 일괄 외부 API 호출
       (네이버 ETF iNAV / 토스 미국주식 / 호주 QuoteAPI / 야후 일반)
    3. 결과를 각 ETF 의 holdings_cache.items 의 항목에 current_price/change_pct 등으로 채움
    4. 업데이트된 holdings_cache 를 다시 저장

호출 주기:
    평일 08:00 ~ 17:00 매 10분 KST (infra/cron/crontab 참조)

이 배치는 가벼움 — 외부 API 호출이 unique 종목 수에 비례하고, ETF 별 mongodb 업데이트만 함.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

# 프로젝트 루트를 Python 경로에 추가 (scripts/ 안의 다른 스크립트와 동일한 패턴)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.component_price_service import enrich_component_prices
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

    processed = 0
    updated = 0
    failures = 0

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
        processed += 1

        try:
            # cumulative_base_date 는 ticker_detail 이 결정하는 값이라 여기선 빈값.
            # 우리는 current_price/previous_close/change_pct 만 채우면 된다.
            priced_items, price_as_of_date = enrich_component_prices(
                holdings_items,
                price_fetch_limit=None,  # 전 종목 처리
            )
            holdings_cache["items"] = priced_items
            holdings_cache["price_updated_at"] = datetime.now(ZoneInfo("Asia/Seoul")).isoformat()
            if price_as_of_date:
                holdings_cache["price_as_of_date"] = price_as_of_date
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
