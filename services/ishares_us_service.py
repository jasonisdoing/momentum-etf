"""iShares·BlackRock 미국 ETF 공식 상품 명단과 전체 구성종목 수집."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from config import CACHE_TTL_SLOW
from services.ishares_holdings_service import fetch_ishares_product_map, parse_ishares_holdings
from utils.http_session import shared_session
from utils.ttl_cache import TtlCache

_BASE_URL = "https://www.ishares.com"
_PRODUCT_MAP_CACHE = TtlCache(CACHE_TTL_SLOW, name="ishares_us_products", max_entries=1)


def _fetch_product_map() -> dict[str, str]:
    return fetch_ishares_product_map(
        base_url=_BASE_URL,
        region="us",
        config_path="/templatedata/config/product-screener-v3/data/en/us-ishares/ishares-product-screener-backend-config",
    )


def fetch_ishares_us_holdings(ticker: str) -> dict[str, Any] | None:
    """비 iShares 종목이면 None, 공식 수집 실패면 예외를 낸다."""
    ticker_norm = str(ticker).strip().upper()
    if not ticker_norm:
        raise ValueError("iShares 구성종목 조회에 티커가 필요합니다.")
    path = _PRODUCT_MAP_CACHE.get_or_compute("products", _fetch_product_map).get(ticker_norm)
    if path is None:
        return None
    response = shared_session.get(f"{_BASE_URL}{path}/latest-holdings.csv", timeout=25)
    response.raise_for_status()
    reference_date, items = parse_ishares_holdings(response.text, date_format="%b %d, %Y")
    return {
        "source": "ishares_us_csv",
        "fetched_at": datetime.now().isoformat(),
        "as_of_date": reference_date,
        "holdings_count": len(items),
        "holdings": items,
    }
