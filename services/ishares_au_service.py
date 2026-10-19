"""iShares 호주 ETF 공식 상품 명단과 전체 구성종목 CSV 수집."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import parse_qs, urljoin, urlparse

from bs4 import BeautifulSoup

from config import CACHE_TTL_SLOW
from services.ishares_holdings_service import fetch_ishares_product_map, parse_ishares_holdings
from utils.asx_ticker import strip_asx_prefix
from utils.http_session import shared_session
from utils.ttl_cache import TtlCache

_BASE_URL = "https://www.blackrock.com"
_PRODUCT_MAP_CACHE = TtlCache(CACHE_TTL_SLOW, name="ishares_au_products", max_entries=1)


def _fetch_product_map() -> dict[str, str]:
    return fetch_ishares_product_map(
        base_url=_BASE_URL,
        region="au",
        config_path="/templatedata/config/product-screener-v3/data/en/blk-au/product-screener-backend-config",
    )


def _find_holdings_url(html: str, ticker: str, product_path: str) -> str:
    product_id = product_path.split("/")[3]
    links = set()
    for anchor in BeautifulSoup(html, "html.parser").select("a[href]"):
        url = urljoin(_BASE_URL, anchor["href"])
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        if params.get("fileType") != ["csv"] or params.get("dataType") != ["fund"]:
            continue
        if (
            parsed.scheme != "https"
            or parsed.netloc != "www.blackrock.com"
            or re.fullmatch(rf"/au/products/{re.escape(product_id)}/[^/]+/\d+\.ajax", parsed.path) is None
            or params.get("fileName") != [f"{ticker}_holdings"]
        ):
            raise ValueError(f"iShares 호주 {ticker} 공식 구성종목 CSV의 상품 또는 주소가 다릅니다.")
        links.add(url)
    if len(links) != 1:
        raise ValueError(f"iShares 호주 {ticker} 공식 전체 구성종목 CSV 주소가 없거나 중복됐습니다.")
    return links.pop()


def fetch_ishares_au_holdings(ticker: str) -> dict[str, Any] | None:
    """공식 명단에 없는 상품은 None, 공식 수집 실패는 예외를 낸다."""
    ticker_norm = strip_asx_prefix(ticker)
    if not ticker_norm:
        raise ValueError("iShares 호주 구성종목 조회에 티커가 필요합니다.")
    path = _PRODUCT_MAP_CACHE.get_or_compute("products", _fetch_product_map).get(ticker_norm)
    if path is None:
        return None
    response = shared_session.get(f"{_BASE_URL}{path}", timeout=25)
    response.raise_for_status()
    csv_url = _find_holdings_url(response.text, ticker_norm, path)
    response = shared_session.get(csv_url, timeout=25)
    response.raise_for_status()
    reference_date, items = parse_ishares_holdings(response.text, date_format="%d-%b-%Y")
    return {
        "source": "ishares_au_csv",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "as_of_date": reference_date,
        "holdings_count": len(items),
        "holdings": items,
    }
