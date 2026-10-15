"""Roundhill 미국 ETF 공식 상품 명단과 일별 전체 보유 CSV 수집."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from config import CACHE_TTL_SLOW
from services.filepoint_holdings_service import fetch_filepoint_csv, normalize_filepoint_holding
from utils.http_session import shared_session
from utils.ttl_cache import TtlCache
from utils.us_etf_market_service import load_us_security_exchange_map

_BASE_URL = "https://www.roundhillinvestments.com"
_DATA_URL = f"{_BASE_URL}/assets/data/FilepointRoundhill.40RU.RU"
_PRODUCT_MAP_CACHE = TtlCache(CACHE_TTL_SLOW, name="roundhill_us_products", max_entries=1)
_HOLDINGS_CACHE = TtlCache(CACHE_TTL_SLOW, name="roundhill_us_holdings", max_entries=1)


def _fetch_product_map() -> dict[str, str]:
    response = shared_session.get(f"{_BASE_URL}/etf/", timeout=25)
    response.raise_for_status()
    products = {}
    for anchor in BeautifulSoup(response.content, "html.parser").select("a[href]"):
        url = urljoin(f"{_BASE_URL}/etf/", anchor["href"])
        parsed = urlparse(url)
        match = re.fullmatch(r"/etf/([a-z0-9]+)/", parsed.path)
        if parsed.scheme == "https" and parsed.netloc == "www.roundhillinvestments.com" and match:
            ticker = match[1].upper()
            if anchor.get_text(strip=True) == ticker:
                products[ticker] = url
    if not products:
        raise ValueError("Roundhill 공식 상품 명단에 ETF가 없습니다.")
    return products


def _fetch_holdings_data() -> tuple[str, dict[str, list[dict[str, str]]], str]:
    nav_rows = fetch_filepoint_csv(f"{_DATA_URL}_DailyNAV.csv", {"Fund Ticker", "Rate Date"}, provider="Roundhill")
    reference_date = max(datetime.strptime(row["Rate Date"], "%m/%d/%Y").date() for row in nav_rows)
    rows = fetch_filepoint_csv(
        f"{_DATA_URL}_Holdings_{reference_date.strftime('%m%d%Y')}.csv",
        {"Date", "Account", "StockTicker", "CUSIP", "SecurityName", "Weightings", "MoneyMarketFlag"},
        provider="Roundhill",
    )
    by_account: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        # 공식 화면은 보유 파일의 Date에서 하루를 빼 공시 기준일로 표시한다.
        date = datetime.strptime(row["Date"], "%m/%d/%Y").date() - timedelta(days=1)
        if date != reference_date or not row["Account"].strip():
            raise ValueError("Roundhill 구성종목의 공시 기준일 또는 상품 식별자가 잘못됐습니다.")
        by_account.setdefault(row["Account"].strip(), []).append(row)
    return reference_date.isoformat(), by_account, datetime.now(timezone.utc).isoformat()


def fetch_roundhill_us_holdings(ticker: str) -> dict[str, Any] | None:
    """공식 명단에 없는 상품은 None, 공식 수집 실패는 예외를 낸다."""
    ticker_norm = str(ticker).strip().upper()
    if not ticker_norm:
        raise ValueError("Roundhill 구성종목 조회에 티커가 필요합니다.")
    products = _PRODUCT_MAP_CACHE.get_or_compute("products", _fetch_product_map)
    if ticker_norm not in products:
        return None
    reference_date, by_account, fetched_at = _HOLDINGS_CACHE.get_or_compute("holdings", _fetch_holdings_data)
    rows = by_account.get(ticker_norm)
    if not rows:
        raise ValueError(f"Roundhill {ticker_norm} 공식 전체 보유 CSV에 구성종목이 없습니다.")
    us_listings = load_us_security_exchange_map()
    items = [normalize_filepoint_holding(raw, us_listings, provider="Roundhill") for raw in rows]
    return {
        "source": "roundhill_us_csv",
        "fetched_at": fetched_at,
        "as_of_date": reference_date,
        "holdings_count": len(items),
        "holdings": items,
    }
