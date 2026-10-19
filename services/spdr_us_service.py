"""SPDR·State Street 미국 ETF 공식 상품 명단과 전체 구성종목 수집."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from config import CACHE_TTL_SLOW
from services.spdr_holdings_service import parse_spdr_holdings_xlsx
from services.xlsx_holdings_service import parse_xlsx_table_rows, read_xlsx_rows
from utils.http_session import shared_session
from utils.normalization import normalize_exchange_symbol
from utils.ttl_cache import TtlCache
from utils.us_etf_market_service import load_us_security_exchange_map

_DATA_URL = "https://www.ssga.com/library-content/products/fund-data/etfs/us"
_PRODUCT_MAP_CACHE = TtlCache(CACHE_TTL_SLOW, name="spdr_us_products", max_entries=1)


def _fetch_product_map() -> dict[str, str]:
    response = shared_session.get(f"{_DATA_URL}/spdr-product-data-us-en.xlsx", timeout=25)
    response.raise_for_status()
    products = parse_xlsx_table_rows(read_xlsx_rows(response.content), {"Ticker", "Name", "CUSIP", "Asset Class"})
    result = {}
    for product in products:
        ticker = product["Ticker"].replace("®", "").replace("™", "").upper()
        cusip = product["CUSIP"]
        if not ticker or len(cusip) != 9:
            raise ValueError("SPDR 상품의 티커 또는 CUSIP이 잘못됐습니다.")
        if ticker in result:
            raise ValueError(f"SPDR 상품 티커가 중복됐습니다: {ticker}")
        result[ticker] = cusip
    return result


def _normalize_holding(raw: dict[str, str], us_listings: dict[str, str]) -> dict[str, Any]:
    name = raw["Name"]
    weight = float(raw["Weight"])
    if not name or not math.isfinite(weight):
        raise ValueError("SPDR 구성종목 이름 또는 비중이 잘못됐습니다.")
    ticker = raw["Ticker"]
    if ticker in {"", "-"}:
        ticker = None
    currency = raw["Local Currency"].upper()
    symbol = normalize_exchange_symbol(ticker, suffix="") if ticker else None
    exchange = us_listings.get(symbol) if currency == "USD" else None
    if exchange is None:
        symbol = None
    if raw["Identifier"].startswith(f"999{currency}"):
        name = f"현금 · {name}"
    return {
        "ticker": symbol or ticker or raw["Identifier"] or None,
        "name": name,
        "weight": weight,
        "raw_code": raw["Identifier"] or None,
        "yahoo_symbol": symbol,
        "listing_currency": currency,
        "listing_exchange": exchange,
        "price_lookup_supported": symbol is not None,
    }


def fetch_spdr_us_holdings(ticker: str) -> dict[str, Any] | None:
    """비 SPDR 종목이면 None, 공식 수집 실패면 예외를 낸다."""
    ticker_norm = str(ticker).strip().upper()
    if not ticker_norm:
        raise ValueError("SPDR 구성종목 조회에 티커가 필요합니다.")
    if ticker_norm not in _PRODUCT_MAP_CACHE.get_or_compute("products", _fetch_product_map):
        return None
    response = shared_session.get(f"{_DATA_URL}/holdings-daily-us-en-{ticker_norm.lower()}.xlsx", timeout=25)
    response.raise_for_status()
    reference_date, raw_items = parse_spdr_holdings_xlsx(
        response.content, ticker_norm, {"Name", "Ticker", "Identifier", "Weight", "Local Currency"}
    )
    us_listings = load_us_security_exchange_map()
    items = [_normalize_holding(raw, us_listings) for raw in raw_items]
    return {
        "source": "spdr_us_xlsx",
        "fetched_at": datetime.now().isoformat(),
        "as_of_date": reference_date,
        "holdings_count": len(items),
        "holdings": items,
    }
