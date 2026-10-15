"""SPDR 호주 ETF 공식 상품 명단과 일별 전체 구성종목 XLSX 수집."""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from config import CACHE_TTL_SLOW
from services.component_price_service import infer_yahoo_symbol_currency
from services.spdr_holdings_service import parse_spdr_holdings_xlsx
from utils.asx_ticker import ensure_asx_prefix, strip_asx_prefix
from utils.http_session import shared_session
from utils.normalization import resolve_bloomberg_listing
from utils.ttl_cache import TtlCache
from utils.us_etf_market_service import load_us_security_exchange_map

_BASE_URL = "https://www.ssga.com"
_PRODUCT_MAP_CACHE = TtlCache(CACHE_TTL_SLOW, name="spdr_au_products", max_entries=1)


def _fetch_product_map() -> dict[str, str]:
    response = shared_session.get(
        f"{_BASE_URL}/bin/v1/ssmp/fund/fundfinder",
        params={"country": "au", "language": "en_gb", "role": "individual", "product": "", "ui": "fund-finder"},
        timeout=25,
    )
    response.raise_for_status()
    payload = response.json()
    if payload["status"] != 200:
        raise ValueError("SPDR 호주 공식 상품 명단 조회가 실패했습니다.")
    products = {}
    for product in payload["data"]["funds"]["etfs"]["datas"]:
        ticker = product["fundTicker"].strip().upper()
        url = urljoin(_BASE_URL, product["fundUri"])
        parsed = urlparse(url)
        if (
            re.fullmatch(r"[A-Z0-9]+", ticker) is None
            or parsed.scheme != "https"
            or parsed.netloc != "www.ssga.com"
            or not parsed.path.startswith("/au/en_gb/individual/etfs/")
            or ticker in products
        ):
            raise ValueError("SPDR 호주 공식 상품의 티커 또는 주소가 잘못됐습니다.")
        products[ticker] = url
    if not products:
        raise ValueError("SPDR 호주 공식 상품 명단에 ETF가 없습니다.")
    return products


def _find_holdings_url(html: str, ticker: str) -> str:
    path = f"/library-content/products/fund-data/etfs/apac/holdings-daily-au-en-{ticker.lower()}.xlsx"
    urls = {
        urljoin(_BASE_URL, anchor["href"])
        for anchor in BeautifulSoup(html, "html.parser").select("a[href]")
        if anchor["href"] == path
    }
    if urls != {f"{_BASE_URL}{path}"}:
        raise ValueError(f"SPDR 호주 {ticker} 공식 전체 구성종목 XLSX 주소가 없습니다.")
    return urls.pop()


def _normalize_holding(raw: dict[str, str], us_listings: dict[str, str]) -> dict[str, Any]:
    label = raw["Ticker"].strip()
    name = raw["Name"].strip()
    currency = raw["Local Currency"].strip().upper()
    weight_text = raw["Weight (%)"].strip()
    weight = None if weight_text in {"", "-"} else float(weight_text)
    if not name or (weight is not None and not math.isfinite(weight)):
        raise ValueError("SPDR 호주 구성종목 이름 또는 비중이 잘못됐습니다.")
    is_cash = label.startswith("CASH_")
    is_future = "Futures" in name
    base, separator, market = label.rpartition("-")
    listing_label = base if separator and market == "US" else f"{base} {market}" if separator else label
    symbol, exchange = (
        resolve_bloomberg_listing(listing_label, currency=currency, us_listings=us_listings)
        if label and not (is_cash or is_future)
        else (None, None)
    )
    ticker = symbol or label or raw["SEDOL"] or raw["ISIN"] or None
    if symbol:
        ticker = ensure_asx_prefix(symbol[:-3]) if symbol.endswith(".AX") else symbol
        if symbol.endswith((".KS", ".KQ")):
            ticker = symbol[:-3]
    return {
        "ticker": ticker,
        "name": f"현금 · {name}" if is_cash else name,
        "weight": weight,
        "raw_code": raw["SEDOL"] if raw["ISIN"] == "Unassigned" else raw["ISIN"],
        "yahoo_symbol": symbol,
        "listing_currency": infer_yahoo_symbol_currency(symbol) if symbol else currency or None,
        "listing_exchange": exchange,
        "listing_country": raw["Trade Country Name"],
        "asset_class": "Cash" if is_cash else "Future" if is_future else None,
        "price_lookup_supported": symbol is not None,
    }


def fetch_spdr_au_holdings(ticker: str) -> dict[str, Any] | None:
    """공식 명단에 없는 상품은 None, 공식 수집 실패는 예외를 낸다."""
    ticker_norm = strip_asx_prefix(ticker).strip().upper()
    if not ticker_norm:
        raise ValueError("SPDR 호주 구성종목 조회에 티커가 필요합니다.")
    url = _PRODUCT_MAP_CACHE.get_or_compute("products", _fetch_product_map).get(ticker_norm)
    if url is None:
        return None
    response = shared_session.get(url, timeout=25)
    response.raise_for_status()
    holdings_url = _find_holdings_url(response.text, ticker_norm)
    response = shared_session.get(holdings_url, timeout=25)
    response.raise_for_status()
    reference_date, rows = parse_spdr_holdings_xlsx(
        response.content,
        ticker_norm,
        {"Name", "Ticker", "ISIN", "SEDOL", "Weight (%)", "Local Currency", "Trade Country Name"},
    )
    us_listings = load_us_security_exchange_map()
    items = [_normalize_holding(raw, us_listings) for raw in rows]
    return {
        "source": "spdr_au_xlsx",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "as_of_date": reference_date,
        "holdings_count": len(items),
        "holdings": items,
    }
