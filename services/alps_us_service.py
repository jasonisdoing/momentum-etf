"""ALPS 미국 ETF 공식 상품 명단과 전체 구성종목 API 수집."""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from config import CACHE_TTL_SLOW
from services.component_price_service import infer_yahoo_symbol_currency
from utils.asx_ticker import ensure_asx_prefix
from utils.http_session import shared_session
from utils.normalization import resolve_bloomberg_listing
from utils.ttl_cache import TtlCache
from utils.us_etf_market_service import load_us_security_exchange_map

_BASE_URL = "https://www.alpsfunds.com"
_API_URL = "https://secure.alpsinc.com/MarketingAPI/api/v1/Holding"
_PRODUCT_MAP_CACHE = TtlCache(CACHE_TTL_SLOW, name="alps_us_products", max_entries=1)


def _fetch_product_map() -> set[str]:
    # 공식 상세 페이지가 불러오는 공통 상품 명단에서 ETF만 읽는다.
    response = shared_session.get(f"{_BASE_URL}/exchange-traded-funds/enfr", timeout=25)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    urls = {
        urljoin(_BASE_URL, script["src"])
        for script in soup.select("script[src]")
        if "template_fund_map" in script["src"]
    }
    if len(urls) != 1:
        raise ValueError("ALPS 공식 상품 명단 주소가 없거나 중복됐습니다.")
    url = urls.pop()
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc != "www.alpsfunds.com":
        raise ValueError("ALPS 공식 상품 명단의 호스트가 다릅니다.")
    response = shared_session.get(url, timeout=25)
    response.raise_for_status()
    match = re.search(r"\bvar etf_map=(\[.*?\]),oef_map=", response.text)
    if match is None:
        raise ValueError("ALPS 공식 명단에 ETF 데이터가 없습니다.")
    data = json.loads(re.sub(r"([,{])(\w+):", r'\1"\2":', match[1]))
    products = set()
    for product in data:
        tickers = product["tickerArray"]
        if not isinstance(tickers, list) or not tickers:
            raise ValueError("ALPS 공식 ETF 티커 목록이 잘못됐습니다.")
        for ticker in tickers:
            if not isinstance(ticker, str) or re.fullmatch(r"[A-Z0-9]+", ticker) is None or ticker in products:
                raise ValueError("ALPS 공식 ETF 티커가 잘못됐거나 중복됐습니다.")
            products.add(ticker)
    if not products:
        raise ValueError("ALPS 공식 상품 명단에 ETF가 없습니다.")
    return products


def _parse_holdings(payload: Any, ticker: str) -> tuple[str, list[dict[str, Any]]]:
    required = {"fundsymbol", "fullpartial", "asofdate", "name", "holdingsymbol", "weight", "holdingtypeabbrev"}
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"ALPS {ticker} 전체 구성종목이 비었거나 잘못됐습니다.")
    for raw in payload:
        if not isinstance(raw, dict) or not required.issubset(raw):
            raise ValueError(f"ALPS {ticker} 구성종목 필수 필드가 없습니다.")
        if raw["fundsymbol"] != ticker or raw["fullpartial"] != "full":
            raise ValueError(f"ALPS {ticker} 전체 구성종목의 상품 또는 목록 종류가 다릅니다.")
        if not raw["asofdate"] or raw["asofdate"] != payload[0]["asofdate"]:
            raise ValueError(f"ALPS {ticker} 구성종목 기준일이 없거나 서로 다릅니다.")
    return datetime.fromisoformat(payload[0]["asofdate"]).date().isoformat(), payload


def _normalize_holding(raw: dict[str, Any], us_listings: dict[str, str]) -> dict[str, Any]:
    label = str(raw["holdingsymbol"] or "").strip()
    name = str(raw["name"] or "").strip()
    identifier = str(raw.get("isin") or raw.get("cusip") or raw.get("sedol") or "").strip()
    weight = float(raw["weight"]) * 100
    if not name or not math.isfinite(weight):
        raise ValueError("ALPS 구성종목 이름 또는 비중이 잘못됐습니다.")
    asset_class = raw["holdingtypeabbrev"]
    is_cash = asset_class == "CASHEQVL"
    symbol, exchange = (
        resolve_bloomberg_listing(label, currency=None, us_listings=us_listings)
        if label and asset_class == "COMMON"
        else (None, None)
    )
    ticker = symbol or label or identifier or None
    if symbol:
        ticker = ensure_asx_prefix(symbol[:-3]) if symbol.endswith(".AX") else symbol
        if symbol.endswith((".KS", ".KQ")):
            ticker = symbol[:-3]
    return {
        "ticker": ticker,
        "name": f"현금 · {name}" if is_cash else name,
        "weight": weight,
        "raw_code": identifier or None,
        "yahoo_symbol": symbol,
        "listing_currency": infer_yahoo_symbol_currency(symbol) if symbol else None,
        "listing_exchange": exchange,
        "asset_class": "Cash" if is_cash else raw.get("holdingtype"),
        "price_lookup_supported": symbol is not None,
    }


def fetch_alps_us_holdings(ticker: str) -> dict[str, Any] | None:
    """공식 명단에 없는 상품은 None, 공식 수집 실패는 예외를 낸다."""
    ticker_norm = str(ticker).strip().upper()
    if not ticker_norm:
        raise ValueError("ALPS 구성종목 조회에 티커가 필요합니다.")
    products = _PRODUCT_MAP_CACHE.get_or_compute("products", _fetch_product_map)
    if ticker_norm not in products:
        return None
    response = shared_session.get(
        f"{_BASE_URL}/_hcms/api/getData",
        params={"api_url": f"{_API_URL}/{ticker_norm}/Full"},
        timeout=25,
    )
    response.raise_for_status()
    reference_date, rows = _parse_holdings(response.json(), ticker_norm)
    us_listings = load_us_security_exchange_map()
    items = [_normalize_holding(raw, us_listings) for raw in rows]
    return {
        "source": "alps_us_api",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "as_of_date": reference_date,
        "holdings_count": len(items),
        "holdings": items,
    }
