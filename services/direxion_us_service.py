"""Direxion 미국 ETF 공식 상품 명단과 일별 전체 구성종목 수집."""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from config import CACHE_TTL_SLOW
from services.etf_holdings_service import DEFAULT_USER_AGENT
from utils.http_session import shared_session
from utils.normalization import resolve_bloomberg_listing
from utils.ttl_cache import TtlCache
from utils.us_etf_market_service import load_us_security_exchange_map

_BASE_URL = "https://www.direxion.com"
_HEADERS = {"User-Agent": DEFAULT_USER_AGENT}
_PRODUCT_MAP_CACHE = TtlCache(CACHE_TTL_SLOW, name="direxion_us_products", max_entries=1)
_REQUEST_CONFIG_CACHE = TtlCache(CACHE_TTL_SLOW, name="direxion_us_request", max_entries=1)
_QUERY = """
query GetDailyHoldings($Ticker: String!) {
  getDailyHoldings(Ticker: $Ticker) {
    FundName Ticker TradeDate
    Holdings { AccountTicker SecurityDescription StockTicker Cusip HoldingPercent }
  }
}
"""


def _fetch_product_map() -> dict[str, str]:
    response = shared_session.get(f"{_BASE_URL}/all-etfs", headers=_HEADERS, timeout=25)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    data = soup.select_one("script#__NEXT_DATA__")
    if data is None:
        raise ValueError("Direxion 공식 상품 명단의 데이터가 없습니다.")
    funds = json.loads(data.get_text())["props"]["pageProps"]["ssrData"]["funds"]
    products = {}
    for fund in funds:
        if fund["Level1"] != "ETF":
            continue
        ticker = fund["Ticker"]
        url = urljoin(_BASE_URL, fund["Url"])
        parsed = urlparse(url)
        if (
            re.fullmatch(r"[A-Z0-9]+", ticker) is None
            or parsed.scheme != "https"
            or parsed.netloc != "www.direxion.com"
            or not parsed.path.startswith("/product/")
            or ticker in products
        ):
            raise ValueError("Direxion 공식 상품 티커 또는 상세 주소가 잘못됐습니다.")
        products[ticker] = url
    displayed = {anchor.get_text(strip=True) for anchor in soup.select("td.overview__ticker a")}
    if not products or set(products) != displayed:
        raise ValueError("Direxion 공식 ETF 명단이 비었거나 일부만 반환됐습니다.")
    return products


def _fetch_request_config(product_url: str) -> tuple[str, str]:
    """공식 화면의 공개 요청 설정을 읽어 주소·키를 코드에 고정하지 않는다."""
    response = shared_session.get(product_url, headers=_HEADERS, timeout=25)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    scripts = [
        script["src"]
        for script in soup.select("script[src]")
        if re.fullmatch(r"/_next/static/chunks/\d+-[a-f0-9]+\.js", script["src"])
    ]
    pattern = (
        r'url:"(https://[a-z0-9]+\.appsync-api\.us-east-1\.amazonaws\.com/graphql)",'
        r'region:"us-east-1",auth:\{type:[^,]+,apiKey:"([^\"]+)"'
    )
    for path in reversed(scripts):
        response = shared_session.get(urljoin(_BASE_URL, path), headers=_HEADERS, timeout=25)
        response.raise_for_status()
        match = re.search(pattern, response.text)
        if match and "getDailyHoldings" in response.text:
            return match[1], match[2]
    raise ValueError("Direxion 공식 전체 구성종목의 공개 요청 설정이 없습니다.")


def _parse_holdings(payload: Any, ticker: str) -> tuple[str, list[dict[str, Any]]]:
    if not isinstance(payload, dict) or payload.get("errors"):
        raise ValueError(f"Direxion {ticker} 공식 구성종목 API가 오류를 반환했습니다.")
    fund = payload["data"]["getDailyHoldings"]
    if not isinstance(fund, dict) or fund["Ticker"] != ticker or not fund["FundName"]:
        raise ValueError(f"Direxion {ticker} 공식 구성종목의 상품 식별자가 다릅니다.")
    reference_date = datetime.strptime(fund["TradeDate"], "%m%d%Y").date().isoformat()
    rows = fund["Holdings"]
    required = {"AccountTicker", "SecurityDescription", "StockTicker", "Cusip", "HoldingPercent"}
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Direxion {ticker} 전체 구성종목이 비었거나 잘못됐습니다.")
    if any(not isinstance(row, dict) or not required.issubset(row) or row["AccountTicker"] != ticker for row in rows):
        raise ValueError(f"Direxion {ticker} 구성종목 필수 필드 또는 상품 식별자가 잘못됐습니다.")
    return reference_date, rows


def _normalize_holding(raw: dict[str, Any], us_listings: dict[str, str]) -> dict[str, Any]:
    name = str(raw["SecurityDescription"] or "").strip()
    label = str(raw["StockTicker"] or "").strip()
    identifier = str(raw["Cusip"] or "").strip()
    weight = raw["HoldingPercent"]
    if not name or (weight is not None and (isinstance(weight, bool) or not math.isfinite(float(weight)))):
        raise ValueError("Direxion 구성종목 이름 또는 비중이 잘못됐습니다.")
    is_cash = identifier.startswith("X9USD")
    is_swap = "SWAP ASSET LEG" in name
    # 스왑 노출과 현금을 보존하고, 미국 공통 명단에서 확인한 주식에만 시세를 연결한다.
    symbol, exchange = (
        resolve_bloomberg_listing(label, currency="USD", us_listings=us_listings)
        if label and not is_cash and not is_swap
        else (None, None)
    )
    return {
        "ticker": symbol or label or identifier or None,
        "name": f"현금 · {name}" if is_cash else name,
        "weight": float(weight) if weight is not None else None,
        "raw_code": identifier or None,
        "yahoo_symbol": symbol,
        "listing_currency": "USD" if symbol else None,
        "listing_exchange": exchange,
        "asset_class": "Cash" if is_cash else "Swap" if is_swap else None,
        "price_lookup_supported": symbol is not None,
    }


def fetch_direxion_us_holdings(ticker: str) -> dict[str, Any] | None:
    """공식 명단에 없는 상품은 None, 공식 수집 실패는 예외를 낸다."""
    ticker_norm = str(ticker).strip().upper()
    if not ticker_norm:
        raise ValueError("Direxion 구성종목 조회에 티커가 필요합니다.")
    product_url = _PRODUCT_MAP_CACHE.get_or_compute("products", _fetch_product_map).get(ticker_norm)
    if product_url is None:
        return None
    api_url, public_key = _REQUEST_CONFIG_CACHE.get_or_compute("api", lambda: _fetch_request_config(product_url))
    response = shared_session.post(
        api_url,
        headers={**_HEADERS, "x-api-key": public_key},
        json={"query": _QUERY, "variables": {"Ticker": ticker_norm}},
        timeout=25,
    )
    response.raise_for_status()
    reference_date, rows = _parse_holdings(response.json(), ticker_norm)
    us_listings = load_us_security_exchange_map()
    items = [_normalize_holding(raw, us_listings) for raw in rows]
    return {
        "source": "direxion_us_api",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "as_of_date": reference_date,
        "holdings_count": len(items),
        "holdings": items,
    }
