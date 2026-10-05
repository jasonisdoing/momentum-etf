"""VanEck 미국 ETF 공식 상품 명단과 전체 구성종목 수집."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Any
from urllib.parse import parse_qs, urljoin, urlparse

from bs4 import BeautifulSoup

from config import CACHE_TTL_SLOW
from utils.asx_ticker import ensure_asx_prefix
from utils.http_session import shared_session
from utils.normalization import resolve_bloomberg_listing
from utils.ttl_cache import TtlCache
from utils.us_etf_market_service import load_us_security_exchange_map

_BASE_URL = "https://www.vaneck.com"
_PRODUCT_MAP_CACHE = TtlCache(CACHE_TTL_SLOW, name="vaneck_us_products", max_entries=1)


def _official_url(value: str, prefix: str) -> str:
    url = urljoin(_BASE_URL, value)
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.netloc != "www.vaneck.com" or not parsed.path.startswith(prefix):
        raise ValueError(f"VanEck 공식 주소가 잘못됐습니다: {value}")
    return url


def _fetch_product_map() -> dict[str, str]:
    response = shared_session.get(f"{_BASE_URL}/us/en/etf-fund-finder/", timeout=25)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    search_data = soup.select_one("script.fund-search-data")
    page_id = soup.select_one("#page-id")
    if search_data is None or page_id is None:
        raise ValueError("VanEck 공식 상품 명단의 주소 또는 페이지 식별자가 없습니다.")
    paths = {str(row["Values"][0]).upper(): row["Url"] for row in json.loads(search_data.get_text())}
    response = shared_session.post(
        f"{_BASE_URL}/Main/FundListingUs/GetFundData",
        data={
            "filterJson": json.dumps(
                {
                    "InvType": ["etf"],
                    "Strategies": [],
                    "Funds": [],
                    "ShareClass": [],
                    "TableType": "search",
                    "CurrentPageId": page_id["value"],
                }
            )
        },
        timeout=25,
    )
    response.raise_for_status()
    payload = response.json()
    if payload["Success"] is not True or payload["HasError"]:
        raise ValueError("VanEck 공식 상품 명단 수집이 실패했습니다.")
    result = {}
    for product in payload["Result"]["FundSet"]:
        if product["TickerGroup"] != "ETF":
            continue
        ticker = str(product["FundID"]).strip().upper()
        if not ticker or ticker not in paths or ticker in result:
            raise ValueError(f"VanEck 상품 티커 또는 상세 주소가 잘못됐습니다: {ticker}")
        result[ticker] = _official_url(paths[ticker], "/us/en/investments/")
    if not result:
        raise ValueError("VanEck 공식 상품 명단에 ETF가 없습니다.")
    return result


def _dataset_url(html: str, ticker: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    identity = soup.select_one("ve-fundticker")
    if identity is None or identity.get_text(strip=True) != ticker:
        raise ValueError(f"VanEck {ticker} 상품 페이지의 티커가 다릅니다.")
    urls = []
    for script in soup.select('script[type="application/ld+json"]'):
        for node in json.loads(script.get_text()).get("@graph", []):
            if node.get("@type") == "Dataset":
                urls.append(_official_url(node["distribution"]["contentUrl"], "/Main/FundDatasetBlock/Get/"))
    if len(urls) != 1 or parse_qs(urlparse(urls[0]).query).get("ticker") != [ticker]:
        raise ValueError(f"VanEck {ticker} 구성종목 데이터 주소가 잘못됐습니다.")
    return urls[0]


def _normalize_holding(raw: dict[str, Any], us_listings: dict[str, str]) -> dict[str, Any]:
    label = str(raw["Label"]).strip()
    name = str(raw["HoldingName"]).strip()
    asset_class = raw["AssetClass"]
    weight = float(raw["Weight"].replace(",", ""))
    currency = raw["CurrencyCode"]
    if not math.isfinite(weight):
        raise ValueError("VanEck 구성종목 비중이 잘못됐습니다.")
    symbol, exchange = (
        resolve_bloomberg_listing(label, currency=currency, us_listings=us_listings)
        if asset_class == "Stock"
        else (None, None)
    )
    ticker = label if label not in {"", "--"} else None
    if asset_class in {"Cash", "Cash Bal"}:
        name = f"현금 · {name or label}"
    if not name:
        raise ValueError("VanEck 구성종목 이름이 비었습니다.")
    if symbol:
        ticker = ensure_asx_prefix(symbol[:-3]) if symbol.endswith(".AX") else symbol
        if symbol.endswith((".KS", ".KQ")):
            ticker = symbol[:-3]
    return {
        "ticker": ticker,
        "name": name,
        "weight": weight,
        "raw_code": raw["ISIN"] or raw["CUSIP"] or raw["FIGI"] or None,
        "yahoo_symbol": symbol,
        "listing_currency": currency,
        "listing_exchange": exchange,
        "asset_class": asset_class,
        "price_lookup_supported": symbol is not None,
    }


def fetch_vaneck_us_holdings(ticker: str) -> dict[str, Any] | None:
    """VanEck 종목이 아니면 None, 공식 수집 실패면 예외를 낸다."""
    ticker_norm = str(ticker).strip().upper()
    if not ticker_norm:
        raise ValueError("VanEck 구성종목 조회에 티커가 필요합니다.")
    url = _PRODUCT_MAP_CACHE.get_or_compute("products", _fetch_product_map).get(ticker_norm)
    if url is None:
        return None
    response = shared_session.get(url, timeout=25)
    response.raise_for_status()
    response = shared_session.get(_dataset_url(response.text, ticker_norm), timeout=25)
    response.raise_for_status()
    groups = response.json()["HoldingsList"]
    if len(groups) != 1 or not groups[0]["Holdings"]:
        raise ValueError(f"VanEck {ticker_norm} 구성종목이 비었거나 기준일이 여러 개입니다.")
    reference_date = datetime.fromisoformat(groups[0]["AsOfDate"]).date().isoformat()
    rows = groups[0]["Holdings"]
    if any(raw["Ticker"] != ticker_norm or raw["AsOfDate"] != groups[0]["AsOfDate"] for raw in rows):
        raise ValueError(f"VanEck {ticker_norm} 구성종목의 티커 또는 기준일이 다릅니다.")
    us_listings = load_us_security_exchange_map()
    items = [_normalize_holding(raw, us_listings) for raw in rows]
    return {
        "source": "vaneck_us_api",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "as_of_date": reference_date,
        "holdings_count": len(items),
        "holdings": items,
    }
