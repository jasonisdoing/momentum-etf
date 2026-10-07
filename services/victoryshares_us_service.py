"""VictoryShares 미국 ETF 공식 상품 명단과 전체 구성종목 API 수집."""

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

_BASE_URL = "https://www.vcm.com"
_PRODUCT_PATH = "/products/victoryshares-etfs/victoryshares-etfs-list/"
_PRODUCT_MAP_CACHE = TtlCache(CACHE_TTL_SLOW, name="victoryshares_us_products", max_entries=1)
_LISTED_SECURITY_TYPES = {"COMMON STOCK", "COMMON STOCK LIKE", "FOREIGN STOCK", "EXCHANGE TRADED FUND"}


def _fetch_product_map() -> dict[str, str]:
    response = shared_session.get(f"{_BASE_URL}/products-fa/victoryshares-etfs", timeout=25)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    products = {}
    for row in soup.select("tr"):
        cells = row.find_all("td", recursive=False)
        if len(cells) < 2:
            continue
        anchor = cells[1].select_one("a[href]")
        if anchor is None:
            continue
        url = urljoin(_BASE_URL, anchor["href"])
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.netloc != "www.vcm.com" or not parsed.path.startswith(_PRODUCT_PATH):
            continue
        ticker = cells[0].get_text(strip=True)
        if re.fullmatch(r"[A-Z0-9]+", ticker) is None:
            raise ValueError("VictoryShares 공식 상품 명단의 티커가 잘못됐습니다.")
        if ticker in products and products[ticker] != url:
            raise ValueError(f"VictoryShares 공식 상품 티커의 주소가 중복됐습니다: {ticker}")
        products[ticker] = url
    if not products:
        raise ValueError("VictoryShares 공식 상품 명단에 ETF가 없습니다.")
    return products


def _read_request_config(html: str, ticker: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    fund = soup.select_one("#fundID")
    config = soup.select_one("#productDetailConfigJson")
    public_key = soup.select_one("#productDetailKey")
    if fund is None or fund.get("value") != ticker:
        raise ValueError(f"VictoryShares {ticker} 공식 상품 페이지의 티커가 다릅니다.")
    if config is None or public_key is None or not public_key.get("value"):
        raise ValueError(f"VictoryShares {ticker} 공식 구성종목의 공개 요청 설정이 없습니다.")
    url = json.loads(config["value"])["allholdings"]
    if url != f"https://investorapi.vcm.com/search/product/{ticker}/AllHoldings":
        raise ValueError(f"VictoryShares {ticker} 공식 전체 구성종목 API 주소가 다릅니다.")
    return url, public_key["value"]


def _parse_holdings(payload: Any, ticker: str) -> tuple[str, list[dict[str, Any]]]:
    required = {"holding_name", "as_of_date", "portfolio_percentage"}
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"VictoryShares {ticker} 전체 구성종목 목록이 비었거나 잘못됐습니다.")
    for raw in payload:
        if not isinstance(raw, dict) or not required.issubset(raw):
            raise ValueError(f"VictoryShares {ticker} 구성종목 필수 필드가 없습니다.")
        if not isinstance(raw["as_of_date"], str) or raw["as_of_date"] != payload[0]["as_of_date"]:
            raise ValueError(f"VictoryShares {ticker} 구성종목의 기준일이 없거나 서로 다릅니다.")
    return datetime.strptime(payload[0]["as_of_date"], "%m/%d/%Y").date().isoformat(), payload


def _normalize_holding(raw: dict[str, Any], us_listings: dict[str, str]) -> dict[str, Any]:
    name = str(raw["holding_name"] or "").strip()
    label = str(raw.get("stock_symbol") or "").strip()
    identifier = str(raw.get("isin") or "").strip()
    asset_class = raw.get("security_type")
    weight = float(raw["portfolio_percentage"])
    if not name or not math.isfinite(weight):
        raise ValueError("VictoryShares 구성종목 이름 또는 비중이 잘못됐습니다.")
    # 공식 응답의 US 표기와 일부 U 축약은 미국 공통 상장 명단으로 확인한 경우에만 변환한다.
    base, separator, market = label.rpartition(" ")
    listing_label = base if separator and market in {"US", "U"} and base in us_listings else label
    symbol, exchange = (
        resolve_bloomberg_listing(listing_label, currency=None, us_listings=us_listings)
        if label and asset_class in _LISTED_SECURITY_TYPES
        else (None, None)
    )
    ticker = symbol or label or identifier or None
    if symbol:
        ticker = ensure_asx_prefix(symbol[:-3]) if symbol.endswith(".AX") else symbol
        if symbol.endswith((".KS", ".KQ")):
            ticker = symbol[:-3]
    is_cash = name == "CASH AND CASH EQUIVALENTS"
    return {
        "ticker": ticker,
        "name": f"현금 · {name}" if is_cash else name,
        "weight": weight,
        "raw_code": identifier or None,
        "yahoo_symbol": symbol,
        "listing_currency": infer_yahoo_symbol_currency(symbol) if symbol else None,
        "listing_exchange": exchange,
        "asset_class": "Cash" if is_cash else asset_class,
        "price_lookup_supported": symbol is not None,
    }


def fetch_victoryshares_us_holdings(ticker: str) -> dict[str, Any] | None:
    """공식 명단에 없는 상품은 None, 공식 수집 실패는 예외를 낸다."""
    ticker_norm = str(ticker).strip().upper()
    if not ticker_norm:
        raise ValueError("VictoryShares 구성종목 조회에 티커가 필요합니다.")
    url = _PRODUCT_MAP_CACHE.get_or_compute("products", _fetch_product_map).get(ticker_norm)
    if url is None:
        return None
    response = shared_session.get(url, timeout=25)
    response.raise_for_status()
    api_url, public_key = _read_request_config(response.text, ticker_norm)
    response = shared_session.get(api_url, headers={"x-api-key": public_key}, timeout=25)
    response.raise_for_status()
    reference_date, rows = _parse_holdings(response.json(), ticker_norm)
    us_listings = load_us_security_exchange_map()
    items = [_normalize_holding(raw, us_listings) for raw in rows]
    return {
        "source": "victoryshares_us_api",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "as_of_date": reference_date,
        "holdings_count": len(items),
        "holdings": items,
    }
