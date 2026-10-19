"""Global X 호주 ETF 공식 전체 보유 XLSX 수집."""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from config import CACHE_TTL_SLOW
from services.globalx_page_service import iter_globalx_page_props, read_globalx_page_records
from services.xlsx_holdings_service import parse_xlsx_table_rows, read_xlsx_rows
from utils.asx_ticker import ensure_asx_prefix, strip_asx_prefix
from utils.http_session import shared_session
from utils.normalization import resolve_bloomberg_listing
from utils.ttl_cache import TtlCache
from utils.us_etf_market_service import load_us_security_exchange_map

_BASE_URL = "https://www.globalxetfs.com.au"
_PRODUCT_MAP_CACHE = TtlCache(CACHE_TTL_SLOW, name="globalx_au_products", max_entries=1)


def _fetch_product_map() -> dict[str, str]:
    response = shared_session.get(f"{_BASE_URL}/funds/", timeout=25)
    response.raise_for_status()
    records = read_globalx_page_records(BeautifulSoup(response.text, "html.parser"))
    products = {}
    for value in records.values():
        for row in iter_globalx_page_props(value):
            if not {"ticker", "fund_name", "url"} <= row.keys():
                continue
            ticker = row["ticker"]
            parsed = urlparse(urljoin(_BASE_URL, row["url"]))
            if (
                re.fullmatch(r"[A-Z0-9]+", ticker) is None
                or parsed.scheme != "https"
                or parsed.netloc != "www.globalxetfs.com.au"
                or parsed.path.rstrip("/") != f"/funds/{ticker.lower()}"
                or ticker in products
            ):
                raise ValueError("Global X 호주 공식 상품 명단의 티커 또는 주소가 잘못됐습니다.")
            products[ticker] = urljoin(_BASE_URL, row["url"])
    if not products:
        raise ValueError("Global X 호주 공식 상품 명단이 비었습니다.")
    return products


def _find_holdings_file(html: str, ticker: str) -> tuple[str, str, str]:
    records = read_globalx_page_records(BeautifulSoup(html, "html.parser"))
    settings = [
        props
        for value in records.values()
        for props in iter_globalx_page_props(value)
        if "pcfData" in props and "ticker" in props and "date" in props
    ]
    if len(settings) != 1 or settings[0]["ticker"] != ticker:
        raise ValueError(f"Global X 호주 {ticker} 공식 전체 보유 설정이 없거나 중복됐습니다.")
    setting = settings[0]
    product = setting["data"]["product"]
    if product["ticker"] != ticker:
        raise ValueError(f"Global X 호주 {ticker} 공식 상품 식별자가 다릅니다.")
    url = product["pcf"]
    if not isinstance(url, str):
        raise ValueError(f"Global X 호주 {ticker} 공식 전체 보유 XLSX 주소가 없습니다.")
    parsed = urlparse(url)
    match = re.fullmatch(rf"/GXAU_{re.escape(ticker)}_FULL_PCF_(\d{{8}})\.xlsx", parsed.path)
    if parsed.scheme != "https" or parsed.netloc != "files.globalxetfs.com.au" or match is None:
        raise ValueError(f"Global X 호주 {ticker} 공식 전체 보유 파일의 주소 또는 상품명이 다릅니다.")
    reference_date = datetime.strptime(match[1], "%Y%m%d").date().isoformat()
    if setting["date"] != reference_date:
        raise ValueError(f"Global X 호주 {ticker} 공식 페이지와 파일의 기준일이 다릅니다.")
    return url, reference_date, f"Global X {product['name']}"


def _normalize_holding(raw: dict[str, str], us_listings: dict[str, str]) -> dict[str, Any]:
    label = raw["Bloomberg Ticker"].strip()
    name = raw["Component Name"].strip()
    currency = raw["Local CCY"].strip().upper()
    weight = float(raw["Weight"]) * 100
    if not name or not math.isfinite(weight):
        raise ValueError("Global X 호주 구성종목의 이름 또는 비중이 잘못됐습니다.")
    asset_class = "Equity" if label.endswith(" Equity") else None
    symbol, exchange = (
        resolve_bloomberg_listing(label, currency=currency, us_listings=us_listings)
        if asset_class == "Equity"
        else (None, None)
    )
    ticker = symbol or label or raw["ISIN"] or raw["SEDOL"] or None
    if symbol and symbol.endswith(".AX"):
        ticker = ensure_asx_prefix(symbol[:-3])
    elif symbol and symbol.endswith((".KS", ".KQ")):
        ticker = symbol[:-3]
    return {
        "ticker": ticker,
        "name": name,
        "weight": weight,
        "raw_code": raw["ISIN"] or raw["SEDOL"] or label or None,
        "yahoo_symbol": symbol,
        "listing_currency": currency,
        "listing_exchange": exchange,
        "asset_class": asset_class,
        "price_lookup_supported": symbol is not None,
    }


def _parse_holdings(content: bytes, ticker: str, reference_date: str, fund_name: str) -> list[dict[str, Any]]:
    rows = read_xlsx_rows(content)
    metadata = {row["A"]: row.get("B", "") for row in rows if "A" in row}
    file_date = datetime.strptime(metadata["Trade Date"], "%Y%m%d").date().isoformat()
    if file_date != reference_date or metadata["Fund Name"] != fund_name:
        raise ValueError(f"Global X 호주 {ticker} 공식 XLSX의 상품명 또는 기준일이 다릅니다.")
    raw = parse_xlsx_table_rows(rows, {"Component Name", "ISIN", "SEDOL", "Bloomberg Ticker", "Local CCY", "Weight"})
    if len(raw) != int(metadata["Number of Securities"]):
        raise ValueError(f"Global X 호주 {ticker} 공식 전체 구성종목 수가 다릅니다.")
    us_listings = load_us_security_exchange_map()
    items = [_normalize_holding(row, us_listings) for row in raw]
    cash, nav = float(metadata["Cash Component (Base CCY)"]), float(metadata["Net Asset Value (Base CCY)"])
    if not math.isfinite(cash) or not math.isfinite(nav) or nav <= 0:
        raise ValueError(f"Global X 호주 {ticker} 공식 현금 또는 순자산이 잘못됐습니다.")
    currency = metadata["Base CCY"]
    items.append(
        {
            "ticker": currency,
            "name": f"현금 · {currency}",
            "weight": cash / nav * 100,
            "raw_code": None,
            "yahoo_symbol": None,
            "listing_currency": currency,
            "listing_exchange": None,
            "asset_class": "Cash",
            "price_lookup_supported": False,
        }
    )
    return items


def fetch_globalx_au_holdings(ticker: str) -> dict[str, Any] | None:
    """공식 명단에 없는 상품은 None, 공식 수집 실패는 예외를 낸다."""
    ticker_norm = strip_asx_prefix(ticker)
    if not ticker_norm:
        raise ValueError("Global X 호주 구성종목 조회에 티커가 필요합니다.")
    url = _PRODUCT_MAP_CACHE.get_or_compute("products", _fetch_product_map).get(ticker_norm)
    if url is None:
        return None
    response = shared_session.get(url, timeout=25)
    response.raise_for_status()
    holdings_url, reference_date, fund_name = _find_holdings_file(response.text, ticker_norm)
    response = shared_session.get(holdings_url, timeout=25)
    response.raise_for_status()
    items = _parse_holdings(response.content, ticker_norm, reference_date, fund_name)
    return {
        "source": "globalx_au_xlsx",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "as_of_date": reference_date,
        "holdings_count": len(items),
        "holdings": items,
    }
