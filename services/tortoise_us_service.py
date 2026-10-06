"""Tortoise 미국 ETF 공식 상품 명단과 일별 전체 구성종목 수집."""

from __future__ import annotations

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

_BASE_URL = "https://tortoisecapital.com"
_PRODUCT_MAP_CACHE = TtlCache(CACHE_TTL_SLOW, name="tortoise_us_products", max_entries=1)


def _fetch_product_map() -> dict[str, str]:
    response = shared_session.get(f"{_BASE_URL}/", timeout=25)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    products = {}
    for anchor in soup.select("a[href]"):
        match = re.match(r"^([A-Z0-9]+)\s+Tortoise\b", anchor.get_text(" ", strip=True))
        url = urljoin(_BASE_URL, anchor["href"])
        parsed = urlparse(url)
        if (
            match is not None
            and parsed.scheme == "https"
            and parsed.netloc == "tortoisecapital.com"
            and re.fullmatch(r"/etf/[a-z0-9-]+/", parsed.path)
        ):
            ticker = match[1]
            if ticker in products and products[ticker] != url:
                raise ValueError(f"Tortoise 공식 상품 티커의 주소가 중복됐습니다: {ticker}")
            products[ticker] = url
    if not products:
        raise ValueError("Tortoise 공식 상품 명단에 ETF가 없습니다.")
    return products


def _parse_holdings(html: str, ticker: str) -> tuple[str, list[dict[str, str]]]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.select_one("h1")
    if title is None or not title.get_text(" ", strip=True).endswith(f"({ticker})"):
        raise ValueError(f"Tortoise {ticker} 공식 구성종목의 티커가 다릅니다.")
    metadata = soup.select_one("#holdings .section-header .meta")
    match = re.fullmatch(
        r"(\d+) total,\s*(\d{1,2}/\d{1,2}/\d{2})", metadata.get_text(" ", strip=True) if metadata else ""
    )
    if match is None:
        raise ValueError(f"Tortoise {ticker} 전체 구성종목 건수 또는 기준일이 없습니다.")
    tables = soup.select("#holdings table")
    if len(tables) != 1:
        raise ValueError(f"Tortoise {ticker} 전체 구성종목 표가 없거나 중복됐습니다.")
    table = tables[0]
    headers = [cell.get_text(" ", strip=True) for cell in table.select("thead th")]
    required = {"Security Name", "Stock Ticker", "CUSIP", "Weight"}
    if not required.issubset(headers) or len(headers) != len(set(headers)):
        raise ValueError(f"Tortoise {ticker} 구성종목 표의 필수 컬럼이 없거나 중복됐습니다.")
    rows = []
    # 화면에서 숨긴 행도 공식 HTML에 있으므로 전체 tbody를 수집한다.
    for row in table.select("tbody tr"):
        values = [cell.get_text(" ", strip=True) for cell in row.find_all("td", recursive=False)]
        if len(values) != len(headers):
            raise ValueError(f"Tortoise {ticker} 구성종목 표의 행 구조가 잘못됐습니다.")
        rows.append(dict(zip(headers, values, strict=True)))
    if not rows or len(rows) != int(match[1]):
        raise ValueError(f"Tortoise {ticker} 전체 건수와 수집 건수가 다릅니다.")
    return datetime.strptime(match[2], "%m/%d/%y").date().isoformat(), rows


def _normalize_holding(raw: dict[str, str], us_listings: dict[str, str]) -> dict[str, Any]:
    label = raw["Stock Ticker"].strip()
    name = raw["Security Name"].strip()
    identifier = raw["CUSIP"].strip()
    weight_text = raw["Weight"].strip()
    if not name or not weight_text.endswith("%"):
        raise ValueError("Tortoise 구성종목 이름 또는 비중이 잘못됐습니다.")
    weight = float(weight_text[:-1].replace(",", ""))
    if not math.isfinite(weight):
        raise ValueError("Tortoise 구성종목 비중이 잘못됐습니다.")
    is_cash = identifier == "Cash&Other"
    symbol, exchange = (
        (None, None)
        if is_cash or not label
        else resolve_bloomberg_listing(label, currency=None, us_listings=us_listings)
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
        "asset_class": "Cash" if is_cash else None,
        "price_lookup_supported": symbol is not None,
    }


def fetch_tortoise_us_holdings(ticker: str) -> dict[str, Any] | None:
    """공식 명단에 없는 상품은 None, 공식 수집 실패는 예외를 낸다."""
    ticker_norm = str(ticker).strip().upper()
    if not ticker_norm:
        raise ValueError("Tortoise 구성종목 조회에 티커가 필요합니다.")
    url = _PRODUCT_MAP_CACHE.get_or_compute("products", _fetch_product_map).get(ticker_norm)
    if url is None:
        return None
    response = shared_session.get(url, timeout=25)
    response.raise_for_status()
    reference_date, rows = _parse_holdings(response.text, ticker_norm)
    us_listings = load_us_security_exchange_map()
    items = [_normalize_holding(raw, us_listings) for raw in rows]
    return {
        "source": "tortoise_us_html",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "as_of_date": reference_date,
        "holdings_count": len(items),
        "holdings": items,
    }
