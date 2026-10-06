"""Global X 미국 ETF 공식 상품 명단과 전체 구성종목 CSV 수집."""

from __future__ import annotations

import csv
import io
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

_BASE_URL = "https://www.globalxetfs.com"
_PRODUCT_MAP_CACHE = TtlCache(CACHE_TTL_SLOW, name="globalx_us_products", max_entries=1)
_FOOTER = (
    "The information contained herein may not be reproduced, redistributed or used to create any derivative works."
)


def _fetch_product_map() -> dict[str, str]:
    response = shared_session.get(f"{_BASE_URL}/explore", timeout=25)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    products = {}
    for anchor in soup.select("a[href]"):
        parsed = urlparse(urljoin(_BASE_URL, anchor["href"]))
        match = re.fullmatch(r"/funds/([A-Za-z0-9]+)", parsed.path)
        if parsed.scheme == "https" and parsed.netloc == "www.globalxetfs.com" and match:
            ticker = match[1].upper()
            products[ticker] = f"{_BASE_URL}/funds/{ticker.lower()}"
    if not products:
        raise ValueError("Global X 공식 상품 명단에 ETF가 없습니다.")
    return products


def _find_holdings_url(html: str, ticker: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.select_one("h1")
    if title is None or title.get_text(strip=True) != ticker:
        raise ValueError(f"Global X {ticker} 공식 상품 페이지의 티커가 다릅니다.")
    links = {
        urljoin(_BASE_URL, anchor["href"])
        for anchor in soup.select("a[href]")
        if anchor.get_text(" ", strip=True) == "Full Holdings (.csv)"
    }
    if len(links) != 1:
        raise ValueError(f"Global X {ticker} 전체 구성종목 CSV 주소가 없거나 중복됐습니다.")
    url = links.pop()
    parsed = urlparse(url)
    match = re.fullmatch(rf"/funds/holdings/{re.escape(ticker.lower())}_full-holdings_(\d{{8}})\.csv", parsed.path)
    if parsed.scheme != "https" or parsed.netloc != "assets.globalxetfs.com" or match is None:
        raise ValueError(f"Global X {ticker} 공식 CSV의 호스트 또는 상품명이 다릅니다.")
    return url, datetime.strptime(match[1], "%Y%m%d").date().isoformat()


def _parse_holdings(text: str, ticker: str, reference_date: str) -> list[dict[str, str]]:
    reader = csv.reader(io.StringIO(text))
    title = next(reader, [])
    date_row = next(reader, [])
    match = re.fullmatch(r"Fund Holdings Data as of (\d{2}/\d{2}/\d{4})", date_row[0]) if len(date_row) == 1 else None
    if len(title) != 1 or not title[0].startswith("Global X ") or match is None:
        raise ValueError(f"Global X {ticker} 구성종목 CSV의 상품명 또는 기준일이 없습니다.")
    if datetime.strptime(match[1], "%m/%d/%Y").date().isoformat() != reference_date:
        raise ValueError(f"Global X {ticker} 공식 CSV 주소와 내용의 기준일이 다릅니다.")
    headers = next(reader, [])
    required = {"% of Net Assets", "Ticker", "Name", "SEDOL", "Market Price ($)", "Shares Held", "Market Value ($)"}
    if len(headers) != len(required) or set(headers) != required:
        raise ValueError(f"Global X {ticker} 구성종목 CSV 필수 열이 잘못됐습니다.")
    rows = []
    footer_seen = False
    for values in reader:
        if not values:
            continue
        if values == [_FOOTER] and not footer_seen:
            footer_seen = True
            continue
        if footer_seen or len(values) != len(headers):
            raise ValueError(f"Global X {ticker} 구성종목 CSV 행 구조가 잘못됐습니다.")
        rows.append(dict(zip(headers, values, strict=True)))
    if not rows or not footer_seen:
        raise ValueError(f"Global X {ticker} 전체 구성종목 CSV가 비었거나 완전하지 않습니다.")
    return rows


def _normalize_holding(raw: dict[str, str], us_listings: dict[str, str]) -> dict[str, Any]:
    label = raw["Ticker"].strip()
    name = raw["Name"].strip()
    identifier = raw["SEDOL"].strip()
    weight = float(raw["% of Net Assets"].strip())
    if not name or not math.isfinite(weight):
        raise ValueError("Global X 구성종목 이름 또는 비중이 잘못됐습니다.")
    symbol, exchange = (
        resolve_bloomberg_listing(label, currency=None, us_listings=us_listings) if label else (None, None)
    )
    ticker = symbol or label or identifier or None
    if symbol:
        ticker = ensure_asx_prefix(symbol[:-3]) if symbol.endswith(".AX") else symbol
        if symbol.endswith((".KS", ".KQ")):
            ticker = symbol[:-3]
    return {
        "ticker": ticker,
        "name": f"현금 · {name}" if name == "CASH" else name,
        "weight": weight,
        "raw_code": identifier or None,
        "yahoo_symbol": symbol,
        "listing_currency": infer_yahoo_symbol_currency(symbol) if symbol else None,
        "listing_exchange": exchange,
        "asset_class": "Cash" if name == "CASH" else None,
        "price_lookup_supported": symbol is not None,
    }


def fetch_globalx_us_holdings(ticker: str) -> dict[str, Any] | None:
    """공식 명단에 없는 상품은 None, 공식 수집 실패는 예외를 낸다."""
    ticker_norm = str(ticker).strip().upper()
    if not ticker_norm:
        raise ValueError("Global X 구성종목 조회에 티커가 필요합니다.")
    url = _PRODUCT_MAP_CACHE.get_or_compute("products", _fetch_product_map).get(ticker_norm)
    if url is None:
        return None
    response = shared_session.get(url, timeout=25)
    response.raise_for_status()
    csv_url, reference_date = _find_holdings_url(response.text, ticker_norm)
    response = shared_session.get(csv_url, timeout=25)
    response.raise_for_status()
    rows = _parse_holdings(response.content.decode("utf-8-sig"), ticker_norm, reference_date)
    us_listings = load_us_security_exchange_map()
    items = [_normalize_holding(raw, us_listings) for raw in rows]
    return {
        "source": "globalx_us_csv",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "as_of_date": reference_date,
        "holdings_count": len(items),
        "holdings": items,
    }
