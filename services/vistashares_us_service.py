"""VistaShares 미국 ETF 공식 상품 명단과 전체 구성종목 CSV 수집."""

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

_BASE_URL = "https://www.vistashares.com"
_PRODUCT_MAP_CACHE = TtlCache(CACHE_TTL_SLOW, name="vistashares_us_products", max_entries=1)


def _fetch_product_map() -> dict[str, str]:
    response = shared_session.get(f"{_BASE_URL}/", timeout=25)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    result = {}
    for anchor in soup.select("a[href]"):
        url = urljoin(_BASE_URL, anchor["href"])
        parsed = urlparse(url)
        match = re.fullmatch(r"/etf/([a-z0-9]+)/?", parsed.path)
        if parsed.scheme == "https" and parsed.netloc == "www.vistashares.com" and match:
            result[match[1].upper()] = f"{_BASE_URL}/etf/{match[1]}/"
    if not result:
        raise ValueError("VistaShares 공식 상품 명단에 ETF가 없습니다.")
    return result


def _holdings_url(html: str, ticker: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    urls = []
    for form in soup.select("#holdings form[action]"):
        identity = form.select_one('input[name="etf"]')
        url = urljoin(_BASE_URL, form["action"])
        parsed = urlparse(url)
        if (
            identity is not None
            and identity.get("value") == ticker
            and str(form.get("method", "")).upper() == "GET"
            and parsed.scheme == "https"
            and parsed.netloc == "www.vistashares.com"
            and parsed.path.rstrip("/") == "/csv/top-holdings"
        ):
            urls.append(url)
    if len(urls) != 1:
        raise ValueError(f"VistaShares {ticker} 공식 전체 구성종목 다운로드 주소가 잘못됐습니다.")
    return urls[0]


def _normalize_holding(raw: dict[str, str], us_listings: dict[str, str]) -> dict[str, Any]:
    label = raw["StockTicker"].strip()
    name = raw["SecurityName"].strip()
    identifier = raw["CUSIP"].strip()
    weight_text = raw["Weightings"].strip()
    if not name or not weight_text.endswith("%") or raw["MoneyMarketFlag"] not in {"", "Y", "N"}:
        raise ValueError("VistaShares 구성종목 이름·비중·현금성 자산 표기가 잘못됐습니다.")
    weight = float(weight_text[:-1].replace(",", ""))
    if not math.isfinite(weight):
        raise ValueError("VistaShares 구성종목 비중이 잘못됐습니다.")
    is_cash = identifier.startswith("CASH") or identifier == "Cash&Other"
    is_money_market = raw["MoneyMarketFlag"] == "Y"
    symbol, exchange = (
        (None, None)
        if is_cash or is_money_market
        else resolve_bloomberg_listing(label, currency=None, us_listings=us_listings)
    )
    ticker = label or None
    currency = infer_yahoo_symbol_currency(symbol) if symbol else None
    asset_class = "Security"
    if is_cash:
        name = f"현금 · {name}"
        currency = identifier[4:] if identifier.startswith("CASH") else None
        asset_class = "Cash"
    elif is_money_market:
        name = f"현금성 · {name}"
        asset_class = "Money Market"
    if symbol:
        ticker = ensure_asx_prefix(symbol[:-3]) if symbol.endswith(".AX") else symbol
        if symbol.endswith((".KS", ".KQ")):
            ticker = symbol[:-3]
    return {
        "ticker": ticker,
        "name": name,
        "weight": weight,
        "raw_code": identifier or None,
        "yahoo_symbol": symbol,
        "listing_currency": currency,
        "listing_exchange": exchange,
        "asset_class": asset_class,
        "price_lookup_supported": symbol is not None,
    }


def fetch_vistashares_us_holdings(ticker: str) -> dict[str, Any] | None:
    """VistaShares 종목이 아니면 None, 공식 수집 실패면 예외를 낸다."""
    ticker_norm = str(ticker).strip().upper()
    if not ticker_norm:
        raise ValueError("VistaShares 구성종목 조회에 티커가 필요합니다.")
    url = _PRODUCT_MAP_CACHE.get_or_compute("products", _fetch_product_map).get(ticker_norm)
    if url is None:
        return None
    response = shared_session.get(url, timeout=25)
    response.raise_for_status()
    url = _holdings_url(response.text, ticker_norm)
    response = shared_session.get(url, params={"etf": ticker_norm}, timeout=25)
    response.raise_for_status()
    reader = csv.DictReader(io.StringIO(response.content.decode("utf-8-sig")))
    required = {"Date", "Account", "StockTicker", "CUSIP", "SecurityName", "Weightings", "MoneyMarketFlag"}
    if not required.issubset(reader.fieldnames or []):
        raise ValueError(f"VistaShares {ticker_norm} 구성종목 CSV 필수 열이 없습니다.")
    rows = list(reader)
    if not rows or any(row["Account"] != ticker_norm or row["Date"] != rows[0]["Date"] for row in rows):
        raise ValueError(f"VistaShares {ticker_norm} 구성종목이 비었거나 티커·기준일이 다릅니다.")
    reference_date = datetime.strptime(rows[0]["Date"], "%m/%d/%Y").date().isoformat()
    us_listings = load_us_security_exchange_map()
    items = [_normalize_holding(row, us_listings) for row in rows]
    return {
        "source": "vistashares_us_csv",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "as_of_date": reference_date,
        "holdings_count": len(items),
        "holdings": items,
    }
