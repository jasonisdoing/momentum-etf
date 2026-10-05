"""First Trust 미국 ETF 공식 상품 명단과 전체 구성종목 수집."""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import parse_qs, urljoin, urlparse

from bs4 import BeautifulSoup

from config import CACHE_TTL_SLOW
from services.component_price_service import infer_yahoo_symbol_currency
from utils.http_session import shared_session
from utils.normalization import resolve_bloomberg_listing
from utils.ttl_cache import TtlCache
from utils.us_etf_market_service import load_us_security_exchange_map

_BASE_URL = "https://www.ftportfolios.com"
_PRODUCT_MAP_CACHE = TtlCache(CACHE_TTL_SLOW, name="firsttrust_us_products", max_entries=1)


def _fetch_product_map() -> dict[str, str]:
    response = shared_session.get(f"{_BASE_URL}/Retail/etf/etflist.aspx", timeout=25)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    products = {}
    for anchor in soup.select("a[href]"):
        parsed = urlparse(urljoin(_BASE_URL, anchor["href"]))
        tickers = parse_qs(parsed.query).get("Ticker", [])
        if (
            parsed.scheme == "https"
            and parsed.netloc == "www.ftportfolios.com"
            and parsed.path.lower() == "/retail/etf/etfsummary.aspx"
            and len(tickers) == 1
            and re.fullmatch(r"[A-Z0-9]+", tickers[0])
        ):
            products[tickers[0]] = f"{_BASE_URL}/Retail/Etf/EtfHoldings.aspx"
    if not products:
        raise ValueError("First Trust 공식 상품 명단에 ETF가 없습니다.")
    return products


def _required_text(soup: BeautifulSoup, field: str) -> str:
    element = soup.select_one(f'span[id$="_{field}"]')
    if element is None:
        raise ValueError(f"First Trust 공식 구성종목에 {field}가 없습니다.")
    return element.get_text(" ", strip=True)


def _parse_holdings(html: str, ticker: str) -> tuple[str, list[dict[str, str]]]:
    soup = BeautifulSoup(html, "html.parser")
    identity = _required_text(soup, "lblPageHeader")
    if not identity.endswith(f"({ticker})"):
        raise ValueError(f"First Trust {ticker} 공식 구성종목의 티커가 다릅니다.")
    date_match = re.fullmatch(
        r"Holdings of the Fund as of (\d{1,2}/\d{1,2}/\d{4})", _required_text(soup, "lblHoldingsTitle")
    )
    count_match = re.fullmatch(
        r"Total Number of Holdings \(excluding cash\):\s*(\d+)", _required_text(soup, "lblHoldingsCount")
    )
    if date_match is None or count_match is None:
        raise ValueError(f"First Trust {ticker} 구성종목 기준일 또는 전체 건수가 잘못됐습니다.")
    tables = []
    required = {"Security Name", "Identifier", "CUSIP", "Weighting"}
    for table in soup.select("table.fundSilverGrid"):
        values = [
            [cell.get_text(" ", strip=True) for cell in row.find_all(["td", "th"], recursive=False)]
            for row in table.select("tr")
        ]
        if values and required.issubset(values[0]):
            tables.append(values)
    if len(tables) != 1:
        raise ValueError(f"First Trust {ticker} 전체 구성종목 표가 없거나 중복됐습니다.")
    headers, *values = tables[0]
    if not values or len(headers) != len(set(headers)) or any(len(row) != len(headers) for row in values):
        raise ValueError(f"First Trust {ticker} 구성종목 표 구조가 잘못됐습니다.")
    rows = [dict(zip(headers, row, strict=True)) for row in values]
    security_count = sum(not _is_cash(row["Identifier"]) for row in rows)
    if security_count != int(count_match[1]):
        raise ValueError(f"First Trust {ticker} 전체 건수와 수집 건수가 다릅니다.")
    return datetime.strptime(date_match[1], "%m/%d/%Y").date().isoformat(), rows


def _is_cash(label: str) -> bool:
    return re.fullmatch(r"\$[A-Z]{3}", label) is not None


def _normalize_holding(raw: dict[str, str], us_listings: dict[str, str]) -> dict[str, Any]:
    label = raw["Identifier"].strip()
    name = raw["Security Name"].strip()
    weight_text = raw["Weighting"].strip()
    if not name or not weight_text.endswith("%"):
        raise ValueError("First Trust 구성종목 이름 또는 비중이 잘못됐습니다.")
    weight = float(weight_text[:-1].replace(",", ""))
    if not math.isfinite(weight):
        raise ValueError("First Trust 구성종목 비중이 잘못됐습니다.")
    is_cash = _is_cash(label)
    # 공식 해외 상장 식별자를 공통 상장 판별 형식으로 변환한다. 미지원 시장은 원본만 보존한다.
    listing_label = re.sub(r"\.(JP|CN|FP)$", r" \1", label)
    symbol, exchange = (
        (None, None)
        if is_cash or not label
        else resolve_bloomberg_listing(listing_label, currency=None, us_listings=us_listings)
    )
    return {
        "ticker": symbol or label or raw["CUSIP"].strip() or None,
        "name": f"현금 · {name}" if is_cash else name,
        "weight": weight,
        "raw_code": raw["CUSIP"].strip() or None,
        "yahoo_symbol": symbol,
        "listing_currency": label[1:] if is_cash else infer_yahoo_symbol_currency(symbol) if symbol else None,
        "listing_exchange": exchange,
        "asset_class": "Cash" if is_cash else raw.get("Classification") or None,
        "price_lookup_supported": symbol is not None,
    }


def fetch_firsttrust_us_holdings(ticker: str) -> dict[str, Any] | None:
    """공식 명단에 없는 상품은 None, 공식 수집 실패는 예외를 낸다."""
    ticker_norm = str(ticker).strip().upper()
    if not ticker_norm:
        raise ValueError("First Trust 구성종목 조회에 티커가 필요합니다.")
    products = _PRODUCT_MAP_CACHE.get_or_compute("products", _fetch_product_map)
    url = products.get(ticker_norm)
    if url is None:
        return None
    response = shared_session.get(url, params={"Print": "Y", "Ticker": ticker_norm}, timeout=25)
    response.raise_for_status()
    reference_date, rows = _parse_holdings(response.text, ticker_norm)
    us_listings = load_us_security_exchange_map()
    items = [_normalize_holding(row, us_listings) for row in rows]
    return {
        "source": "firsttrust_us_html",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "as_of_date": reference_date,
        "holdings_count": len(items),
        "holdings": items,
    }
