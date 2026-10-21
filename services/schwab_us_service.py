"""Schwab 미국 ETF 공식 상품 명단(사이트맵)과 전체 보유 페이지(allholdings) 수집.

사이트가 일반 HTTP 요청을 막아(403) 브라우저 TLS 지문을 흉내 내는 curl_cffi 로 받는다.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any

from bs4 import BeautifulSoup
from curl_cffi import requests as curl_requests

from config import CACHE_TTL_SLOW
from utils.normalization import resolve_bloomberg_listing
from utils.ttl_cache import TtlCache
from utils.us_etf_market_service import load_us_security_exchange_map

_BASE_URL = "https://www.schwabassetmanagement.com"
_SITEMAP_URL = f"{_BASE_URL}/sitemap.xml"
_HOLDINGS_URL = f"{_BASE_URL}/allholdings/{{ticker}}?page={{page}}"
_SESSION = curl_requests.Session(impersonate="chrome")
_PRODUCT_CACHE = TtlCache(CACHE_TTL_SLOW, name="schwab_us_products", max_entries=1)
_PRODUCT_PATH = re.compile(r"<loc>https://www\.schwabassetmanagement\.com/products/([a-z0-9]{3,6})</loc>")
_HEADERS = ["Fund Name", "CUSIP", "Symbol", "Quantity", "% of Assets", "Market Value"]


def _fetch_products() -> set[str]:
    """Schwab 상품 페이지(`/products/<티커>`)가 있는 티커 — 사이트맵이 공식 명단이다."""
    response = _SESSION.get(_SITEMAP_URL, timeout=40)
    response.raise_for_status()
    products = {slug.upper() for slug in _PRODUCT_PATH.findall(response.text)}
    if not products:
        raise ValueError("Schwab 사이트맵에 상품 페이지가 없습니다.")
    return products


def _fetch_page(ticker: str, page: int) -> tuple[list[list[str]], str, int]:
    """한 페이지의 (행, 기준일 원문, 마지막 페이지 번호)."""
    response = _SESSION.get(_HOLDINGS_URL.format(ticker=ticker, page=page), timeout=40)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    table = soup.find("table")
    if table is None:
        raise ValueError(f"Schwab {ticker} 전체 보유 표가 없습니다(page={page}).")
    headers = [th.get_text(" ", strip=True) for th in table.find_all("th")]
    if headers != _HEADERS:
        raise ValueError(f"Schwab {ticker} 전체 보유 표 컬럼이 바뀌었습니다: {headers}")
    rows = [[td.get_text(" ", strip=True) for td in tr.find_all("td")] for tr in table.find_all("tr")[1:]]
    as_of_node = soup.select_one("section.view__header--as-of-date")
    as_of = re.fullmatch(r"As of (\d{1,2}/\d{1,2}/\d{2})", as_of_node.get_text(" ", strip=True)) if as_of_node else None
    if as_of is None:
        raise ValueError(f"Schwab {ticker} 전체 보유 기준일이 없습니다(page={page}).")
    pages = [int(value) for value in re.findall(r"\?page=(\d+)", response.text)]
    return rows, as_of.group(1), max(pages, default=0)


def _is_us_equity_fund(rows: list[list[str]], us_listings: dict[str, str]) -> bool:
    """미국 주식 펀드인지 — 비중의 80% 이상이 미국 상장 명단에 있는 티커면 그렇다.

    Schwab 표의 Symbol 에는 거래소 표기가 없다. 해외 펀드(SCHF)는 현지 코드라 미국 티커와 겹치면 엉뚱한
    회사가 된다(로슈 'ROP' → 로퍼). 반대로 미국 펀드의 아일랜드·스위스 법인(ACN·LIN·CB)은 CUSIP 이
    영문(CINS)이고, 해외 펀드의 현지 주식도 CUSIP 자리에 숫자 코드가 와서 행 단위로는 가를 수 없다.
    그래서 펀드 단위로 미국 명단과 대조할지 정한다(실측: SCHD·SCHB 96% 이상, SCHF 30% 남짓).
    """
    total = listed = 0.0
    for row in rows:
        weight = abs(float(row[4].rstrip("%").replace(",", ""))) if row[4].endswith("%") else 0.0
        total += weight
        if row[2] and resolve_bloomberg_listing(row[2], currency="USD", us_listings=us_listings)[0]:
            listed += weight
    return total > 0 and listed / total >= 0.8


def _normalize_holding(row: list[str], us_listings: dict[str, str], *, us_fund: bool) -> dict[str, Any]:
    if len(row) != len(_HEADERS):
        raise ValueError(f"Schwab 구성종목 행 구조가 잘못됐습니다: {row}")
    name, cusip, label, _quantity, weight_text, _value = row
    if not name or not weight_text.endswith("%"):
        raise ValueError(f"Schwab 구성종목 이름 또는 비중이 잘못됐습니다: {row}")
    weight = float(weight_text[:-1].replace(",", ""))
    if not math.isfinite(weight):
        raise ValueError(f"Schwab 구성종목 비중이 잘못됐습니다: {row}")
    # Schwab 은 현금을 CUSIP '999USDZ92'(US DOLLAR) 처럼 999 로 시작하는 내부 코드로 표기한다.
    is_cash = cusip.startswith("999")
    symbol, exchange = (
        resolve_bloomberg_listing(label, currency="USD", us_listings=us_listings)
        if label and us_fund and not is_cash
        else (None, None)
    )
    return {
        "ticker": symbol or label or None,
        "name": f"현금 · {name}" if is_cash else name,
        "weight": weight,
        "raw_code": cusip or None,
        "yahoo_symbol": symbol,
        "listing_currency": "USD" if symbol else None,
        "listing_exchange": exchange,
        "asset_class": "Cash" if is_cash else None,
        "price_lookup_supported": symbol is not None,
    }


def fetch_schwab_us_holdings(ticker: str) -> dict[str, Any] | None:
    """공식 명단에 없는 상품은 None, 공식 수집 실패는 예외를 낸다."""
    ticker_norm = str(ticker).strip().upper()
    if not ticker_norm:
        raise ValueError("Schwab 구성종목 조회에 티커가 필요합니다.")
    if ticker_norm not in _PRODUCT_CACHE.get_or_compute("products", _fetch_products):
        return None
    rows, as_of, last_page = _fetch_page(ticker_norm, 0)
    for page in range(1, last_page + 1):
        page_rows, page_as_of, _ = _fetch_page(ticker_norm, page)
        if page_as_of != as_of:
            raise ValueError(f"Schwab {ticker_norm} 전체 보유 페이지의 기준일이 다릅니다: {as_of} / {page_as_of}")
        rows.extend(page_rows)
    if not rows:
        raise ValueError(f"Schwab {ticker_norm} 전체 보유 표에 구성종목이 없습니다.")
    us_listings = load_us_security_exchange_map()
    us_fund = _is_us_equity_fund(rows, us_listings)
    items = [_normalize_holding(row, us_listings, us_fund=us_fund) for row in rows]
    return {
        "source": "schwab_us_html",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "as_of_date": datetime.strptime(as_of, "%m/%d/%y").date().isoformat(),
        "holdings_count": len(items),
        "holdings": items,
    }
