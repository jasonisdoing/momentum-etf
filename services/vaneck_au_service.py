"""VanEck 호주 ETF 공식 상품 명단과 전체 보유 XLSX 수집."""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from config import CACHE_TTL_SLOW
from services.component_price_service import infer_yahoo_symbol_currency
from services.xlsx_holdings_service import parse_xlsx_table_rows, read_xlsx_rows
from utils.asx_ticker import ensure_asx_prefix, strip_asx_prefix
from utils.http_session import shared_session
from utils.normalization import resolve_bloomberg_listing
from utils.ttl_cache import TtlCache
from utils.us_etf_market_service import load_us_security_exchange_map

_BASE_URL = "https://www.vaneck.com.au"
_PRODUCT_MAP_CACHE = TtlCache(CACHE_TTL_SLOW, name="vaneck_au_products", max_entries=1)
# 공식 ETF 목록의 상품 링크 — /etf/<분류>/<티커 소문자>/snapshot
_PRODUCT_PATH = re.compile(r"(/etf/[a-z-]+/([a-z0-9]{2,6})/)snapshot/?")
_COLUMNS = {"Security name", "Ticker", "Shares", "Market Value", "% of Net Assets"}


def _fetch_product_map() -> dict[str, str]:
    """티커 → 상품 페이지 주소. 공식 ETF 목록 페이지(`/etf/`)의 상품 링크에서 읽는다."""
    response = shared_session.get(f"{_BASE_URL}/etf/", timeout=25)
    response.raise_for_status()
    products: dict[str, str] = {}
    for anchor in BeautifulSoup(response.text, "html.parser").select("a.fund-link[href]"):
        match = _PRODUCT_PATH.fullmatch(anchor["href"])
        if match is None:
            continue
        path, ticker = match[1], match[2].upper()
        if products.setdefault(ticker, path) != path:
            raise ValueError(f"VanEck 호주 상품 명단에서 {ticker} 주소가 둘 이상입니다.")
    if not products:
        raise ValueError("VanEck 호주 공식 상품 명단이 비었습니다.")
    return products


def _normalize_holding(raw: dict[str, str], us_listings: dict[str, str]) -> dict[str, Any]:
    name = raw["Security name"].strip()
    label = raw["Ticker"].strip()
    weight_text = raw["% of Net Assets"].strip()
    if not name or not weight_text.endswith("%"):
        raise ValueError(f"VanEck 호주 구성종목 이름 또는 비중이 잘못됐습니다: {raw}")
    weight = float(weight_text[:-1].replace(",", ""))
    if not math.isfinite(weight):
        raise ValueError(f"VanEck 호주 구성종목 비중이 잘못됐습니다: {raw}")
    # 현금·기타는 티커 칸이 '--' 다(행 이름 'Other/Cash').
    is_cash = label in {"", "--"}
    symbol, exchange = (
        (None, None) if is_cash else resolve_bloomberg_listing(label, currency=None, us_listings=us_listings)
    )
    ticker = None if is_cash else label
    if symbol:
        ticker = ensure_asx_prefix(symbol[:-3]) if symbol.endswith(".AX") else symbol
        if symbol.endswith((".KS", ".KQ")):
            ticker = symbol[:-3]
    return {
        "ticker": ticker,
        "name": f"현금 · {name}" if is_cash else name,
        "weight": weight,
        "raw_code": None if is_cash else label,
        "yahoo_symbol": symbol,
        "listing_currency": infer_yahoo_symbol_currency(symbol) if symbol else None,
        "listing_exchange": exchange,
        "asset_class": "Cash" if is_cash else None,
        "price_lookup_supported": symbol is not None,
    }


def fetch_vaneck_au_holdings(ticker: str) -> dict[str, Any] | None:
    """공식 명단에 없는 상품은 None, 공식 수집 실패는 예외를 낸다."""
    ticker_norm = strip_asx_prefix(ticker)
    if not ticker_norm:
        raise ValueError("VanEck 호주 구성종목 조회에 티커가 필요합니다.")
    path = _PRODUCT_MAP_CACHE.get_or_compute("products", _fetch_product_map).get(ticker_norm)
    if path is None:
        return None
    response = shared_session.get(urljoin(_BASE_URL, f"{path}downloads/holdings/"), timeout=25)
    response.raise_for_status()
    # 파일 이름이 상품과 기준일을 알려 준다 — 'QUAL_asat_20261021.xlsx'.
    disposition = response.headers.get("content-disposition") or ""
    match = re.search(rf"filename={re.escape(ticker_norm)}_asat_(\d{{8}})\.xlsx", disposition)
    if match is None:
        raise ValueError(f"VanEck 호주 {ticker_norm} 전체 보유 파일의 상품명 또는 기준일이 다릅니다: {disposition}")
    reference_date = datetime.strptime(match[1], "%Y%m%d").date()
    rows = read_xlsx_rows(response.content)
    title = (rows[0].get("A") if rows else "") or ""
    if title != f"All Fund Holdings as at {reference_date:%d/%m/%Y}":
        raise ValueError(f"VanEck 호주 {ticker_norm} 전체 보유 파일 제목의 기준일이 다릅니다: {title}")
    us_listings = load_us_security_exchange_map()
    items = [_normalize_holding(raw, us_listings) for raw in parse_xlsx_table_rows(rows, _COLUMNS)]
    return {
        "source": "vaneck_au_xlsx",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "as_of_date": reference_date.isoformat(),
        "holdings_count": len(items),
        "holdings": items,
    }
