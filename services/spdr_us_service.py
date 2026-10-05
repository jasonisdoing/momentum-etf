"""SPDR·State Street 미국 ETF 공식 상품 명단과 전체 구성종목 수집."""

from __future__ import annotations

import math
from datetime import datetime
from io import BytesIO
from typing import Any
from xml.etree import ElementTree
from zipfile import ZipFile

from config import CACHE_TTL_SLOW
from utils.http_session import shared_session
from utils.ttl_cache import TtlCache
from utils.us_etf_market_service import load_us_security_master

_DATA_URL = "https://www.ssga.com/library-content/products/fund-data/etfs/us"
_PRODUCT_MAP_CACHE = TtlCache(CACHE_TTL_SLOW, name="spdr_us_products", max_entries=1)
_NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _xlsx_rows(content: bytes) -> list[dict[str, str]]:
    """공식 두 파일의 첫 시트를 읽고 빈 행 경계를 보존한다."""
    with ZipFile(BytesIO(content)) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
            strings = ["".join(node.text or "" for node in item.findall(".//s:t", _NS)) for item in root]
        sheet = ElementTree.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    rows: list[dict[str, str]] = []
    previous = 0
    for row in sheet.findall("s:sheetData/s:row", _NS):
        number = int(row.attrib["r"])
        if number > previous + 1:
            rows.append({})
        cells = {}
        for cell in row.findall("s:c", _NS):
            if cell.find("s:f", _NS) is not None:
                raise ValueError("SPDR 공식 파일에 계산되지 않은 수식이 있습니다.")
            kind = cell.attrib.get("t")
            value = cell.findtext("s:v", namespaces=_NS)
            if kind == "s" and value is not None:
                value = strings[int(value)]
            elif kind == "inlineStr":
                value = "".join(node.text or "" for node in cell.findall("s:is/s:t", _NS))
            elif kind == "e":
                raise ValueError("SPDR 공식 파일에 오류 셀이 있습니다.")
            if value is not None and value.strip():
                cells[cell.attrib["r"].rstrip("0123456789")] = value.strip()
        rows.append(cells)
        previous = number
    return rows


def _table_rows(rows: list[dict[str, str]], required: set[str]) -> list[dict[str, str]]:
    header_index = next((index for index, row in enumerate(rows) if required <= set(row.values())), None)
    if header_index is None:
        raise ValueError("SPDR 공식 파일에 필수 컬럼이 없습니다.")
    headers = rows[header_index]
    result = []
    for row in rows[header_index + 1 :]:
        if not row:
            if result:
                break
            continue
        # 상품 명단의 두 번째 헤더에는 필수 컬럼의 값이 없다.
        if not result and not any(row.get(column) for column, title in headers.items() if title in required):
            continue
        result.append({title: row.get(column, "") for column, title in headers.items()})
    if not result:
        raise ValueError("SPDR 공식 파일의 명단이 비었습니다.")
    return result


def _fetch_product_map() -> dict[str, str]:
    response = shared_session.get(f"{_DATA_URL}/spdr-product-data-us-en.xlsx", timeout=25)
    response.raise_for_status()
    products = _table_rows(_xlsx_rows(response.content), {"Ticker", "Name", "CUSIP", "Asset Class"})
    result = {}
    for product in products:
        ticker = product["Ticker"].replace("®", "").replace("™", "").upper()
        cusip = product["CUSIP"]
        if not ticker or len(cusip) != 9:
            raise ValueError("SPDR 상품의 티커 또는 CUSIP이 잘못됐습니다.")
        if ticker in result:
            raise ValueError(f"SPDR 상품 티커가 중복됐습니다: {ticker}")
        result[ticker] = cusip
    return result


def _normalize_holding(raw: dict[str, str], us_listings: dict[str, str]) -> dict[str, Any]:
    name = raw["Name"]
    weight = float(raw["Weight"])
    if not name or not math.isfinite(weight):
        raise ValueError("SPDR 구성종목 이름 또는 비중이 잘못됐습니다.")
    ticker = raw["Ticker"]
    if ticker in {"", "-"}:
        ticker = None
    currency = raw["Local Currency"].upper()
    symbol = _us_symbol(ticker) if ticker else None
    exchange = us_listings.get(symbol) if currency == "USD" else None
    if exchange is None:
        symbol = None
    if raw["Identifier"].startswith(f"999{currency}"):
        name = f"현금 · {name}"
    return {
        "ticker": symbol or ticker or raw["Identifier"] or None,
        "name": name,
        "weight": weight,
        "raw_code": raw["Identifier"] or None,
        "yahoo_symbol": symbol,
        "listing_currency": currency,
        "listing_exchange": exchange,
        "price_lookup_supported": symbol is not None,
    }


def _us_symbol(ticker: str) -> str:
    """SPDR의 BRK.B와 KIS의 BRK/B를 같은 미국 시세 심볼로 변환한다."""
    return ticker.replace(".", "-").replace("/", "-")


def fetch_spdr_us_holdings(ticker: str) -> dict[str, Any] | None:
    """비 SPDR 종목이면 None, 공식 수집 실패면 예외를 낸다."""
    ticker_norm = str(ticker).strip().upper()
    if not ticker_norm:
        raise ValueError("SPDR 구성종목 조회에 티커가 필요합니다.")
    if ticker_norm not in _PRODUCT_MAP_CACHE.get_or_compute("products", _fetch_product_map):
        return None
    response = shared_session.get(f"{_DATA_URL}/holdings-daily-us-en-{ticker_norm.lower()}.xlsx", timeout=25)
    response.raise_for_status()
    rows = _xlsx_rows(response.content)
    identity = {row["A"]: row.get("B", "") for row in rows if row.get("A") in {"Ticker Symbol:", "Holdings:"}}
    if identity.get("Ticker Symbol:") != ticker_norm:
        raise ValueError(f"SPDR {ticker_norm} 구성종목 파일의 티커가 다릅니다.")
    reference_date = datetime.strptime(identity["Holdings:"], "As of %d-%b-%Y").date().isoformat()
    raw_items = _table_rows(rows, {"Name", "Ticker", "Identifier", "Weight", "Local Currency"})
    us_listings = {_us_symbol(row["ticker"]): row["exchange"] for row in load_us_security_master()}
    items = [_normalize_holding(raw, us_listings) for raw in raw_items]
    return {
        "source": "spdr_us_xlsx",
        "fetched_at": datetime.now().isoformat(),
        "as_of_date": reference_date,
        "holdings_count": len(items),
        "holdings": items,
    }
