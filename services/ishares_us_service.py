"""iShares·BlackRock 미국 ETF 공식 상품 명단과 전체 구성종목 수집."""

from __future__ import annotations

import csv
import io
import math
from datetime import datetime
from typing import Any

from config import CACHE_TTL_SLOW
from utils.asx_ticker import ensure_asx_prefix
from utils.http_session import shared_session
from utils.normalization import normalize_exchange_symbol
from utils.ttl_cache import TtlCache

_BASE_URL = "https://www.ishares.com"
_PRODUCT_MAP_CACHE = TtlCache(CACHE_TTL_SLOW, name="ishares_us_products", max_entries=1)
_US_EXCHANGES = {"NASDAQ", "NYSE", "Cboe BZX", "Nyse Mkt Llc", "Non-Nms Quotation Service (Nnqs)"}
# 공통 구성종목 가격 경로가 지원하는 거래소만 시세 심볼로 변환한다.
_EXCHANGE_SUFFIXES = {
    "Asx - All Markets": "AX",
    "Korea Exchange (Stock Market)": "KS",
    "Korea Exchange (Kosdaq)": "KQ",
    "Taiwan Stock Exchange": "TW",
    "Tokyo Stock Exchange": "T",
    "Hong Kong Exchanges And Clearing Ltd": "HK",
    "Shanghai Stock Exchange": "SS",
    "Shenzhen Stock Exchange": "SZ",
    "London Stock Exchange": "L",
}


def _fetch_product_map() -> dict[str, str]:
    response = shared_session.get(
        f"{_BASE_URL}/us/product-screener/product-screener-v3.1.jsn",
        params={
            "dcrPath": "/templatedata/config/product-screener-v3/data/en/us-ishares/ishares-product-screener-backend-config",
            "siteEntryPassthrough": "true",
        },
        timeout=25,
    )
    response.raise_for_status()
    products = response.json()
    if not isinstance(products, dict) or not products:
        raise ValueError("iShares 공식 상품 명단이 비었습니다.")
    result: dict[str, str] = {}
    for product in products.values():
        if "etf" not in product["productView"]:
            continue
        ticker = str(product["localExchangeTicker"]).strip().upper()
        path = str(product["productPageUrl"]).strip()
        if not ticker or not path.startswith(f"/us/products/{product['portfolioId']}/"):
            raise ValueError("iShares 상품의 티커 또는 페이지 주소가 잘못됐습니다.")
        if ticker in result and result[ticker] != path:
            raise ValueError(f"iShares 상품 식별자가 중복됐습니다: {ticker}")
        result[ticker] = path
    if not result:
        raise ValueError("iShares 공식 상품 명단에 ETF가 없습니다.")
    return result


def _resolve_symbol(ticker: str | None, exchange: str, asset_class: str) -> str | None:
    """공식 상장 거래소를 식별한 주식만 시세 조회 심볼로 변환한다."""
    if not ticker or asset_class != "Equity":
        return None
    if exchange in _US_EXCHANGES:
        return normalize_exchange_symbol(ticker, suffix="")
    suffix = _EXCHANGE_SUFFIXES.get(exchange)
    if suffix is None:
        return None
    return normalize_exchange_symbol(ticker, suffix=suffix)


def _normalize_holding(raw: dict[str, str], weight_column: str) -> dict[str, Any]:
    raw_weight = raw[weight_column].strip()
    weight = None if raw_weight == "-" else float(raw_weight.replace(",", ""))
    if weight is not None and not math.isfinite(weight):
        raise ValueError("iShares 구성종목 비중이 잘못됐습니다.")
    raw_ticker = raw.get("Ticker", "").strip()
    ticker = raw_ticker if raw_ticker not in {"", "-"} else None
    identifier = next((raw[key] for key in ("ISIN", "CUSIP") if raw.get(key) not in {None, "", "-"}), None)
    name = raw["Name"].strip()
    if not name:
        raise ValueError("iShares 구성종목 이름이 비었습니다.")
    asset_class = raw["Asset Class"]
    exchange = raw["Exchange"]
    symbol = _resolve_symbol(ticker, exchange, asset_class)
    if symbol and symbol.endswith(".AX"):
        ticker = ensure_asx_prefix(ticker)
    elif symbol and not symbol.endswith((".KS", ".KQ")):
        # 외국 상장 종목을 같은 영문 티커의 미국 ETF로 재간접 확장하지 않는다.
        ticker = symbol
    if asset_class == "Cash Collateral and Margins" or (asset_class == "Cash" and name.endswith(" CASH")):
        name = f"현금 · {name}"
    return {
        "ticker": ticker or identifier,
        "name": name,
        "weight": weight,
        "raw_code": identifier or raw_ticker or None,
        "yahoo_symbol": symbol,
        "listing_currency": raw["Market Currency"].upper(),
        "listing_exchange": exchange,
        "asset_class": asset_class,
        "price_lookup_supported": symbol is not None,
    }


def _parse_holdings(text: str) -> tuple[str, list[dict[str, Any]]]:
    rows = list(csv.reader(io.StringIO(text.lstrip("\ufeff"))))
    date_row = next((row for row in rows if row and row[0] == "Fund Holdings as of"), None)
    if not date_row or len(date_row) != 2:
        raise ValueError("iShares 구성종목 기준일이 없습니다.")
    reference_date = datetime.strptime(date_row[1], "%b %d, %Y").date().isoformat()
    header_index = next(
        (index for index, row in enumerate(rows) if {"Name", "Asset Class", "Exchange"} <= set(row)), None
    )
    if header_index is None:
        raise ValueError("iShares 구성종목 CSV 헤더가 없습니다.")
    headers = rows[header_index]
    weight_column = next((column for column in ("Weight (%)", "Market Weight") if column in headers), None)
    if weight_column is None or "Market Currency" not in headers:
        raise ValueError("iShares 구성종목 비중 또는 상장 통화 컬럼이 없습니다.")
    items = []
    for row in rows[header_index + 1 :]:
        if not row or all(not cell.strip() for cell in row):
            continue
        if len(row) != len(headers):
            raise ValueError("iShares 구성종목 CSV 행의 컬럼 수가 다릅니다.")
        items.append(_normalize_holding(dict(zip(headers, row)), weight_column))
    if not items:
        raise ValueError("iShares 구성종목 CSV가 비었습니다.")
    return reference_date, items


def fetch_ishares_us_holdings(ticker: str) -> dict[str, Any] | None:
    """비 iShares 종목이면 None, 공식 수집 실패면 예외를 낸다."""
    ticker_norm = str(ticker).strip().upper()
    if not ticker_norm:
        raise ValueError("iShares 구성종목 조회에 티커가 필요합니다.")
    path = _PRODUCT_MAP_CACHE.get_or_compute("products", _fetch_product_map).get(ticker_norm)
    if path is None:
        return None
    response = shared_session.get(f"{_BASE_URL}{path}/latest-holdings.csv", timeout=25)
    response.raise_for_status()
    reference_date, items = _parse_holdings(response.text)
    return {
        "source": "ishares_us_csv",
        "fetched_at": datetime.now().isoformat(),
        "as_of_date": reference_date,
        "holdings_count": len(items),
        "holdings": items,
    }
