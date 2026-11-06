"""ProShares 미국 ETF의 공식 전체 보유 표 수집."""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any

from bs4 import BeautifulSoup

from utils.http_session import request_with_retry
from utils.normalization import resolve_bloomberg_listing
from utils.us_etf_market_service import load_us_security_exchange_map

# 공식 전체 표와 미국 주식 상장 식별을 확인한 상품만 연동한다.
_PRODUCTS = {"NOBL": "https://www.proshares.com/our-etfs/strategic/nobl"}
_HEADERS = [
    "Weight",
    "Ticker",
    "Description",
    "Exposure Value (Notional + GL)",
    "Market Value",
    "Shares/Contracts",
    "SEDOL Number",
]


def _parse_holdings(html: str, ticker: str) -> tuple[str, list[dict[str, str]]]:
    soup = BeautifulSoup(html, "html.parser")
    title = soup.select_one("h1")
    if title is None or title.get_text(strip=True) != ticker:
        raise ValueError(f"ProShares {ticker} 공식 상품 페이지의 티커가 다릅니다.")
    sections = soup.select(".fund-detail-table-holdings")
    if len(sections) != 1:
        raise ValueError(f"ProShares {ticker} 전체 보유 영역이 없거나 중복됐습니다.")
    section = sections[0]
    date_node = section.select_one(".section-header-subtitle")
    match = re.fullmatch(r"as of (\d{1,2}/\d{1,2}/\d{4})", date_node.get_text(strip=True)) if date_node else None
    if match is None:
        raise ValueError(f"ProShares {ticker} 구성종목 기준일이 없습니다.")
    tables = section.select("table#holdings")
    if len(tables) != 1 or [cell.get_text(" ", strip=True) for cell in tables[0].select("th")] != _HEADERS:
        raise ValueError(f"ProShares {ticker} 구성종목 표 구조가 잘못됐습니다.")
    rows = []
    for row in tables[0].select("tbody tr"):
        values = [cell.get_text(" ", strip=True) for cell in row.find_all("td", recursive=False)]
        if len(values) != len(_HEADERS):
            raise ValueError(f"ProShares {ticker} 구성종목 행 구조가 잘못됐습니다.")
        rows.append(dict(zip(_HEADERS, values, strict=True)))
    if not rows:
        raise ValueError(f"ProShares {ticker} 전체 구성종목이 비었습니다.")
    return datetime.strptime(match[1], "%m/%d/%Y").date().isoformat(), rows


def _normalize_holding(raw: dict[str, str], us_listings: dict[str, str]) -> dict[str, Any]:
    name = raw["Description"].strip()
    label = raw["Ticker"].strip()
    identifier = raw["SEDOL Number"].strip()
    weight_text = raw["Weight"].strip()
    if not name or (weight_text != "--" and not weight_text.endswith("%")):
        raise ValueError("ProShares 구성종목 이름 또는 비중이 잘못됐습니다.")
    weight = None if weight_text == "--" else float(weight_text[:-1].replace(",", ""))
    if weight is not None and not math.isfinite(weight):
        raise ValueError("ProShares 구성종목 비중이 잘못됐습니다.")
    symbol, exchange = (
        resolve_bloomberg_listing(label, currency="USD", us_listings=us_listings)
        if label not in {"", "--"}
        else (None, None)
    )
    return {
        "ticker": symbol or (label if label not in {"", "--"} else None),
        "name": name,
        "weight": weight,
        "raw_code": identifier if identifier not in {"", "--"} else None,
        "yahoo_symbol": symbol,
        "listing_currency": "USD" if symbol else None,
        "listing_exchange": exchange,
        "asset_class": "Equity" if symbol else None,
        "price_lookup_supported": symbol is not None,
    }


def fetch_proshares_us_holdings(ticker: str) -> dict[str, Any] | None:
    """확인한 상품은 공식 전체 보유 표를 받고, 미등록 상품은 None을 반환한다."""
    ticker_norm = str(ticker).strip().upper()
    if not ticker_norm:
        raise ValueError("ProShares 구성종목 조회에 티커가 필요합니다.")
    url = _PRODUCTS.get(ticker_norm)
    if url is None:
        return None
    response = request_with_retry("GET", url, timeout=25)
    response.raise_for_status()
    reference_date, rows = _parse_holdings(response.text, ticker_norm)
    us_listings = load_us_security_exchange_map()
    items = [_normalize_holding(row, us_listings) for row in rows]
    return {
        "source": "proshares_us_html",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "as_of_date": reference_date,
        "holdings_count": len(items),
        "holdings": items,
    }
