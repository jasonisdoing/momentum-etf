"""Fidelity 미국 ETF 공식 일별 전체 보유종목 XLS 수집."""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import xlrd
from bs4 import BeautifulSoup

from utils.http_session import shared_session
from utils.normalization import resolve_bloomberg_listing
from utils.us_etf_market_service import load_us_security_exchange_map

_REPORT_HOST = "https://www.actionsxchangerepository.fidelity.com"
# 공개 시세 API는 자동 요청을 차단하므로 공식 상세 페이지에서 확인한 CUSIP을 등록한다.
_PRODUCT_CUSIPS = {"FTEC": "316092808"}


def _fetch_report(ticker: str, cusip: str) -> bytes:
    response = shared_session.get(
        "https://fundresearch.fidelity.com/prospectus/eproredirect",
        params={
            "clientId": "Fidelity",
            "applicationId": "ETF",
            "securityIdType": "CUSIP",
            "critical": "N",
            "securityId": cusip,
        },
        timeout=25,
    )
    response.raise_for_status()
    match = re.search(r"window\.location\.href\s*=\s*'([^']+)'", response.text)
    if match is None:
        raise ValueError(f"Fidelity {ticker} 공식 보고서 목록 주소가 없습니다.")
    parsed = urlparse(match[1])
    if (
        parsed.scheme != "https"
        or parsed.netloc != "www.actionsxchangerepository.fidelity.com"
        or parsed.path != "/ShowDocument/ComplianceEnvelope.htm"
        or parsed.query == "_fax=error"
    ):
        raise ValueError(f"Fidelity {ticker} 공식 보고서 목록 주소가 잘못됐습니다.")
    response = shared_session.get(match[1], timeout=25)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    daily = soup.select_one("#DALYTab")
    menu = daily.find_parent("td") if daily is not None else None
    action = str(menu.get("onclick", "")) if menu is not None else ""
    pattern = rf"'true'\s*,\s*'{re.escape(ticker)}_Holdings\.xls'\s*,\s*'(_fax=[^']+)'"
    matches = re.findall(pattern, action)
    if len(matches) != 1:
        raise ValueError(f"Fidelity {ticker} 공식 일별 전체 보유종목 XLS 주소가 없습니다.")
    response = shared_session.get(f"{_REPORT_HOST}/ShowDocument/documentExcel.htm?{matches[0]}", timeout=25)
    response.raise_for_status()
    return response.content


def _parse_report(content: bytes, ticker: str) -> tuple[str, list[dict[str, Any]]]:
    book = xlrd.open_workbook(file_contents=content)
    if book.sheet_names() != ["ETF"]:
        raise ValueError(f"Fidelity {ticker} 공식 XLS 시트가 잘못됐습니다.")
    sheet = book.sheet_by_name("ETF")
    metadata = {sheet.cell_value(index, 0): sheet.cell_value(index, 1) for index in range(4)}
    if metadata.get("Ticker Symbol:") != ticker or not metadata.get("Fund Name:"):
        raise ValueError(f"Fidelity {ticker} 공식 XLS의 티커 또는 이름이 잘못됐습니다.")
    reference_date = datetime.strptime(str(metadata["Holding as of:"]), "%d-%b-%y").date().isoformat()
    required = {"Ticker", "Live Cusip", "Security Name", "Security Type", "Currency", "% of Net Assets"}
    header_index = next((index for index in range(sheet.nrows) if required.issubset(sheet.row_values(index))), None)
    if header_index is None:
        raise ValueError(f"Fidelity {ticker} 공식 XLS의 필수 컬럼이 없습니다.")
    headers = sheet.row_values(header_index)
    if len(headers) != len(set(headers)):
        raise ValueError(f"Fidelity {ticker} 공식 XLS의 컬럼이 중복됐습니다.")
    rows = []
    for index in range(header_index + 1, sheet.nrows):
        raw = dict(zip(headers, sheet.row_values(index), strict=True))
        if raw["Security Name"] == "Total:":
            total = raw["% of Net Assets"]
            if not rows or not isinstance(total, (float, int)) or not math.isfinite(total):
                raise ValueError(f"Fidelity {ticker} 공식 XLS의 합계가 잘못됐습니다.")
            return reference_date, rows
        if not raw["Security Name"]:
            raise ValueError(f"Fidelity {ticker} 공식 XLS의 구성종목 행이 비었습니다.")
        rows.append(raw)
    raise ValueError(f"Fidelity {ticker} 공식 XLS에 전체 목록의 합계 행이 없습니다.")


def _normalize_holding(raw: dict[str, Any], us_listings: dict[str, str]) -> dict[str, Any]:
    label = str(raw["Ticker"]).strip()
    name = str(raw["Security Name"]).strip()
    asset_class = str(raw["Security Type"]).strip()
    currency_name = str(raw["Currency"]).strip()
    weight = raw["% of Net Assets"]
    if not name or not isinstance(weight, (float, int)) or not math.isfinite(weight):
        raise ValueError("Fidelity 구성종목 이름 또는 비중이 잘못됐습니다.")
    # 미국 상장 주식만 공통 명단으로 확인한다. 선물·권리·현금·기타 자산은 원본 목록에 남긴다.
    currency = "USD" if currency_name == "US Dollar" else None
    symbol, exchange = (
        resolve_bloomberg_listing(label, currency=currency, us_listings=us_listings)
        if asset_class == "Common Stock" and currency == "USD" and label
        else (None, None)
    )
    if asset_class == "Currency":
        name = f"현금 · {name}"
    return {
        "ticker": symbol or label or str(raw["Live Cusip"]).strip() or None,
        "name": name,
        "weight": float(weight) * 100,
        "raw_code": str(raw["Live Cusip"]).strip() or None,
        "yahoo_symbol": symbol,
        "listing_currency": currency,
        "listing_exchange": exchange,
        "asset_class": asset_class or None,
        "price_lookup_supported": symbol is not None,
    }


def fetch_fidelity_us_holdings(ticker: str) -> dict[str, Any] | None:
    """공식 식별자가 등록되지 않은 상품은 None, 공식 수집 실패는 예외를 낸다."""
    ticker_norm = str(ticker).strip().upper()
    if not ticker_norm:
        raise ValueError("Fidelity 구성종목 조회에 티커가 필요합니다.")
    cusip = _PRODUCT_CUSIPS.get(ticker_norm)
    if cusip is None:
        return None
    reference_date, rows = _parse_report(_fetch_report(ticker_norm, cusip), ticker_norm)
    us_listings = load_us_security_exchange_map()
    items = [_normalize_holding(raw, us_listings) for raw in rows]
    return {
        "source": "fidelity_us_xls",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "as_of_date": reference_date,
        "holdings_count": len(items),
        "holdings": items,
    }
