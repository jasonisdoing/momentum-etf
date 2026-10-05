"""ARK 미국 ETF 공식 전체 구성종목 CSV 수집."""

from __future__ import annotations

import csv
import io
import math
from datetime import datetime, timezone
from typing import Any

from services.component_price_service import infer_yahoo_symbol_currency
from utils.asx_ticker import ensure_asx_prefix
from utils.http_session import shared_session
from utils.normalization import resolve_bloomberg_listing
from utils.us_etf_market_service import load_us_security_exchange_map

_BASE_URL = "https://assets.ark-funds.com/fund-documents/funds-etf-csv"
# 상품 페이지는 자동 요청을 차단하므로 현재 전체 공시로 확인한 공개 파일 주소를 등록한다.
_CSV_FILES = {
    "ARKG": "ARK_GENOMIC_REVOLUTION_ETF_ARKG_HOLDINGS.csv",
    "ARKK": "ARK_INNOVATION_ETF_ARKK_HOLDINGS.csv",
}


def _parse_holdings(text: str, ticker: str) -> tuple[str, list[dict[str, str]]]:
    reader = csv.reader(io.StringIO(text))
    headers = next(reader, [])
    required = {"date", "fund", "company", "ticker", "cusip", "shares", "market value ($)", "weight (%)"}
    if len(headers) != len(required) or set(headers) != required:
        raise ValueError(f"ARK {ticker} 구성종목 CSV 필수 열이 잘못됐습니다.")
    rows = []
    footer_seen = False
    for values in reader:
        if not values:
            continue
        if len(values) == 1 and values[0].startswith("Investors should carefully consider"):
            footer_seen = True
            continue
        if footer_seen or len(values) != len(headers):
            raise ValueError(f"ARK {ticker} 구성종목 CSV 행 구조가 잘못됐습니다.")
        raw = dict(zip(headers, values, strict=True))
        if raw["fund"] != ticker or (rows and raw["date"] != rows[0]["date"]):
            raise ValueError(f"ARK {ticker} 구성종목의 티커 또는 기준일이 다릅니다.")
        rows.append(raw)
    if not rows:
        raise ValueError(f"ARK {ticker} 공식 구성종목이 비었습니다.")
    reference_date = datetime.strptime(rows[0]["date"], "%m/%d/%Y").date().isoformat()
    return reference_date, rows


def _normalize_holding(raw: dict[str, str], us_listings: dict[str, str]) -> dict[str, Any]:
    label = raw["ticker"].strip()
    name = raw["company"].strip()
    identifier = raw["cusip"].strip()
    weight_text = raw["weight (%)"].strip()
    if not name or not weight_text.endswith("%"):
        raise ValueError("ARK 구성종목 이름 또는 비중이 잘못됐습니다.")
    weight = float(weight_text[:-1])
    if not math.isfinite(weight):
        raise ValueError("ARK 구성종목 비중이 잘못됐습니다.")
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
        "name": name,
        "weight": weight,
        "raw_code": identifier or None,
        "yahoo_symbol": symbol,
        "listing_currency": infer_yahoo_symbol_currency(symbol) if symbol else None,
        "listing_exchange": exchange,
        "price_lookup_supported": symbol is not None,
    }


def fetch_ark_us_holdings(ticker: str) -> dict[str, Any] | None:
    """등록된 ARK 상품이 아니면 None, 공식 수집 실패면 예외를 낸다."""
    ticker_norm = str(ticker).strip().upper()
    if not ticker_norm:
        raise ValueError("ARK 구성종목 조회에 티커가 필요합니다.")
    filename = _CSV_FILES.get(ticker_norm)
    if filename is None:
        return None
    response = shared_session.get(f"{_BASE_URL}/{filename}", timeout=25)
    response.raise_for_status()
    reference_date, rows = _parse_holdings(response.content.decode("utf-8-sig"), ticker_norm)
    us_listings = load_us_security_exchange_map()
    items = [_normalize_holding(raw, us_listings) for raw in rows]
    return {
        "source": "ark_us_csv",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "as_of_date": reference_date,
        "holdings_count": len(items),
        "holdings": items,
    }
