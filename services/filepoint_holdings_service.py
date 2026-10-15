"""Filepoint 보유 CSV의 공통 검증과 구성종목 정규화."""

from __future__ import annotations

import csv
import io
import math
from typing import Any

from services.component_price_service import infer_yahoo_symbol_currency
from utils.asx_ticker import ensure_asx_prefix
from utils.http_session import shared_session
from utils.normalization import resolve_bloomberg_listing


def fetch_filepoint_csv(url: str, required: set[str], *, provider: str) -> list[dict[str, str]]:
    response = shared_session.get(url, timeout=25)
    response.raise_for_status()
    reader = csv.DictReader(io.StringIO(response.content.decode("utf-8-sig")))
    headers = reader.fieldnames or []
    if not required.issubset(headers) or len(headers) != len(set(headers)):
        raise ValueError(f"{provider} 공식 CSV 필수 컬럼이 없거나 중복됐습니다.")
    rows = list(reader)
    if not rows or any(None in row or any(row[field] is None for field in required) for row in rows):
        raise ValueError(f"{provider} 공식 CSV가 비었거나 행 구조가 잘못됐습니다.")
    return rows


def normalize_filepoint_holding(raw: dict[str, str], us_listings: dict[str, str], *, provider: str) -> dict[str, Any]:
    label = raw["StockTicker"].strip()
    name = raw["SecurityName"].strip()
    identifier = raw["CUSIP"].strip()
    weight_text = raw["Weightings"].strip()
    if not name or not weight_text.endswith("%"):
        raise ValueError(f"{provider} 구성종목 이름 또는 비중이 잘못됐습니다.")
    weight = float(weight_text[:-1].replace(",", ""))
    if not math.isfinite(weight):
        raise ValueError(f"{provider} 구성종목 비중이 잘못됐습니다.")
    is_cash = raw["MoneyMarketFlag"] == "Y" or identifier.upper().startswith("CASH")
    is_swap = " TRS " in label or "SWAP" in name.upper() or "SWP" in label
    is_bond = name.startswith("United States Treasury Bill")
    symbol, exchange = (
        resolve_bloomberg_listing(label, currency=None, us_listings=us_listings)
        if label and not (is_cash or is_swap or is_bond)
        else (None, None)
    )
    ticker = symbol or label or identifier or None
    if symbol:
        ticker = ensure_asx_prefix(symbol[:-3]) if symbol.endswith(".AX") else symbol
        if symbol.endswith((".KS", ".KQ")):
            ticker = symbol[:-3]
    return {
        "ticker": ticker,
        "name": f"현금 · {name}" if is_cash else name,
        "weight": weight,
        "raw_code": identifier or None,
        "yahoo_symbol": symbol,
        "listing_currency": infer_yahoo_symbol_currency(symbol) if symbol else None,
        "listing_exchange": exchange,
        "asset_class": "Cash" if is_cash else "Swap" if is_swap else "Bond" if is_bond else None,
        "price_lookup_supported": symbol is not None,
    }
