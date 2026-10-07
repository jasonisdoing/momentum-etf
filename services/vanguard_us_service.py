"""Vanguard 미국 ETF 공식 상품 명단과 전체 보유 목록 수집."""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any

from config import CACHE_TTL_SLOW
from services.component_price_service import infer_yahoo_symbol_currency
from services.etf_holdings_service import DEFAULT_USER_AGENT
from utils.http_session import shared_session
from utils.normalization import resolve_bloomberg_listing
from utils.ttl_cache import TtlCache
from utils.us_etf_market_service import load_us_security_exchange_map

_PRODUCTS_URL = "https://advisors.vanguard.com/investments/fund-data/all-products"
_HOLDINGS_URL = "https://investor.vanguard.com/irr/funds/profile/{ticker}-AdditionalFundData"
_HEADERS = {"User-Agent": DEFAULT_USER_AGENT}
_PRODUCT_MAP_CACHE = TtlCache(CACHE_TTL_SLOW, name="vanguard_us_products", max_entries=1)
_LISTED_SECURITY_TYPES = {"EQ.STOCK", "EQ.REIT", "EQ.DRCPT", "EQ.PREF", "EQ.FSH"}


def _fetch_product_map() -> dict[str, dict[str, Any]]:
    response = shared_session.get(_PRODUCTS_URL, headers=_HEADERS, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or not payload:
        raise ValueError("Vanguard 미국 공식 상품 명단이 비었거나 잘못됐습니다.")
    products = {}
    for fund_id, product in payload.items():
        classifications = product["classifications"]
        if classifications["isETF"] is not True:
            continue
        ticker = product["ticker"]
        if (
            not isinstance(ticker, str)
            or re.fullmatch(r"[A-Z0-9]+", ticker) is None
            or product["portId"] != fund_id
            or not isinstance(classifications["isDomestic"], bool)
            or ticker in products
        ):
            raise ValueError("Vanguard 미국 공식 ETF의 티커 또는 분류가 잘못됐습니다.")
        products[ticker] = {"is_domestic": classifications["isDomestic"]}
    if not products:
        raise ValueError("Vanguard 미국 공식 상품 명단에 ETF가 없습니다.")
    return products


def _parse_holdings(payload: Any, ticker: str) -> tuple[str, list[dict[str, Any]]]:
    if not isinstance(payload, dict) or payload["historicalPrice"]["ticker"] != ticker:
        raise ValueError(f"Vanguard {ticker} 공식 구성종목 응답의 티커가 다릅니다.")
    details = payload["holdingDetails"]
    reference_date = datetime.strptime(details["asOfDate"], "%m/%d/%Y").date().isoformat()
    rows = []
    for section, holdings in details.items():
        if not section.endswith("Holdings"):
            continue
        if not isinstance(holdings, list):
            raise ValueError(f"Vanguard {ticker} 전체 구성종목 목록의 구조가 잘못됐습니다: {section}")
        for raw in holdings:
            if not isinstance(raw, dict) or not {"securityType", "securityLongDescription"}.issubset(raw):
                raise ValueError(f"Vanguard {ticker} 구성종목 필수 필드가 없습니다.")
            rows.append(raw)
    if not rows:
        raise ValueError(f"Vanguard {ticker} 공식 전체 구성종목이 비었습니다.")
    return reference_date, rows


def _holding_weight(raw: dict[str, Any]) -> float | None:
    text = raw.get("marketValuePercentage")
    # 공식 공시에서 비중을 제공하지 않는 권리 등은 값이 없는 상태 그대로 유지한다.
    if text is None:
        return None
    if not isinstance(text, str) or not text.endswith("%"):
        raise ValueError("Vanguard 구성종목 비중이 잘못됐습니다.")
    weight = float(text[:-1])
    if not math.isfinite(weight):
        raise ValueError("Vanguard 구성종목 비중이 잘못됐습니다.")
    return weight


def _normalize_holding(raw: dict[str, Any], *, is_domestic: bool, us_listings: dict[str, str]) -> dict[str, Any]:
    name = str(raw["securityLongDescription"] or "").strip()
    label = str(raw.get("ticker") or "").strip()
    isin = str(raw.get("isin") or "").strip()
    asset_class = raw["securityType"]
    if not name:
        raise ValueError("Vanguard 구성종목 이름이 비었습니다.")
    # 해외 원주와 미국 ADR은 티커가 같을 수 있다. 미국 분류나 미국 ISIN이 확인된 주식만 연결한다.
    symbol, exchange = (
        resolve_bloomberg_listing(label, currency="USD", us_listings=us_listings)
        if label and asset_class in _LISTED_SECURITY_TYPES and (is_domestic or isin.startswith("US"))
        else (None, None)
    )
    identifier = isin or raw.get("cusip") or raw.get("sedol") or raw.get("securityId") or None
    is_cash = asset_class == "MM.CASH"
    return {
        "ticker": symbol or label or identifier,
        "name": f"현금 · {name}" if is_cash else name,
        "weight": _holding_weight(raw),
        "raw_code": identifier,
        "yahoo_symbol": symbol,
        "listing_currency": infer_yahoo_symbol_currency(symbol) if symbol else None,
        "listing_exchange": exchange,
        "asset_class": "Cash" if is_cash else asset_class,
        "price_lookup_supported": symbol is not None,
    }


def fetch_vanguard_us_holdings(ticker: str) -> dict[str, Any] | None:
    """공식 명단에 없는 상품은 None, 공식 수집 실패는 예외를 낸다."""
    ticker_norm = str(ticker).strip().upper()
    if not ticker_norm:
        raise ValueError("Vanguard 미국 구성종목 조회에 티커가 필요합니다.")
    product = _PRODUCT_MAP_CACHE.get_or_compute("products", _fetch_product_map).get(ticker_norm)
    if product is None:
        return None
    response = shared_session.get(_HOLDINGS_URL.format(ticker=ticker_norm), headers=_HEADERS, timeout=30)
    response.raise_for_status()
    reference_date, rows = _parse_holdings(response.json(), ticker_norm)
    us_listings = load_us_security_exchange_map()
    items = [_normalize_holding(raw, is_domestic=product["is_domestic"], us_listings=us_listings) for raw in rows]
    return {
        "source": "vanguard_us_api",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "as_of_date": reference_date,
        "holdings_count": len(items),
        "holdings": items,
    }
