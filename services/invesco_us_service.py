"""Invesco 미국 ETF 공식 상품 명단과 전체 구성종목 수집."""

from __future__ import annotations

import math
from datetime import datetime
from html import unescape
from typing import Any

from config import CACHE_TTL_SLOW
from utils.http_session import shared_session
from utils.ttl_cache import TtlCache

_API_BASE = "https://dng-api.invesco.com"
_HEADERS = {
    "Accept": "application/json",
    "Origin": "https://www.invesco.com",
    "Referer": "https://www.invesco.com/us/en/financial-products/etfs.html",
}
_PRODUCT_MAP_CACHE = TtlCache(CACHE_TTL_SLOW, name="invesco_us_products", max_entries=1)


def _get_json(path: str, params: dict[str, Any]) -> dict[str, Any]:
    response = shared_session.get(f"{_API_BASE}{path}", params=params, headers=_HEADERS, timeout=25)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Invesco 응답이 JSON 객체가 아닙니다.")
    return payload


def _fetch_product_map() -> dict[str, str]:
    payload = _get_json(
        "/product/search",
        {
            "q": "_suggest_:*",
            "fq": [
                'countryCode:"US"',
                'language:"en_us"',
                'accountType:"ETF"',
                'contentType:"Product"',
                'shareClassStatus:"open"',
            ],
            "fl": "ticker,cusip",
            "rows": 2000,
            "start": 0,
        },
    )["response"]
    products = payload["docs"]
    if not products or len(products) != payload["numFound"]:
        raise ValueError("Invesco ETF 상품 명단이 비었거나 일부만 반환됐습니다.")
    result: dict[str, str] = {}
    for product in products:
        ticker = str(product["ticker"]).strip().upper()
        cusip = str(product["cusip"]).strip()
        if not ticker or len(cusip) != 9:
            raise ValueError("Invesco 상품의 티커 또는 CUSIP이 잘못됐습니다.")
        if ticker in result and result[ticker] != cusip:
            raise ValueError(f"Invesco 상품 식별자가 중복됐습니다: {ticker}")
        result[ticker] = cusip
    return result


def _resolve_cusip(ticker: str) -> str | None:
    """공식 상품 명단에 속한 티커만 Invesco로 판정한다."""
    return _PRODUCT_MAP_CACHE.get_or_compute("products", _fetch_product_map).get(ticker)


def _normalize_holding(raw: dict[str, Any]) -> dict[str, Any]:
    weight = raw["percentageOfTotalNetAssets"]
    if weight is not None and not math.isfinite(float(weight)):
        raise ValueError("Invesco 구성종목 비중이 잘못됐습니다.")
    name = unescape(str(raw["issuerName"]))
    if raw["securityTypeCode"] in {"CURR", "UCURR"}:
        # 공통 구성종목 가격 경로가 현금을 종목 시세로 조회하지 않도록 표준 표시를 쓴다.
        name = f"현금 · {name}"
    return {
        "ticker": raw["ticker"],
        "name": name,
        "weight": float(weight) if weight is not None else None,
        "raw_code": raw["cusip"],
    }


def fetch_invesco_us_holdings(ticker: str) -> dict[str, Any] | None:
    """비 Invesco 종목이면 None, 공식 수집 실패면 예외를 낸다."""
    ticker_norm = str(ticker).strip().upper()
    if not ticker_norm:
        raise ValueError("Invesco 구성종목 조회에 티커가 필요합니다.")
    cusip = _resolve_cusip(ticker_norm)
    if cusip is None:
        return None
    payload = _get_json(
        f"/cache/v1/accounts/en_US/shareclasses/{cusip}/holdings/fund",
        {"idType": "cusip", "productType": "ETF"},
    )
    rows = payload["holdings"]
    if payload["cusip"] != cusip or not rows or len(rows) != payload["totalNumberOfHoldings"]:
        raise ValueError(f"Invesco {ticker_norm} 구성종목이 비었거나 일부만 반환됐습니다.")
    reference_date = payload["effectiveBusinessDate"]
    datetime.strptime(reference_date, "%Y-%m-%d")
    items = [_normalize_holding(raw) for raw in rows]
    return {
        "source": "invesco_us_api",
        "fetched_at": datetime.now().isoformat(),
        "as_of_date": reference_date,
        "holdings_count": len(items),
        "holdings": items,
    }
