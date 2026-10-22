"""WisdomTree 미국 ETF 공식 상품 명단과 전체 보유 API 수집.

사이트가 Cloudflare 로 일반 HTTP 요청을 막아(403) 브라우저 TLS 지문을 흉내 내는 공용 요청(`browser_get`)으로 받는다.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any

from config import CACHE_TTL_SLOW
from services.component_price_service import infer_yahoo_symbol_currency
from utils.http_session import browser_get
from utils.normalization import resolve_bloomberg_listing
from utils.ttl_cache import TtlCache
from utils.us_etf_market_service import load_us_security_exchange_map

_BASE_URL = "https://www.wisdomtree.com"
_PRODUCT_LIST_URL = f"{_BASE_URL}/investments/etfs"
_HOLDINGS_URL = f"{_BASE_URL}/api/fund-holdings/{{class_id}}"
_PRODUCT_MAP_CACHE = TtlCache(CACHE_TTL_SLOW, name="wisdomtree_us_products", max_entries=1)
# 상품 목록 페이지의 메뉴 데이터 — `{"_id":"<상품ID>.product", ..., "ticker":"GDE"}`.
_FUTURES_NAME = re.compile(r"\bFUTR?\b|\(CBT\)")
_PRODUCT_PATTERN = re.compile(r'"_id":"(\d+)\.product","_type":"product".{0,900}?"ticker":"([A-Z0-9.]+)"')


def _fetch_product_map() -> dict[str, str]:
    """티커 → WisdomTree 상품 ID(wtClassID). 공식 ETF 목록 페이지의 상품 링크에서 읽는다."""
    response = browser_get(_PRODUCT_LIST_URL, timeout=40)
    text = response.text.replace('\\"', '"')
    products: dict[str, str] = {}
    for match in _PRODUCT_PATTERN.finditer(text):
        class_id, ticker = match.groups()
        if products.setdefault(ticker, class_id) != class_id:
            raise ValueError(f"WisdomTree 상품 명단에서 {ticker} 의 상품 ID 가 둘 이상입니다.")
    if not products:
        raise ValueError("WisdomTree 공식 상품 명단에 ETF가 없습니다.")
    return products


def _fetch_holdings(class_id: str, ticker: str) -> list[dict[str, Any]]:
    response = browser_get(_HOLDINGS_URL.format(class_id=class_id), timeout=30)
    rows = response.json()
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"WisdomTree {ticker} 구성종목 응답이 비었거나 형식이 잘못됐습니다.")
    for row in rows:
        if not isinstance(row, dict) or row.get("fundTicker") != ticker:
            raise ValueError(f"WisdomTree {ticker} 구성종목 응답의 상품 식별자가 다릅니다.")
    return rows


def _normalize_holding(raw: dict[str, Any], us_listings: dict[str, str], total_value: float) -> dict[str, Any]:
    name = str(raw.get("securityName") or "").strip()
    label = str(raw.get("securityTicker") or "").strip()
    group = str(raw.get("assetGroup") or "").strip().upper()
    is_cash = group == "CA"
    is_currency = group == "CC"
    # 비중 = 평가금액 / 전체 평가금액. 공시 `wgt` 는 상품에 따라 자산군마다 따로 100%를 맞춰
    # (GDE 현금 2건이 50%씩, USFR 현금이 99.9%) 펀드 내 비중이 아니다. 평가금액이 없는 환헤지
    # 계약(CC)만 공시 `wgt`(헤지 비율)를 그대로 쓴다.
    value = raw.get("marketValueBase")
    if isinstance(value, (int, float)) and math.isfinite(value):
        weight = float(value) / total_value * 100.0
    elif is_currency and isinstance(raw.get("wgt"), (int, float)) and math.isfinite(raw["wgt"]):
        weight = float(raw["wgt"]) * 100.0
    else:
        raise ValueError(f"WisdomTree 구성종목 평가금액이 없습니다: {name}")
    if not name:
        raise ValueError("WisdomTree 구성종목 이름이 비었습니다.")
    # 선물은 티커 없이 이름으로만 온다 — 'GOLD 100 OZ FUTR DEC26', 'US 10YR ULTRA FUT DEC26', 'US 5YR NOTE (CBT) DEC26'.
    is_futures = not label and _FUTURES_NAME.search(name.upper()) is not None
    symbol, exchange = (
        resolve_bloomberg_listing(label, currency=None, us_listings=us_listings)
        if label and not (is_cash or is_currency or is_futures)
        else (None, None)
    )
    figi = str(raw.get("figi") or "").strip()
    return {
        "ticker": symbol or label or None,
        "name": f"현금 · {name}" if is_cash else name,
        "weight": weight,
        "raw_code": None if figi.startswith("FIGI-PENDING") else figi or None,
        "yahoo_symbol": symbol,
        "listing_currency": infer_yahoo_symbol_currency(symbol) if symbol else None,
        "listing_exchange": exchange,
        "asset_class": "Cash" if is_cash else "Currency" if is_currency else "Futures" if is_futures else None,
        "price_lookup_supported": symbol is not None,
    }


def fetch_wisdomtree_us_holdings(ticker: str) -> dict[str, Any] | None:
    """공식 명단에 없는 상품은 None, 공식 수집 실패는 예외를 낸다."""
    ticker_norm = str(ticker).strip().upper()
    if not ticker_norm:
        raise ValueError("WisdomTree 구성종목 조회에 티커가 필요합니다.")
    products = _PRODUCT_MAP_CACHE.get_or_compute("products", _fetch_product_map)
    class_id = products.get(ticker_norm)
    if class_id is None:
        return None
    rows = _fetch_holdings(class_id, ticker_norm)
    dates = {str(row.get("dt") or "")[:10] for row in rows}
    if len(dates) != 1 or not next(iter(dates)):
        raise ValueError(f"WisdomTree {ticker_norm} 구성종목의 기준일이 하나가 아닙니다: {sorted(dates)}")
    total_value = sum(
        float(row["marketValueBase"])
        for row in rows
        if isinstance(row.get("marketValueBase"), (int, float)) and math.isfinite(row["marketValueBase"])
    )
    if total_value <= 0:
        raise ValueError(f"WisdomTree {ticker_norm} 구성종목의 평가금액 합계가 0 이하입니다.")
    us_listings = load_us_security_exchange_map()
    items = [_normalize_holding(raw, us_listings, total_value) for raw in rows]
    return {
        "source": "wisdomtree_us_api",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "as_of_date": next(iter(dates)),
        "holdings_count": len(items),
        "holdings": items,
    }
