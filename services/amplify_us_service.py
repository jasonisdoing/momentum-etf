"""Amplify 미국 ETF의 공식 전체 보유내역 수집."""

from __future__ import annotations

import math
from datetime import date, datetime, timezone
from typing import Any

from utils.http_session import request_with_retry
from utils.normalization import resolve_bloomberg_listing
from utils.us_etf_market_service import load_us_security_exchange_map

_DOCUMENT_ROOT = "projects/amplify-etfs-data-feed/databases/(default)/documents"
_API_ROOT = f"https://firestore.googleapis.com/v1/{_DOCUMENT_ROOT}"
_SUPPORTED_TICKERS = {"DIVO"}
_ASSET_CLASSES = {"equity": "Equity", "option": "Option", "money_market": "Money Market", "cash": "Cash"}


def _normalize_holding(fields: dict[str, Any], us_listings: dict[str, str]) -> dict[str, Any]:
    label = fields["StockTicker"]["stringValue"].strip()
    name = fields["SecurityName"]["stringValue"].strip()
    identifier = fields["CUSIP"]["stringValue"].strip()
    kind = fields["holding_type"]["stringValue"].strip()
    raw_weight = fields["Weightings"]["stringValue"].strip()
    if not name or not kind or not raw_weight.endswith("%"):
        raise ValueError("Amplify 구성종목 이름·자산 유형·비중이 잘못됐습니다.")
    weight = float(raw_weight[:-1].replace(",", ""))
    if not math.isfinite(weight):
        raise ValueError("Amplify 구성종목 비중이 유효한 숫자가 아닙니다.")
    # 현금과 옵션은 공시 목록에 남기고, 시세는 미국 상장 명단에서 확인된 증권만 연결한다.
    symbol, exchange = (
        resolve_bloomberg_listing(label, currency="USD", us_listings=us_listings)
        if label and kind in {"equity", "internal_fund"}
        else (None, None)
    )
    return {
        "ticker": symbol or label or identifier or None,
        "name": name,
        "weight": weight,
        "raw_code": identifier or None,
        "yahoo_symbol": symbol,
        "listing_currency": "USD" if symbol else None,
        "listing_exchange": exchange,
        "asset_class": _ASSET_CLASSES.get(kind, kind),
        "price_lookup_supported": symbol is not None,
    }


def fetch_amplify_us_holdings(ticker: str) -> dict[str, Any] | None:
    """공식 전체 보유 페이지와 같은 최신 일별 문서를 읽는다."""
    ticker_norm = ticker.strip().upper()
    if not ticker_norm:
        raise ValueError("Amplify 구성종목 조회에 티커가 필요합니다.")
    if ticker_norm not in _SUPPORTED_TICKERS:
        return None

    path = f"funds/{ticker_norm}/holdings"
    response = request_with_retry(
        "GET", f"{_API_ROOT}/{path}", params={"pageSize": 1, "orderBy": "__name__ desc"}, timeout=25
    )
    response.raise_for_status()
    documents = response.json().get("documents")
    if not isinstance(documents, list) or len(documents) != 1:
        raise ValueError(f"Amplify {ticker_norm} 공식 보유 문서를 찾을 수 없습니다.")
    doc = documents[0]
    fields = doc["fields"]
    as_of = date.fromisoformat(fields["asOfDate"]["stringValue"]).isoformat()
    if doc["name"] != f"{_DOCUMENT_ROOT}/{path}/{as_of}" or fields["type"]["stringValue"] != "regular":
        raise ValueError(f"Amplify {ticker_norm} 공식 보유 문서의 상품·기준일·유형이 다릅니다.")
    rows = fields["holdings"]["arrayValue"].get("values")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Amplify {ticker_norm} 공식 전체 보유내역이 비었습니다.")

    listings = load_us_security_exchange_map()
    items = [_normalize_holding(row["mapValue"]["fields"], listings) for row in rows]
    items.sort(key=lambda item: item["weight"], reverse=True)
    return {
        "source": "amplify_us_api",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "as_of_date": as_of,
        "holdings_count": len(items),
        "holdings": items,
    }
