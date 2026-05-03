"""미국 개별주 시가총액 리스트 — 네이버 금융 API 기반."""

from __future__ import annotations

import logging
import time
from datetime import date
from typing import Any

import pandas as pd
import requests

from config import NAVER_FINANCE_HEADERS, NAVER_US_STOCK_MARKET_VALUE_URL
from services.price_service import get_realtime_snapshot
from utils.data_loader import get_today_str
from utils.index_constituents_loader import load_index_constituents, load_index_meta
from utils.market_service import load_ticker_pool_map
from utils.portfolio_io import load_all_holding_tickers

logger = logging.getLogger(__name__)

_SUPPORTED_MARKETS = {"NYS", "NSQ"}
_NAVER_US_PAGE_SIZE_MAX = 200


def _parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _parse_int(value: Any) -> int | None:
    parsed = _parse_float(value)
    if parsed is None:
        return None
    return int(parsed)


def _fetch_us_market_value_page(market: str, start_idx: int, page_size: int) -> list[dict[str, Any]]:
    params = {
        "nation": "USA",
        "tradeType": market,
        "orderType": "marketValue",
        "startIdx": str(start_idx),
        "pageSize": str(page_size),
    }
    try:
        resp = requests.get(NAVER_US_STOCK_MARKET_VALUE_URL, params=params, headers=NAVER_FINANCE_HEADERS, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        logger.error("네이버 미국 주식 리스트 조회 실패 (market=%s, page_size=%s): %s", market, page_size, exc)
        raise RuntimeError(f"네이버 미국 주식 리스트 조회에 실패했습니다: {exc}") from exc

    if not isinstance(payload, list):
        raise RuntimeError("네이버 미국 주식 리스트 응답 형식이 올바르지 않습니다.")
    return payload


def load_us_stock_market(market: str, limit: int, min_market_cap_ukm: int = 0) -> dict[str, Any]:
    """네이버 API에서 미국 시가총액 상위 종목 리스트를 가져온다."""
    if market not in _SUPPORTED_MARKETS:
        raise ValueError(f"지원하지 않는 마켓입니다: {market}")
    if limit not in (50, 100, 150, 200):
        raise ValueError(f"지원하지 않는 시가총액 상위 개수입니다: {limit}")

    # 미국 풀만 매칭 (호주 동일 심볼과 혼동 방지)
    ticker_pool_map = load_ticker_pool_map(country_code="us")
    held_tickers = load_all_holding_tickers(country_code="us")

    target_count = limit
    min_market_cap_usd = min_market_cap_ukm * 100_000_000
    rows: list[dict[str, Any]] = []
    
    start_idx = 0
    page_size = 100
    
    while len(rows) < target_count:
        items = _fetch_us_market_value_page(market, start_idx=start_idx, page_size=page_size)
        if not items:
            break
            
        for item in items:
            market_cap = _parse_float(item.get("marketValue"))
            if market_cap is None or market_cap < min_market_cap_usd:
                continue

            raw_ticker = str(item.get("symbolCode") or "").strip().upper()
            if not raw_ticker:
                continue
            ticker = raw_ticker.replace(".", "-") if "." in raw_ticker else raw_ticker

            exchange = item.get("stockExchangeType") or {}
            exchange_code = str(exchange.get("code") or market).strip().upper()
            current_price = _parse_float(item.get("currentPrice") or item.get("closePrice"))
            change_pct = _parse_float(item.get("fluctuationsRatio"))

            rows.append(
                {
                    "rank": 0,
                    "ticker": ticker,
                    "name": item.get("koreanCodeName") or item.get("englishCodeName") or ticker,
                    "english_name": item.get("englishCodeName") or "",
                    "industry": item.get("reutersIndustryName") or "",
                    "market": exchange_code,
                    "ticker_pools": ", ".join(ticker_pool_map.get(ticker, [])),
                    "is_held": ticker in held_tickers,
                    "current_price": current_price,
                    "change_pct": change_pct,
                    "volume": _parse_int(item.get("accumulatedTradingVolume")),
                    "market_cap": market_cap,
                    "return_3m_base_date": None,
                    "return_3m_base_price": None,
                    "return_3m_pct": None,
                }
            )
            if len(rows) >= target_count:
                break
                
        start_idx += page_size

    _apply_us_realtime_overlay(rows)
    rows.sort(key=lambda row: (-(row["market_cap"] or 0), row["ticker"]))
    for idx, row in enumerate(rows[:limit], start=1):
        row["rank"] = idx

    return {
        "market": market,
        "total_count": len(rows),
        "count": len(rows[:limit]),
        "rows": rows[:limit],
    }


def load_index_stock_market(index: str, min_market_cap_ukm: int = 0) -> dict[str, Any]:
    """S&P500 또는 NASDAQ100 구성종목을 JSON에서 읽어 실시간 가격을 더해 반환한다.
    시가총액은 JSON에 저장된 값(yfinance 기준)을 사용한다."""
    constituents = load_index_constituents(index)
    meta = load_index_meta(index)

    ticker_pool_map = load_ticker_pool_map(country_code="us")
    held_tickers = load_all_holding_tickers(country_code="us")

    min_market_cap_usd = min_market_cap_ukm * 100_000_000

    rows: list[dict[str, Any]] = []
    for item in constituents:
        ticker = str(item.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        market_cap = item.get("market_cap")
        if min_market_cap_usd > 0 and (market_cap is None or market_cap < min_market_cap_usd):
            continue
        rows.append(
            {
                "rank": 0,
                "ticker": ticker,
                "name": item.get("name") or ticker,
                "english_name": item.get("name") or "",
                "industry": item.get("industry") or item.get("sector") or "",
                "sector": item.get("sector") or "",
                "market": "",
                "ticker_pools": ", ".join(ticker_pool_map.get(ticker, [])),
                "is_held": ticker in held_tickers,
                "current_price": None,
                "change_pct": None,
                "volume": item.get("volume"),
                "market_cap": market_cap,
                "return_3m_base_date": item.get("return_3m_base_date"),
                "return_3m_base_price": item.get("return_3m_base_price"),
                "return_3m_pct": item.get("return_3m_pct"),
            }
        )

    _apply_us_realtime_overlay(rows)
    rows.sort(key=lambda r: (-(r["market_cap"] or 0), r["ticker"]))

    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx

    return {
        "index": index.upper(),
        "updated_at": meta.get("updated_at", ""),
        "total_count": len(rows),
        "count": len(rows),
        "rows": rows,
    }


def fetch_naver_us_stock_info_map(tickers: set[str] | list[str] | tuple[str, ...]) -> dict[str, dict[str, Any]]:
    """네이버 미국 종목 API에서 요청 티커의 이름·거래소·업종 정보를 조회한다."""
    targets = {str(ticker or "").strip().upper() for ticker in tickers if str(ticker or "").strip()}
    if not targets:
        return {}

    found: dict[str, dict[str, Any]] = {}
    for market in ("NSQ", "NYS"):
        start_idx = 0
        while targets - set(found):
            page = _fetch_us_market_value_page(market, start_idx=start_idx, page_size=_NAVER_US_PAGE_SIZE_MAX)
            if not page:
                break

            for item in page:
                raw_ticker = str(item.get("symbolCode") or "").strip().upper()
                ticker = raw_ticker.replace(".", "-") if "." in raw_ticker else raw_ticker
                if ticker not in targets or ticker in found:
                    continue
                exchange = item.get("stockExchangeType") or {}
                found[ticker] = {
                    "ticker": ticker,
                    "name": item.get("koreanCodeName") or item.get("englishCodeName") or ticker,
                    "english_name": item.get("englishCodeName") or "",
                    "market": str(exchange.get("code") or market).strip().upper(),
                    "industry": item.get("reutersIndustryName") or "",
                    "market_cap": _parse_float(item.get("marketValue")),
                    "dividend_yield_ttm": _parse_float(item.get("dividendYieldRatio")),
                    "dividend_per_share_ttm": _parse_float(item.get("dividend")),
                    "listing_date": item.get("listedAt"),
                }

            if len(page) < _NAVER_US_PAGE_SIZE_MAX:
                break
            start_idx += 1

    return found


def _apply_us_realtime_overlay(rows: list[dict[str, Any]]) -> None:
    """Toss API를 통해 실시간 가격 정보를 오버레이합니다."""
    tickers = [str(row.get("ticker") or "").strip().upper() for row in rows if row.get("ticker")]
    if not tickers:
        return

    try:
        from services.price_service import get_realtime_snapshot
        snapshot = get_realtime_snapshot("us", tickers)
    except Exception as exc:
        logger.warning("미국 개별주 실시간 가격 오버레이 실패: %s", exc)
        return

    for row in rows:
        ticker = str(row.get("ticker") or "").strip().upper()
        realtime = snapshot.get(ticker)
        if not realtime:
            continue

        now_val = realtime.get("nowVal")
        if now_val is not None:
            row["current_price"] = now_val

        change_rate = realtime.get("changeRate")
        if change_rate is not None:
            row["change_pct"] = change_rate

        base_price = _parse_float(row.get("return_3m_base_price"))
        if base_price is not None and base_price > 0 and row.get("current_price") is not None:
            row["return_3m_pct"] = round(((float(row["current_price"]) / base_price) - 1.0) * 100.0, 4)
