"""미국 개별주 시가총액 리스트 — 네이버 금융 API 기반."""

from __future__ import annotations

import logging
from typing import Any

import requests

from config import NAVER_FINANCE_HEADERS, NAVER_US_STOCK_MARKET_VALUE_URL
from services.price_service import get_realtime_snapshot
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
        logger.error("네이버 미국 주식 리스트 조회 실패 (market=%s, limit=%s): %s", market, limit, exc)
        raise RuntimeError(f"네이버 미국 주식 리스트 조회에 실패했습니다: {exc}") from exc

    if not isinstance(payload, list):
        raise RuntimeError("네이버 미국 주식 리스트 응답 형식이 올바르지 않습니다.")
    return payload


def load_us_stock_market(market: str, limit: int) -> dict[str, Any]:
    """네이버 API에서 미국 시가총액 상위 종목 리스트를 가져온다."""
    if market not in _SUPPORTED_MARKETS:
        raise ValueError(f"지원하지 않는 마켓입니다: {market}")
    if limit not in (50, 100, 150, 200):
        raise ValueError(f"지원하지 않는 시가총액 상위 개수입니다: {limit}")

    # 미국 풀만 매칭 (호주 동일 심볼과 혼동 방지)
    ticker_pool_map = load_ticker_pool_map(country_code="us")
    held_tickers = load_all_holding_tickers(country_code="us")

    rows: list[dict[str, Any]] = []
    for item in _fetch_us_market_value_page(market, start_idx=0, page_size=limit):
        raw_ticker = str(item.get("symbolCode") or "").strip().upper()
        if not raw_ticker:
            continue
        ticker = raw_ticker.replace(".", "-") if "." in raw_ticker else raw_ticker

        exchange = item.get("stockExchangeType") or {}
        exchange_code = str(exchange.get("code") or market).strip().upper()
        current_price = _parse_float(item.get("currentPrice") or item.get("closePrice"))
        change_pct = _parse_float(item.get("fluctuationsRatio"))
        market_cap = _parse_float(item.get("marketValue"))

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
            }
        )

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
                ticker = str(item.get("symbolCode") or "").strip().upper()
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
    """미국 개별주 리스트에 기존 실시간 가격 조회 결과를 반영한다."""
    tickers = [str(row.get("ticker") or "").strip().upper() for row in rows if str(row.get("ticker") or "").strip()]
    if not tickers:
        return

    try:
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

        volume = realtime.get("volume")
        if volume is not None:
            row["volume"] = volume
