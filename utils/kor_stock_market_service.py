"""한국 개별주 시가총액 리스트 — 네이버 금융 API 기반."""

from __future__ import annotations

import logging
from typing import Any

import requests

from config import NAVER_FINANCE_HEADERS
from utils.market_service import load_ticker_pool_map
from utils.portfolio_io import load_all_holding_tickers

logger = logging.getLogger(__name__)

_NAVER_STOCK_LIST_URL = "https://m.stock.naver.com/api/stocks/marketValue"


def _parse_number(value: str | None) -> int | None:
    """쉼표가 포함된 숫자 문자열을 int로 변환한다."""
    if not value:
        return None
    try:
        return int(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _parse_float(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def load_kor_stock_market(
    market: str,
    limit: int,
    min_market_cap: int,
) -> dict[str, Any]:
    """네이버 API에서 시가총액 상위 종목 리스트를 가져온다.

    Args:
        market: "KOSPI" 또는 "KOSDAQ"
        limit: 가져올 종목 수 (최대 100)
        min_market_cap: 최소 시가총액(억)
    """
    if market not in ("KOSPI", "KOSDAQ"):
        raise ValueError(f"지원하지 않는 마켓입니다: {market}")
    if min_market_cap < 0:
        raise ValueError(f"최소 시가총액은 음수일 수 없습니다: {min_market_cap}")

    page_size = 100
    url = f"{_NAVER_STOCK_LIST_URL}/{market}?page=1&pageSize={page_size}"

    try:
        resp = requests.get(url, headers=NAVER_FINANCE_HEADERS, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as exc:
        logger.error("네이버 주식 리스트 조회 실패 (market=%s): %s", market, exc)
        raise RuntimeError(f"네이버 주식 리스트 조회에 실패했습니다: {exc}") from exc

    raw_stocks = payload.get("stocks") or []
    total_count = payload.get("totalCount", 0)

    # 종목풀 및 보유 정보 로드
    ticker_pool_map = load_ticker_pool_map()
    held_tickers = load_all_holding_tickers()

    rows: list[dict[str, Any]] = []
    for item in raw_stocks:
        # 종목 유형 필터링 (stockEndType이 'stock'인 것만 포함)
        stock_type = str(item.get("stockEndType", "")).lower()
        name = item.get("stockName", "")

        # ETF, ETN 제외 (필드값 및 명칭 키워드 체크)
        if stock_type != "stock" or any(k in name.upper() for k in ["ETF", "ETN"]):
            continue

        ticker = item.get("itemCode", "")
        close_price = _parse_number(item.get("closePrice"))
        change_ratio = _parse_float(item.get("fluctuationsRatio"))
        volume = _parse_number(item.get("accumulatedTradingVolume"))
        # 시가총액은 백만 원 단위 → 억 단위로 변환
        market_cap_million = _parse_number(item.get("marketValue"))
        market_cap_eok = round(market_cap_million / 100) if market_cap_million else None
        if market_cap_eok is None or market_cap_eok < min_market_cap:
            continue

        compare_code = (item.get("compareToPreviousPrice") or {}).get("code", "")
        # code "5"=하락 → 등락률을 음수로
        if compare_code == "5" and change_ratio is not None and change_ratio > 0:
            change_ratio = -change_ratio

        rows.append(
            {
                "rank": 0,
                "ticker": ticker,
                "name": name,
                "ticker_pools": ", ".join(ticker_pool_map.get(ticker, [])),
                "is_held": ticker in held_tickers,
                "current_price": close_price,
                "change_pct": change_ratio,
                "volume": volume,
                "market_cap": market_cap_eok,
            }
        )
    rows = rows[: min(limit, 100)]
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx

    return {
        "market": market,
        "total_count": total_count,
        "count": len(rows),
        "rows": rows,
    }
