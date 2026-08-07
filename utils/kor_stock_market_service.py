"""한국 개별주 시가총액 리스트 — 네이버 금융 API 기반."""

from __future__ import annotations

import logging
import math
from typing import Any

import pandas as pd
import requests

from config import NAVER_FINANCE_HEADERS
from utils.market_service import load_ticker_pool_map
from utils.naver_chart import fetch_naver_daily_ohlc
from utils.portfolio_io import load_all_holding_tickers
from services.price_service import get_realtime_snapshot

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


def _fetch_market_value_page(market: str, page: int, page_size: int) -> dict[str, Any]:
    url = f"{_NAVER_STOCK_LIST_URL}/{market}?page={page}&pageSize={page_size}"

    try:
        resp = requests.get(url, headers=NAVER_FINANCE_HEADERS, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as exc:
        logger.error("네이버 주식 리스트 조회 실패 (market=%s, page=%s): %s", market, page, exc)
        raise RuntimeError(f"네이버 주식 리스트 조회에 실패했습니다: {exc}") from exc


def load_kor_stock_market(
    market: str,
    limit: int,
    min_market_cap_jo: int,
) -> dict[str, Any]:
    """네이버 API에서 시가총액 상위 종목 리스트를 가져온다.

    Args:
        market: "KOSPI" 또는 "KOSDAQ"
        limit: 가져올 종목 수 (최대 200)
        min_market_cap_jo: 최소 시가총액(조)
    """
    if market not in ("KOSPI", "KOSDAQ"):
        raise ValueError(f"지원하지 않는 마켓입니다: {market}")
    if limit <= 0:
        raise ValueError(f"가져올 종목 수는 1 이상이어야 합니다: {limit}")
    if min_market_cap_jo < 0:
        raise ValueError(f"최소 시가총액은 음수일 수 없습니다: {min_market_cap_jo}")

    min_market_cap_eok = min_market_cap_jo * 10000

    page_size = 100
    first_payload = _fetch_market_value_page(market, page=1, page_size=page_size)
    total_count = int(first_payload.get("totalCount") or 0)
    total_pages = max(1, math.ceil(total_count / page_size)) if total_count > 0 else 1

    # 종목풀 및 보유 정보 로드
    ticker_pool_map = load_ticker_pool_map()
    held_tickers = load_all_holding_tickers()

    target_count = min(limit, 200)
    rows: list[dict[str, Any]] = []
    payload = first_payload
    for page in range(1, total_pages + 1):
        stocks = payload.get("stocks") or []
        for item in stocks:
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
            # 네이버 marketValue는 이미 억 단위다.
            market_cap_eok = _parse_number(item.get("marketValue"))
            if market_cap_eok is None or market_cap_eok < min_market_cap_eok:
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
            if len(rows) >= target_count:
                break
        if len(rows) >= target_count:
            break
        if page >= total_pages:
            break
        payload = _fetch_market_value_page(market, page=page + 1, page_size=page_size)

    rows = rows[:target_count]
    _apply_kor_realtime_overlay(rows)
    _apply_kor_history_metrics(rows)
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx

    return {
        "market": market,
        "total_count": total_count,
        "count": len(rows),
        "rows": rows,
    }


def _apply_kor_realtime_overlay(rows: list[dict[str, Any]]) -> None:
    """한국 개별주 리스트에 네이버 실시간/장전 가격을 반영한다."""
    tickers = [str(row.get("ticker") or "").strip().upper() for row in rows if str(row.get("ticker") or "").strip()]
    if not tickers:
        return

    try:
        snapshot = get_realtime_snapshot("kor", tickers)
    except Exception as exc:
        logger.warning("한국 개별주 실시간 가격 오버레이 실패: %s", exc)
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


def _calculate_period_return(close: pd.Series, latest_price: float, months: int) -> float | None:
    target_date = pd.Timestamp.today().normalize() - pd.DateOffset(months=months)
    base_candidates = close[close.index >= target_date]
    if close.empty:
        return None
    base_date = close.index.min() if base_candidates.empty else base_candidates.index.min()
    base_price = float(close.loc[base_date])
    if base_price <= 0:
        return None
    return round(((latest_price / base_price) - 1.0) * 100.0, 4)


def _calculate_mdd(close: pd.Series, months: int) -> float | None:
    target_date = pd.Timestamp.today().normalize() - pd.DateOffset(months=months)
    period = close[close.index >= target_date]
    if period.empty:
        period = close
    if period.empty:
        return None
    drawdown = (period / period.cummax() - 1.0) * 100.0
    return round(float(drawdown.min()), 4)


def _apply_kor_history_metrics(rows: list[dict[str, Any]]) -> None:
    """한국 개별주 일봉으로 기간 수익률과 MDD를 보강한다."""
    for row in rows:
        ticker = str(row.get("ticker") or "").strip()
        if not ticker:
            continue
        history = fetch_naver_daily_ohlc(ticker, count=280)
        if history is None or history.empty or "Close" not in history.columns:
            row.update({"return_1m_pct": None, "return_3m_pct": None, "return_12m_pct": None, "mdd_12m_pct": None})
            continue
        close = pd.to_numeric(history["Close"], errors="coerce").dropna()
        close = close[close > 0]
        if close.empty:
            row.update({"return_1m_pct": None, "return_3m_pct": None, "return_12m_pct": None, "mdd_12m_pct": None})
            continue
        latest_price = float(row.get("current_price") or close.iloc[-1])
        row["return_1m_pct"] = _calculate_period_return(close, latest_price, 1)
        row["return_3m_pct"] = _calculate_period_return(close, latest_price, 3)
        row["return_12m_pct"] = _calculate_period_return(close, latest_price, 12)
        row["mdd_12m_pct"] = _calculate_mdd(close, 12)
# 티커→시가총액 맵은 화면 재방문마다 페이지 순회를 반복하지 않게 짧은 TTL 로 캐시한다.
_MARKET_CAP_CACHE: dict[str, tuple[float, dict[str, int]]] = {}
_MARKET_CAP_CACHE_TTL_SEC = 600.0


def load_kor_market_caps(tickers: list[str]) -> dict[str, int]:
    """티커 → 시가총액(억 원) 맵 — 네이버 marketValue 리스트(위 화면과 같은 소스).

    KOSPI·KOSDAQ 를 시총 상위부터 순회하며 요청 티커를 찾는다. 요청 티커를 모두
    찾으면 조기 종료한다. 목록에 없는 티커(순위권 밖)는 맵에서 빠진다 — 값이 없으면
    화면은 '-' 로 둔다(임의 보정 없음).
    """
    import time as _time

    wanted = {str(t or "").strip() for t in tickers if str(t or "").strip()}
    if not wanted:
        return {}

    now = _time.monotonic()
    cached = _MARKET_CAP_CACHE.get("kor")
    if cached is not None and now - cached[0] < _MARKET_CAP_CACHE_TTL_SEC:
        return {t: cap for t, cap in cached[1].items() if t in wanted}

    caps: dict[str, int] = {}
    page_size = 100
    for market in ("KOSPI", "KOSDAQ"):
        try:
            first = _fetch_market_value_page(market, page=1, page_size=page_size)
        except RuntimeError:
            continue  # 한 시장 실패가 다른 시장까지 막지 않게 — 실패분은 맵에서 빠진다.
        total_count = int(first.get("totalCount") or 0)
        total_pages = max(1, math.ceil(total_count / page_size)) if total_count > 0 else 1
        payload = first
        for page in range(1, total_pages + 1):
            for item in payload.get("stocks") or []:
                ticker = str(item.get("itemCode") or "").strip()
                cap = _parse_number(item.get("marketValue"))
                if ticker and cap is not None:
                    caps[ticker] = cap
            if wanted <= set(caps):
                break
            if page < total_pages:
                payload = _fetch_market_value_page(market, page=page + 1, page_size=page_size)
        if wanted <= set(caps):
            break

    _MARKET_CAP_CACHE["kor"] = (now, caps)
    return {t: cap for t, cap in caps.items() if t in wanted}
