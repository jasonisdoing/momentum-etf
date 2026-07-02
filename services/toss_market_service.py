"""토스증권 시장지표(mini-chart)·차트(c-chart) 연동 서비스.

/live-24h 의 나스닥 100 선물·달러 환율 카드용. 비공식 API 이므로 스키마 변경에 취약하다
— 실패 시 예외를 그대로 올리고, 호출부가 카드 단위로 격리 처리한다.

    - 실시간 지표: wts-cert-api /api/v3/dashboard/wts/overview/indicator/mini-chart
        (주가지수/선물/환율/채권/원자재 latestPrice + basePrice(전일 기준가), REAL_TIME 피드)
    - 캔들: wts-info-api /api/v2/c-chart/us-s/{code}/{interval}?count=N
        (예: RFU.NQc1 min:15 × 96 = 최근 24시간, 최신→과거 순으로 반환됨)
"""

from __future__ import annotations

import threading
import time
from typing import Any

import requests

from config import TOSS_INVEST_API_BASE_URL, TOSS_INVEST_CERT_API_BASE_URL, TOSS_INVEST_HEADERS
from utils.logger import get_app_logger

logger = get_app_logger()

_INDICATOR_TTL_SECONDS = 5.0
_indicator_cache: tuple[float, dict[str, dict[str, Any]]] | None = None
_indicator_lock = threading.Lock()


def fetch_toss_indicator_prices() -> dict[str, dict[str, Any]]:
    """mini-chart 의 모든 지표를 {code: {latest, base, name}} 로 반환한다 (TTL 캐시)."""
    global _indicator_cache
    now = time.time()
    with _indicator_lock:
        if _indicator_cache is not None and now - _indicator_cache[0] < _INDICATOR_TTL_SECONDS:
            return _indicator_cache[1]

    url = f"{TOSS_INVEST_CERT_API_BASE_URL}/api/v3/dashboard/wts/overview/indicator/mini-chart"
    resp = requests.get(url, headers=TOSS_INVEST_HEADERS, timeout=8)
    resp.raise_for_status()
    index_map = (resp.json().get("result") or {}).get("indexMap") or {}

    result: dict[str, dict[str, Any]] = {}
    for items in index_map.values():
        for item in items or []:
            code = str(item.get("code") or "").strip()
            price = item.get("price") or {}
            if not code or code in result:
                continue
            result[code] = {
                "name": str(item.get("displayName") or code),
                "latest": price.get("latestPrice"),
                "base": price.get("basePrice"),
            }
    if not result:
        raise RuntimeError("토스 mini-chart 응답에 지표가 없습니다.")

    with _indicator_lock:
        _indicator_cache = (now, result)
    return result


def fetch_toss_candles(code: str, interval: str = "min:15", count: int = 96) -> list[dict[str, float]]:
    """토스 c-chart 캔들을 과거→최신 순 [{o,h,l,c}] 로 반환한다."""
    url = f"{TOSS_INVEST_API_BASE_URL}/api/v2/c-chart/us-s/{code}/{interval}"
    resp = requests.get(url, headers=TOSS_INVEST_HEADERS, params={"count": int(count)}, timeout=8)
    resp.raise_for_status()
    payload = resp.json()
    candles_raw = ((payload.get("result") or {}).get("candles")) or []
    if not candles_raw:
        raise RuntimeError(f"토스 캔들 응답이 비어 있습니다: {code}")

    candles: list[dict[str, float]] = []
    for c in reversed(candles_raw):  # API 는 최신→과거 순
        try:
            candles.append(
                {
                    "o": float(c["open"]),
                    "h": float(c["high"]),
                    "l": float(c["low"]),
                    "c": float(c["close"]),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    return candles


def fetch_toss_latest_daily_close(code: str) -> tuple[str, float]:
    """가장 최근 일봉(형성 중 포함)의 (날짜 YYYY-MM-DD, 종가)를 반환한다."""
    url = f"{TOSS_INVEST_API_BASE_URL}/api/v2/c-chart/us-s/{code}/day:1"
    resp = requests.get(url, headers=TOSS_INVEST_HEADERS, params={"count": 1}, timeout=8)
    resp.raise_for_status()
    candles = ((resp.json().get("result") or {}).get("candles")) or []
    if not candles:
        raise RuntimeError(f"토스 최신 일봉 응답이 비어 있습니다: {code}")
    latest = candles[0]  # 최신→과거 순
    return str(latest["dt"])[:10], float(latest["close"])
