"""네이버 국내 업종 분류 조회 (한국 종목의 업종 단일 소스).

한국 종목의 업종은 yfinance 대신 네이버를 쓴다.

- yfinance 는 국내 종목 셋 중 하나(911개 중 312개)에 분류가 아예 없어서, 그만큼이
  업종 상한(`max_per_industry`) 계산에서 통째로 빠졌다.
- 용어도 네이버가 실제 사업에 가깝다(가비아 = IT서비스, GS = 석유와가스).
- 한국어 원본이라 번역 사전이 필요 없다. 미국·호주는 계속 yfinance 영문을 쓴다.

API 두 개를 쓴다(비공식 — 네이버 앱/웹이 쓰는 경로).

    /api/stock/{ticker}/integration   → industryCode (숫자)
    /api/stocks/industry/{code}       → groupInfo.name (업종명)

코드→이름은 전체 79종뿐이라 프로세스 안에서 캐시한다. 종목당 요청은 integration 1회다.
호출이 잦으면 네이버가 잠깐 거절하므로 짧은 재시도를 둔다.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.request

from utils.logger import get_app_logger

logger = get_app_logger()

_BASE = "https://m.stock.naver.com/api"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    ),
    "Referer": "https://m.stock.naver.com/",
    "Accept": "application/json, text/plain, */*",
}
_TIMEOUT_SECONDS = 15
_MAX_ATTEMPTS = 3

# 업종 코드 → 이름. 전체 79종이라 프로세스 캐시로 충분하다.
_INDUSTRY_NAME_CACHE: dict[int, str] = {}
_CACHE_LOCK = threading.Lock()


def _get_json(url: str) -> dict:
    last_error: Exception | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            request = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
                return json.load(response)
        except Exception as exc:
            last_error = exc
            if attempt < _MAX_ATTEMPTS:
                time.sleep(1.0 * attempt)
    raise RuntimeError(f"네이버 조회 실패: {url}") from last_error


def industry_name_by_code(code: int) -> str:
    """업종 코드 → 업종명. 조회 실패 시 빈 문자열(임의 값으로 채우지 않는다)."""
    with _CACHE_LOCK:
        cached = _INDUSTRY_NAME_CACHE.get(code)
    if cached is not None:
        return cached

    try:
        payload = _get_json(f"{_BASE}/stocks/industry/{code}")
        name = str((payload.get("groupInfo") or {}).get("name") or "").strip()
    except Exception as exc:
        logger.warning("네이버 업종명 조회 실패 (code=%s): %s", code, exc)
        return ""

    if name:
        with _CACHE_LOCK:
            _INDUSTRY_NAME_CACHE[code] = name
    return name


def fetch_industry(ticker: str) -> str:
    """한국 종목의 업종명. 조회 실패·미분류면 빈 문자열.

    빈 문자열은 '미설정'을 뜻한다 — 호출자가 그 상태를 그대로 두어야 한다
    (임의 값으로 메우면 업종 상한이 엉뚱하게 묶인다).
    """
    code = str(ticker or "").strip()
    if not code:
        return ""

    try:
        payload = _get_json(f"{_BASE}/stock/{code}/integration")
    except Exception as exc:
        logger.warning("네이버 종목 정보 조회 실패 (%s): %s", code, exc)
        return ""

    industry_code = payload.get("industryCode")
    if industry_code is None:
        return ""
    try:
        return industry_name_by_code(int(industry_code))
    except (TypeError, ValueError):
        return ""
