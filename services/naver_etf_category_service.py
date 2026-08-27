"""네이버 국내 ETF 업종 분류 조회 (한국 ETF 업종의 단일 소스).

한국 ETF 는 업종이 비어 있어서 업종 상한(`max_per_industry`)이 아예 적용되지 않았다.
개별주는 네이버 업종(`services/naver_industry_service`), 미국·호주는 yfinance 분류를
쓰는데 ETF 만 구멍이었다.

네이버 ETF 화면(`stock.naver.com/market/stock/kr/etf/priceTop`)의 「(주식)섹터」 대분류
아래 중분류가 GICS 계열 11개다. 그 목록을 그대로 받아 티커 → 업종명 맵을 만든다.

    /api/stockSecurity/etfs/v2/domestic
        ?listingType=tradingValueDesc&size=100&index=0
        &largeCategoryCode=0401&middleCategoryCode=0401001

응답은 코드만 주고 **이름은 주지 않는다** — 이름은 화면 드롭다운에만 있어서 아래
`_MIDDLE_CATEGORIES` 에 박아 둔다(2026-08-28 화면에서 확인한 순서 그대로).
분류에 없는 ETF(채권·파생·해외 시장대표 등)는 맵에서 빠진다 — 업종이 없는 게 맞고,
임의 값으로 채우면 업종 상한이 엉뚱하게 묶인다.
"""

from __future__ import annotations

import json
import time
import urllib.request
from typing import Any

from utils.logger import get_app_logger

logger = get_app_logger()

_URL = (
    "https://stock.naver.com/api/stockSecurity/etfs/v2/domestic"
    "?listingType=tradingValueDesc&size={size}&index={index}"
    "&largeCategoryCode={large}&middleCategoryCode={middle}"
)
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    ),
    "Referer": "https://stock.naver.com/",
    "Accept": "application/json, text/plain, */*",
}
_TIMEOUT_SECONDS = 15
_MAX_ATTEMPTS = 3
_PAGE_SIZE = 100  # API 상한
_PAGE_SLEEP_SECONDS = 0.15

# 「(주식)섹터」 대분류.
_LARGE_CATEGORY_CODE = "0401"

# 중분류 코드 → 업종명. 코드는 대분류 + 3자리이고, 이름은 API 가 주지 않아 여기 둔다.
_MIDDLE_CATEGORIES: dict[str, str] = {
    "0401001": "소재",
    "0401002": "필수소비재",
    "0401003": "경기소비재",
    "0401004": "산업재",
    "0401005": "에너지",
    "0401006": "유틸리티",
    "0401007": "IT",
    "0401008": "통신서비스",
    "0401009": "헬스케어",
    "0401010": "금융",
    "0401011": "부동산",
}


def industry_options() -> tuple[str, ...]:
    """이 서비스가 채울 수 있는 업종 이름 목록 (화면 표기 순서)."""
    return tuple(_MIDDLE_CATEGORIES.values())


def _get_json(url: str) -> dict[str, Any]:
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
    raise RuntimeError(f"네이버 ETF 분류 조회 실패: {url}") from last_error


def _fetch_category(middle_code: str) -> list[str]:
    """한 중분류의 종목코드 전부. `hasNext` 를 따라 끝까지 넘긴다."""
    codes: list[str] = []
    index = 0
    while True:
        payload = _get_json(_URL.format(size=_PAGE_SIZE, index=index, large=_LARGE_CATEGORY_CODE, middle=middle_code))
        for item in payload.get("items") or []:
            code = str(item.get("itemCode") or "").strip()
            if code:
                codes.append(code)
        if not payload.get("hasNext"):
            return codes
        index += 1
        # 페이지가 이상하게 길어지면(응답 형식 변경 등) 무한 루프가 되지 않게 끊는다.
        if index > 50:
            logger.warning("[NAVER-ETF] %s 분류 페이지가 50쪽을 넘어 중단합니다", middle_code)
            return codes
        time.sleep(_PAGE_SLEEP_SECONDS)


def fetch_etf_industry_map() -> dict[str, str]:
    """국내 ETF 종목코드 → 업종명. 분류에 없는 ETF 는 결과에서 빠진다.

    한 분류가 실패해도 나머지는 살린다 — 업종은 표시·상한 계산용이라 전부 없는 것보다
    있는 만큼이라도 채우는 쪽이 낫다. 실패한 분류는 경고로 남긴다.
    """
    result: dict[str, str] = {}
    for middle_code, name in _MIDDLE_CATEGORIES.items():
        try:
            codes = _fetch_category(middle_code)
        except Exception as exc:
            logger.warning("[NAVER-ETF] %s(%s) 분류 조회 실패 — 건너뜁니다: %s", name, middle_code, exc)
            continue
        for code in codes:
            # 같은 ETF 가 두 분류에 걸리면 먼저 만난 쪽을 유지한다(화면 순서 = 우선순위).
            result.setdefault(code, name)
        time.sleep(_PAGE_SLEEP_SECONDS)
    logger.info("[NAVER-ETF] 국내 ETF 업종 %d건 수집 (분류 %d종)", len(result), len(_MIDDLE_CATEGORIES))
    return result
