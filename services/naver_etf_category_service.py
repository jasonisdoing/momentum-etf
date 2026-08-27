"""네이버 국내 ETF 업종 분류 조회 (한국 ETF 업종의 단일 소스).

한국 ETF 는 업종이 비어 있어서 업종 상한(`max_per_industry`)이 아예 적용되지 않았다.
개별주는 네이버 업종(`services/naver_industry_service`), 미국·호주는 yfinance 분류를
쓰는데 ETF 만 구멍이었다.

API 두 개를 쓴다(비공식 — 네이버 ETF 화면이 쓰는 경로).

    /api/stockSecurity/etfs/v2/domestic/themes        → 대분류·중분류 코드와 이름 전체
    /api/stockSecurity/etfs/v2/domestic?...&middleCategoryCode=  → 그 분류의 종목 목록

**우선순위**: 한 ETF 가 여러 분류에 걸리므로 좁은 분류부터 잡는다
(`_INDUSTRY_LARGE_CATEGORIES` 순서). 예) TIGER 코리아원자력 →
트렌드「원자력」(먼저) ≫ (주식)섹터「에너지」. 「에너지」로 뭉치면 원자력과 태양광이
업종 상한 한 칸을 다투는데 실제로는 다른 테마다.

분류에 없는 ETF(대표지수·레버리지·인버스 등)는 맵에서 빠진다 — 업종이 없는 게 맞고,
임의 값으로 채우면 업종 상한이 엉뚱하게 묶인다.
"""

from __future__ import annotations

import json
import time
import urllib.request
from typing import Any

from utils.logger import get_app_logger

logger = get_app_logger()

_BASE = "https://stock.naver.com/api/stockSecurity/etfs/v2/domestic"
_THEMES_URL = f"{_BASE}/themes"
_LIST_URL = _BASE + "?listingType=tradingValueDesc&size={size}&index={index}&middleCategoryCode={middle}"
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
_MAX_PAGES = 50  # 응답 형식이 바뀌어도 무한 루프가 되지 않게

# 업종으로 쓸 대분류 — **앞에 있을수록 우선**한다. 앞에서 못 찾은 종목만 다음으로 내려간다.
# 좁고 구체적인 분류를 먼저 잡고, 섹터가 없는 상품(대표지수·파생·배당전략)을 뒤에서 받는다.
# 여기 없는 대분류는 업종으로 쓰지 않는다: 배율·투자국가·투자전략·단일종목·국내운용사·
# 채권·원자재·통화·부동산·멀티에셋·단기자금.
_INDUSTRY_LARGE_CATEGORIES: tuple[str, ...] = (
    "0701",  # 트렌드 — 원자력·조선·건설·화장품·데이터센터 … 가장 구체적
    "0601",  # 혁신기술 — AI·K-반도체·2차전지·로봇·방산 …
    "0401",  # (주식)섹터 — 소재·IT·헬스케어 … GICS 계열
    # 아래 둘은 업종이라기보다 '무엇을 사는 상품인가' 다. 위 셋에 없는 종목만 내려온다.
    "0501",  # (주식)지수 — 코스피200·코스닥150 … 같은 지수를 사는 정방향·레버리지·인버스가 한 묶음이 된다
    "0609",  # 배당 — 고배당·월배당·리츠·커버드콜
)


def _get_json(url: str) -> Any:
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


def fetch_industry_categories() -> list[tuple[str, str]]:
    """업종으로 쓸 (중분류 코드, 업종명) 목록 — **우선순위 순**.

    이름은 네이버가 준다(하드코딩하지 않는다). 대분류가 응답에서 사라지면 그만큼 빠질 뿐
    나머지는 그대로 쓴다 — 없는 코드를 지어내지 않는다.
    """
    groups = _get_json(_THEMES_URL) or []
    by_large = {str(group.get("largeCategoryCode") or ""): group for group in groups}
    ordered: list[tuple[str, str]] = []
    for large_code in _INDUSTRY_LARGE_CATEGORIES:
        group = by_large.get(large_code)
        if not group:
            logger.warning("[NAVER-ETF] 대분류 %s 가 응답에 없습니다 — 건너뜁니다", large_code)
            continue
        for middle in group.get("middleCategories") or []:
            code = str(middle.get("code") or "").strip()
            name = str(middle.get("name") or "").strip()
            if code and name:
                ordered.append((code, name))
    return ordered


def _fetch_category_tickers(middle_code: str) -> list[str]:
    """한 중분류의 종목코드 전부. `hasNext` 를 따라 끝까지 넘긴다."""
    codes: list[str] = []
    index = 0
    while True:
        payload = _get_json(_LIST_URL.format(size=_PAGE_SIZE, index=index, middle=middle_code))
        for item in payload.get("items") or []:
            code = str(item.get("itemCode") or "").strip()
            if code:
                codes.append(code)
        if not payload.get("hasNext"):
            return codes
        index += 1
        if index >= _MAX_PAGES:
            logger.warning("[NAVER-ETF] %s 분류가 %d쪽을 넘어 중단합니다", middle_code, _MAX_PAGES)
            return codes
        time.sleep(_PAGE_SLEEP_SECONDS)


def fetch_etf_industry_map() -> dict[str, str]:
    """국내 ETF 종목코드 → 업종명. 분류에 없는 ETF 는 결과에서 빠진다.

    한 분류가 실패해도 나머지는 살린다 — 업종은 표시·상한 계산용이라 전부 없는 것보다
    있는 만큼이라도 채우는 쪽이 낫다. 실패한 분류는 경고로 남긴다.
    """
    result: dict[str, str] = {}
    categories = fetch_industry_categories()
    for middle_code, name in categories:
        try:
            tickers = _fetch_category_tickers(middle_code)
        except Exception as exc:
            logger.warning("[NAVER-ETF] %s(%s) 분류 조회 실패 — 건너뜁니다: %s", name, middle_code, exc)
            continue
        for ticker in tickers:
            # 우선순위 — 먼저 잡힌 분류(좁은 쪽)를 유지한다.
            result.setdefault(ticker, name)
        time.sleep(_PAGE_SLEEP_SECONDS)
    logger.info("[NAVER-ETF] 국내 ETF 업종 %d건 수집 (분류 %d종)", len(result), len(categories))
    return result
