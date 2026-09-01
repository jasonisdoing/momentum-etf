"""국내 상장 ETF 의 과세 구분 — `/kor-market-etf` 의 모두/과세/비과세 토글이 쓴다.

국내 주식형 ETF 는 매매차익이 **비과세**지만, 그 밖(해외지수·파생·원자재·채권 등)은
매매차익에 배당소득세가 붙는다. 이름만으로는 못 가른다 — `KODEX 레버리지` 는 국내
주식을 담지만 파생이라 과세다.

KIS 종목정보파일에는 이 구분이 없어 **네이버 ETF 분류**(`etfTabCode`)를 쓴다. 인증도
호출 한도도 없고 1,100여 종목을 한 번에 준다.

    1 국내 시장지수 · 2 국내 업종/테마     → 국내주식형 = 비과세
    3 국내 파생 · 4 해외 주식 · 5 원자재 · 6 채권 · 7 기타 → 과세

분류가 없는 종목은 **모른다**로 둔다(None) — 비과세로 넘겨짚으면 세금 계산이 틀린다.
"""

from __future__ import annotations

from typing import Any

import requests

from utils.logger import get_app_logger

logger = get_app_logger()

_URL = "https://finance.naver.com/api/sise/etfItemList.nhn"
_HEADERS = {"User-Agent": "Mozilla/5.0"}
_TIMEOUT = 20

# 매매차익 비과세 — 국내 주식형만.
_TAX_FREE_TAB_CODES = frozenset({1, 2})


def load_etf_tax_free_map() -> dict[str, bool]:
    """티커 → 비과세 여부. 조회 실패나 분류 없는 종목은 맵에서 빠진다."""
    try:
        response = requests.get(_URL, timeout=_TIMEOUT, headers=_HEADERS)
        response.raise_for_status()
        payload: Any = response.json()
    except Exception as exc:
        logger.warning("[국내 ETF] 네이버 ETF 분류 조회 실패 — 과세 구분을 채우지 않는다: %s", exc)
        return {}

    items = ((payload or {}).get("result") or {}).get("etfItemList") or []
    result: dict[str, bool] = {}
    for item in items:
        ticker = str(item.get("itemcode") or "").strip()
        tab = item.get("etfTabCode")
        if not ticker or not isinstance(tab, int):
            continue
        result[ticker] = tab in _TAX_FREE_TAB_CODES
    logger.info("[국내 ETF] 네이버 ETF 분류 %d종목 (비과세 %d)", len(result), sum(result.values()))
    return result
