"""네이버 해외증시 일봉 — 미국 종목 가격 캐시의 **보강용** 소스.

주 소스는 yfinance 다. 그런데 야후가 봉을 만들어 두고 **종가만 비우는** 날이 있다
(2026-08-28: Open·High·Low·Volume 은 있는데 Close·Adj Close 가 NaN). 그 행이 캐시에
그대로 들어가면 그날이 거래일로 잡힌 채 종가가 없어, 백테스트가 보유 평가액을 0 으로
매기는 등 화면마다 다른 방식으로 어긋난다.

같은 봉을 네이버는 온전히 들고 있다 — 시가·고가·저가는 yfinance 와 소수점까지 같고
종가만 채워져 있다. 그래서 빈 곳만 여기서 메운다. 인증도 호출 한도도 없다.

**심볼 접미사는 규칙으로 못 만든다.** 거래소가 같아도 갈린다:
    SPY · VOO · GLD · TAN → 접미사 없음      (AMX)
    SCHD · BWET · JEPI    → `.K`             (AMX)
    QQQ · AAPL · NVDA     → `.O`             (NSQ)
그래서 후보를 순서대로 눌러 보고 찾은 값을 캐시한다(토스 심볼 매핑과 같은 방식).

클래스 주식(`BRK-B`)은 접미사로 안 되고 표기 자체가 다르다 — 하이픈을 빼고 클래스
문자를 **소문자**로 붙인 로이터 코드를 쓴다(`BRKb` · `BFb` · `LENb` · `HEIa`).
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import requests

from utils.logger import get_app_logger

logger = get_app_logger()

_BASE_URL = "https://api.stock.naver.com/stock"
_HEADERS = {"User-Agent": "Mozilla/5.0"}
_TIMEOUT = 15

# 눌러 볼 접미사 — 앞의 것부터. 무접미사가 가장 흔해 먼저 본다.
_SYMBOL_SUFFIXES = ("", ".O", ".K", ".N")

# 티커 → 네이버 심볼. 못 찾은 티커는 None 으로 남겨 다시 두드리지 않는다.
_SYMBOL_CACHE: dict[str, str | None] = {}

_FIELD_MAP = {
    "Open": "openPrice",
    "High": "highPrice",
    "Low": "lowPrice",
    "Close": "closePrice",
}


def _symbol_bases(ticker: str) -> list[str]:
    """눌러 볼 표기 후보 — 보통은 티커 그대로, 클래스 주식은 로이터 표기도 함께.

    `BRK-B` 는 접미사를 아무리 붙여도 안 나오고 `BRKb` 로만 나온다(하이픈 제거 +
    클래스 문자 소문자). `BF-B`·`LEN-B`·`HEI-A` 도 같다.
    """
    bases = [ticker]
    if "-" in ticker:
        head, _, tail = ticker.rpartition("-")
        if head and len(tail) == 1 and tail.isalpha():
            bases.append(f"{head}{tail.lower()}")
    return bases


def resolve_symbol(ticker: str) -> str | None:
    """미국 티커 → 네이버 심볼. 못 찾으면 None(캐시해 재조회하지 않는다)."""
    key = str(ticker or "").strip().upper()
    if not key:
        return None
    if key in _SYMBOL_CACHE:
        return _SYMBOL_CACHE[key]

    for base in _symbol_bases(key):
        for suffix in _SYMBOL_SUFFIXES:
            candidate = f"{base}{suffix}"
            try:
                response = requests.get(f"{_BASE_URL}/{candidate}/basic", timeout=_TIMEOUT, headers=_HEADERS)
            except Exception as exc:
                logger.debug("[네이버 해외] %s 심볼 조회 실패: %s", candidate, exc)
                continue
            if response.status_code == 200:
                _SYMBOL_CACHE[key] = candidate
                return candidate

    _SYMBOL_CACHE[key] = None
    return None


def _to_float(value: Any) -> float | None:
    try:
        number = float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def fetch_daily_ohlc(ticker: str, days: int = 30) -> pd.DataFrame | None:
    """최근 ``days`` 거래일의 일봉. 열은 Open·High·Low·Close (거래량은 주지 않는다).

    가격이 하나라도 빠진 행은 **버린다** — 반쪽짜리 봉을 메우려다 또 반쪽을 넣지 않는다.
    """
    symbol = resolve_symbol(ticker)
    if not symbol:
        return None
    try:
        response = requests.get(
            f"{_BASE_URL}/{symbol}/price",
            params={"pageSize": int(days), "page": 1},
            timeout=_TIMEOUT,
            headers=_HEADERS,
        )
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        logger.debug("[네이버 해외] %s 일봉 조회 실패: %s", symbol, exc)
        return None

    rows: dict[pd.Timestamp, dict[str, float]] = {}
    for item in payload if isinstance(payload, list) else []:
        traded_at = str(item.get("localTradedAt") or "")[:10]
        if not traded_at:
            continue
        values = {column: _to_float(item.get(field)) for column, field in _FIELD_MAP.items()}
        if any(value is None for value in values.values()):
            continue
        rows[pd.Timestamp(traded_at)] = values

    if not rows:
        return None
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()
