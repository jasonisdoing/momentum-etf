"""호주(ASX) 종목 티커 표기 규칙 — 시스템 공통.

시스템 내부에서 유통되는 호주 종목 티커는 `ASX:NAB` 처럼 항상 접두사를 붙인다.
호주 시장에는 미국과 겹치는 티커가 많아(예: IVV) 접두사 없이는 구분되지 않는다.
DB 저장, 화면 표시, 국가 분류 모두 이 표기를 그대로 쓴다.

접두사를 벗기는 것은 **외부로 나갈 때뿐**이다.
- yfinance: `NAB.AX` (`to_yahoo_symbol`)
- BetaShares CSV / Vanguard API / QuoteAPI: `NAB` (`strip_asx_prefix`)
"""

from __future__ import annotations

ASX_PREFIX = "ASX:"
YAHOO_AU_SUFFIX = ".AX"


def normalize_ticker(ticker: object) -> str:
    """앞뒤 공백을 제거하고 대문자로 통일한다."""
    return str(ticker or "").strip().upper()


def is_asx_ticker(ticker: object) -> bool:
    """시스템 표준 표기(`ASX:` 접두사)의 호주 종목인지."""
    return normalize_ticker(ticker).startswith(ASX_PREFIX)


def strip_asx_prefix(ticker: object) -> str:
    """`ASX:NAB` → `NAB`. 접두사가 없으면 그대로 둔다."""
    return normalize_ticker(ticker).removeprefix(ASX_PREFIX)


def ensure_asx_prefix(ticker: object) -> str:
    """호주 종목 티커에 `ASX:` 접두사를 붙인다.

    `NAB` / `NAB.AX` / `ASX:NAB` 어느 형태로 들어와도 `ASX:NAB` 을 돌려준다.
    빈 값은 그대로 빈 문자열이다. 호출자가 이미 호주 종목임을 아는 경우에만 쓴다
    (티커 형식만 보고 호주 여부를 판단하지는 않는다).
    """
    normalized = normalize_ticker(ticker)
    if not normalized:
        return ""
    if normalized.startswith(ASX_PREFIX):
        return normalized
    if normalized.endswith(YAHOO_AU_SUFFIX):
        return f"{ASX_PREFIX}{normalized[: -len(YAHOO_AU_SUFFIX)]}"
    return f"{ASX_PREFIX}{normalized}"


def to_yahoo_symbol(ticker: object) -> str:
    """`ASX:NAB` → `NAB.AX` (yfinance 조회용). 빈 값은 빈 문자열."""
    base = strip_asx_prefix(ticker)
    if not base:
        return ""
    return base if base.endswith(YAHOO_AU_SUFFIX) else f"{base}{YAHOO_AU_SUFFIX}"


def from_yahoo_symbol(symbol: object) -> str | None:
    """`NAB.AX` → `ASX:NAB`. 호주 심볼이 아니면 None (판단은 호출자 몫)."""
    normalized = normalize_ticker(symbol)
    if not normalized.endswith(YAHOO_AU_SUFFIX):
        return None
    return ensure_asx_prefix(normalized)
