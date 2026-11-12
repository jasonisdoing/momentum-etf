"""사고팔기 전략의 편집 설정과 회차 매수 수량 계산."""

from __future__ import annotations

from datetime import datetime, time, timedelta
from math import isfinite
from typing import Any

# 전략 정의(풀·종목·계좌·파라미터)는 DB(`system_config.strategy_trade_settings.strategies`)가 단일 소스다 —
# 전략의 종목풀은 가격 출처이며, 대상은 계좌 국가(`TRADE_COUNTRIES`)의 개별주·ETF다.


#: 사고팔기를 운영할 수 있는 계좌·종목풀 국가 — 계좌 하나는 국가 하나이고 그 국가의 풀만 쓴다.
TRADE_COUNTRIES: tuple[str, ...] = ("kor", "us")

#: 국가별 거래 통화 — 매수에 쓰는 예수금의 통화다.
TRADE_CURRENCIES = {"kor": "KRW", "us": "USD"}


def is_eligible_pool(pool_config: dict[str, Any], countries: tuple[str, ...]) -> bool:
    """``countries`` 의 개별주·ETF 종목풀인지 확인한다."""
    return str(pool_config.get("country_code") or "").strip().lower() in countries and str(
        pool_config.get("pool_kind") or ""
    ).strip().lower() in {"stock", "etf"}


#: 1호 매수 시각 — 본장 개장 뒤·마감 전 이 시간(분). 국장·미장, 개별주·ETF 모두 같다.
FIRST_ENTRY_OFFSET_MINUTES = 20


def _first_entry_windows(country: str) -> tuple[tuple[time, time], ...]:
    """그 시장 본장 기준 1호 매수 구간(현지 시각) — 개장 20분 뒤부터 20분, 마감 20분 전부터 마감까지.

    미장 본장은 서머타임에 따라 한국시간이 바뀌어 현지 시각으로 두고, 화면·슬랙에는 한국시간으로 바꿔 보인다
    (미장은 한국시간 겨울 23:50·05:40, 서머타임 22:50·04:40). 그 시각에 실제로 거래 중인지는
    `config.INSTRUMENT_TRADING_WINDOWS` 가 판정한다.
    """
    from config import MARKET_SCHEDULES

    day = datetime(2000, 1, 1)
    offset = timedelta(minutes=FIRST_ENTRY_OFFSET_MINUTES)
    session_open = datetime.combine(day, MARKET_SCHEDULES[country]["open"])
    session_close = datetime.combine(day, MARKET_SCHEDULES[country]["close"])
    return (
        ((session_open + offset).time(), (session_open + 2 * offset).time()),
        ((session_close - offset).time(), session_close.time()),
    )


#: 1호 매수 구간(그 시장 현지 시각, (시작, 끝) 목록) — 보유 0인데 이 구간 안이면 가격과 상관없이 현재가로 1호를 산다.
#: 구간마다 보유가 없을 때만이라 앞 구간에서 샀으면 다음 구간은 건너뛴다.
FIRST_ENTRY_WINDOWS: dict[str, tuple[tuple[time, time], ...]] = {
    country: _first_entry_windows(country) for country in TRADE_COUNTRIES
}

# 화면에서 편집하는 파라미터 키 — DB 문서에 이 키로 저장된다.
EDITABLE_CONFIG_KEYS: tuple[str, ...] = ("trigger_pct", "round_size", "rounds")

# 편집 가능 범위 — 화면 입력과 API 가 같은 한계를 쓴다.
PCT_MIN, PCT_MAX = 0.1, 50.0
# 회차 크기(`round_size`) — 국가와 무관하게 원화 금액(만원) 정수로 저장한다. 미장은 환율로 달러 수량을 환산한다.
ROUND_SIZE_UNIT_KRW = 10_000
DEFAULT_ROUND_SIZE = 100
ROUND_SIZE_MAX = (2**53 - 1) // ROUND_SIZE_UNIT_KRW
ROUNDS_MIN, ROUNDS_MAX = 1, 12


def validate_strategy_trade_config(values: dict[str, Any]) -> dict[str, float | int]:
    """편집 파라미터를 검증해 정규화된 dict 를 반환한다. 실패 시 ValueError."""
    if not isinstance(values, dict):
        raise ValueError("설정 형식이 올바르지 않습니다.")

    cleaned: dict[str, float | int] = {}

    raw_pct = values.get("trigger_pct")
    if not isinstance(raw_pct, (int, float)) or isinstance(raw_pct, bool):
        raise ValueError("'매매 간격' 은 숫자여야 합니다.")
    pct = round(float(raw_pct), 2)
    if not PCT_MIN <= pct <= PCT_MAX:
        raise ValueError(f"'매매 간격' 은 {PCT_MIN}~{PCT_MAX}(%) 사이여야 합니다.")
    cleaned["trigger_pct"] = pct

    size = values.get("round_size")
    if type(size) is not int or not 1 <= size <= ROUND_SIZE_MAX:
        raise ValueError("'회차 금액·수량'은 유효한 양의 정수여야 합니다.")
    cleaned["round_size"] = size

    raw_rounds = values.get("rounds")
    if not isinstance(raw_rounds, (int, float)) or isinstance(raw_rounds, bool) or float(raw_rounds) != int(raw_rounds):
        raise ValueError("'회차 수' 는 정수여야 합니다.")
    rounds = int(raw_rounds)
    if not ROUNDS_MIN <= rounds <= ROUNDS_MAX:
        raise ValueError(f"'회차 수' 는 {ROUNDS_MIN}~{ROUNDS_MAX} 사이여야 합니다.")
    cleaned["rounds"] = rounds

    return cleaned


def round_buy_quantity(size: int, price: float, krw_rate: float) -> int:
    """회차 매수 수량 — 회차 금액(만원)으로 지정가에서 살 수 있는 최대 정수 주수. `krw_rate` 는 종목 통화→원 환율(원화 1.0)."""
    if not isfinite(price) or price <= 0 or not isfinite(krw_rate) or krw_rate <= 0:
        raise ValueError("회차 매수가와 환율은 양수여야 합니다.")
    return int((size * ROUND_SIZE_UNIT_KRW) // (price * krw_rate))
