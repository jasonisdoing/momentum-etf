"""사고팔기 전략의 편집 설정과 회차 매수 수량 계산."""

from __future__ import annotations

from datetime import time
from math import isfinite
from typing import Any

# 전략 정의(풀·종목·계좌·파라미터)는 DB(`system_config.strategy_trade_settings.strategies`)가 단일 소스다 —
# 전략의 종목풀은 가격 출처이며, 대상은 한국 개별주·ETF다.


def is_eligible_pool(pool_config: dict[str, Any]) -> bool:
    """사고팔기를 켤 수 있는 국내 개별주·ETF 종목풀인지 확인한다."""
    return str(pool_config.get("country_code") or "").strip().lower() == "kor" and str(
        pool_config.get("pool_kind") or ""
    ).strip().lower() in {"stock", "etf"}


# 1호 장 마감 매수 시각(KST, 종목 종류별) — 보유 0인데 장중 1호 진입선에 못 닿았으면 이 시각부터 그 종류의
# 마지막 거래 시간이 끝나기 전까지 현재가로 1호를 산다. ETF 는 KRX 정규장 마감(15:30) 전,
# 개별주는 NXT 애프터마켓 마감(20:00) 전이다(`config.INSTRUMENT_TRADING_WINDOWS`).
FIRST_ENTRY_CLOSE_TIMES = {"etf": time(15, 0), "stock": time(19, 30)}

# 화면에서 편집하는 파라미터 키 — DB 문서에 이 키로 저장된다.
EDITABLE_CONFIG_KEYS: tuple[str, ...] = ("trigger_pct", "round_amount_manwon", "rounds")

# 편집 가능 범위 — 화면 입력과 API 가 같은 한계를 쓴다.
PCT_MIN, PCT_MAX = 0.1, 50.0
DEFAULT_ROUND_AMOUNT_MANWON = 100
WON_PER_MANWON = 10_000
AMOUNT_MAX_MANWON = (2**53 - 1) // WON_PER_MANWON
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

    amount = values.get("round_amount_manwon")
    if type(amount) is not int or not 1 <= amount <= AMOUNT_MAX_MANWON:
        raise ValueError("'회차 금액(만원)'은 유효한 양의 정수여야 합니다.")
    cleaned["round_amount_manwon"] = amount

    raw_rounds = values.get("rounds")
    if not isinstance(raw_rounds, (int, float)) or isinstance(raw_rounds, bool) or float(raw_rounds) != int(raw_rounds):
        raise ValueError("'회차 수' 는 정수여야 합니다.")
    rounds = int(raw_rounds)
    if not ROUNDS_MIN <= rounds <= ROUNDS_MAX:
        raise ValueError(f"'회차 수' 는 {ROUNDS_MIN}~{ROUNDS_MAX} 사이여야 합니다.")
    cleaned["rounds"] = rounds

    return cleaned


def round_buy_quantity(amount_manwon: int, price: float) -> int:
    """회차 금액으로 지정가에서 살 수 있는 최대 정수 주수."""
    if not isfinite(price) or price <= 0:
        raise ValueError("회차 매수가는 양수여야 합니다.")
    return int((amount_manwon * WON_PER_MANWON) // price)
