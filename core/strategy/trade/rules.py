"""사고팔기(분할 매수·매도) 규칙 — 운용 현황 화면·슬랙 신호·주문이 함께 쓰는 단일 소스.

한 종목을 매매 간격(%) 하나로 사다리처럼 나눠 산다.

- 1호는 가격 기준 없이, 보유가 없으면 정해진 구간에 현재가로 산다(구간은 `strategy_trade_config.FIRST_ENTRY_WINDOWS`).
- 다음 회차 매수가 = **마지막 보유 회차의 체결가** × (1 − 간격%). 사다리 선이 아니라 실제 체결가가 기준이다.
- 회차 매도가 = 그 회차의 체결가 × (1 + 간격%). 팔 수 있는 회차는 가장 깊은(마지막) 보유 회차 하나뿐이다.
- 신호는 한 번에 하나만 낸다 — 팔 회차가 있으면 매도가 먼저이고, 매수는 그 매도가 체결된 뒤에 본다.
- 지정가는 국가·종목 유형·가격대에 맞는 호가 단위(국장 KRX, 미장 0.01달러)로 반올림한다.
"""

from __future__ import annotations

from math import isfinite, log
from typing import Literal

InstrumentKind = Literal["stock", "etf"]
Country = Literal["kor", "us"]

# KRX 유가증권·코스닥 주권의 가격대별 호가 단위 (상한 미포함).
_STOCK_TICK_BANDS = ((2_000, 1), (5_000, 5), (20_000, 10), (50_000, 50), (200_000, 100), (500_000, 500))
# 미국 호가 단위 — 1달러 이상 0.01달러, 그 미만 0.0001달러(주문 단가는 소수 4자리까지 받는다).
_US_TICK_BANDS = ((1.0, 0.0001),)
_US_TICK_DEFAULT = 0.01
_US_PRICE_DIGITS = 4


def price_digits(country: Country) -> int:
    """가격 표시·저장 소수 자릿수 — 국장은 원 단위 정수, 미장은 달러 소수 둘째 자리(호가 단위 0.0001 은 호가에만 쓴다)."""
    if country not in ("kor", "us"):
        raise ValueError(f"지원하지 않는 국가입니다: {country}")
    return 0 if country == "kor" else 2


def round_price(price: float, country: Country) -> int | float:
    """표시용 가격 반올림 — 국장은 정수, 미장은 소수 둘째 자리."""
    digits = price_digits(country)
    return round(price) if digits == 0 else round(price, digits)


def tick_size(price: float, *, kind: InstrumentKind, country: Country) -> int | float:
    if not isfinite(price) or price <= 0:
        raise ValueError("호가는 유한한 양수여야 합니다.")
    if kind not in ("stock", "etf"):
        raise ValueError(f"지원하지 않는 종목 유형입니다: {kind}")
    if country == "us":
        return next((unit for upper, unit in _US_TICK_BANDS if price < upper), _US_TICK_DEFAULT)
    if country != "kor":
        raise ValueError(f"지원하지 않는 국가입니다: {country}")
    if kind == "etf":
        return 1 if price < 2_000 else 5
    return next((unit for upper, unit in _STOCK_TICK_BANDS if price < upper), 1_000)


def round_to_tick(price: float, *, kind: InstrumentKind, country: Country) -> int | float:
    """전략 기준가를 가장 가까운 유효 호가로 맞춘다 — 국장은 정수, 미장은 소수 4자리."""
    unit = tick_size(price, kind=kind, country=country)
    ticks = max(1, round(price / unit))
    return ticks * unit if country == "kor" else round(ticks * unit, _US_PRICE_DIGITS)


def is_valid_quote(price: int | float, *, kind: InstrumentKind, country: Country) -> bool:
    if isinstance(price, bool) or not isfinite(price) or price <= 0:
        return False
    unit = tick_size(price, kind=kind, country=country)
    if country == "kor":
        return type(price) is int and price % unit == 0
    return abs(price / unit - round(price / unit)) < 1e-6


def down_ratio(trigger_pct: float) -> float:
    return 1.0 - float(trigger_pct) / 100.0


def up_ratio(trigger_pct: float) -> float:
    return 1.0 + float(trigger_pct) / 100.0


def projected_buy_limit(
    last_fill: float, steps: int, trigger_pct: float, *, kind: InstrumentKind, country: Country
) -> int | float:
    """화면에 미리 보여주는 매수 지정가 — 마지막 보유 회차 체결가에서 ``steps`` 번 내린 가격(다음 회차는 1).

    다음 회차 이후는 앞 회차가 실제로 얼마에 체결될지 몰라 간격%를 겹쳐 계산한 예상값이다.
    """
    return round_to_tick(last_fill * down_ratio(trigger_pct) ** steps, kind=kind, country=country)


def sell_limit(fill: float, trigger_pct: float, *, kind: InstrumentKind, country: Country) -> int | float:
    """회차 체결가에서 간격%만큼 오른 매도 지정가."""
    return round_to_tick(fill * up_ratio(trigger_pct), kind=kind, country=country)


def ladder_positions(fills: list[float], trigger_pct: float) -> list[int]:
    """보유 회차들의 사다리 위치(표시용) — 1호 체결가를 위치 1로, 위치 k = 1호 × (1 − 간격%)^(k−1) 중 가장 가까운 칸.

    회차 사이에 건너뛴 칸이 있으면 위치가 벌어진다. 위치는 앞 회차보다 항상 깊다(같거나 얕으면 바로 다음 칸).
    매매 판정(다음 매수·매도·회차 수)에는 쓰지 않는다 — 화면이 간격을 보여 주는 데만 쓴다.
    """
    if not fills:
        return []
    positions: list[int] = []
    for fill in fills:
        position = ladder_position(fill, fills[0], trigger_pct)
        positions.append(max(position, positions[-1] + 1) if positions else 1)
    return positions


def ladder_position(price: float, first_fill: float, trigger_pct: float) -> int:
    """가격의 사다리 위치(표시용) — 1호 체결가를 위치 1로, 1호 × (1 − 간격%)^(k−1) 중 가장 가까운 k.

    1호보다 비싸면 0 이하가 된다(그 위 칸).
    """
    return round(log(price / first_fill) / log(down_ratio(trigger_pct))) + 1
