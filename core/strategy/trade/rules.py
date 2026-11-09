"""사고팔기(분할 매수·매도) 규칙 — 운용 현황 화면·슬랙 신호·주문이 함께 쓰는 단일 소스.

한 종목을 매매 간격(%) 하나로 사다리처럼 나눠 산다.

- 1호 진입가 = 직전 거래일 종가 × (1 − 간격%). 보유가 없을 때만 이 기준을 쓴다.
- 다음 회차 매수가 = **마지막 보유 회차의 체결가** × (1 − 간격%). 사다리 선이 아니라 실제 체결가가 기준이다.
- 회차 매도가 = 그 회차의 체결가 × (1 + 간격%). 팔 수 있는 회차는 가장 깊은(마지막) 보유 회차 하나뿐이다.
- 신호는 한 번에 하나만 낸다 — 팔 회차가 있으면 매도가 먼저이고, 매수는 그 매도가 체결된 뒤에 본다.
- 지정가는 종목 유형과 가격대에 맞는 KRX 호가 단위로 반올림한다.
"""

from __future__ import annotations

from math import isfinite
from typing import Literal

InstrumentKind = Literal["stock", "etf"]

# KRX 유가증권·코스닥 주권의 가격대별 호가 단위 (상한 미포함).
_STOCK_TICK_BANDS = ((2_000, 1), (5_000, 5), (20_000, 10), (50_000, 50), (200_000, 100), (500_000, 500))


def tick_size(price: float, *, kind: InstrumentKind) -> int:
    if not isfinite(price) or price <= 0:
        raise ValueError("호가는 유한한 양수여야 합니다.")
    if kind == "etf":
        return 1 if price < 2_000 else 5
    if kind != "stock":
        raise ValueError(f"지원하지 않는 종목 유형입니다: {kind}")
    return next((unit for upper, unit in _STOCK_TICK_BANDS if price < upper), 1_000)


def round_to_tick(price: float, *, kind: InstrumentKind) -> int:
    """전략 기준가를 가장 가까운 유효 호가로 맞춘다."""
    unit = tick_size(price, kind=kind)
    return max(unit, round(price / unit) * unit)


def is_valid_quote(price: int, *, kind: InstrumentKind) -> bool:
    return type(price) is int and price > 0 and price % tick_size(price, kind=kind) == 0


def down_ratio(trigger_pct: float) -> float:
    return 1.0 - float(trigger_pct) / 100.0


def up_ratio(trigger_pct: float) -> float:
    return 1.0 + float(trigger_pct) / 100.0


def projected_buy_limit(last_fill: float, steps: int, trigger_pct: float, *, kind: InstrumentKind) -> int:
    """화면에 미리 보여주는 매수 지정가 — 마지막 보유 회차 체결가에서 ``steps`` 번 내린 가격(다음 회차는 1).

    다음 회차 이후는 앞 회차가 실제로 얼마에 체결될지 몰라 간격%를 겹쳐 계산한 예상값이다.
    """
    return round_to_tick(last_fill * down_ratio(trigger_pct) ** steps, kind=kind)


def first_buy_limit(prev_close: float, trigger_pct: float, *, kind: InstrumentKind) -> int:
    """보유가 없을 때 1호 매수 지정가 — 직전 거래일 종가 기준."""
    return round_to_tick(prev_close * down_ratio(trigger_pct), kind=kind)


def preview_buy_limit(prev_close: float, round_no: int, trigger_pct: float, *, kind: InstrumentKind) -> int:
    """보유가 없을 때 화면에 미리 보여주는 k호 매수 지정가 — 전일 종가에서 k번 내린 가격."""
    return round_to_tick(prev_close * down_ratio(trigger_pct) ** round_no, kind=kind)


def sell_limit(fill: float, trigger_pct: float, *, kind: InstrumentKind) -> int:
    """회차 체결가에서 간격%만큼 오른 매도 지정가."""
    return round_to_tick(fill * up_ratio(trigger_pct), kind=kind)
