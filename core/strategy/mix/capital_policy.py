"""고정 원화 기준금액의 회수·채우기 판정. 화면과 합성 재생이 공유한다."""

from __future__ import annotations

import math
from decimal import Decimal
from typing import Literal, NamedTuple


class CapitalTradeDecision(NamedTuple):
    quantity: int
    reason: Literal["none", "exit", "target_reduction", "harvest", "refill"]


def capital_trade_quantity(
    *,
    held: int,
    price: float,
    target_amount: float,
    harvest_pct: float,
    refill_pct: float,
    previous_target_amount: float,
) -> int:
    """화면 주문 수량 — 판정 사유는 재생 엔진과 같은 함수에서 정한다."""
    return capital_trade_decision(
        held=held,
        price=price,
        target_amount=target_amount,
        harvest_pct=harvest_pct,
        refill_pct=refill_pct,
        previous_target_amount=previous_target_amount,
    ).quantity


def capital_trade_decision(
    *,
    held: int,
    price: float,
    target_amount: float,
    harvest_pct: float,
    refill_pct: float,
    previous_target_amount: float,
) -> CapitalTradeDecision:
    """목표 금액은 가격과 같은 통화. 문턱은 내림 전 금액에 적용하고 주문은 정수로 낸다."""
    if not all(math.isfinite(v) for v in (held, price, target_amount, previous_target_amount, harvest_pct, refill_pct)):
        raise ValueError("회수·채우기 입력은 유한한 숫자여야 합니다.")
    if (
        held < 0
        or price <= 0
        or target_amount < 0
        or previous_target_amount < 0
        or harvest_pct < 0
        or not 0 <= refill_pct <= 100
    ):
        raise ValueError("회수·채우기 입력 범위가 올바르지 않습니다.")
    if target_amount == 0:
        return CapitalTradeDecision(-held, "exit")
    target = math.floor(target_amount / price)
    difference = target - held
    # 청산 여부는 개별 슬리브 신호가 아니라 동일 종목의 합산 기준금액 감소로 판단한다.
    if difference < 0 and target_amount < previous_target_amount:
        return CapitalTradeDecision(difference, "target_reduction")
    # 금액과 비율을 십진수로 비교해 100 × 1.1 같은 경계의 이진 부동소수점 오차를 없앤다.
    value = Decimal(held) * Decimal(str(price))
    basis = Decimal(str(target_amount))
    if difference < 0 and value >= basis * (1 + Decimal(str(harvest_pct)) / 100):
        return CapitalTradeDecision(difference, "harvest")
    # 100은 미보유를 포함해 모든 채우기를 끈다. 0은 작은 부족분도 표시한다.
    if difference > 0 and refill_pct < 100 and value <= basis * (1 - Decimal(str(refill_pct)) / 100):
        return CapitalTradeDecision(difference, "refill")
    return CapitalTradeDecision(0, "none")


def internal_target_weights(*, strategy: str, settings: dict) -> dict[str, float] | float:
    """포트폴리오는 저장 배분, 슬롯 전략은 고정 1/N이며 엔진의 드리프트를 쓰지 않는다."""
    if strategy == "portfolio":
        return {str(row["ticker"]): float(row["weight_pct"]) for row in settings["weights"]}
    count = int(settings["top_n"])
    if count <= 0:
        raise ValueError("전략 슬롯 수는 양수여야 합니다.")
    return 100.0 / count
