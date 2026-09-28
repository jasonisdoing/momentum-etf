"""고정 원화 기준금액의 목표 주수 배분과 회수·채우기 판정. 화면과 합성 재생이 공유한다."""

from __future__ import annotations

import math
from decimal import Decimal
from typing import Literal, NamedTuple


class CapitalTradeDecision(NamedTuple):
    quantity: int
    reason: Literal["none", "exit", "target_reduction", "harvest", "refill"]


def allocate_target_shares(amounts: dict[str, float], prices: dict[str, float], budget: float) -> dict[str, int]:
    """비싼 종목부터 첫 1주를 확보하고 남은 예산을 목표 금액에 맞춰 배분한다."""
    if not math.isfinite(budget) or budget < 0:
        raise ValueError("주수 배분 예산이 올바르지 않습니다.")
    for ticker, amount in amounts.items():
        price = prices.get(ticker)
        if price is None or not all(math.isfinite(value) for value in (amount, price)) or amount < 0 or price <= 0:
            raise ValueError(f"{ticker}: 목표 금액 또는 가격이 올바르지 않습니다.")
    quantities = {ticker: 0 for ticker in amounts}
    remaining = budget
    selected = []
    for ticker in sorted(amounts, key=lambda key: (-prices[key], key)):
        if amounts[ticker] <= 0 or prices[ticker] > remaining + 1e-9:
            continue
        quantities[ticker] = 1
        remaining -= prices[ticker]
        selected.append(ticker)
    if not selected:
        return quantities

    limits = {ticker: max(1, math.floor(amounts[ticker] / prices[ticker])) for ticker in selected}
    extra_cost = math.fsum((limits[ticker] - 1) * prices[ticker] for ticker in selected)
    if extra_cost <= remaining + 1e-9:
        quantities.update(limits)
        return quantities

    low, high = 0.0, 1.0
    for _ in range(48):
        scale = (low + high) / 2
        cost = math.fsum(
            (max(1, min(limits[ticker], math.floor(scale * amounts[ticker] / prices[ticker]))) - 1) * prices[ticker]
            for ticker in selected
        )
        if cost <= remaining:
            low = scale
        else:
            high = scale
    for ticker in selected:
        quantities[ticker] = max(1, min(limits[ticker], math.floor(low * amounts[ticker] / prices[ticker])))
    remaining = budget - math.fsum(quantities[ticker] * prices[ticker] for ticker in selected)
    while True:
        candidates = [
            ticker for ticker in selected if quantities[ticker] < limits[ticker] and prices[ticker] <= remaining + 1e-9
        ]
        if not candidates:
            break
        ticker = min(candidates, key=lambda key: (quantities[key] * prices[key] / amounts[key], -prices[key], key))
        quantities[ticker] += 1
        remaining -= prices[ticker]
    return quantities


def capital_trade_quantity(
    *,
    held: int,
    price: float,
    target_amount: float,
    harvest_pct: float,
    refill_pct: float,
    previous_target_amount: float,
    target_quantity: int,
) -> int:
    """화면 주문 수량 — 판정 사유는 재생 엔진과 같은 함수에서 정한다."""
    return capital_trade_decision(
        held=held,
        price=price,
        target_amount=target_amount,
        harvest_pct=harvest_pct,
        refill_pct=refill_pct,
        previous_target_amount=previous_target_amount,
        target_quantity=target_quantity,
    ).quantity


def capital_trade_decision(
    *,
    held: int,
    price: float,
    target_amount: float,
    harvest_pct: float,
    refill_pct: float,
    previous_target_amount: float,
    target_quantity: int,
) -> CapitalTradeDecision:
    """목표 금액은 가격과 같은 통화. 문턱은 원래 금액, 주문은 공통 배분 수량을 쓴다."""
    if not all(math.isfinite(v) for v in (held, price, target_amount, previous_target_amount, harvest_pct, refill_pct)):
        raise ValueError("회수·채우기 입력은 유한한 숫자여야 합니다.")
    if (
        held < 0
        or price <= 0
        or target_amount < 0
        or previous_target_amount < 0
        or harvest_pct < 0
        or not 0 <= refill_pct <= 100
        or target_quantity < 0
    ):
        raise ValueError("회수·채우기 입력 범위가 올바르지 않습니다.")
    if target_amount == 0:
        return CapitalTradeDecision(-held, "exit")
    difference = target_quantity - held
    # 청산 여부는 개별 슬리브 신호가 아니라 동일 종목의 합산 기준금액 감소로 판단한다.
    if difference < 0 and (target_amount < previous_target_amount or target_quantity == 0):
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
