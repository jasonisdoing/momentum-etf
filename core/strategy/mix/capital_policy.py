"""고정 원화 기준금액의 회수·채우기 판정. 화면과 합성 재생이 공유한다."""

from __future__ import annotations

import math


def capital_trade_quantity(
    *, held: int, price: float, target_amount: float, harvest_pct: float, refill_pct: float, force_exit: bool
) -> int:
    """목표 금액은 가격과 같은 통화. 문턱은 내림 전 금액에 적용하고 주문은 정수로 낸다."""
    if not all(math.isfinite(v) for v in (held, price, target_amount, harvest_pct, refill_pct)):
        raise ValueError("회수·채우기 입력은 유한한 숫자여야 합니다.")
    if held < 0 or price <= 0 or target_amount < 0 or harvest_pct < 0 or not 0 <= refill_pct <= 100:
        raise ValueError("회수·채우기 입력 범위가 올바르지 않습니다.")
    if force_exit or target_amount == 0:
        return -held
    target = math.floor(target_amount / price)
    difference = target - held
    value = held * price
    if difference < 0 and value >= target_amount * (1 + harvest_pct / 100):
        return difference
    # 100은 미보유를 포함해 모든 채우기를 끈다. 0은 작은 부족분도 표시한다.
    if difference > 0 and refill_pct < 100 and value <= target_amount * (1 - refill_pct / 100):
        return difference
    return 0


def internal_target_weights(*, strategy: str, settings: dict) -> dict[str, float] | float:
    """포트폴리오는 저장 배분, 슬롯 전략은 고정 1/N이며 엔진의 드리프트를 쓰지 않는다."""
    if strategy == "portfolio":
        return {str(row["ticker"]): float(row["weight_pct"]) for row in settings["weights"]}
    count = int(settings["top_n"])
    if count <= 0:
        raise ValueError("전략 슬롯 수는 양수여야 합니다.")
    return 100.0 / count
