"""자산 화면 전용 정수 수량 배분. 합성 전략의 배분 규칙과 독립적으로 유지한다."""

from dataclasses import dataclass
from math import floor, fsum, isfinite


@dataclass(frozen=True)
class AssetShareTarget:
    key: str
    target_amount: float
    price: float


def allocate_asset_shares(targets: list[AssetShareTarget], budget: float) -> dict[str, int]:
    """내림 후 최대 잉여 순으로 채운다. 비싼 종목을 못 사면 살 수 있는 종목을 계속 찾는다.

    한 번씩 올린 뒤에도 예산이 남으면 목표 주수 대비 부족분 순으로 추가한다.
    살 수 있는 종목이 없어질 때 종료하며, 예산에 포함되지 않은 목표 현금은 쓰지 않는다.
    """
    if not isfinite(budget) or budget < 0:
        raise ValueError("자산 배분 예산은 0 이상의 유한한 금액이어야 합니다.")
    if len({item.key for item in targets}) != len(targets):
        raise ValueError("자산 배분 대상에 중복 종목이 있습니다.")
    for item in targets:
        if not isfinite(item.price) or item.price <= 0 or not isfinite(item.target_amount) or item.target_amount < 0:
            raise ValueError(f"자산 배분 가격·목표금액이 올바르지 않습니다: {item.key}")
    active = [item for item in targets if item.target_amount > 0]
    quantities = {item.key: floor(item.target_amount / item.price) for item in targets}
    remaining = budget - fsum(quantities[item.key] * item.price for item in targets)
    if remaining < 0:
        raise ValueError("목표 내림 수량이 자산 배분 예산을 초과합니다. 목표비중과 현금비중을 확인하세요.")
    while True:
        affordable = [item for item in active if item.price <= remaining]
        if not affordable:
            break
        selected = min(affordable, key=lambda item: (quantities[item.key] - item.target_amount / item.price, item.key))
        # 한 종목만 살 수 있을 때는 반복 없이 남은 예산으로 가능한 주수를 배정한다.
        count = floor(remaining / selected.price) if len(affordable) == 1 else 1
        quantities[selected.key] += count
        remaining -= count * selected.price
    return quantities
