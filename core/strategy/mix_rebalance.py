"""합성 월초 재배분 — 전략 내부 주식·현금 비율을 유지하는 공용 계산."""

from math import fsum, isfinite


def rebalance_sleeves(
    values: dict[str, float],
    reserved_cash: float,
    weights: dict[str, float],
    stock_ratios: dict[str, float],
    slippage: dict[str, tuple[float, float]],
) -> tuple[dict[str, float], float, float]:
    """비용 차감 후 목표 배분으로 이동한다. 반환은 슬리브 금액·유보 현금·비용이다.

    비율·슬리피지는 0~1 단위. 전략 내부 현금 이동은 무비용이고 주식 증감분에만
    매수·매도 슬리피지를 적용한다. 실제 계좌 보유는 사용하지 않는다.
    """
    keys = set(values)
    if set(stock_ratios) != keys or set(slippage) != keys or set(weights) != keys | {"cash"}:
        raise ValueError("합성 재배분의 슬리브 키가 일치하지 않습니다.")
    if any(not isfinite(v) or v < 0 for v in [*values.values(), reserved_cash, *weights.values()]):
        raise ValueError("합성 재배분 금액·비율은 유한한 음이 아닌 값이어야 합니다.")
    if abs(fsum(weights.values()) - 1) > 1e-9:
        raise ValueError("합성 목표 배분 합계가 100%가 아닙니다.")
    for key in keys:
        if not 0 <= stock_ratios[key] <= 1 or any(not 0 <= rate < 1 for rate in slippage[key]):
            raise ValueError("주식 비율 또는 슬리피지가 올바르지 않습니다.")
    total = fsum([*values.values(), reserved_cash])
    if total <= 0:
        raise ValueError("합성 재배분 총액이 0입니다.")

    def cost(net: float) -> float:
        costs = []
        for key, value in values.items():
            diff = net * weights[key] - value
            rate = slippage[key][0 if diff > 0 else 1]
            costs.append(abs(diff) * stock_ratios[key] * rate)
        return fsum(costs)

    # 목표 자체가 비용 차감 후 총액에 비례하므로 자금 보존식을 이분법으로 푼다.
    low, high = 0.0, total
    for _ in range(80):
        net = (low + high) / 2
        if net + cost(net) > total:
            high = net
        else:
            low = net
    net = low
    return {key: net * weights[key] for key in values}, net * weights["cash"], total - net
