"""목표 금액을 **정수 주수**로 배분한다 — 합성 운용 현황·계좌 상세·자산 도우미 공용.

주식은 소수점으로 못 산다. 목표 비중을 그대로 주수로 옮기면 `4.503주` 같은 값이 나온다.
규칙은 세 단계다.

  1. **내림.** 목표를 넘겨 사지 않는다. 못 채운 나머지는 현금으로 둔다.
  2. **유지.** 보유가 딱 「내림+1」이면 그 값을 그대로 목표로 둔다. 3 단계가 어제 그 종목에
     한 주를 얹어 준 결과인데, 소수부는 시세 따라 계속 움직여 순위가 뒤집힌다. 매번 다시
     세우면 어제 산 것을 오늘 팔라고 하게 된다. 그보다 많이 들고 있으면(손으로 산 10주 같은)
     내림까지 팔라는 지시가 그대로 난다. 단, 예산을 넘으면 소수부가 작은 종목부터
     유지한 추가 1주를 취소한다 — 목표 대비 올림 폭이 가장 큰 것부터 줄인다.
  3. **최대잉여법.** 남는 돈으로 부족분(소수부)이 큰 종목부터 한 주씩 채운다. 내림만 하면
     1주 값이 비싼 종목에서 큰 돈이 논다(VOO 는 3.98주가 3주가 되어 총자산의 5% 가 현금으로
     남았다). 살 수 없는 종목이 나오면 **거기서 멈춘다** — 건너뛰고 더 싼 종목을 사면 부족분과
     상관없이 싼 것만 담긴다(부족분 0.045 인 BMY 를 사려고 0.601 인 DELL 을 건너뛰는 식).

예전에는 예산이 남는 한 「목표 대비 가장 덜 채워진」 종목에 계속 한 주씩 주는 방식이었다.
그러면 한 종목이 자기 목표를 크게 넘긴다 — DELL 이 1.6주 목표에 2주(125%)를 받고, 그 돈
때문에 BMY 가 13주 목표에 11주로 깎였다.

균등 슬롯(모멘텀·신고가 진입)은 여기를 쓰지 않는다. 슬롯마다 몫이 정해져 있어 슬롯별 내림이
맞고, 그 규칙은 `utils/slot_backtest.py` 안에 있다.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum, isfinite

from config import BACKTEST_INITIAL_CAPITAL


def backtest_initial_capital(pool: str) -> float:
    """종목풀의 백테스트 시작 자본(그 시장 통화). 통화를 모르면 에러 — 임의 기본값을 쓰지 않는다.

    백테스트가 정수 주수로 돌려면 "1주"를 셀 기준 금액이 있어야 한다. 계좌 잔고가 아니라
    통화별 상수를 쓰는 이유는 `config.BACKTEST_INITIAL_CAPITAL` 주석에 적어 두었다.
    """
    from utils.cash_model import currency_for_country
    from utils.settings_loader import get_ticker_type_settings

    country = str((get_ticker_type_settings(pool) or {}).get("country_code") or "").strip().lower()
    currency = currency_for_country(country)
    if currency not in BACKTEST_INITIAL_CAPITAL:
        raise ValueError(f"'{pool}'({currency}) 의 백테스트 시작 자본이 config 에 없습니다.")
    return float(BACKTEST_INITIAL_CAPITAL[currency])


@dataclass(frozen=True)
class ShareTarget:
    """배분 대상 한 종목."""

    key: str
    """식별자(티커 등)."""
    target_amount: float
    """목표 금액. 예산과 같은 통화여야 한다."""
    price: float
    """1주 값. 목표 금액과 같은 통화여야 한다(환율은 호출부에서 맞춘다)."""
    held: int = 0
    """지금 계좌에 있는 주수. 「내림+1」이면 그대로 유지한다(2 단계). 모르면 0."""


def allocate_integer_shares(targets: list[ShareTarget], budget: float) -> dict[str, int]:
    """목표 금액을 정수 주수로 배분한다. 총 배정액은 ``budget`` 을 넘지 않는다.

    Args:
        targets: 종목별 목표 금액·1주 값·현재 보유 주수.
        budget: 주식에 쓸 총액(= 총자산 × 주식 목표비중). **보유 현금이 아니다** —
            초과 보유분을 팔면 그 대금이 부족분 매수에 쓰이므로, 지금 현금이 얼마인지는
            목표 수량을 정하는 데 쓰지 않는다.

    Returns:
        {식별자: 주수}. 배정이 0 인 종목도 키는 있다.
    """
    if not isfinite(budget) or budget < 0:
        raise ValueError("주수 배분 예산은 0 이상의 유한한 금액이어야 합니다.")
    quantities = {item.key: 0 for item in targets}
    # (부족분, 식별자, 1주 값) — 3 단계에서 부족분이 큰 순서로 한 주씩 준다.
    remainders: list[tuple[float, str, float]] = []
    retained: list[tuple[float, str, float]] = []
    floor_costs: list[float] = []
    for item in targets:
        if item.price <= 0 or item.target_amount <= 0:
            continue
        floored = int(item.target_amount // item.price)
        floor_costs.append(floored * item.price)
        if item.held == floored + 1:
            quantities[item.key] = item.held  # 2. 유지
            retained.append((item.target_amount / item.price - floored, item.key, item.price))
        else:
            quantities[item.key] = floored  # 1. 내림
            remainders.append((item.target_amount / item.price - floored, item.key, item.price))

    if fsum(floor_costs) > budget:
        raise ValueError("목표 내림 수량만으로 배분 예산을 초과합니다. 목표 비중과 배정 예산을 확인하세요.")

    def allocated_amount() -> float:
        """누적 차감 오차 없이 현재 목표 수량의 총액을 계산한다."""
        return fsum(quantities[item.key] * item.price for item in targets if quantities[item.key])

    # 보유 유지보다 예산이 우선이다. 내림 몫은 지키고, 유지한 추가 주수만 줄인다.
    spent = allocated_amount()
    for _, key, _ in sorted(retained):
        if spent <= budget:
            break
        quantities[key] -= 1
        spent = allocated_amount()

    # 3. 최대잉여법 — 못 사는 종목이 나오면 거기서 멈춘다.
    for shortfall, key, price in sorted(remainders, reverse=True):
        del shortfall
        quantities[key] += 1
        if allocated_amount() > budget:
            quantities[key] -= 1
            break

    return quantities
