"""목표 금액을 **정수 주수**로 배분한다 — 운용 현황·백테스트·튜닝 공용.

주식은 소수점으로 못 산다. 목표 비중을 그대로 주수로 옮기면 `4.503주` 같은 값이 나오는데,
이걸 종목마다 따로 반올림하면 두 가지가 깨진다.

  · 위로 반올림한 종목이 제 목표 금액보다 더 쓴다 → 총 매수액이 예산을 넘어 못 산다
    (실제로 "현금 $172 인데 $238 짜리를 사라"는 지시가 나왔다)
  · 아래로 반올림한 몫이 현금으로 남아 놀아도 아무도 다시 쓰지 않는다

그래서 **순차 배분**(의석 배분의 divisor method, D'Hondt 계열)을 쓴다. 몫을 미리 확정해
나누는 게 아니라 **한 주씩** 준다. 매 회차 "1주당 부족분이 가장 큰" 종목을 골라 한 주를
배정하고, 남은 예산으로 살 수 없는 종목은 후보에서 빠진다.

  · 1주 값이 종목마다 다른 문제가 규칙 안에 들어간다 — 최대잉여법처럼 "잉여 순번은 높은데
    돈이 모자란다"를 예외로 처리할 필요가 없다.
  · 총 배정액이 예산을 절대 넘지 않는다(살 수 있을 때만 사므로).
  · 남는 예산은 어떤 종목의 1주 값보다도 작을 때뿐이다 → 현금 비중이 목표에 최대한 붙는다.
"""

from __future__ import annotations

from dataclasses import dataclass

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


def allocate_integer_shares(targets: list[ShareTarget], budget: float) -> dict[str, int]:
    """목표 금액을 정수 주수로 배분한다. 총 배정액은 ``budget`` 을 넘지 않는다.

    Args:
        targets: 종목별 목표 금액과 1주 값.
        budget: 주식에 쓸 총액(= 총자산 × 주식 목표비중). **보유 현금이 아니다** —
            초과 보유분을 팔면 그 대금이 부족분 매수에 쓰이므로, 지금 현금이 얼마인지는
            목표 수량을 정하는 데 쓰지 않는다.

    Returns:
        {식별자: 주수}. 배정이 0 인 종목도 키는 있다.
    """
    quantities = {item.key: 0 for item in targets}
    usable = [item for item in targets if item.price > 0 and item.target_amount > 0]
    remaining = float(budget)

    while True:
        # 남은 예산으로 살 수 있고, 아직 목표 금액에 못 미친 종목만 후보.
        best: ShareTarget | None = None
        best_score = 0.0
        for item in usable:
            if item.price > remaining:
                continue
            # 1주당 부족분 — 금액이 아니라 비율로 봐야 비싼 종목이 불리해지지 않는다.
            score = (item.target_amount - quantities[item.key] * item.price) / item.price
            if score > best_score:
                best, best_score = item, score
        if best is None:
            break
        quantities[best.key] += 1
        remaining -= best.price

    return quantities
