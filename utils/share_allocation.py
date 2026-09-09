"""목표 금액을 정수 주수로 환산한다. 실제 보유는 목표 배분에 사용하지 않는다."""

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


def allocate_integer_shares(targets: list[ShareTarget], budget: float) -> dict[str, int]:
    """목표 금액을 정수 주수로 배분한다 — **종목별 내림만**, 단주 잔여는 현금으로 남긴다.

    예전에는 내림 후 남는 예산을 소수부가 큰 종목부터 한 주씩 더 줬다(최대잉여법).
    그 +1주가 어느 종목에 가는지는 장중 가격 미세 변동에 아주 민감해서, 합성 오늘의
    액션이 A −1주 / B +1주 식으로 널뛰고 알림이 반복됐다 — 2026-09 제거. 남는 단주
    잔여(종목당 1주 가격 미만)는 현금으로 남아 다음 재배분·진입 예산에 다시 포함된다.

    Args:
        targets: 종목별 목표 금액·1주 값.
        budget: 주식에 쓸 총액(= 총자산 × 주식 목표비중). **보유 현금이 아니다** —
            초과 보유분을 팔면 그 대금이 부족분 매수에 쓰이므로, 지금 현금이 얼마인지는
            목표 수량을 정하는 데 쓰지 않는다. 내림 수량만으로 이 예산을 넘는 입력은
            목표 비중이 잘못됐다는 뜻이라 오류다.

    Returns:
        {식별자: 주수}. 배정이 0 인 종목도 키는 있다.
    """
    if not isfinite(budget) or budget < 0:
        raise ValueError("주수 배분 예산은 0 이상의 유한한 금액이어야 합니다.")
    quantities = {item.key: 0 for item in targets}
    floor_costs: list[float] = []
    for item in targets:
        if item.price <= 0 or item.target_amount <= 0:
            continue
        floored = int(item.target_amount // item.price)
        floor_costs.append(floored * item.price)
        quantities[item.key] = floored

    if fsum(floor_costs) > budget:
        raise ValueError("목표 내림 수량만으로 배분 예산을 초과합니다. 목표 비중과 배정 예산을 확인하세요.")

    return quantities
