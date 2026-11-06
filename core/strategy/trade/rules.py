"""사고팔기(분할 매수·매도) 규칙 — 운용 현황 화면과 백테스트가 함께 쓰는 단일 소스.

한 종목을 매매 간격(%) 하나로 사다리처럼 나눠 산다.

- 1호 진입가 = 직전 거래일 종가 × (1 − 간격%). 보유가 없을 때만 이 기준을 쓴다.
- 다음 회차 매수가 = **마지막 보유 회차의 체결가** × (1 − 간격%). 사다리 선이 아니라 실제 체결가가 기준이다.
- 회차 매도가 = 그 회차의 체결가 × (1 + 간격%). 팔 수 있는 회차는 가장 깊은(마지막) 보유 회차 하나뿐이다.
- 신호는 한 번에 하나만 낸다 — 팔 회차가 있으면 매도가 먼저이고, 매수는 그 매도가 체결된 뒤에 본다.
- 지정가는 ETF 호가 단위(5원)로 반올림한다.
- 전략마다 매수 한도(원)가 있다. 이미 산 회차의 매수금액에 다음 회차 금액을 더해 한도를 넘으면
  그 회차부터는 사지 않는다(`within_budget`).
"""

from __future__ import annotations

TICK_SIZE = 5


def round_to_tick(price: float) -> float:
    """KRX ETF 호가 단위(5원)로 반올림한다 — 지정가는 이 단위로만 주문할 수 있다."""
    return round(price / TICK_SIZE) * TICK_SIZE


def down_ratio(trigger_pct: float) -> float:
    return 1.0 - float(trigger_pct) / 100.0


def up_ratio(trigger_pct: float) -> float:
    return 1.0 + float(trigger_pct) / 100.0


def next_buy_limit(last_fill: float, trigger_pct: float) -> float:
    """다음 회차 매수 지정가 — 마지막 보유 회차 체결가에서 간격%만큼 내린 가격."""
    return round_to_tick(last_fill * down_ratio(trigger_pct))


def projected_buy_limit(last_fill: float, steps: int, trigger_pct: float) -> float:
    """화면에 미리 보여주는 매수 지정가 — 마지막 보유 회차 체결가에서 ``steps`` 번 내린 가격(다음 회차는 1).

    다음 회차 이후는 앞 회차가 실제로 얼마에 체결될지 몰라 간격%를 겹쳐 계산한 예상값이다.
    """
    return round_to_tick(last_fill * down_ratio(trigger_pct) ** steps)


def first_buy_limit(prev_close: float, trigger_pct: float) -> float:
    """보유가 없을 때 1호 매수 지정가 — 직전 거래일 종가 기준."""
    return round_to_tick(prev_close * down_ratio(trigger_pct))


def preview_buy_limit(prev_close: float, round_no: int, trigger_pct: float) -> float:
    """보유가 없을 때 화면에 미리 보여주는 k호 매수 지정가 — 전일 종가에서 k번 내린 가격."""
    return round_to_tick(prev_close * down_ratio(trigger_pct) ** round_no)


def within_budget(invested_cost: float, price: float, quantity: int, budget: float) -> bool:
    """다음 회차를 사도 한도(원)를 넘지 않는지 — 넘으면 그 회차부터 사지 않는다. 매도는 한도와 무관하다."""
    return invested_cost + price * quantity <= budget + 1e-9


def sell_limit(fill: float, trigger_pct: float) -> float:
    """회차 체결가에서 간격%만큼 오른 매도 지정가."""
    return round_to_tick(fill * up_ratio(trigger_pct))
