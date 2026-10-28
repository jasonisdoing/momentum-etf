"""사고팔기(분할 매수·매도) 규칙 — 운용 현황 화면과 백테스트가 함께 쓰는 단일 소스.

한 종목을 매매 간격(%) 하나로 사다리처럼 나눠 산다.

- 1호 진입가 = 직전 거래일 종가 × (1 − 간격%). 보유가 없을 때만 이 기준을 쓴다.
- k호 진입가 = 1호 체결가 × (1 − 간격%)^(k−1).
- k호 매도가 = k호 진입가 × (1 + 간격%).
- 지정가는 ETF 호가 단위(5원)로 반올림한다.

계좌 보유에서 회차를 거꾸로 맞추는 계산(`first_entry_from_average`)도 여기 둔다 — 평균단가는
회차가 1호부터 연속으로 찬 사다리의 평균이라는 전제에서만 1호가로 되돌려진다.
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


def first_entry_from_average(avg_price: float, held_rounds: int, trigger_pct: float) -> float:
    """보유 평균단가와 보유 회차 수에서 1호 진입가를 되돌린다 (m 회차 평균 = P1 × Σ r^i ÷ m)."""
    ratio = down_ratio(trigger_pct)
    ladder_sum = sum(ratio**i for i in range(held_rounds))
    return avg_price * held_rounds / ladder_sum


def entry_price(first_entry: float, round_no: int, trigger_pct: float) -> float:
    """k호 진입가(공칭값, 호가 반올림 전)."""
    return first_entry * down_ratio(trigger_pct) ** (round_no - 1)


def buy_limit(first_entry: float, round_no: int, trigger_pct: float) -> float:
    """1호가를 알 때 k호 매수 지정가."""
    return round_to_tick(entry_price(first_entry, round_no, trigger_pct))


def first_buy_limit(prev_close: float, trigger_pct: float) -> float:
    """보유가 없을 때 1호 매수 지정가 — 직전 거래일 종가 기준."""
    return round_to_tick(prev_close * down_ratio(trigger_pct))


def preview_buy_limit(prev_close: float, round_no: int, trigger_pct: float) -> float:
    """보유가 없을 때 화면에 미리 보여주는 k호 매수 지정가 — 전일 종가에서 k번 내린 가격."""
    return round_to_tick(prev_close * down_ratio(trigger_pct) ** round_no)


def sell_limit(entry: float, trigger_pct: float) -> float:
    """진입가에서 간격%만큼 오른 매도 지정가."""
    return round_to_tick(entry * up_ratio(trigger_pct))
