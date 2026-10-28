"""전략 사고팔기(분할 매수/매도) 설정 정의 — 단일 소스.

전략 규칙
---------
**전략 여러 개를 독립 운용**한다. 전략마다 사고팔기를 켠 종목풀의 ETF 하나를 고르고, 각 전략은
자기 종목 **하나만** 설정된 회차 수까지 분할 매수한다. 운용 계좌와 매수 한도는 전략이 아니라 그
종목풀의 합성 슬리브가 정한다(전략마다 계좌를 따로 고르지 않는다).

- 자금 배분: 회차마다 같은 **수량**(``round_quantity``, 전략별 편집)을 산다.
  수량이 같아야 계좌의 (보유 수량, 평균단가)만으로 회차 상태를 역산할 수 있다.
- 매수 한도: 슬리브 한도(비중 × 합성 기준금액)를 전략 수로 균등 분할한 금액. 이미 산 금액에 다음
  회차 금액을 더해 한도를 넘으면 그 회차부터 사지 않는다(``rules.within_budget``).
- 파라미터는 전략당 ``trigger_pct``(매매 간격 %) · ``round_quantity``(회차당 주수) ·
  ``rounds``(회차 수)다.
- 1호 진입(보유 0일 때): 종목이 **전일 종가 대비** ``trigger_pct`` 이상 하락하면 매수.
  장중에 닿지 못했으면 ``FIRST_ENTRY_CLOSE_TIME`` 부터 정규장 마감 전까지 현재가로 매수한다
  (보유 0인 날을 그대로 넘기지 않는다).
- 2호 이후 진입: **마지막 매수가 대비** ``trigger_pct`` 이상 하락하면 같은 종목을 추가 매수.
  → 진입가 사다리는 1호가 P 에서 P·(1-x%)^(k-1) 로 결정적이다.
- 매도: 각 회차가 **자기 진입가 대비** ``trigger_pct`` 이상 오르면 그 회차 수량만 매도.
  사다리 구조상 항상 마지막(가장 싼) 회차부터 팔린다.
- 리셋: 전량 매도되면 1호 진입 대기로 돌아간다(기준은 다시 전일 종가).

회차 상태 복원 (배치·원장 없음)
------------------------------
회차당 수량이 고정이므로 보유 수량 ÷ 회차 수량 = 진입한 회차 수(m)이고,
평균단가 = 1호가 × (1 + r + … + r^(m-1)) / m (r = 1-간격%) 에서 1호가를 역산한다.
매도가 마지막 회차부터 일어나고 재진입도 같은 사다리 자리에 걸리므로
열린 회차는 항상 1호부터 연속 구간이다 — 계좌만 읽으면 전체 상태가 나온다.
전제: 회차당 수량 고정 + 지정가 체결. (수량이 회차 수량의 배수가 아니면 화면에 경고)

``trigger_pct``·``round_quantity``·``rounds`` 는 화면에서 편집하며
DB(``system_config.strategy_trade_settings.strategies``)가 단일 소스다.
검증은 ``validate_strategy_trade_config`` 한 곳에서만 한다.
"""

from __future__ import annotations

from datetime import time
from typing import Any

# 전략 정의(풀·종목·계좌·파라미터)는 DB(`system_config.strategy_trade_settings.strategies`)가 단일 소스다 —
# 사고팔기를 켠 종목풀에서 그 풀의 ETF 를 골라 전략을 만든다(호가 단위는 ETF 라 5원 고정).


def is_eligible_pool(pool_config: dict[str, Any]) -> bool:
    """사고팔기를 켤 수 있는 종목풀인지 — 국장 전용이라 한국 ETF 풀만 된다(원 단위 호가·한국 장 시간)."""
    return (
        str(pool_config.get("country_code") or "").strip().lower() == "kor"
        and str(pool_config.get("pool_kind") or "").strip().lower() == "etf"
    )


# 1호 장 마감 매수 시각(KST) — 보유 0인데 장중 1호 진입선에 못 닿았으면 이 시각부터 정규장 마감 전까지
# 현재가로 1호를 산다. 국장 전용 전략이라 한국 시간표의 마감과 짝을 이룬다.
FIRST_ENTRY_CLOSE_TIME = time(15, 0)

# 화면에서 편집하는 파라미터 키 — DB 문서에 이 키로 저장된다.
EDITABLE_CONFIG_KEYS: tuple[str, ...] = ("trigger_pct", "round_quantity", "rounds")

# 편집 가능 범위 — 화면 입력과 API 가 같은 한계를 쓴다.
PCT_MIN, PCT_MAX = 0.1, 50.0
QTY_MIN, QTY_MAX = 1, 1_000_000
ROUNDS_MIN, ROUNDS_MAX = 1, 12


def validate_strategy_trade_config(values: dict[str, Any]) -> dict[str, float | int]:
    """편집 파라미터를 검증해 정규화된 dict 를 반환한다. 실패 시 ValueError."""
    if not isinstance(values, dict):
        raise ValueError("설정 형식이 올바르지 않습니다.")

    cleaned: dict[str, float | int] = {}

    raw_pct = values.get("trigger_pct")
    if not isinstance(raw_pct, (int, float)) or isinstance(raw_pct, bool):
        raise ValueError("'매매 간격' 은 숫자여야 합니다.")
    pct = round(float(raw_pct), 2)
    if not PCT_MIN <= pct <= PCT_MAX:
        raise ValueError(f"'매매 간격' 은 {PCT_MIN}~{PCT_MAX}(%) 사이여야 합니다.")
    cleaned["trigger_pct"] = pct

    raw_qty = values.get("round_quantity")
    if not isinstance(raw_qty, (int, float)) or isinstance(raw_qty, bool) or float(raw_qty) != int(raw_qty):
        raise ValueError("'회차당 수량' 은 정수여야 합니다.")
    qty = int(raw_qty)
    if not QTY_MIN <= qty <= QTY_MAX:
        raise ValueError(f"'회차당 수량' 은 {QTY_MIN}~{QTY_MAX:,}(주) 사이여야 합니다.")
    cleaned["round_quantity"] = qty

    raw_rounds = values.get("rounds")
    if not isinstance(raw_rounds, (int, float)) or isinstance(raw_rounds, bool) or float(raw_rounds) != int(raw_rounds):
        raise ValueError("'회차 수' 는 정수여야 합니다.")
    rounds = int(raw_rounds)
    if not ROUNDS_MIN <= rounds <= ROUNDS_MAX:
        raise ValueError(f"'회차 수' 는 {ROUNDS_MIN}~{ROUNDS_MAX} 사이여야 합니다.")
    cleaned["rounds"] = rounds

    return cleaned
