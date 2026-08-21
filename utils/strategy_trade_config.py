"""전략 사고팔기(분할 매수/매도) 설정 정의 — 단일 소스.

전략 규칙
---------
**전략 4개(KODEX 200 / KODEX 코스닥150 / 삼성전자 / SK하이닉스)를 같은 계좌에서
독립 운용**한다. 각 전략은 자기 종목 **하나만** 설정된 회차 수까지 분할 매수한다.

- 자금 배분: 회차마다 같은 **수량**(``round_quantity``, 전략별 편집)을 산다.
  수량이 같아야 계좌의 (보유 수량, 평균단가)만으로 회차 상태를 역산할 수 있다.
- 파라미터는 전략당 ``trigger_pct``(매매 간격 %) · ``round_quantity``(회차당 주수) ·
  ``rounds``(회차 수)다.
- 1호 진입(보유 0일 때): 종목이 **전일 종가 대비** ``trigger_pct`` 이상 하락하면 매수.
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

(이전 구조 — 코스피200/코스닥150 두 전략을 같은 지수 추종 ETF 4종 회차로 운용 —
는 이 종목별 단일 티커 사다리 구조로 대체됐다.)
"""

from __future__ import annotations

from typing import Any

# 전략 정의 — id 순서가 화면 표시 순서(상단부터)다. 전략당 종목 하나(티커 고정).
# seed_config 는 최초 1회 DB 를 심을 때만 쓴다. 평상시 읽기 경로는 DB 만 본다
# (값이 없으면 시드로 슬쩍 넘어가지 않고 에러를 낸다).
# round_quantity 시드는 회차당 약 550만원 기준 어림값 — 화면에서 계좌에 맞게 조정한다.
STRATEGIES: dict[str, dict[str, Any]] = {
    # is_etf: 호가 단위 결정용 — ETF 는 5원 고정, 주식은 가격대별(50원~1,000원).
    "kospi200": {
        "label": "KODEX 200",
        "ticker": "069500",
        "name": "KODEX 200",
        "is_etf": True,
        "seed_config": {"trigger_pct": 5.0, "round_quantity": 50, "rounds": 4},
    },
    "kosdaq150": {
        "label": "KODEX 코스닥150",
        "ticker": "229200",
        "name": "KODEX 코스닥150",
        "is_etf": True,
        "seed_config": {"trigger_pct": 5.0, "round_quantity": 400, "rounds": 4},
    },
    "samsung": {
        "label": "삼성전자",
        "ticker": "005930",
        "name": "삼성전자",
        "is_etf": False,
        "seed_config": {"trigger_pct": 3.0, "round_quantity": 20, "rounds": 4},
    },
    "hynix": {
        "label": "SK하이닉스",
        "ticker": "000660",
        "name": "SK하이닉스",
        "is_etf": False,
        "seed_config": {"trigger_pct": 3.0, "round_quantity": 3, "rounds": 4},
    },
}
STRATEGY_IDS: tuple[str, ...] = tuple(STRATEGIES)

# 이 전략을 운용하는 계좌. 실제 보유 수량·평균단가를 여기서 읽는다(전 전략 공용).
ACCOUNT_ID = "kor_account"

# 미보유 종목 현재가의 폴백용 가격 캐시(종목풀) 키 — 1순위는 실시간 시세다.
# ETF 는 kor_kr 풀, 삼성전자·SK하이닉스는 kospi200 풀 캐시에 있다.
PRICE_CACHE_TICKER_TYPES: tuple[str, ...] = ("kor_kr", "kospi200")

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
