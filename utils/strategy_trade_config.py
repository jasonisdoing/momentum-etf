"""전략 사고팔기(분할 매수/매도) 설정 정의 — 단일 소스.

전략 규칙
---------
지수 추종 ETF 6개를 회차별로 나눠 담는 분할 매수/매도 전략이다.
**전략 2개(코스피200 / 코스닥150)를 같은 계좌에서 독립 운용**한다.

- 회차별 배분: 1호~6호가 각각 다른 ETF 를 담당한다(전략별 고정, 아래 정의).
- 자금 배분: 총 투입금을 회차 수로 균등 분할한다.
- 파라미터는 전략당 **``trigger_pct`` 하나**다 — 진입·추가·매도가 같은 간격을 쓴다.
- 1호 진입: 판정 지수(코스피/코스닥)가 **일간** ``trigger_pct`` 이상 하락한 날 매수.
- 2~6호 진입: **직전 회차 종목**이 자기 진입가 대비 ``trigger_pct`` 이상
  하락하면 다음 회차를 매수.
- 매도: 각 회차 종목이 자기 진입가 대비 ``trigger_pct`` 이상 오르면
  그 회차만 개별 매도한다.
- 리셋: 전 회차가 모두 매도되면 다음 거래일부터 1호 진입 대기로 돌아간다.

``trigger_pct`` 는 **전략별로 화면에서 편집**하며
DB(``system_config.strategy_trade_settings.strategies``)가 단일 소스다.
검증은 ``validate_strategy_trade_config`` 한 곳에서만 한다.
회차 티커는 전략별 ``round_tickers`` 코드 고정값이다.
"""

from __future__ import annotations

from typing import Any

# 전략 정의 — id 순서가 화면 표시 순서(상단부터)다.
# round_tickers: 회차 → 담당 ETF(1호부터).
#
# **두 전략의 회차별 운용사를 맞춘다** — 같은 호차면 같은 브랜드라 화면에서 바로 대응된다.
# 순서는 '두 지수 중 작은 쪽 시총' 기준이다. 양쪽 다 거래 가능해야 하므로 작은 쪽이 병목이다.
#   KODEX 52,364 · TIGER 13,837 · RISE 3,309 · ACE 1,558 (억)
#
# 제외한 것
#   KIWOOM 478 · HANARO 213 · PLUS 103 (억) — 5~7호로 쓰던 자리다. 그 깊이까지 내려가는
#     일이 드물고, 시총이 작아 분할 매매에 불리해 4회차까지만 둔다.
#   TR(총수익)형 — 분배금 재투자로 가격 경로가 일반형과 미세하게 다르다.
#   SOL — 코스닥150은 있으나 코스피200은 TR형만 있어 짝이 안 맞는다.
#   IBK — 양쪽 다 있지만 코스피200 거래량이 하루 1,463주 수준이라 분할 매매에 부적합.
# seed_config 는 최초 1회 DB 를 심을 때만 쓴다. 평상시 읽기 경로는 DB 만 본다
# (값이 없으면 시드로 슬쩍 넘어가지 않고 에러를 낸다).
STRATEGIES: dict[str, dict[str, Any]] = {
    "kospi200": {
        "label": "코스피200",
        "index_ticker": "^KS11",
        "index_name": "코스피",
        "round_tickers": (
            ("069500", "KODEX 200"),
            ("102110", "TIGER 200"),
            ("148020", "RISE 200"),
            ("105190", "ACE 200"),
        ),
        "seed_config": {"trigger_pct": 5.0},
    },
    "kosdaq150": {
        "label": "코스닥150",
        # 코스닥150 지수 자체는 가격 소스가 없어(네이버/야후 모두 미제공) 코스닥 종합을 쓴다.
        "index_ticker": "^KQ11",
        "index_name": "코스닥",
        "round_tickers": (
            ("229200", "KODEX 코스닥150"),
            ("232080", "TIGER 코스닥150"),
            ("270810", "RISE 코스닥150"),
            ("354500", "ACE 코스닥150"),
        ),
        "seed_config": {"trigger_pct": 5.0},
    },
}
STRATEGY_IDS: tuple[str, ...] = tuple(STRATEGIES)

MAX_ROUNDS = 4

# 이 전략을 운용하는 계좌. 실제 보유 수량·평균단가를 여기서 읽는다(두 전략 공용).
ACCOUNT_ID = "kor_account"

# 미보유 회차 현재가의 폴백용 가격 캐시(종목풀) 키 — 1순위는 실시간 시세다.
PRICE_CACHE_TICKER_TYPE = "kor_kr"

# 화면에서 편집하는 파라미터 키 — DB 문서에 이 키로 저장된다.
# 진입 하락·추가 하락·매도 상승이 모두 이 한 값을 쓴다(전략당 1개로 관리).
EDITABLE_PCT_KEYS: tuple[str, ...] = ("trigger_pct",)

# 편집 가능 범위 (%) — 화면 입력과 API 가 같은 한계를 쓴다.
PCT_MIN, PCT_MAX = 0.1, 50.0


def validate_strategy_trade_config(values: dict[str, Any]) -> dict[str, float]:
    """편집 파라미터(%)를 검증해 정규화된 dict 를 반환한다. 실패 시 ValueError."""
    if not isinstance(values, dict):
        raise ValueError("설정 형식이 올바르지 않습니다.")

    labels = {"trigger_pct": "매매 간격"}
    cleaned: dict[str, float] = {}
    for key in EDITABLE_PCT_KEYS:
        raw = values.get(key)
        if not isinstance(raw, (int, float)) or isinstance(raw, bool):
            raise ValueError(f"'{labels[key]}' 는 숫자여야 합니다.")
        value = round(float(raw), 2)
        if not PCT_MIN <= value <= PCT_MAX:
            raise ValueError(f"'{labels[key]}' 는 {PCT_MIN}~{PCT_MAX}(%) 사이여야 합니다.")
        cleaned[key] = value
    return cleaned
