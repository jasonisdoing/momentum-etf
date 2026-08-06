"""전략 사고팔기(분할 매수/매도) 설정 정의 — 단일 소스.

전략 규칙
---------
코스피200 ETF 6개를 회차별로 나눠 담는 분할 매수/매도 전략이다.

- 회차별 배분: 1호~6호가 각각 다른 ETF 를 담당한다(아래 ``ROUND_TICKERS``).
- 자금 배분: 총 투입금을 회차 수로 균등 분할한다.
- 1호 진입: 코스피 지수가 **일간** ``entry_drop_pct`` 이상 하락한 날 매수.
- 2~6호 진입: **직전 회차 종목**이 자기 진입가 대비 ``add_drop_pct`` 이상
  하락하면 다음 회차를 매수.
- 매도: 각 회차 종목이 자기 진입가 대비 ``take_profit_pct`` 이상 오르면
  그 회차만 개별 매도한다(+5% 든 +10% 든 도달 시 매도).
- 리셋: 전 회차가 모두 매도되면 다음 거래일부터 1호 진입 대기로 돌아간다.

진입 트리거·추가 매수 간격·매도 목표(%)는 **화면에서 편집**하며 DB
(``system_config.strategy_trade_settings``)가 단일 소스다. 검증은
``validate_strategy_trade_config`` 한 곳에서만 한다.
회차 수·티커 목록은 ``ROUND_TICKERS`` 에 묶여 있어 코드 고정값이다.
"""

from __future__ import annotations

from typing import Any

# 회차 → 담당 ETF. 순서가 곧 회차(1호부터)다. 코스피200 추종 6종.
ROUND_TICKERS: tuple[tuple[str, str], ...] = (
    ("069500", "KODEX 200"),
    ("102110", "TIGER 200"),
    ("148020", "RISE 200"),
    ("105190", "ACE 200"),
    ("152100", "PLUS 200"),
    ("293180", "HANARO 200"),
)

# 진입 판정 기준 지수 — 코스피 지수(일간 하락률).
INDEX_TICKER = "^KS11"
INDEX_NAME = "코스피"

MAX_ROUNDS = len(ROUND_TICKERS)

# 회차별 ETF 일봉이 들어 있는 가격 캐시(종목풀) 키.
PRICE_CACHE_TICKER_TYPE = "kor_kr"

# 이 전략을 운용하는 계좌. 실제 보유 수량·평균단가를 여기서 읽는다.
ACCOUNT_ID = "kor_account"

# 화면에서 편집하는 파라미터 키 — DB 문서에 이 키로 저장된다.
EDITABLE_PCT_KEYS: tuple[str, ...] = ("entry_drop_pct", "add_drop_pct", "take_profit_pct")

# 최초 1회 DB 를 심을 때만 쓰는 시드값. 평상시 읽기 경로는 DB 만 본다
# (값이 없으면 이 값으로 슬쩍 넘어가지 않고 에러를 낸다 — 그럴듯한 값이 화면에 떴다가
#  그대로 저장돼 실제 설정을 덮어쓰는 것을 막기 위한 것이다).
SEED_CONFIG: dict[str, float] = {
    "entry_drop_pct": 5.0,
    "add_drop_pct": 5.0,
    "take_profit_pct": 5.0,
}

# 편집 가능 범위 (%) — 화면 입력과 API 가 같은 한계를 쓴다.
PCT_MIN, PCT_MAX = 0.1, 50.0


def validate_strategy_trade_config(values: dict[str, Any]) -> dict[str, float]:
    """편집 파라미터를 검증해 정규화된 dict 를 반환한다. 실패 시 ValueError."""
    if not isinstance(values, dict):
        raise ValueError("설정 형식이 올바르지 않습니다.")

    labels = {
        "entry_drop_pct": "1호 진입 하락률",
        "add_drop_pct": "추가 진입 하락률",
        "take_profit_pct": "매도 목표 상승률",
    }
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


def round_ticker_names() -> dict[str, str]:
    """티커 → 종목명 매핑."""
    return {code: name for code, name in ROUND_TICKERS}
