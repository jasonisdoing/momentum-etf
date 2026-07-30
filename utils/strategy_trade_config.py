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

파라미터(진입 트리거/추가 매수 간격/매도 목표/최대 회차/티커 목록)는 화면에서
편집하며, 검증은 ``validate_strategy_trade_config`` 한 곳에서만 한다.
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

# 전략 파라미터 — 화면에서 편집하지 않는 고정값.
DEFAULT_CONFIG: dict[str, Any] = {
    "entry_drop_pct": 5.0,
    "add_drop_pct": 5.0,
    "take_profit_pct": 5.0,
    "rounds": MAX_ROUNDS,
    "tickers": [code for code, _ in ROUND_TICKERS],
}


def round_ticker_names() -> dict[str, str]:
    """티커 → 종목명 매핑."""
    return {code: name for code, name in ROUND_TICKERS}
