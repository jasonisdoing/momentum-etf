"""모멘텀 ETF 파라미터 스윕 백테스트 (alpha).

실행 예시::

    python backtest.py kor_kr


BACKTEST_CONFIG 의 스윕 범위를 따라 (TOP_N_HOLD, HOLDING_BONUS_SCORE,
FIRST_MA_TYPE, FIRST_MA_MONTHS, SECOND_MA_TYPE, SECOND_MA_MONTHS) 6개
파라미터 조합을 모두 돌려 BACKTEST_MONTHS 개월 백테스트 결과를 CAGR
내림차순으로 정렬하여 ``ztickers/<order>_<pool>/results/backtest_<YYYY-MM-DD>.log``
파일에 기록한다.

MongoDB 캐시가 반드시 있어야 한다 (로컬 전용).

알파 버전 가정:
    * 점수 역전 시 다음 거래일 시초가(Open)에 체결.
    * 단주 매수, 잔여는 현금 보유.
    * 매도 금액 전액을 대상 종목 목표 비중(1/N)까지 매수하되, 현금 부족 시 잔액만 사용.
    * 수수료/슬리피지/세금 = 0.
    * 상장폐지/거래정지 종목은 풀에 없는 것으로 간주.
    * HOLDING_BONUS_SCORE 는 백테스트 내부에서만 적용 (rankings.py 미변경).
"""

from __future__ import annotations

import sys

from core.strategy.backtest import run_backtest
from utils.env import load_env_if_present

load_env_if_present()


BACKTEST_CONFIG: dict[str, dict] = {
    "kor_kr": {
        "BACKTEST_MONTHS": 12,
        "INITIAL_KRW_AMOUNT": 100_000_000,
        "TOP_N_HOLD": [3, 4, 5, 6, 7],
        "HOLDING_BONUS_SCORE": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "FIRST_MA_TYPE": ["SMA"],
        "FIRST_MA_MONTHS": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "SECOND_MA_TYPE": ["ALMA"],
        "SECOND_MA_MONTHS": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    },
    "kor_us": {
        "BACKTEST_MONTHS": 12,
        "INITIAL_KRW_AMOUNT": 100_000_000,
        "TOP_N_HOLD": [3, 4, 5, 6, 7],
        "HOLDING_BONUS_SCORE": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "FIRST_MA_TYPE": ["SMA"],
        "FIRST_MA_MONTHS": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "SECOND_MA_TYPE": ["ALMA"],
        "SECOND_MA_MONTHS": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    },
    "aus": {
        "BACKTEST_MONTHS": 12,
        "INITIAL_KRW_AMOUNT": 100_000_000,
        "TOP_N_HOLD": [3, 4, 5, 6, 7],
        "HOLDING_BONUS_SCORE": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "FIRST_MA_TYPE": ["SMA"],
        "FIRST_MA_MONTHS": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "SECOND_MA_TYPE": ["ALMA"],
        "SECOND_MA_MONTHS": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    },
    "us": {
        "BACKTEST_MONTHS": 12,
        "INITIAL_KRW_AMOUNT": 100_000_000,
        "TOP_N_HOLD": [3, 4, 5, 6, 7],
        "HOLDING_BONUS_SCORE": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "FIRST_MA_TYPE": ["SMA"],
        "FIRST_MA_MONTHS": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "SECOND_MA_TYPE": ["ALMA"],
        "SECOND_MA_MONTHS": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    },
    "kor": {
        "BACKTEST_MONTHS": 12,
        "INITIAL_KRW_AMOUNT": 100_000_000,
        "TOP_N_HOLD": [3, 4, 5, 6, 7],
        "HOLDING_BONUS_SCORE": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "FIRST_MA_TYPE": ["SMA"],
        "FIRST_MA_MONTHS": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "SECOND_MA_TYPE": ["ALMA"],
        "SECOND_MA_MONTHS": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    },
}


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print("Usage: python backtest.py <pool_id>", file=sys.stderr)
        print(f"사용 가능한 pool_id: {sorted(BACKTEST_CONFIG.keys())}", file=sys.stderr)
        return 2

    pool_id = argv[1].strip().lower()
    run_backtest(pool_id, BACKTEST_CONFIG)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
