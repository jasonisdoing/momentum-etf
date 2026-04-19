"""백테스트 파라미터 스윕 설정."""

from __future__ import annotations


BACKTEST_CONFIG: dict[str, dict] = {
    "kor_kr": {
        "BACKTEST_MONTHS": 12,
        "INITIAL_KRW_AMOUNT": 100_000_000,
        "TOP_N_HOLD": [5],
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
        "TOP_N_HOLD": [5],
        "HOLDING_BONUS_SCORE": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "FIRST_MA_TYPE": ["SMA"],
        "FIRST_MA_MONTHS": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "SECOND_MA_TYPE": ["ALMA"],
        "SECOND_MA_MONTHS": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    },
}
