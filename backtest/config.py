"""백테스트 파라미터 스윕 설정."""

from __future__ import annotations


BACKTEST_MONTHS = 12
INITIAL_KRW_AMOUNT = 100_000_000

# 슬리피지는 % 단위로 입력한다.
SLIPPAGE_CONFIG: dict[str, dict[str, float]] = {
    "kor_kr": {
        "BUY_PCT": 0.5,
        "SELL_PCT": 0.5,
    },
    "kor_us": {
        "BUY_PCT": 0.5,
        "SELL_PCT": 0.5,
    },
    "aus": {
        "BUY_PCT": 1.0,
        "SELL_PCT": 1.0,
    },
    "us": {
        "BUY_PCT": 0.5,
        "SELL_PCT": 0.5,
    },
    "kor": {
        "BUY_PCT": 0.5,
        "SELL_PCT": 0.5,
    },
}


BACKTEST_CONFIG: dict[str, dict] = {
    "kor_kr": {
        "TOP_N_HOLD": [5],
        "HOLDING_BONUS_SCORE": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200],
        "FIRST_MA_TYPE": ["SMA"],
        "FIRST_MA_MONTHS": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "SECOND_MA_TYPE": ["ALMA"],
        "SECOND_MA_MONTHS": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    },
    "kor_us": {
        "TOP_N_HOLD": [3, 4, 5, 6, 7],
        "HOLDING_BONUS_SCORE": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200],
        "FIRST_MA_TYPE": ["SMA"],
        "FIRST_MA_MONTHS": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "SECOND_MA_TYPE": ["ALMA"],
        "SECOND_MA_MONTHS": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    },
    "aus": {
        "TOP_N_HOLD": [3, 4, 5, 6, 7],
        "HOLDING_BONUS_SCORE": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200],
        "FIRST_MA_TYPE": ["SMA"],
        "FIRST_MA_MONTHS": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "SECOND_MA_TYPE": ["ALMA"],
        "SECOND_MA_MONTHS": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    },
    "us": {
        "TOP_N_HOLD": [3, 4, 5, 6, 7],
        "HOLDING_BONUS_SCORE": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200],
        "FIRST_MA_TYPE": ["SMA"],
        "FIRST_MA_MONTHS": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "SECOND_MA_TYPE": ["ALMA"],
        "SECOND_MA_MONTHS": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    },
    "kor": {
        "TOP_N_HOLD": [5],
        "HOLDING_BONUS_SCORE": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200],
        "FIRST_MA_TYPE": ["SMA"],
        "FIRST_MA_MONTHS": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "SECOND_MA_TYPE": ["ALMA"],
        "SECOND_MA_MONTHS": [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    },
}
