from __future__ import annotations

import numpy as np

# 공통 행 색상 설정 (상태명 -> HEX 색상)
# 필요 시 국가별 튜닝에서 개별 상태 색상을 지정하세요.
DEFAULT_ROW_COLORS: dict[str, str] = {}


TUNING_CONFIG: dict[str, dict] = {
    "coin": {
        "MA_RANGE": np.arange(1, 6, 1),
        "PORTFOLIO_TOPN": [2, 3],
        "REPLACE_SCORE_THRESHOLD": [0.5],
        "TEST_MONTHS_RANGE": 3,
        "ROW_COLORS": DEFAULT_ROW_COLORS,
    },
    "aus": {
        "MA_RANGE": np.arange(15, 31, 1),
        "PORTFOLIO_TOPN": [7],
        "REPLACE_SCORE_THRESHOLD": [0.5],
        "TEST_MONTHS_RANGE": 12,
        "ROW_COLORS": DEFAULT_ROW_COLORS,
    },
    "kor": {
        "MA_RANGE": np.arange(15, 31, 1),
        "PORTFOLIO_TOPN": np.arange(7, 11, 1),
        "REPLACE_SCORE_THRESHOLD": [0.5],
        "TEST_MONTHS_RANGE": 12,
        "ROW_COLORS": DEFAULT_ROW_COLORS,
    },
    "us": {
        "MA_RANGE": np.arange(5, 31, 1),
        "PORTFOLIO_TOPN": np.arange(5, 11, 1),
        "REPLACE_SCORE_THRESHOLD": [0.5],
        "TEST_MONTHS_RANGE": 12,
        "ROW_COLORS": DEFAULT_ROW_COLORS,
    },
}


__all__ = ["TUNING_CONFIG", "DEFAULT_ROW_COLORS"]
