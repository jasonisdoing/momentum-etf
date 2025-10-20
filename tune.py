"""계정별 전략 파라미터 튜닝 실행 스크립트."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from utils.account_registry import get_account_settings, get_strategy_rules
from logic.tune.runner import run_account_tuning
from utils.logger import get_app_logger

RESULTS_DIR = Path(__file__).resolve().parent / "data" / "results"

TUNING_CONFIG: dict[str, dict] = {
    "aus": {
        "_설명": "최소한의 최적 범위",
        "MA_RANGE": np.arange(30, 51, 1),
        "MA_TYPE": ["SMA"],
        "PORTFOLIO_TOPN": [3, 4],
        "REPLACE_SCORE_THRESHOLD": [0.5, 1, 1.5],
        "OVERBOUGHT_SELL_THRESHOLD": np.arange(5, 15, 1),  # 5~25
        "COOLDOWN_DAYS": [1, 2],
    },
    "kor": {
        "_설명": "최소한의 최적 범위",
        "MA_RANGE": np.arange(45, 51, 1),
        # "MA_RANGE": [15],
        "MA_TYPE": ["SMA"],
        "PORTFOLIO_TOPN": [6, 7],
        "REPLACE_SCORE_THRESHOLD": [0.5, 1, 1.5],
        "OVERBOUGHT_SELL_THRESHOLD": np.arange(10, 21, 1),  # 15~25
        "COOLDOWN_DAYS": [1],
    },
    # "kor": {
    #     "_설명": "최대 삼세한 넓은 범위",
    #     "MA_RANGE": np.arange(10, 71, 1),  # 10~70
    #     "MA_TYPE": ["SMA", "EMA", "WMA", "DEMA", "TEMA", "HMA"],
    #     "PORTFOLIO_TOPN": np.arange(5, 8, 1),  # 5~7
    #     "REPLACE_SCORE_THRESHOLD": np.arange(0, 2.1, 0.1),  # 0~2.0
    #     "OVERBOUGHT_SELL_THRESHOLD": np.arange(1, 21, 1),  # 1~21
    #     "COOLDOWN_DAYS": np.arange(0, 3, 1),  # 0~2
    # },
    "us": {
        "MA_RANGE": np.arange(5, 31, 1),
        "PORTFOLIO_TOPN": np.arange(5, 11, 1),
        "REPLACE_SCORE_THRESHOLD": np.arange(0, 2.1, 0.1),
        "COOLDOWN_DAYS": np.arange(0, 6, 1),  # 0~5일
    },
}


def main() -> None:
    logger = get_app_logger()

    if len(sys.argv) < 2:
        print("Usage: python tune.py <account_id>")
        raise SystemExit(1)

    account_id = sys.argv[1].strip().lower()

    try:
        get_account_settings(account_id)
        get_strategy_rules(account_id)
    except Exception as exc:  # pragma: no cover - 잘못된 입력 방어 전용 처리
        raise SystemExit(f"계정 설정을 로드하는 중 오류가 발생했습니다: {exc}")

    output = run_account_tuning(
        account_id,
        output_path=None,
        results_dir=RESULTS_DIR,
        tuning_config=TUNING_CONFIG,
    )
    if output is None:
        logger.error("튜닝이 실패하여 결과를 저장하지 않습니다.")


if __name__ == "__main__":
    main()
