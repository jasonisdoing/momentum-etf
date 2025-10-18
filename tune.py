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
        "MA_RANGE": np.arange(10, 60, 10),  # 10~50
        "PORTFOLIO_TOPN": np.arange(3, 6, 1),  # 5~7
        "REPLACE_SCORE_THRESHOLD": np.arange(0, 4, 1),  # 0~3
        "OVERBOUGHT_SELL_THRESHOLD": np.arange(5, 35, 5),  # 5~30
        "COOLDOWN_DAYS": np.arange(1, 4, 1),  # 1~3
    },
    # "kor": {
    #     "MA_RANGE": [50],
    #     "PORTFOLIO_TOPN": np.arange(5, 8, 1),
    #     "REPLACE_SCORE_THRESHOLD": [0],
    #     "OVERBOUGHT_SELL_THRESHOLD": [20],
    #     "COOLDOWN_DAYS": np.arange(0, 6, 1),  # 0~5일
    # },
    "kor": {
        "MA_RANGE": np.arange(10, 80, 10),  # 10~70
        "PORTFOLIO_TOPN": [5, 6, 7],  # 5~7
        "REPLACE_SCORE_THRESHOLD": np.arange(0, 4, 1),  # 0~3
        "OVERBOUGHT_SELL_THRESHOLD": np.arange(5, 35, 5),  # 5~30
        "COOLDOWN_DAYS": np.arange(1, 4, 1),  # 1~3
    },
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
