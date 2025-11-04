"""계정별 전략 파라미터 튜닝 실행 스크립트."""

from __future__ import annotations

import sys
import numpy as np
from pathlib import Path

from utils.account_registry import get_account_settings, get_strategy_rules
from logic.tune.runner import run_account_tuning
from utils.logger import get_app_logger


# 튜닝·최적화 작업이 공유하는 계정별 파라미터 탐색 설정
TUNING_CONFIG: dict[str, dict] = {
    "a1": {
        "MA_RANGE": np.arange(30, 65, 5),
        "MA_TYPE": ["SMA"],
        "PORTFOLIO_TOPN": [7],
        "REPLACE_SCORE_THRESHOLD": [0, 1, 2, 3, 4, 5],
        "OVERBOUGHT_SELL_THRESHOLD": [100],
        "CORE_HOLDINGS": ["ASX:HNDQ", "ASX:IOO"],
        "COOLDOWN_DAYS": [3, 4, 5],
        "STOP_LOSS_PCT": np.arange(3, 11, 1),
        "OPTIMIZATION_METRIC": "CAGR",  # "CAGR", "Sharpe", "SDR" 중 선택
    },
    "k1": {
        "MA_RANGE": np.arange(80, 125, 5),
        "MA_TYPE": ["HMA"],
        "PORTFOLIO_TOPN": [12],
        "REPLACE_SCORE_THRESHOLD": [0, 1, 2, 3, 4, 5],
        "OVERBOUGHT_SELL_THRESHOLD": np.arange(90, 101, 1),
        "CORE_HOLDINGS": [442580, 315960],
        "COOLDOWN_DAYS": [3, 4, 5],
        "STOP_LOSS_PCT": np.arange(3, 11, 1),
        "OPTIMIZATION_METRIC": "CAGR",  # "CAGR", "Sharpe", "SDR" 중 선택
    },
    # "k1": {
    #     "MA_RANGE": np.arange(80, 125, 5),
    #     "MA_TYPE": ["HMA"],
    #     "PORTFOLIO_TOPN": [12],
    #     "REPLACE_SCORE_THRESHOLD": [0, 1, 2, 3, 4, 5],
    #     "OVERBOUGHT_SELL_THRESHOLD": np.arange(90, 101, 1),
    #     "CORE_HOLDINGS": [442580, 395160, 102970],
    #     # "CORE_HOLDINGS": [],
    #     "COOLDOWN_DAYS": [1, 2, 3, 4, 5],
    #     "OPTIMIZATION_METRIC": "CAGR",  # "CAGR", "Sharpe", "SDR" 중 선택
    # },
}


RESULTS_DIR = Path(__file__).resolve().parent / "data" / "results"


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
