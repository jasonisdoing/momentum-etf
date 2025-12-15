"""계정별 전략 파라미터 튜닝 실행 스크립트."""

from __future__ import annotations

import sys
from pathlib import Path

from logic.tune.runner import run_account_tuning
from utils.account_registry import get_account_settings, get_strategy_rules
from utils.data_loader import MissingPriceDataError
from utils.logger import get_app_logger

# 튜닝·최적화 작업이 공유하는 계정별 파라미터 탐색 설정
TUNING_CONFIG: dict[str, dict] = {
    "kor1": {
        "PORTFOLIO_TOPN": [8],
        "MA_RANGE": [20, 25, 30, 35, 40, 45, 50],  # 범위가 넓어지면 과최적화 위험 증가
        "MA_TYPE": ["EMA"],
        "REPLACE_SCORE_THRESHOLD": [0, 1, 2, 3, 4, 5],
        "STOP_LOSS_PCT": [5, 6, 7, 8, 9, 10],
        "OVERBOUGHT_SELL_THRESHOLD": [84, 86, 88, 90],
        "TRAILING_STOP_PCT": [0],
        "COOLDOWN_DAYS": [0, 1, 2, 3],
        "CORE_HOLDINGS": [],
        "OPTIMIZATION_METRIC": "CAGR",  # "CAGR", "Sharpe", "SDR" 중 선택
    },
    "kor2": {
        "PORTFOLIO_TOPN": [5],
        "MA_RANGE": [20, 25, 30, 35, 40, 45, 50],  # 범위가 넓어지면 과최적화 위험 증가
        "MA_TYPE": ["EMA"],
        "REPLACE_SCORE_THRESHOLD": [0, 1, 2, 3, 4, 5],
        "STOP_LOSS_PCT": [5],
        "OVERBOUGHT_SELL_THRESHOLD": [85, 86, 87, 88, 89, 90, 91, 92, 93],
        "TRAILING_STOP_PCT": [0],
        "COOLDOWN_DAYS": [0, 1, 2, 3],
        "CORE_HOLDINGS": [],
        "OPTIMIZATION_METRIC": "CAGR",  # "CAGR", "Sharpe", "SDR" 중 선택
    },
    "us1": {
        "PORTFOLIO_TOPN": [8],
        "MA_RANGE": [20, 25, 30, 35, 40, 45, 50],  # 범위가 넓어지면 과최적화 위험 증가
        # "MA_TYPE": ["SMA", "EMA"],
        "MA_TYPE": ["SMA"],
        "REPLACE_SCORE_THRESHOLD": [0, 1, 2, 3, 4, 5],
        "STOP_LOSS_PCT": [5, 6, 7, 8, 9, 10],
        "OVERBOUGHT_SELL_THRESHOLD": [80, 82, 84, 86],
        "TRAILING_STOP_PCT": [0],
        "COOLDOWN_DAYS": [0, 1, 2, 3],
        "CORE_HOLDINGS": [],
        "OPTIMIZATION_METRIC": "CAGR",  # "CAGR", "Sharpe", "SDR" 중 선택
    },
}


# TUNING_CONFIG: dict[str, dict] = {
#     "k1": {
#         "PORTFOLIO_TOPN": [10],
#         "MA_RANGE": [25, 30, 35, 40, 45, 50],  # 범위가 넓어지면 과최적화 위험 증가
#         "MA_TYPE": ["EMA"],
#         "REPLACE_SCORE_THRESHOLD": [2, 3],
#         "STOP_LOSS_PCT": [5, 6, 7, 8, 9, 10],
#         "OVERBOUGHT_SELL_THRESHOLD": [86],
#         "TRAILING_STOP_PCT": [0],
#         "COOLDOWN_DAYS": [2],
#         "CORE_HOLDINGS": [],
#         "OPTIMIZATION_METRIC": "CAGR",  # "CAGR", "Sharpe", "SDR" 중 선택
#     }
# }

# 매달 전체 테스트
# TUNING_CONFIG: dict[str, dict] = {
#     "k1": {
#         "PORTFOLIO_TOPN": [8],
#         "MA_RANGE": [25, 30, 35, 40, 45, 50],  # 범위가 넓어지면 과최적화 위험 증가
#         # "MA_TYPE": ["EMA"],
#         "MA_TYPE": ["SMA", "EMA", "WMA", "DEMA", "TEMA", "HMA"],
#         "REPLACE_SCORE_THRESHOLD": [1, 2, 3, 4, 5],
#         "STOP_LOSS_PCT": [5, 6, 7, 8, 9, 10],
#         "OVERBOUGHT_SELL_THRESHOLD": [83, 84, 85, 86, 87, 88, 89, 90],
#         "TRAILING_STOP_PCT": [0],
#         "COOLDOWN_DAYS": [0, 1, 2, 3],
#         "CORE_HOLDINGS": [],
#         "OPTIMIZATION_METRIC": "CAGR",  # "CAGR", "Sharpe", "SDR" 중 선택
#     }
# }


RESULTS_DIR = Path(__file__).resolve().parent / "zaccounts"


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

    try:
        output = run_account_tuning(
            account_id,
            output_path=None,
            results_dir=RESULTS_DIR,
            tuning_config=TUNING_CONFIG,
        )
    except MissingPriceDataError as exc:
        logger.error(str(exc))
        raise SystemExit(1)
    if output is None:
        logger.error("튜닝이 실패하여 결과를 저장하지 않습니다.")


if __name__ == "__main__":
    main()
