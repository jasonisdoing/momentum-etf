"""계정별 전략 파라미터 튜닝 실행 스크립트."""

from __future__ import annotations

import sys
from pathlib import Path

from core.tune.runner import run_account_tuning
from utils.account_registry import get_account_settings, get_strategy_rules
from utils.data_loader import MissingPriceDataError
from utils.logger import get_app_logger

# =========================================================
# 계좌별 성격 맞춤형 설정
# =========================================================
ACCOUNT_TUNING_CONFIG = {
    # "BUCKET_TOPN": [2],
    # "MA_MONTH": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    # "MA_TYPE": ["HMA"],
    # "SELL_ON_NEGATIVE_SCORE": [True, False],
    # "REPLACEMENT_MODE": ["DAILY", "WEEKLY"],
    # "REBALANCE_MODE": ["DAILY", "WEEKLY", "TWICE_A_MONTH", "MONTHLY", "QUARTERLY"],
    # "OPTIMIZATION_METRIC": "CAGR",
    "kor_kr": {
        "BUCKET_TOPN": [2],
        "MA_MONTH": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "MA_TYPE": ["HMA"],
        "SELL_ON_NEGATIVE_SCORE": [True, False],
        "REPLACEMENT_MODE": ["DAILY"],
        "REBALANCE_MODE": ["MONTHLY"],
        "OPTIMIZATION_METRIC": "SDR",
    },
    "kor_pension": {
        "BUCKET_TOPN": [2],  # 연금 1개, ISA 1개
        "MA_MONTH": [6],
        "MA_TYPE": ["HMA"],
        "SELL_ON_NEGATIVE_SCORE": [True, False],
        "REPLACEMENT_MODE": ["WEEKLY"],
        "REBALANCE_MODE": ["MONTHLY"],
        "OPTIMIZATION_METRIC": "CAGR",
    },
    "kor_us": {
        "BUCKET_TOPN": [2],
        "MA_MONTH": [6],
        "MA_TYPE": ["HMA"],
        "SELL_ON_NEGATIVE_SCORE": [True, False],
        "REPLACEMENT_MODE": ["WEEKLY"],
        "REBALANCE_MODE": ["MONTHLY"],
        "OPTIMIZATION_METRIC": "CAGR",
    },
    "us": {
        "BUCKET_TOPN": [2],
        "MA_MONTH": [6],
        "MA_TYPE": ["HMA"],
        "SELL_ON_NEGATIVE_SCORE": [True, False],
        "REPLACEMENT_MODE": ["WEEKLY"],
        "REBALANCE_MODE": ["MONTHLY"],
        "OPTIMIZATION_METRIC": "CAGR",
    },
    "aus": {
        "BUCKET_TOPN": [2],
        "MA_MONTH": [6],
        "MA_TYPE": ["HMA"],
        "SELL_ON_NEGATIVE_SCORE": [True, False],
        "REPLACEMENT_MODE": ["WEEKLY"],
        "REBALANCE_MODE": ["MONTHLY"],
        "OPTIMIZATION_METRIC": "CAGR",
    },
}


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

    account_config = ACCOUNT_TUNING_CONFIG.get(account_id, {})

    try:
        output = run_account_tuning(
            account_id,
            output_path=None,
            results_dir=RESULTS_DIR,
            tuning_config={account_id: account_config},
        )
    except MissingPriceDataError as exc:
        logger.error(str(exc))
        raise SystemExit(1)
    if output is None:
        logger.error("튜닝이 실패하여 결과를 저장하지 않습니다.")


if __name__ == "__main__":
    main()
