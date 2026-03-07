"""계정별 전략 파라미터 튜닝 실행 스크립트."""

from __future__ import annotations

import sys
from pathlib import Path

from core.tune.runner import run_account_tuning
from utils.account_registry import get_account_settings
from utils.data_loader import MissingPriceDataError
from utils.logger import get_app_logger

# =========================================================
# 계좌별 성격 맞춤형 설정
# =========================================================
ACCOUNT_TUNING_CONFIG = {
    # "TOPN": [10],
    # "MA_MONTH": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    # "MA_TYPE": ["HMA"],
    # "COOLDOWN": [1, 2, 3],
    # "REBALANCE_MODE": ["DAILY", "WEEKLY", "TWICE_A_MONTH", "MONTHLY", "QUARTERLY"],
    # "OPTIMIZATION_METRIC": "CAGR",
    "kor_kr": {
        # +------+--------+--------+------+--------+---------------+---------+--------+---------------+--------+-----------------+-----------------+
        # | 전략 | MA개월 | MA타입 | TOPN | 쿨다운 |   리밸런스    | CAGR(%) | MDD(%) | 12개월 4일(%) | Sharpe | SDR(Sharpe/MDD) | Trades(거래 수) |
        # +------+--------+--------+------+--------+---------------+---------+--------+---------------+--------+-----------------+-----------------+
        # | MAPS |     12 |  HMA   |    5 |   4    |    WEEKLY     |  146.12 | -21.79 |        145.36 |   2.49 |           0.114 |             320 |
        # +------+--------+--------+------+--------+---------------+---------+--------+---------------+--------+-----------------+-----------------+
        "STRATEGY": ["MAPS"],
        "TOPN": [5],
        "MA_MONTH": [12],
        "MA_TYPE": ["HMA"],
        "COOLDOWN": [1, 2, 3, 4, 5],
        "REBALANCE_MODE": ["WEEKLY", "TWICE_A_MONTH", "MONTHLY", "QUARTERLY"],
        "OPTIMIZATION_METRIC": "CAGR",
    },
    "kor_pension": {
        # +------+--------+--------+------+--------+---------------+---------+--------+---------------+--------+-----------------+-----------------+
        # | 전략 | MA개월 | MA타입 | TOPN | 쿨다운 |   리밸런스    | CAGR(%) | MDD(%) | 12개월 4일(%) | Sharpe | SDR(Sharpe/MDD) | Trades(거래 수) |
        # +------+--------+--------+------+--------+---------------+---------+--------+---------------+--------+-----------------+-----------------+
        # | MAPS |     12 |  HMA   |    5 |   2    |    WEEKLY     |   51.62 | -13.42 |         51.40 |   1.67 |           0.124 |             375 |
        # +------+--------+--------+------+--------+---------------+---------+--------+---------------+--------+-----------------+-----------------+
        "STRATEGY": ["MAPS"],
        "TOPN": [5],
        "MA_MONTH": [12],
        "MA_TYPE": ["HMA"],
        "COOLDOWN": [1, 2, 3, 4, 5],
        "REBALANCE_MODE": ["WEEKLY", "TWICE_A_MONTH", "MONTHLY", "QUARTERLY"],
        "OPTIMIZATION_METRIC": "CAGR",
    },
    "us": {
        # +------+--------+--------+------+--------+---------------+---------+--------+---------------+--------+-----------------+-----------------+
        # | 전략 | MA개월 | MA타입 | TOPN | 쿨다운 |   리밸런스    | CAGR(%) | MDD(%) | 12개월 4일(%) | Sharpe | SDR(Sharpe/MDD) | Trades(거래 수) |
        # +------+--------+--------+------+--------+---------------+---------+--------+---------------+--------+-----------------+-----------------+
        # | MAPS |     12 |  HMA   |    5 |   3    | TWICE_A_MONTH |   49.39 | -13.68 |         45.53 |   1.43 |           0.104 |             335 |
        # +------+--------+--------+------+--------+---------------+---------+--------+---------------+--------+-----------------+-----------------+
        "STRATEGY": ["MAPS"],
        "TOPN": [5],
        "MA_MONTH": [12],
        "MA_TYPE": ["HMA"],
        "COOLDOWN": [1, 2, 3, 4, 5],
        "REBALANCE_MODE": ["WEEKLY", "TWICE_A_MONTH", "MONTHLY", "QUARTERLY"],
        "OPTIMIZATION_METRIC": "CAGR",
    },
    "aus": {
        # +------+--------+--------+------+--------+---------------+---------+--------+---------------+--------+-----------------+-----------------+
        # | 전략 | MA개월 | MA타입 | TOPN | 쿨다운 |   리밸런스    | CAGR(%) | MDD(%) | 12개월 4일(%) | Sharpe | SDR(Sharpe/MDD) | Trades(거래 수) |
        # +------+--------+--------+------+--------+---------------+---------+--------+---------------+--------+-----------------+-----------------+
        # | MAPS |     12 |  HMA   |    6 |   5    |    MONTHLY    |   71.54 | -16.38 |         67.03 |   1.72 |           0.105 |             293 |
        # +------+--------+--------+------+--------+---------------+---------+--------+---------------+--------+-----------------+-----------------+
        "STRATEGY": ["MAPS"],
        "TOPN": [6],
        "MA_MONTH": [12],
        "MA_TYPE": ["HMA"],
        "COOLDOWN": [1, 2, 3, 4, 5],
        "REBALANCE_MODE": ["WEEKLY", "TWICE_A_MONTH", "MONTHLY", "QUARTERLY"],
        "OPTIMIZATION_METRIC": "CAGR",
    },
    "kor_us": {
        # +------+--------+--------+------+--------+---------------+---------+--------+---------------+--------+-----------------+-----------------+
        # | 전략 | MA개월 | MA타입 | TOPN | 쿨다운 |   리밸런스    | CAGR(%) | MDD(%) | 12개월 4일(%) | Sharpe | SDR(Sharpe/MDD) | Trades(거래 수) |
        # +------+--------+--------+------+--------+---------------+---------+--------+---------------+--------+-----------------+-----------------+
        # | MAPS |     12 |  HMA   |   10 |   3    |   QUARTERLY   |   54.18 |  -6.74 |         53.95 |   2.97 |           0.440 |             119 |
        # +------+--------+--------+------+--------+---------------+---------+--------+---------------+--------+-----------------+-----------------+
        "STRATEGY": ["MAPS"],
        "TOPN": [10],
        "MA_MONTH": [12],
        "MA_TYPE": ["HMA"],
        "COOLDOWN": [1, 2, 3, 4, 5],
        "REBALANCE_MODE": ["WEEKLY", "TWICE_A_MONTH", "MONTHLY", "QUARTERLY"],
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
