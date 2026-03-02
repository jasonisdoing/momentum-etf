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
    "aus": {
        # 6개월 + MONTHLY가 최적
        "BUCKET_TOPN": [2],
        # "MA_MONTH": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        "MA_MONTH": [6],  # 중기 추세
        "MA_TYPE": ["HMA"],
        # "REBALANCE_MODE": ["WEEKLY", "TWICE_A_MONTH", "MONTHLY", "QUARTERLY"], # 다음 쿼터에 오픈
        "REBALANCE_MODE": ["MONTHLY"],
    },
    "kor_isa": {
        # 아무 개월 + QUARTERLY가 최적이지만 다음 쿼터까지 MONTHLY 로 유지
        "BUCKET_TOPN": [1],  # 절세계좌 금액이 적어서 1 * 5 종목
        "MA_MONTH": [3],  # 종목이 많지 않고 고정 종목이라 의미 없음
        "MA_TYPE": ["HMA"],
        # "REBALANCE_MODE": ["WEEKLY", "TWICE_A_MONTH", "MONTHLY", "QUARTERLY"], # 다음 쿼터에 오픈
        "REBALANCE_MODE": ["MONTHLY"],
    },
    "kor_kr": {
        # 3개월 + WEEKLY가 최적
        "BUCKET_TOPN": [2],
        "MA_MONTH": [3],  # 단기 추세
        "MA_TYPE": ["HMA"],
        "REBALANCE_MODE": ["WEEKLY"],
    },
    "kor_pension": {
        # 5개월 + TWICE_A_MONTH가 최적
        "BUCKET_TOPN": [1],  # 절세계좌 금액이 적어서 1 * 5 종목
        "MA_MONTH": [5],  # 중기 추세
        "MA_TYPE": ["HMA"],
        "REBALANCE_MODE": ["TWICE_A_MONTH"],
    },
    "us": {
        # 4개월 + TWICE_A_MONTH 가 최적
        "BUCKET_TOPN": [2],
        "MA_MONTH": [4],  # 중단기 추세
        "MA_TYPE": ["HMA"],
        "REBALANCE_MODE": ["TWICE_A_MONTH"],
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
