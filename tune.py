"""계정별 전략 파라미터 튜닝 실행 스크립트."""

from __future__ import annotations

import sys
from pathlib import Path

from core.tune.runner import run_account_tuning
from utils.account_registry import get_account_settings
from utils.data_loader import MissingPriceDataError, format_missing_price_data_guidance
from utils.logger import get_app_logger

# =========================================================
# 계좌별 성격 맞춤형 설정
# =========================================================
# ["DAILY", "WEEKLY", "TWICE_A_MONTH", "MONTHLY", "QUARTERLY"],
ACCOUNT_TUNING_CONFIG = {
    "kor_account": {
        "TUNE_MONTHS": 12,
        "OPTIMIZATION_METRIC": "CAGR",
        "TOPN": [6],
        "REBALANCE_MODE": ["TWICE_A_MONTH"],
        "MY_TYPE": ["ALMA"],
        "MY_MONTHS": [3],
    },
    "isa_account": {
        "TUNE_MONTHS": 12,
        "OPTIMIZATION_METRIC": "CAGR",
        "TOPN": [4, 5],
        "REBALANCE_MODE": ["TWICE_A_MONTH"],
        "MY_TYPE": ["ALMA"],
        "MY_MONTHS": [3],
    },
    "pension_account": {
        "TUNE_MONTHS": 12,
        "OPTIMIZATION_METRIC": "CAGR",
        "TOPN": [4, 5],
        "REBALANCE_MODE": ["TWICE_A_MONTH"],
        "MY_TYPE": ["ALMA"],
        "MY_MONTHS": [3],
    },
    "core_account": {
        "TUNE_MONTHS": 12,
        "OPTIMIZATION_METRIC": "CAGR",
        "TOPN": [4],
        "REBALANCE_MODE": ["TWICE_A_MONTH"],
        "MY_TYPE": ["ALMA"],
        "MY_MONTHS": [3],
    },
    "aus_account": {
        "TUNE_MONTHS": 12,
        "OPTIMIZATION_METRIC": "CAGR",
        "TOPN": [5, 6, 7],
        "REBALANCE_MODE": ["TWICE_A_MONTH"],
        "MY_TYPE": ["ALMA"],
        "MY_MONTHS": [3],
    },
    "ma_type_options": {
        "HMA": {
            "desc": "HMA: 가격 변곡점에 즉각 반응하여 추세 전환을 가장 빠르게 포착합니다.",
            "key_factor": "속도(Speed)",
            "strategy": "단기 스윙 및 빠른 추세 전환 대응",
        },
        "DEMA": {
            "desc": "DEMA: 이중 지수 계산으로 지연을 줄이며 중장기 추세의 방향성을 유지합니다.",
            "key_factor": "지속성(Continuity)",
            "strategy": "시장 지수 추종 및 안정적 추세 보유",
        },
        "ALMA": {
            "desc": "ALMA: 가우시안 필터를 통해 노이즈를 제거하고 정교한 진입·이탈 시점을 제공합니다.",
            "key_factor": "정밀도(Precision)",
            "strategy": "변동성 제어 및 고정밀 추세 추종",
        },
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
        for line in format_missing_price_data_guidance(exc, target_id=account_id):
            logger.error(line)
        raise SystemExit(1)
    if output is None:
        logger.error("튜닝이 실패하여 결과를 저장하지 않습니다.")


if __name__ == "__main__":
    main()
