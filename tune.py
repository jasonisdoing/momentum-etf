"""계정별 전략 파라미터 튜닝 실행 스크립트."""

from __future__ import annotations

import sys
from pathlib import Path

from old_logic.tune.runner import run_account_tuning
from utils.account_registry import get_account_settings, get_strategy_rules
from utils.data_loader import MissingPriceDataError
from utils.logger import get_app_logger

# =========================================================
# 계좌별 성격 맞춤형 설정
# =========================================================
ACCOUNT_TUNING_CONFIG = {
    "kor_kr": {
        "BUCKET_TOPN": [2],
        "REPLACE_SCORE_THRESHOLD": [0],
        "MA_MONTH": [3, 6, 9, 12],
        "MA_TYPE": ["SMA", "EMA", "WMA", "DEMA", "TEMA", "HMA"],
    },
    "kor_us": {
        "BUCKET_TOPN": [1],
        "REPLACE_SCORE_THRESHOLD": [0],
        "MA_MONTH": [3, 6, 9, 12],
        "MA_TYPE": ["SMA", "EMA", "WMA", "DEMA", "TEMA", "HMA"],
    },
    "aus": {
        "BUCKET_TOPN": [2],
        "REPLACE_SCORE_THRESHOLD": [0],
        "MA_MONTH": [3, 6, 9, 12],
        "MA_TYPE": ["SMA", "EMA", "WMA", "DEMA", "TEMA", "HMA"],
    },
    # 🇺🇸 미국 직투: 테스트 중
    "us": {
        "BUCKET_TOPN": [2],
        "REPLACE_SCORE_THRESHOLD": [0],
        "MA_MONTH": [3, 6, 9, 12],
        "MA_TYPE": ["SMA", "EMA", "WMA", "DEMA", "TEMA", "HMA"],
    },
}

# =========================================================
# 공통 설정 (나머지 변수 통제)
# =========================================================
COMMON_TUNING_CONFIG = {
    # 1. 손절: 5~15
    "STOP_LOSS_PCT": [10],
    # 2. RSI: 100
    "OVERBOUGHT_SELL_THRESHOLD": [100],
    # 3. 쿨다운: 0
    "COOLDOWN_DAYS": [0],
    # 4. 목표: 수익률 극대화
    "OPTIMIZATION_METRIC": "CAGR",  # CAGR, SHARPE, SDR 중 선택
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

    # 공통 설정과 계정별 설정을 조합
    merged_config = COMMON_TUNING_CONFIG.copy()
    account_config = ACCOUNT_TUNING_CONFIG.get(account_id, {})
    merged_config.update(account_config)

    try:
        output = run_account_tuning(
            account_id,
            output_path=None,
            results_dir=RESULTS_DIR,
            tuning_config={account_id: merged_config},
        )
    except MissingPriceDataError as exc:
        logger.error(str(exc))
        raise SystemExit(1)
    if output is None:
        logger.error("튜닝이 실패하여 결과를 저장하지 않습니다.")


if __name__ == "__main__":
    main()
