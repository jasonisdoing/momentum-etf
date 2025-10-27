"""계정별 전략 파라미터 튜닝 실행 스크립트."""

from __future__ import annotations

import sys
from pathlib import Path

from config import ACCOUNT_PARAMETER_SEARCH_CONFIG
from utils.account_registry import get_account_settings, get_strategy_rules
from logic.tune.runner import run_account_tuning
from utils.logger import get_app_logger

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
        tuning_config=ACCOUNT_PARAMETER_SEARCH_CONFIG,
    )
    if output is None:
        logger.error("튜닝이 실패하여 결과를 저장하지 않습니다.")


if __name__ == "__main__":
    main()
