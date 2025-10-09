"""계정별 전략 파라미터 튜닝 실행 스크립트."""

from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime

import numpy as np

from utils.account_registry import (
    get_account_settings,
    get_strategy_rules,
    list_available_accounts,
)
from logic.tune.runner import run_account_tuning
from utils.logger import get_app_logger

RESULTS_DIR = Path(__file__).resolve().parent / "data" / "results"

TUNING_CONFIG: dict[str, dict] = {
    "aus": {
        "MA_RANGE": np.arange(1, 51, 1),
        "PORTFOLIO_TOPN": [7],
        "REPLACE_SCORE_THRESHOLD": np.arange(0, 1.1, 0.1),
    },
    "kor": {
        "MA_RANGE": np.arange(1, 51, 1),
        "PORTFOLIO_TOPN": [10],
        "REPLACE_SCORE_THRESHOLD": np.arange(0, 1.1, 0.1),
    },
    "us": {
        "MA_RANGE": np.arange(5, 31, 1),
        "PORTFOLIO_TOPN": np.arange(5, 11, 1),
        "REPLACE_SCORE_THRESHOLD": np.arange(0, 2.1, 0.1),
    },
}


def _available_account_choices() -> list[str]:
    choices = list_available_accounts()
    if not choices:
        raise SystemExit("계정 설정(JSON)이 존재하지 않습니다. data/settings/account/*.json 파일을 확인하세요.")
    return choices


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MomentumEtf 전략 파라미터 튜닝 실행기",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("account", choices=_available_account_choices(), help="실행할 계정 ID")
    parser.add_argument(
        "--output",
        help="튜닝 결과 저장 경로 (기본값: data/results/tune_<account>.json)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        help="Optuna 시도 횟수 제한 (기본값: 탐색 공간 전체)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        help="월별 튜닝 시간 제한(초)",
    )
    return parser


def main() -> None:
    logger = get_app_logger()

    parser = build_parser()
    args = parser.parse_args()

    account_id = args.account.lower()

    try:
        get_account_settings(account_id)
        get_strategy_rules(account_id)
    except Exception as exc:  # pragma: no cover - 잘못된 입력 방어 전용 처리
        parser.error(f"계정 설정을 로드하는 중 오류가 발생했습니다: {exc}")

    if args.output:
        output_path = Path(args.output)
    else:
        # 파일명에 YYYYMMDD 형식의 날짜 추가
        date_str = datetime.now().strftime("%Y%m%d")
        output_path = RESULTS_DIR / f"tune_{account_id}_{date_str}.json"

    output = run_account_tuning(
        account_id,
        output_path=output_path,
        results_dir=RESULTS_DIR,
        tuning_config=TUNING_CONFIG,
        n_trials=args.n_trials,
        timeout=args.timeout,
    )
    if output is None:
        logger.error("튜닝이 실패하여 결과를 저장하지 않습니다.")
    else:
        logger.info("✅ 튜닝 결과를 '%s'에 저장했습니다.", output)


if __name__ == "__main__":
    main()
