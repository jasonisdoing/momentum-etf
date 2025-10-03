"""국가별 전략 파라미터 튜닝 실행 스크립트."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from utils.country_registry import (
    get_country_settings,
    get_strategy_rules,
    list_available_countries,
)
from logic.tune.runner import run_country_tuning

RESULTS_DIR = Path(__file__).resolve().parent / "data" / "results"
TUNING_MONTHS_RANGE = 12

TUNING_CONFIG: dict[str, dict] = {
    "aus": {
        "MA_RANGE": np.arange(10, 51, 5),
        "PORTFOLIO_TOPN": np.arange(5, 8, 1),
        "REPLACE_SCORE_THRESHOLD": [0.5],
    },
    "kor": {
        "MA_RANGE": np.arange(10, 51, 1),
        "PORTFOLIO_TOPN": np.arange(5, 11, 1),
        "REPLACE_SCORE_THRESHOLD": [0.5],
    },
    "us": {
        "MA_RANGE": np.arange(5, 31, 1),
        "PORTFOLIO_TOPN": np.arange(5, 11, 1),
        "REPLACE_SCORE_THRESHOLD": np.arange(0, 2.5, 0.5),
    },
}


def _available_country_choices() -> list[str]:
    choices = list_available_countries()
    if not choices:
        raise SystemExit("국가 설정(JSON)이 존재하지 않습니다. settings/country/*.json 파일을 확인하세요.")
    return choices


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MomentumEtf 전략 파라미터 튜닝 실행기",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("country", choices=_available_country_choices(), help="실행할 국가 코드")
    parser.add_argument(
        "--output",
        help="튜닝 결과 저장 경로 (기본값: data/results/tune_<country>.txt)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    country = args.country.lower()

    try:
        get_country_settings(country)
        get_strategy_rules(country)
    except Exception as exc:  # pragma: no cover - guard for invalid inputs
        parser.error(f"국가 설정을 로드하는 중 오류가 발생했습니다: {exc}")

    output_path = Path(args.output) if args.output else RESULTS_DIR / f"tune_{country}.txt"

    run_country_tuning(
        country,
        output_path=output_path,
        results_dir=RESULTS_DIR,
        tuning_config=TUNING_CONFIG,
        months_range=TUNING_MONTHS_RANGE,
    )
    print(f"\n✅ 튜닝 결과를 '{output_path}'에 저장했습니다.")


if __name__ == "__main__":
    main()
