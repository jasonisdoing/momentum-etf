"""국가별 백테스트 실행 스크립트."""


from __future__ import annotations

import argparse
from pathlib import Path

from utils.country_registry import (
    get_country_settings,
    get_strategy_rules,
    list_available_countries,
)
from logic.backtest.reporting import dump_backtest_log, print_backtest_summary
from logic.recommend.output import print_run_header

from constants import TEST_INITIAL_CAPITAL, TEST_MONTHS_RANGE

RESULTS_DIR = Path(__file__).resolve().parent / "data" / "results"


def _available_country_choices() -> list[str]:
    choices = list_available_countries()
    if not choices:
        raise SystemExit("국가 설정(JSON)이 존재하지 않습니다. settings/country/*.json 파일을 확인하세요.")
    return choices


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MomentumEtf 국가 백테스트 실행기",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("country", choices=_available_country_choices(), help="실행할 국가 코드")
    parser.add_argument(
        "--output",
        help="백테스트 로그 저장 경로 (기본값: data/results/backtest_<country>.txt)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    country = args.country.lower()

    try:
        country_settings = get_country_settings(country)
        get_strategy_rules(country)
    except Exception as exc:  # pragma: no cover - guard for invalid inputs
        parser.error(f"국가 설정을 로드하는 중 오류가 발생했습니다: {exc}")

    print_run_header(country, date_str=None)

    from logic.backtest.country_runner import run_country_backtest

    result = run_country_backtest(country)
    target_path = Path(args.output) if args.output else RESULTS_DIR / f"backtest_{country}.txt"

    generated_path = dump_backtest_log(
        result,
        country_settings,
        country=country,
        results_dir=target_path.parent,
    )

    if generated_path != target_path:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        generated_path.replace(target_path)
    else:
        target_path = generated_path

    print_backtest_summary(
        summary=result.summary,
        country=country,
        test_months_range=getattr(result, "months_range", TEST_MONTHS_RANGE),
        initial_capital_krw=result.initial_capital,
        portfolio_topn=result.portfolio_topn,
        ticker_summaries=getattr(result, "ticker_summaries", []),
        core_start_dt=result.start_date,
    )
    print(f"\n✅ 백테스트 로그를 '{target_path}'에 저장했습니다.")


if __name__ == "__main__":
    main()
