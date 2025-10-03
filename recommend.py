"""국가별 추천 실행 스크립트."""

from __future__ import annotations

import argparse
from pathlib import Path

from utils.country_registry import (
    get_country_settings,
    get_strategy_rules,
    list_available_countries,
)
from logic.recommend.output import (
    dump_json,
    invoke_country_pipeline,
    print_result_summary,
    print_run_header,
)

RESULTS_DIR = Path(__file__).resolve().parent / "data" / "results"


def _available_country_choices() -> list[str]:
    choices = list_available_countries()
    if not choices:
        raise SystemExit("국가 설정(JSON)이 존재하지 않습니다. settings/country/*.json 파일을 확인하세요.")
    return choices


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MomentumEtf 국가 추천 실행기",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("country", choices=_available_country_choices(), help="실행할 국가 코드")
    parser.add_argument(
        "--date",
        help="YYYY-MM-DD 형식의 기준일 (미지정 시 최신 거래일)",
    )
    parser.add_argument(
        "--output",
        help="결과 JSON 저장 경로",
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

    print_run_header(country, date_str=args.date)
    items = invoke_country_pipeline(country, date_str=args.date)
    print_result_summary(items, country, args.date)

    output_path = (
        Path(args.output) if args.output else RESULTS_DIR / f"recommendation_{country}.json"
    )
    dump_json(items, output_path)
    print(f"\n✅ {country.upper()} 결과를 '{output_path}'에 저장했습니다.")


if __name__ == "__main__":
    main()
