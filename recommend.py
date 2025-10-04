"""계정별 추천 실행 스크립트."""

from __future__ import annotations

import argparse
from pathlib import Path

from utils.account_registry import (
    get_account_settings,
    get_strategy_rules,
    list_available_accounts,
)
from logic.recommend.output import (
    dump_json,
    invoke_account_pipeline,
    print_result_summary,
    print_run_header,
)

RESULTS_DIR = Path(__file__).resolve().parent / "data" / "results"


def _available_account_choices() -> list[str]:
    choices = list_available_accounts()
    if not choices:
        raise SystemExit("계정 설정(JSON)이 존재하지 않습니다. settings/account/*.json 파일을 확인하세요.")
    return choices


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MomentumEtf 계정 추천 실행기",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("account", choices=_available_account_choices(), help="실행할 계정 ID")
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

    account_id = args.account.lower()

    try:
        get_account_settings(account_id)
        get_strategy_rules(account_id)
    except Exception as exc:  # pragma: no cover - 잘못된 입력 방어 전용 처리
        parser.error(f"계정 설정을 로드하는 중 오류가 발생했습니다: {exc}")

    print_run_header(account_id, date_str=args.date)
    items = invoke_account_pipeline(account_id, date_str=args.date)
    print_result_summary(items, account_id, args.date)

    output_path = (
        Path(args.output) if args.output else RESULTS_DIR / f"recommendation_{account_id}.json"
    )
    dump_json(items, output_path)
    print(f"\n✅ {account_id.upper()} 결과를 '{output_path}'에 저장했습니다.")


if __name__ == "__main__":
    main()
