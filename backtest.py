"""국가별 백테스트 실행 스크립트."""

from __future__ import annotations

import argparse
from pathlib import Path

from utils.account_registry import (
    get_account_settings,
    get_strategy_rules,
    list_available_accounts,
)
from logic.backtest.reporting import dump_backtest_log, print_backtest_summary
from logic.recommend.output import print_run_header
from utils.logger import get_app_logger

from utils.settings_loader import get_backtest_months_range

RESULTS_DIR = Path(__file__).resolve().parent / "data" / "results"


def _available_account_choices() -> list[str]:
    choices = list_available_accounts()
    if not choices:
        raise SystemExit("계정 설정(JSON)이 존재하지 않습니다. data/settings/account/*.json 파일을 확인하세요.")
    return choices


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MomentumEtf 계정 백테스트 실행기",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("account", choices=_available_account_choices(), help="실행할 계정 ID")
    parser.add_argument(
        "--output",
        help="백테스트 로그 저장 경로 (기본값: data/results/backtest_<account>.txt)",
    )
    return parser


def main() -> None:
    logger = get_app_logger()

    parser = build_parser()
    args = parser.parse_args()

    account_id = args.account.lower()

    try:
        account_settings = get_account_settings(account_id)
        get_strategy_rules(account_id)
    except Exception as exc:  # pragma: no cover - 잘못된 입력 방어 전용 처리
        parser.error(f"계정 설정을 로드하는 중 오류가 발생했습니다: {exc}")

    print_run_header(account_id, date_str=None)

    from logic.backtest.account_runner import run_account_backtest

    result = run_account_backtest(account_id)
    target_path = Path(args.output) if args.output else RESULTS_DIR / f"backtest_{account_id}.txt"

    generated_path = dump_backtest_log(
        result,
        account_settings,
        results_dir=target_path.parent,
    )

    if generated_path != target_path:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        generated_path.replace(target_path)
    else:
        target_path = generated_path

    print_backtest_summary(
        summary=result.summary,
        account_id=account_id,
        country_code=result.country_code,
        test_months_range=getattr(result, "months_range", get_backtest_months_range()),
        initial_capital_krw=result.initial_capital_krw,
        portfolio_topn=result.portfolio_topn,
        ticker_summaries=getattr(result, "ticker_summaries", []),
        core_start_dt=result.start_date,
    )
    logger.info("✅ 백테스트 로그를 '%s'에 저장했습니다.", target_path)


if __name__ == "__main__":
    main()
