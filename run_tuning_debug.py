#!/usr/bin/env python3
"""Run tuning with debug exports to compare recorded vs live backtests."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from logic.tune.runner import run_account_tuning, WORKERS  # noqa: E402
from utils.account_registry import get_account_settings, get_strategy_rules  # noqa: E402
from utils.logger import get_app_logger  # noqa: E402
from tune import TUNING_CONFIG  # noqa: E402

RESULTS_DIR = ROOT_DIR / "data" / "results"
DEBUG_ROOT = RESULTS_DIR / "tuning_debug_sessions"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run tuning with debug artifacts for later comparison.",
        allow_abbrev=False,
    )
    parser.add_argument("account", help="Account ID (e.g., k1)")
    parser.add_argument(
        "--top-n",
        type=int,
        default=1,
        help="Number of top combinations (per months_range) to capture in detail",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    account_id = args.account.strip().lower()
    if not account_id:
        parser.error("Account ID is required.")

    try:
        get_account_settings(account_id)
        get_strategy_rules(account_id)
    except Exception as exc:  # pragma: no cover
        parser.error(f"계정 설정을 로드하는 중 오류가 발생했습니다: {exc}")

    DEBUG_ROOT.mkdir(parents=True, exist_ok=True)
    session_dir = DEBUG_ROOT / f"{account_id}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    logger = get_app_logger()
    logger.info("디버그 세션 디렉토리: %s", session_dir)

    # 강제 순차 실행 (macOS 세마포어 제한 회피)
    import logic.tune.runner as tuning_runner  # noqa: E402

    tuning_runner.WORKERS = 1

    output = run_account_tuning(
        account_id,
        results_dir=RESULTS_DIR,
        tuning_config=TUNING_CONFIG,
        months_range=None,
        n_trials=None,
        timeout=None,
        debug_export_dir=session_dir,
        debug_capture_top_n=args.top_n,
    )

    if output is None:
        logger.error("튜닝이 실패했습니다. 디버그 아티팩트를 확인하세요: %s", session_dir)
    else:
        logger.info("✅ 튜닝 결과: %s", output)
        logger.info("디버그 아티팩트가 %s 에 저장되었습니다.", session_dir)
        print(f"\nDebug session ready at: {session_dir}")

        diff_path = session_dir / "diff_summary.json"
        if diff_path.exists():
            try:
                diff_rows = json.loads(diff_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                diff_rows = []
            if diff_rows:
                print("\nRecorded vs Live Differences:")

                def fmt(value: float | None) -> str:
                    if value is None or (isinstance(value, float) and (value != value)):
                        return "-"
                    return f"{value:.2f}"

                def fmt_delta(value: float | None) -> str:
                    if value is None or (isinstance(value, float) and (value != value)):
                        return "-"
                    return f"{value:+.2f}"

                for row in diff_rows:
                    print(
                        f" - MONTHS={row['months_range']} MA={row['ma_period']} TOPN={row['topn']} "
                        f"TH={row['threshold']:.3f} | Recorded CAGR={fmt(row['recorded_cagr'])} "
                        f"Live CAGR={fmt(row['live_cagr'])} (Δ={fmt_delta(row['live_minus_recorded'])})"
                    )
            else:
                print("No diff summary produced (check capture settings).")


if __name__ == "__main__":
    main()
