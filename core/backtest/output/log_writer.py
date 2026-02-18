from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from core.backtest.output.daily_report import _generate_daily_report_lines
from core.backtest.output.summary_report import print_backtest_summary

if TYPE_CHECKING:
    from core.backtest.domain import AccountBacktestResult

DEFAULT_RESULTS_DIR = Path("zaccounts").resolve()


def dump_backtest_log(
    result: AccountBacktestResult, account_settings: dict[str, Any], *, results_dir: Path | str | None = None
) -> Path:
    account_id = result.account_id
    if results_dir:
        base_dir = Path(results_dir) / account_id / "results"
    else:
        base_dir = DEFAULT_RESULTS_DIR / account_id / "results"
    base_dir.mkdir(parents=True, exist_ok=True)

    path = base_dir / f"backtest_{pd.Timestamp.now().strftime('%Y-%m-%d')}.log"
    lines = []

    # 레거시 형식 준수: 한국어 헤더 및 섹션 번호 1번 부여
    now_str = pd.Timestamp.now().strftime("%Y-%m-%dT%H:%M:%S")
    lines.append(f"백테스트 로그 생성: {now_str}")
    lines.append("1. ========= 일자별 성과 상세 ==========")

    daily_lines = _generate_daily_report_lines(result, account_settings)
    lines.extend(daily_lines)

    summary_lines = print_backtest_summary(
        summary=result.summary,
        account_id=account_id,
        country_code=result.country_code,
        backtest_start_date=result.backtest_start_date,
        initial_capital_krw=result.initial_capital_krw,
        bucket_topn=result.bucket_topn,
        ticker_summaries=getattr(result, "ticker_summaries", []),
        core_start_dt=result.start_date,
        emit_to_logger=False,
        section_start_index=2,  # 요약 섹션은 2번부터 시작
    )
    lines.extend(summary_lines)

    with path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return path
