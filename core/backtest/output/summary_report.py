from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from core.backtest.output.formatters import BUCKET_NAMES, _is_finite_number, _usd_money
from core.backtest.output.summary_formatter import format_period_return_with_listing_date
from utils.account_registry import get_account_settings
from utils.formatters import format_pct_change
from utils.logger import get_app_logger
from utils.report import format_kr_money, render_table_eaw
from utils.settings_loader import resolve_strategy_params

if TYPE_CHECKING:
    pass

logger = get_app_logger()


def print_backtest_summary(
    *,
    summary: dict[str, Any],
    account_id: str,
    country_code: str,
    backtest_start_date: str,
    initial_capital_krw: float,
    bucket_topn: int,
    ticker_summaries: list[dict[str, Any]],
    core_start_dt: pd.Timestamp,
    emit_to_logger: bool = True,
    section_start_index: int = 1,
) -> list[str]:
    """Return a formatted summary for an account backtest run."""

    account_settings = get_account_settings(account_id)
    currency = str(summary.get("currency") or account_settings.get("currency", "KRW")).upper()

    strategy_cfg = account_settings.get("strategy", {}) or {}
    strategy_tuning = resolve_strategy_params(strategy_cfg)
    merged_strategy = dict(strategy_tuning)

    replace_threshold = strategy_tuning.get("REPLACE_SCORE_THRESHOLD", 0.0)

    initial_capital_local = float(
        summary.get("initial_capital_local", summary.get("initial_capital", initial_capital_krw))
    )
    initial_capital_krw_value = float(summary.get("initial_capital_krw", initial_capital_krw))
    final_value_local = float(summary.get("final_value_local", summary.get("final_value", 0.0)))
    final_value_krw_value = float(summary.get("final_value_krw", final_value_local))

    money_formatter = _usd_money if currency == "USD" else format_kr_money

    output_lines: list[str] = []
    section_counter = section_start_index

    def add(line: str = "") -> None:
        output_lines.append(line)

    def ensure_blank_line() -> None:
        if output_lines and output_lines[-1] != "":
            add("")

    def add_section_heading(title: str) -> None:
        nonlocal section_counter
        ensure_blank_line()
        add(f"{section_counter}. ========= {title} ==========")
        section_counter += 1

    ma_month = merged_strategy.get("MA_MONTH")
    momentum_label = f"{ma_month}개월" if ma_month is not None else "N/A"

    # [수정] 전체 종목 수 한도 표시를 명확히 함
    holdings_limit = summary.get("holdings_limit") or (bucket_topn * 5)  # 5버킷 기준 기본값
    try:
        holdings_limit = int(holdings_limit)
    except (TypeError, ValueError):
        holdings_limit = bucket_topn

    used_settings = {
        "버킷당 종목 수 (Bucket TopN)": bucket_topn,
        "전체 종목 수 한도 (Total Limit)": holdings_limit,
        "모멘텀 스코어 MA 기간": momentum_label,
        "교체 매매 점수 임계값": replace_threshold,
    }

    add_section_heading("주별 성과 요약")
    weekly_summary_rows = summary.get("weekly_summary") or []
    if isinstance(weekly_summary_rows, list) and weekly_summary_rows:
        headers = ["주차(종료일)", "보유종목", "평가금액", "주간 수익률", "누적 수익률"]
        table_rows = []
        for item in weekly_summary_rows:
            week_label = item.get("week_end") or "-"
            value = item.get("value")
            held_count = item.get("held_count", 0)
            max_topn = item.get("max_topn", 0)
            weekly_ret = item.get("weekly_return_pct")
            cum_ret = item.get("cumulative_return_pct")
            value_display = money_formatter(value) if _is_finite_number(value) else "-"
            holdings_display = f"{held_count}/{max_topn}" if max_topn > 0 else "-"
            table_rows.append(
                [
                    week_label,
                    holdings_display,
                    value_display,
                    format_pct_change(weekly_ret) if _is_finite_number(weekly_ret) else "-",
                    format_pct_change(cum_ret) if _is_finite_number(cum_ret) else "-",
                ]
            )
        aligns = ["left", "center", "right", "right", "right"]
        output_lines.extend(render_table_eaw(headers, table_rows, aligns))
    else:
        add("| 주별 성과 데이터를 찾을 수 없습니다.")

    add_section_heading("월별 성과 요약")
    if (
        "monthly_returns" in summary
        and isinstance(summary["monthly_returns"], (pd.Series, pd.DataFrame))
        and not summary["monthly_returns"].empty
    ):
        monthly_returns = summary["monthly_returns"]
        yearly_returns = summary.get("yearly_returns", pd.Series(dtype=float))
        pivot_df = (
            monthly_returns.mul(100)
            .to_frame("return")
            .pivot_table(index=monthly_returns.index.year, columns=monthly_returns.index.month, values="return")
        )
        if not yearly_returns.empty:
            yearly_series = yearly_returns.mul(100)
            yearly_series.index = yearly_series.index.year
            pivot_df["연간"] = yearly_series

        headers = ["구분"] + [f"{m}월" for m in range(1, 13)] + ["연간"]
        rows_data = []
        all_years = sorted(pivot_df.index.unique())
        bench_monthly_returns_map = summary.get("benchmark_monthly_returns") or {}
        bench_pivots = {}

        for bench_name, bench_m_ret in bench_monthly_returns_map.items():
            if bench_m_ret is not None and not bench_m_ret.empty:
                b_pivot = (
                    bench_m_ret.mul(100)
                    .to_frame("return")
                    .pivot_table(index=bench_m_ret.index.year, columns=bench_m_ret.index.month, values="return")
                )
                b_annual = bench_m_ret.groupby(bench_m_ret.index.year).apply(lambda x: (1 + x).prod() - 1).mul(100)
                b_pivot["연간"] = b_annual
                bench_pivots[bench_name] = b_pivot
                all_years = sorted(list(set(all_years) | set(b_pivot.index)))

        for year in all_years:
            if year in pivot_df.index:
                row = pivot_df.loc[year]
                row_data = [str(year)]
                for month in range(1, 13):
                    val = row.get(month)
                    row_data.append(format_pct_change(val) if pd.notna(val) else "-")
                yearly_val = row.get("연간")
                row_data.append(format_pct_change(yearly_val) if pd.notna(yearly_val) else "-")
                rows_data.append(row_data)

            for bench_name, b_pivot in bench_pivots.items():
                if year in b_pivot.index:
                    b_row = b_pivot.loc[year]
                    b_row_data = [f"{bench_name}"]
                    for month in range(1, 13):
                        val = b_row.get(month)
                        b_row_data.append(format_pct_change(val) if pd.notna(val) else "-")
                    b_yearly_val = b_row.get("연간")
                    b_row_data.append(format_pct_change(b_yearly_val) if pd.notna(b_yearly_val) else "-")
                    rows_data.append(b_row_data)

        aligns = ["left"] + ["right"] * (len(headers) - 1)
        output_lines.extend(render_table_eaw(headers, rows_data, aligns))
    else:
        add("| 월별 성과 데이터가 없어 표시할 수 없습니다.")

    add_section_heading("종목별 성과 요약")
    if ticker_summaries:
        headers = ["티커", "종목명", "총 기여도", "실현손익", "미실현손익", "거래횟수", "승률", "기간수익률"]
        sorted_summaries = sorted(ticker_summaries, key=lambda x: x["total_contribution"], reverse=True)
        rows = [
            [
                s["ticker"],
                s["name"],
                money_formatter(s["total_contribution"]),
                money_formatter(s["realized_profit"]),
                money_formatter(s["unrealized_profit"]),
                f"{s['total_trades']}회",
                f"{s['win_rate']:.1f}%",
                format_period_return_with_listing_date(s, core_start_dt),
            ]
            for s in sorted_summaries
        ]
        aligns = ["right", "left", "right", "right", "right", "right", "right", "right"]
        output_lines.extend(render_table_eaw(headers, rows, aligns))
    else:
        add("| 종목별 성과 데이터가 없어 표시할 수 없습니다.")

    add_section_heading("버킷별 성과 요약")
    bucket_summaries = summary.get("bucket_summary") or []
    if isinstance(bucket_summaries, list) and bucket_summaries:
        headers = ["버킷", "종목 수", "총 기여도", "실현손익", "미실현손익"]
        rows = []
        for b in bucket_summaries:
            bid = b.get("bucket_id")
            b_name = BUCKET_NAMES.get(bid, str(bid))
            b_display = f"{bid}. {b_name}" if isinstance(bid, int) else str(bid)
            rows.append(
                [
                    b_display,
                    f"{b.get('ticker_count', 0)}개",
                    money_formatter(b.get("total_contribution", 0.0)),
                    money_formatter(b.get("realized_profit", 0.0)),
                    money_formatter(b.get("unrealized_profit", 0.0)),
                ]
            )
        aligns = ["left", "right", "right", "right", "right"]
        output_lines.extend(render_table_eaw(headers, rows, aligns))
    else:
        add("| 버킷별 성과 데이터를 찾을 수 없습니다.")

    add_section_heading("지표 설명")
    add("  - Sharpe: 위험(변동성) 대비 수익률. 높을수록 좋음 (기준: >1 양호, >2 우수).")
    add(
        "  - SDR (Sharpe/MDD): Sharpe를 MDD(%)로 나눈 값. 수익 대비 최대 낙폭 효율성. 높을수록 우수 (기준: >0.2 양호, >0.3 우수)."
    )

    add_section_heading("백테스트 결과 요약")
    add(f"| 계정: {account_id.upper()} ({country_code.upper()})")
    try:
        start_dt, end_dt = pd.to_datetime(summary["start_date"]), pd.to_datetime(summary["end_date"])
        delta = end_dt - start_dt
        years, remaining = delta.days // 365, delta.days % 365
        months, days = remaining // 30, remaining % 30
        period_str = (
            f"{years}년 {months}개월 {days}일"
            if years > 0
            else (f"{months}개월 {days}일" if months > 0 else f"{days}일")
        )
    except Exception:
        period_str = ""
    add(f"| 기간: {summary['start_date']} ~ {summary['end_date']} ({period_str})")
    add(f"| 거래 수(Trades): {int(summary.get('turnover', 0))}회")
    add("\n[ 전략 설정 ]")
    for k, v in used_settings.items():
        add(f"| {k}: {v}")
    add("\n[ 자산 현황 ]")
    add(f"| 초기 자본: {money_formatter(initial_capital_local)}")
    if currency != "KRW":
        add(f"| 초기 자본 (KRW): {format_kr_money(initial_capital_krw_value)}")
    add(f"| 최종 자산: {money_formatter(final_value_local)}")
    if currency != "KRW":
        add(f"| 최종 자산 (KRW): {format_kr_money(final_value_krw_value)}")

    benchmarks_info = summary.get("benchmarks")
    headers = ["구분", "포트폴리오"]
    has_benchmark = False
    if isinstance(benchmarks_info, list) and benchmarks_info:
        has_benchmark = True
        for bench in benchmarks_info:
            headers.append(f"{bench.get('name') or bench.get('ticker') or '-'}(벤치마크)")
    elif summary.get("benchmark_cum_ret_pct") is not None:
        has_benchmark = True
        headers.append(f"{summary.get('benchmark_name') or 'Benchmark'}(벤치마크)")

    row_ret = ["기간수익률", format_pct_change(summary["period_return"])]
    row_cagr = ["CAGR", format_pct_change(summary["cagr"])]
    row_mdd = ["MDD", format_pct_change(-summary["mdd"])]
    row_sharpe = ["Sharpe", f"{summary.get('sharpe', 0.0):.2f}"]
    row_sdr = ["SDR", f"{summary.get('sharpe_to_mdd', 0.0):.3f}"]
    row_trades = ["Trades", f"{int(summary.get('turnover', 0))}"]

    if isinstance(benchmarks_info, list) and benchmarks_info:
        for bench in benchmarks_info:
            row_ret.append(format_pct_change(bench.get("cumulative_return_pct")))
            row_cagr.append(format_pct_change(bench.get("cagr_pct")))
            bm_mdd = bench.get("mdd")
            row_mdd.append(format_pct_change(-bm_mdd) if bm_mdd is not None else "-")
            bm_sharpe = bench.get("sharpe")
            row_sharpe.append(f"{float(bm_sharpe):.2f}" if bm_sharpe is not None else "-")
            bm_sdr = bench.get("sharpe_to_mdd")
            row_sdr.append(f"{float(bm_sdr):.3f}" if bm_sdr is not None else "-")
            row_trades.append("-")
    elif has_benchmark:
        row_ret.append(format_pct_change(summary.get("benchmark_cum_ret_pct")))
        row_cagr.append(format_pct_change(summary.get("benchmark_cagr_pct")))
        for r in [row_mdd, row_sharpe, row_sdr, row_trades]:
            r.append("-")

    aligns = ["left"] + ["right"] * (len(headers) - 1)
    output_lines.extend(
        render_table_eaw(headers, [row_ret, row_cagr, row_mdd, row_sharpe, row_sdr, row_trades], aligns)
    )

    if emit_to_logger:
        for line in output_lines:
            logger.info(line)
    return output_lines
