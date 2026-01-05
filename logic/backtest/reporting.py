"""백테스트 결과를 표시하기 위한 유틸리티 모음."""

from __future__ import annotations

import math
import numbers
from pathlib import Path
from typing import Any

import pandas as pd

from logic.backtest.account import AccountBacktestResult
from logic.entry_point import DECISION_CONFIG
from utils.account_registry import get_account_settings
from utils.data_loader import fetch_naver_etf_inav_snapshot, get_exchange_rate_series
from utils.formatters import format_pct_change
from utils.logger import get_app_logger
from utils.notification import build_summary_line_from_summary_data
from utils.report import format_kr_money, render_table_eaw
from utils.settings_loader import get_account_precision, resolve_strategy_params

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[2] / "zaccounts"
logger = get_app_logger()


# ---------------------------------------------------------------------------
# 콘솔 요약 출력 관련 헬퍼
# ---------------------------------------------------------------------------


def format_period_return_with_listing_date(series_summary: dict[str, Any], core_start_dt: pd.Timestamp) -> str:
    """Format period return percentage, optionally appending listing date."""

    period_return_pct = series_summary.get("period_return_pct", 0.0)
    listing_date = series_summary.get("listing_date")

    if listing_date and core_start_dt:
        listing_dt = pd.to_datetime(listing_date)
        if listing_dt > core_start_dt:
            return f"{format_pct_change(period_return_pct).strip()}({listing_date})"

    return format_pct_change(period_return_pct)


def print_backtest_summary(
    *,
    summary: dict[str, Any],
    account_id: str,
    country_code: str,
    test_months_range: int,
    initial_capital_krw: float,
    portfolio_topn: int,
    ticker_summaries: list[dict[str, Any]],
    core_start_dt: pd.Timestamp,
    category_summaries: list[dict[str, Any]] = [],
    emit_to_logger: bool = True,
    section_start_index: int = 1,
) -> list[str]:
    """Return a formatted summary for an account backtest run.

    Args:
        summary: 백테스트 요약 지표
        account_id: 계정 ID
        country_code: 국가 코드
        test_months_range: 테스트 기간(개월)
        initial_capital_krw: 초기 자본
        portfolio_topn: 포트폴리오 보유 종목 수
        ticker_summaries: 종목별 성과 요약 리스트
        core_start_dt: 백테스트 시작일
        emit_to_logger: True면 logger.info 로 출력도 수행

    Returns:
        출력용 문자열 리스트 (섹션 구분 포함)
    """

    account_settings = get_account_settings(account_id)
    currency = str(summary.get("currency") or account_settings.get("currency", "KRW")).upper()
    precision = account_settings.get("precision", {}).get("amt_precision", 0) or account_settings.get(
        "amt_precision", 0
    )
    try:
        precision = int(precision)
    except (TypeError, ValueError):
        precision = 0

    strategy_cfg = account_settings.get("strategy", {}) or {}
    if not isinstance(strategy_cfg, dict):
        strategy_cfg = {}

    strategy_tuning = resolve_strategy_params(strategy_cfg)

    merged_strategy = dict(strategy_tuning)

    # 검증은 get_account_strategy_sections에서 이미 완료됨 - 바로 사용
    cooldown_days = int(strategy_tuning["COOLDOWN_DAYS"])
    replace_threshold = strategy_tuning["REPLACE_SCORE_THRESHOLD"]

    initial_capital_local = float(
        summary.get("initial_capital_local", summary.get("initial_capital", initial_capital_krw))
    )
    initial_capital_krw_value = float(summary.get("initial_capital_krw", initial_capital_krw))
    final_value_local = float(summary.get("final_value_local", summary.get("final_value", 0.0)))
    final_value_krw_value = float(summary.get("final_value_krw", final_value_local))

    # 통화에 따라 적절한 포맷터 설정
    if currency == "USD":
        money_formatter = _usd_money
    else:
        money_formatter = format_kr_money

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

    # 설정값 계산 (나중에 사용)
    if "MA_PERIOD" not in merged_strategy or merged_strategy.get("MA_PERIOD") is None:
        raise ValueError(f"'{account_id}' 계정 설정에 'strategy.MA_PERIOD' 값이 필요합니다.")
    ma_period = merged_strategy["MA_PERIOD"]
    momentum_label = f"{ma_period}일"

    stop_loss_source = strategy_tuning.get("STOP_LOSS_PCT")
    try:
        holding_stop_loss_pct = float(stop_loss_source if stop_loss_source is not None else portfolio_topn)
    except (TypeError, ValueError):
        holding_stop_loss_pct = float(portfolio_topn)

    if abs(holding_stop_loss_pct - round(holding_stop_loss_pct)) < 1e-6:
        stop_loss_label = f"{int(round(holding_stop_loss_pct))}%"
    else:
        stop_loss_label = f"{holding_stop_loss_pct:.2f}%"

    used_settings = {
        "포트폴리오 종목 수 (TopN)": portfolio_topn,
        "모멘텀 스코어 MA 기간": momentum_label,
        "교체 매매 점수 임계값": replace_threshold,
        "개별 종목 손절매": stop_loss_label,
        "매수/매도 쿨다운": f"{cooldown_days}일",
    }

    def _align_korean_money_for_weekly(text: str) -> str:
        if currency != "KRW" or not isinstance(text, str):
            return text
        stripped = text.strip()
        sign = ""
        if stripped.startswith("-"):
            sign = "-"
            stripped = stripped[1:].strip()

        if "억" in stripped and "만원" in stripped:
            eok_part, remainder = stripped.split("억", 1)
            eok_part = eok_part.strip()
            remainder = remainder.strip()
            man_part = remainder.replace("만원", "").strip()
            man_part_padded = man_part.rjust(5)
            return f"{sign}{eok_part}억 {man_part_padded}만원"
        return text

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
            value_display = _align_korean_money_for_weekly(value_display)
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
        for line in render_table_eaw(headers, table_rows, aligns):
            add(line)
    else:
        add("| 주별 성과 데이터를 찾을 수 없습니다.")

    add_section_heading("월별 성과 요약")
    if "monthly_returns" in summary and not summary["monthly_returns"].empty:
        monthly_returns = summary["monthly_returns"]
        yearly_returns = summary["yearly_returns"]

        pivot_df = (
            monthly_returns.mul(100)
            .to_frame("return")
            .pivot_table(
                index=monthly_returns.index.year,
                columns=monthly_returns.index.month,
                values="return",
            )
        )

        if not yearly_returns.empty:
            yearly_series = yearly_returns.mul(100)
            yearly_series.index = yearly_series.index.year
            pivot_df["연간"] = yearly_series

        headers = ["구분"] + [f"{m}월" for m in range(1, 13)] + ["연간"]
        rows_data: list[list[str]] = []

        # 모든 연도를 수집 (포트폴리오 + 각 벤치마크)
        all_years = sorted(pivot_df.index.unique())

        bench_monthly_returns_map = summary.get("benchmark_monthly_returns") or {}
        bench_pivots = {}

        # 벤치마크 데이터 Pivot 미리 생성
        for bench_name, bench_m_ret in bench_monthly_returns_map.items():
            if bench_m_ret is not None and not bench_m_ret.empty:
                b_pivot = (
                    bench_m_ret.mul(100)
                    .to_frame("return")
                    .pivot_table(
                        index=bench_m_ret.index.year,
                        columns=bench_m_ret.index.month,
                        values="return",
                    )
                )

                # 벤치마크 연간 수익률 (복리)
                b_annual = bench_m_ret.groupby(bench_m_ret.index.year).apply(lambda x: (1 + x).prod() - 1).mul(100)
                b_pivot["연간"] = b_annual

                bench_pivots[bench_name] = b_pivot
                all_years = sorted(list(set(all_years) | set(b_pivot.index)))

        for year in all_years:
            # 1. 포트폴리오 (해당 연도 데이터가 있을 경우)
            if year in pivot_df.index:
                row = pivot_df.loc[year]
                # 첫 번째 컬럼에 연도 혹은 "포트폴리오" 명시
                # 사용자가 "2025" 처럼 첫 컬럼에 연도가 오길 원함 -> 포트폴리오를 기본으로 연도 표시
                row_data = [str(year)]
                for month in range(1, 13):
                    val = row.get(month)
                    row_data.append(format_pct_change(val) if pd.notna(val) else "-")
                yearly_val = row.get("연간")
                row_data.append(format_pct_change(yearly_val) if pd.notna(yearly_val) else "-")
                rows_data.append(row_data)

            # 3. 벤치마크 (해당 연도 데이터가 있을 경우)
            # 벤치마크가 여러 개일 수 있으므로 순회
            for bench_name, b_pivot in bench_pivots.items():
                if year in b_pivot.index:
                    b_row = b_pivot.loc[year]
                    # 벤치마크 이름 표시
                    b_row_data = [f"{bench_name}"]
                    for month in range(1, 13):
                        val = b_row.get(month)
                        b_row_data.append(format_pct_change(val) if pd.notna(val) else "-")
                    b_yearly_val = b_row.get("연간")
                    b_row_data.append(format_pct_change(b_yearly_val) if pd.notna(b_yearly_val) else "-")
                    rows_data.append(b_row_data)

        aligns = ["left"] + ["right"] * (len(headers) - 1)
        for line in render_table_eaw(headers, rows_data, aligns):
            add(line)
    else:
        add("| 월별 성과 데이터가 없어 표시할 수 없습니다.")

    add_section_heading("종목별 성과 요약")
    if ticker_summaries:
        headers = [
            "티커",
            "종목명",
            "카테고리",
            "총 기여도",
            "실현손익",
            "미실현손익",
            "거래횟수",
            "승률",
            "기간수익률",
        ]

        sorted_summaries = sorted(ticker_summaries, key=lambda x: x["total_contribution"], reverse=True)

        rows = [
            [
                s["ticker"],
                s["name"],
                s.get("category", "-"),
                money_formatter(s["total_contribution"]),
                money_formatter(s["realized_profit"]),
                money_formatter(s["unrealized_profit"]),
                f"{s['total_trades']}회",
                f"{s['win_rate']:.1f}%",
                format_period_return_with_listing_date(s, core_start_dt),
            ]
            for s in sorted_summaries
        ]

        aligns = ["right", "left", "left", "right", "right", "right", "right", "right", "right"]
        table_lines = render_table_eaw(headers, rows, aligns)
        for line in table_lines:
            add(line)
    else:
        add("| 종목별 성과 데이터가 없어 표시할 수 없습니다.")

    add_section_heading("카테고리별 성과 요약")
    if category_summaries:
        headers = [
            "카테고리",
            "총 기여도",
            "실현손익",
            "미실현손익",
            "거래횟수",
            "승률",
        ]
        rows = [
            [
                s["category"],
                money_formatter(s["total_contribution"]),
                money_formatter(s["realized_profit"]),
                money_formatter(s["unrealized_profit"]),
                f"{s['total_trades']}회",
                f"{s['win_rate']:.1f}%",
            ]
            for s in category_summaries
        ]
        aligns = ["left", "right", "right", "right", "right", "right"]
        table_lines = render_table_eaw(headers, rows, aligns)
        for line in table_lines:
            add(line)
    else:
        add("| 카테고리별 성과 데이터가 없어 표시할 수 없습니다.")

    add_section_heading("지표 설명")
    add("  - Sharpe: 위험(변동성) 대비 수익률. 높을수록 좋음 (기준: >1 양호, >2 우수).")
    add(
        "  - SDR (Sharpe/MDD): Sharpe를 MDD(%)로 나눈 값. 수익 대비 최대 낙폭 효율성. 높을수록 우수 (기준: >0.2 양호, >0.3 우수)."
    )

    add_section_heading("백테스트 결과 요약")
    # 기본 정보 통합
    add(f"| 계정: {account_id.upper()} ({country_code.upper()})")
    add(f"| 기간: {summary['start_date']} ~ {summary['end_date']} ({test_months_range} 개월)")
    add(f"| 거래 수(Trades): {int(summary.get('turnover', 0))}회")
    add("")
    # 사용된 설정값 통합
    add("[ 전략 설정 ]")
    for key, value in used_settings.items():
        add(f"| {key}: {value}")
    add("")
    # 자산 정보
    add("[ 자산 현황 ]")
    add(f"| 초기 자본: {money_formatter(initial_capital_local)}")
    if currency != "KRW":
        add(f"| 초기 자본 (KRW): {format_kr_money(initial_capital_krw_value)}")
    add(f"| 최종 자산: {money_formatter(final_value_local)}")
    if currency != "KRW":
        add(f"| 최종 자산 (KRW): {format_kr_money(final_value_krw_value)}")

    benchmarks_info = summary.get("benchmarks")

    # 테이블 헤더 생성
    headers = ["구분", "포트폴리오"]
    has_benchmark = False

    if isinstance(benchmarks_info, list) and benchmarks_info:
        has_benchmark = True
        for bench in benchmarks_info:
            name = str(bench.get("name") or bench.get("ticker") or "-").strip()
            headers.append(f"{name}(벤치마크)")
    else:
        # 단일 벤치마크 (레거시 지원)
        bm_ret = summary.get("benchmark_cum_ret_pct")
        if bm_ret is not None:
            has_benchmark = True
            bm_name = summary.get("benchmark_name") or "Benchmark"
            headers.append(f"{bm_name}(벤치마크)")

    # 데이터 행 생성
    row_ret = ["기간수익률", format_pct_change(summary["period_return"])]
    row_cagr = ["CAGR", format_pct_change(summary["cagr"])]
    row_mdd = ["MDD", format_pct_change(-summary["mdd"])]
    row_sharpe = ["Sharpe", f"{summary.get('sharpe', 0.0):.2f}"]
    row_sdr = ["SDR", f"{summary.get('sharpe_to_mdd', 0.0):.3f}"]
    row_trades = ["Trades", f"{int(summary.get('turnover', 0))}"]

    if isinstance(benchmarks_info, list) and benchmarks_info:
        for bench in benchmarks_info:
            # 수익률
            row_ret.append(format_pct_change(bench.get("cumulative_return_pct")))
            # CAGR
            row_cagr.append(format_pct_change(bench.get("cagr_pct")))
            # MDD (벤치마크 객체에 mdd가 없으면 - 표시)
            bm_mdd = bench.get("mdd")
            row_mdd.append(format_pct_change(-bm_mdd) if bm_mdd is not None else "-")
            # Sharpe
            bm_sharpe = bench.get("sharpe")
            row_sharpe.append(f"{float(bm_sharpe):.2f}" if bm_sharpe is not None else "-")
            # SDR
            bm_sdr = bench.get("sharpe_to_mdd")
            row_sdr.append(f"{float(bm_sdr):.3f}" if bm_sdr is not None else "-")
            # Trades (벤치마크는 해당 없음)
            row_trades.append("-")

    elif has_benchmark:
        # 단일 벤치마크 폴백
        row_ret.append(format_pct_change(summary.get("benchmark_cum_ret_pct")))
        row_cagr.append(format_pct_change(summary.get("benchmark_cagr_pct")))
        row_mdd.append("-")
        row_sharpe.append("-")
        row_sdr.append("-")
        row_trades.append("-")

    # 테이블 렌더링
    rows = [row_ret, row_cagr, row_mdd, row_sharpe, row_sdr, row_trades]
    # 정렬: 구분(Left), 나머지(Right)
    aligns = ["left"] + ["right"] * (len(headers) - 1)

    table_lines = render_table_eaw(headers, rows, aligns)
    for line in table_lines:
        add(line)

    if emit_to_logger:
        for line in output_lines:
            logger.info(line)

    return output_lines


# ---------------------------------------------------------------------------
# 백테스트 로그 출력 헬퍼
# ---------------------------------------------------------------------------


def _is_finite_number(value: Any) -> bool:
    if not isinstance(value, numbers.Number):
        return False
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(numeric)


def _format_quantity(amount: float, precision: int) -> str:
    if not _is_finite_number(amount):
        return "-"
    if precision <= 0:
        return f"{int(round(amount)):,}"
    return f"{amount:,.{precision}f}".rstrip("0").rstrip(".")


def _resolve_formatters(account_settings: dict[str, Any], account_id: str = ""):
    if not account_id:
        account_id = str(account_settings.get("account") or "").strip().lower()
    try:
        precision = get_account_precision(account_id)
    except Exception:
        precision = {}

    if not isinstance(precision, dict):
        precision = {}

    currency = str(precision.get("currency", "KRW")).strip().upper()
    qty_precision = int(precision.get("qty_precision", 0))
    price_precision = int(precision.get("price_precision", 0))

    digits = max(price_precision, 0)

    def _format_price(value: float) -> str:
        if not _is_finite_number(value):
            return "-"
        return f"{float(value):,.{digits}f}"

    return currency, _usd_money if currency == "USD" else format_kr_money, _format_price, qty_precision, digits


def _usd_money(value: float) -> str:
    if not _is_finite_number(value):
        return "-"

    is_negative = value < 0
    abs_val = abs(value)

    # 2 decimal places for USD
    if abs_val < 0.01 and abs_val != 0:
        formatted_val = f"{abs_val:,.4f}"
    else:
        formatted_val = f"{abs_val:,.2f}"

    if is_negative:
        return f"-${formatted_val}"
    return f"${formatted_val}"


def _format_date_kor(ts: pd.Timestamp) -> str:
    ts = pd.to_datetime(ts)
    weekday_map = {0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토", 6: "일"}
    weekday = weekday_map.get(ts.weekday(), "")
    return f"{ts.strftime('%Y-%m-%d')}({weekday})"


def _build_daily_table_rows(
    *,
    result: AccountBacktestResult,
    target_date: pd.Timestamp,
    total_value: float,
    total_cash: float,
    price_formatter,
    money_formatter,
    qty_precision: int,
    buy_date_map: dict[str, pd.Timestamp | None],
    holding_days_map: dict[str, int],
    prev_rows_cache: dict[str, pd.Series | None],
    price_overrides: dict[str, float] | None = None,
) -> list[list[str]]:
    entries: list[tuple[tuple[int, int, float, str], list[str]]] = []

    tickers_order: list[str] = []
    if "CASH" in result.ticker_timeseries:
        tickers_order.append("CASH")
    tickers_order.extend(
        sorted(
            [
                str(t)
                for t in result.ticker_timeseries.keys()
                if str(t).upper() != "CASH" and not str(t).startswith("__")
            ],
            key=lambda x: str(x).upper(),
        )
    )

    for _, ticker in enumerate(tickers_order, 1):
        ts = result.ticker_timeseries.get(ticker)
        if ts is None:
            ts = result.ticker_timeseries.get(str(ticker).upper())
        # DataFrame만 처리 (메타데이터 문자열 제외)
        if ts is None or not isinstance(ts, pd.DataFrame) or target_date not in ts.index:
            continue

        row = ts.loc[target_date]
        ticker_key = str(ticker).upper()
        meta = result.ticker_meta.get(ticker_key, {})

        price_val = row.get("price")
        shares_val = row.get("shares")

        avg_cost_val = row.get("avg_cost")

        price = float(price_val) if pd.notna(price_val) else 0.0
        # 실시간 가격 오버라이드 적용
        if price_overrides and ticker_key in price_overrides:
            price = float(price_overrides[ticker_key])

        shares = float(shares_val) if pd.notna(shares_val) else 0.0
        # 가격이 변경되었을 수 있으므로 pv 재계산 (단, shares가 0이면 0)
        pv = price * shares
        avg_cost = float(avg_cost_val) if pd.notna(avg_cost_val) else 0.0

        decision = str(row.get("decision", "")).upper()
        score = row.get("score")
        filter_val = row.get("filter")
        note = str(row.get("note", "") or "")

        is_cash = ticker_key == "CASH"
        if is_cash:
            price = 1.0
            shares = pv if pv else 1.0

        prev_row = prev_rows_cache.get(ticker_key)
        if prev_row is not None:
            prev_price_val = prev_row.get("price")
            prev_price = float(prev_price_val) if pd.notna(prev_price_val) else None
        else:
            prev_price = None

        daily_ret = ((price / prev_price) - 1.0) * 100.0 if prev_price else 0.0
        pv_safe = pv if _is_finite_number(pv) else 0.0
        total_value_safe = total_value if _is_finite_number(total_value) and total_value > 0 else 0.0
        weight = (pv_safe / total_value_safe * 100.0) if total_value_safe > 0 else 0.0

        if ticker_key not in buy_date_map:
            buy_date_map[ticker_key] = None
            holding_days_map[ticker_key] = 0

        if not is_cash:
            if shares > 0:
                if buy_date_map[ticker_key] is None or decision.startswith("BUY"):
                    buy_date_map[ticker_key] = target_date
                    holding_days_map[ticker_key] = 1
                else:
                    holding_days_map[ticker_key] += 1
            else:
                buy_date_map[ticker_key] = None
                holding_days_map[ticker_key] = 0
        else:
            buy_date_map[ticker_key] = None
            holding_days_map[ticker_key] = 0

        holding_days_display = str(holding_days_map.get(ticker_key, 0))

        if is_cash:
            price_display = "1"
        else:
            price_display = price_formatter(price) if _is_finite_number(price) else "-"
        shares_display = "1" if is_cash else _format_quantity(shares, qty_precision)

        # CASH의 미세한 잔액(0.01 미만)은 0으로 처리하여 서식 통일 ($0.0000 -> $0.00)
        if is_cash and abs(pv) < 0.01:
            pv = 0.0

        pv_display = money_formatter(pv)
        cost_basis = avg_cost * shares if _is_finite_number(avg_cost) and shares > 0 else 0.0
        eval_profit_value = 0.0 if is_cash else pv - cost_basis
        # 누적 손익 = 평가 손익 (현재 보유분만 계산, 실현 손익은 제외)
        # 이유: 백테스트 시작일 이전의 실현 손익이 포함되는 문제를 방지
        cumulative_profit_value = eval_profit_value
        evaluated_profit_display = money_formatter(eval_profit_value)
        evaluated_pct = (
            (eval_profit_value / cost_basis * 100.0) if cost_basis > 0 and _is_finite_number(eval_profit_value) else 0.0
        )
        evaluated_pct_display = f"{evaluated_pct:+.1f}%" if cost_basis > 0 else "-"
        # 누적 수익률 = 평가 수익률 (현재 보유분 기준)
        cumulative_pct_display = evaluated_pct_display
        score_display = f"{float(score):.1f}" if _is_finite_number(score) else "-"
        weight_display = f"{weight:.1f}%"
        if is_cash and total_value_safe > 0:
            cash_ratio = (total_cash / total_value_safe) if _is_finite_number(total_cash) else 0.0
            weight_display = f"{cash_ratio * 100.0:.1f}%"

        phrase = note or str(row.get("phrase", ""))

        decision_order = DECISION_CONFIG.get(decision, {}).get("order", 99)
        score_val = float(score) if _is_finite_number(score) else float("-inf")
        sort_key = (
            0 if is_cash else 1,
            decision_order,
            -score_val,  # MAPS 점수
            ticker_key,
        )

        rsi_score = row.get("rsi_score")
        rsi_score_display = f"{float(rsi_score):.1f}" if _is_finite_number(rsi_score) else "-"

        row_data = [
            "0",
            ticker_key,
            str(meta.get("name") or ticker_key),
            str(meta.get("category") or "-"),
            decision or "-",
            holding_days_display,
            price_display,
            f"{daily_ret:+.1f}%",
            shares_display,
            pv_display,
            evaluated_profit_display,
            evaluated_pct_display,
            money_formatter(cumulative_profit_value),
            cumulative_pct_display,
            weight_display,
            score_display,
            rsi_score_display,
            f"{int(filter_val)}일" if _is_finite_number(filter_val) else "-",
            phrase,
        ]
        entries.append((sort_key, row_data))

    entries.sort(key=lambda item: item[0])

    # 카테고리별 최고 점수 필터링을 위해 딕셔너리 형태로 변환
    from logic.backtest.filtering import filter_category_duplicates

    items_for_filter = []
    for sort_key, row_data in entries:
        # row_data: [순위, 티커, 종목명, 카테고리, 상태, ...]
        # sort_key: (is_cash, decision_order, -score, ticker)
        score_val = -sort_key[2] if len(sort_key) > 2 else 0.0  # 음수로 저장되어 있으므로 다시 양수로
        item_dict = {
            "ticker": row_data[1],  # 티커
            "category": row_data[3],  # 카테고리
            "state": row_data[4],  # 상태
            "score": score_val,
            "row_data": row_data,
            "sort_key": sort_key,
        }
        items_for_filter.append(item_dict)

    # 카테고리 정규화 함수
    def normalize_category(category: str) -> str:
        if not category or category == "-":
            return ""
        return str(category).strip().upper()

    # 필터링 적용
    filtered_items = filter_category_duplicates(items_for_filter, category_key_getter=normalize_category)

    # 다시 row 형태로 변환
    sorted_rows: list[list[str]] = []
    current_idx = 1
    for item in filtered_items:
        row = item["row_data"]
        ticker = item["ticker"]
        if str(ticker).upper() == "CASH":
            row[0] = "0"
        else:
            row[0] = str(current_idx)
            current_idx += 1
        sorted_rows.append(row)

    return sorted_rows


def _generate_daily_report_lines(
    result: AccountBacktestResult,
    account_settings: dict[str, Any],
) -> list[str]:
    (
        _currency,
        money_formatter,
        price_formatter,
        qty_precision,
        _price_precision,
    ) = _resolve_formatters(account_settings, result.account_id)

    portfolio_df = result.portfolio_timeseries
    lines: list[str] = []

    price_header = "현재가"

    headers = [
        "#",
        "티커",
        "종목명",
        "카테고리",
        "상태",
        "보유일",
        price_header,
        "일간(%)",
        "수량",
        "금액",
        "평가손익",
        "평가(%)",
        "누적손익",
        "누적(%)",
        "비중",
        "점수",
        "RSI",
        "지속",
        "문구",
    ]
    aligns = [
        "right",  # #
        "left",  # 티커
        "left",  # 종목명
        "left",  # 카테고리
        "center",  # 상태
        "right",  # 보유일
        "right",  # 현재가 계열
        "right",  # 일간(%)
        "right",  # 수량
        "right",  # 금액
        "right",  # 평가손익
        "right",  # 평가(%)
        "right",  # 누적손익
        "right",  # 누적(%)
        "right",  # 비중
        "right",  # 점수
        "right",  # RSI
        "right",  # 지속
        "left",  # 문구
    ]

    buy_date_map: dict[str, pd.Timestamp | None] = {}
    holding_days_map: dict[str, int] = {}
    prev_rows_cache: dict[str, pd.Series | None] = {}

    # 환율 데이터 프리패치 (필요 시)
    fx_series = None
    if _currency != "KRW":
        try:
            # 전체 기간에 대해 한 번에 로딩
            start_dt = portfolio_df.index.min()
            end_dt = portfolio_df.index.max()
            # 여유있게 앞뒤로 조절
            fx_series = get_exchange_rate_series(start_dt, end_dt)
        except Exception as e:
            logger.warning(f"리포팅 중 환율 정보 로드 실패: {e}")

    current_streak = 0

    for target_date in portfolio_df.index:
        portfolio_row = portfolio_df.loc[target_date]

        def _safe_float(value: Any) -> float:
            if _is_finite_number(value):
                return float(value)
            try:
                candidate = float(value)
            except (TypeError, ValueError):
                return 0.0
            return candidate if _is_finite_number(candidate) else 0.0

        def _safe_int(value: Any) -> int:
            if _is_finite_number(value):
                return int(value)
            try:
                candidate = float(value)
            except (TypeError, ValueError):
                return 0
            return int(candidate) if _is_finite_number(candidate) else 0

        total_value = _safe_float(portfolio_row.get("total_value"))
        total_cash = _safe_float(portfolio_row.get("total_cash"))
        total_holdings = _safe_float(portfolio_row.get("total_holdings"))
        daily_profit_loss = _safe_float(portfolio_row.get("daily_profit_loss"))
        daily_return_pct = _safe_float(portfolio_row.get("daily_return_pct"))
        eval_profit_loss = _safe_float(portfolio_row.get("evaluation_profit_loss"))
        eval_return_pct = _safe_float(portfolio_row.get("evaluation_return_pct"))
        cumulative_return_pct = _safe_float(portfolio_row.get("cumulative_return_pct"))
        held_count = _safe_int(portfolio_row.get("held_count"))

        # [Real-time Price Reflection]
        # 마지막 날(오늘)이고 한국 계정이면 실시간 가격(Naver iNAV)을 가져와 반영
        # recommend.py와 유사한 로직
        price_overrides = {}
        is_last_day = target_date == portfolio_df.index[-1]

        # account_settings에서 country_code 가져오기 (없으면 result에서 fallback)
        country_code_check = str(account_settings.get("country_code") or result.country_code).strip().lower()

        if is_last_day and country_code_check in ("kor", "kr"):
            # 현재 보유중이거나 관심있는 티커 목록 수집
            tickers_to_fetch = []
            for ticker_key, ts in result.ticker_timeseries.items():
                if target_date in ts.index:
                    tickers_to_fetch.append(str(ticker_key).upper())

            if tickers_to_fetch:
                try:
                    snapshot = fetch_naver_etf_inav_snapshot(tickers_to_fetch)
                    # 오버라이드 딕셔너리 생성
                    # 또한 Total Value 등 헤더값도 재계산 필요
                    # (Total Cash는 변하지 않음)
                    recalc_holdings_value = 0.0

                    for ticker_key in tickers_to_fetch:
                        if ticker_key in snapshot:
                            new_price = float(snapshot[ticker_key].get("nowVal", 0))
                            if new_price > 0:
                                price_overrides[ticker_key] = new_price

                        # PV 재계산 합산
                        # (오버라이드된 가격이 있으면 그거 쓰고, 없으면 기존 가격)
                        # 주의: shares 정보를 가져와야 함
                        tkr_ts = result.ticker_timeseries.get(ticker_key)
                        if tkr_ts is not None and target_date in tkr_ts.index:
                            tkr_row = tkr_ts.loc[target_date]
                            tkr_shares = _safe_float(tkr_row.get("shares"))
                            if tkr_shares > 0:
                                tkr_price = price_overrides.get(ticker_key) or _safe_float(tkr_row.get("price"))
                                recalc_holdings_value += tkr_shares * tkr_price

                    if price_overrides:
                        # 헤더 값 업데이트
                        total_holdings = recalc_holdings_value
                        total_value = total_cash + total_holdings

                        # 평가 손익 등 파생 지표 재계산은 복잡하므로
                        # daily_profit_loss와 eval_profit_loss 정도만 근사 업데이트
                        # 하지만 정확한 '어제 대비' 수익을 계산하기엔 여기서 어제 Total Value 접근이 필요.
                        # 여기서는 Total Value가 바뀌었으므로 그에 따른 단순 차이만 반영
                        pass

                except Exception as e:
                    logger.warning(f"리포팅 중 실시간 가격 패치 실패: {e}")

        # 환율 적용 로직:

        # 1. 포트폴리오 데이터는 현지 통화 기준 (USD 등)
        # 2. Daily Summary Header는 KRW 기준으로 표시 (User Request: "Header only in Korean Won is correct")
        # 3. Table은 현지 통화 기준 (USD) 유지

        header_money_formatter = money_formatter
        header_values = {
            "principal": float(result.initial_capital),
            "total_equity": total_value,
            "total_holdings_value": total_holdings,
            "total_cash": total_cash,
            "daily_profit_loss": daily_profit_loss,
            "cum_profit_loss": total_value - float(result.initial_capital),
        }

        # 비 KRW 계좌인 경우 KRW로 변환
        if _currency != "KRW":
            try:
                # 해당 일자의 환율 조회 (없으면 가장 최근 값)
                target_result = fx_series.asof(target_date)
                if pd.notna(target_result):
                    rate = float(target_result)
                    header_money_formatter = format_kr_money
                    # 환율 적용 (KRW 환산)
                    # 원금은 초기 환율 기준 고정일 수도 있지만,
                    # 일별 리포트에서는 '현재 가치'를 보여주는게 맞으므로 매일 변동될 수도 있고,
                    # 혹은 원금 자체는 초기 환율로 박아두고 평가금액만 변동시킬 수도 있음.
                    # 여기서는 심플하게 '현재 환율'로 모든 가치를 환산하여 보여줌.
                    # 단, 원금의 경우 '투입 당시 KRW'를 보여주는 게 더 직관적일 수 있으나
                    # 이미 result.initial_capital_krw가 있으므로 그걸 쓰는 게 나을 수도.
                    # 하지만 result.initial_capital_krw는 고정값.

                    # Header: 원금 -> initial_capital * initial_fx (fixed) or current_fx?
                    # User: "원금: 7만원" -> This implies fixed initial KRW amount.
                    # So use result.initial_capital_krw for principal.

                    header_values["principal"] = float(result.initial_capital_krw)  # 고정된 초기 KRW 원금

                    header_values["total_equity"] = total_value * rate
                    header_values["total_holdings_value"] = total_holdings * rate
                    header_values["total_cash"] = total_cash * rate
                    header_values["daily_profit_loss"] = daily_profit_loss * rate
                    header_values["cum_profit_loss"] = (total_value * rate) - float(result.initial_capital_krw)
            except Exception:
                pass

        summary_data = {
            "principal": header_values["principal"],
            "total_equity": header_values["total_equity"],
            "total_holdings_value": header_values["total_holdings_value"],
            "total_cash": header_values["total_cash"],
            "daily_profit_loss": header_values["daily_profit_loss"],
            "daily_return_pct": daily_return_pct,
            "eval_profit_loss": eval_profit_loss,  # 이건 개별 종목 합산이라 애매하지만, 일단 테이블 값이랑 맞추려면 USD가 맞나? 아님 헤더니까 KRW?
            # 헤더: "평가: -0.84%(-593원)" -> KRW여야 함.
            # eval_profit_loss는 (total_holdings - total_cost).
            # total_cost도 KRW로 변환 필요하지만 복잡함.
            # 단순히 total_equity - total_cash - total_cost(converted)? No.
            # 약식으로: eval_profit_loss(USD) * rate 사용.
            "eval_return_pct": eval_return_pct,
            "cum_profit_loss": header_values["cum_profit_loss"],
            "cum_return_pct": cumulative_return_pct,
            "held_count": held_count,
            "portfolio_topn": int(result.portfolio_topn),
        }

        # eval_profit_loss KRW 변환 (비 KRW 계좌 시)
        if _currency != "KRW" and summary_data["eval_profit_loss"]:
            try:
                target_result = fx_series.asof(target_date)
                if pd.notna(target_result):
                    summary_data["eval_profit_loss"] = eval_profit_loss * float(target_result)
            except Exception:
                pass

        prefix = f"{_format_date_kor(target_date)} |"
        summary_line = build_summary_line_from_summary_data(
            summary_data,
            header_money_formatter,
            use_html=False,
            prefix=prefix,
        )

        rows = _build_daily_table_rows(
            result=result,
            target_date=target_date,
            total_value=total_value,
            total_cash=total_cash,
            price_formatter=price_formatter,
            money_formatter=money_formatter,
            qty_precision=qty_precision,
            buy_date_map=buy_date_map,
            holding_days_map=holding_days_map,
            prev_rows_cache=prev_rows_cache,
            price_overrides=price_overrides,
        )

        table_lines = render_table_eaw(headers, rows, aligns)

        lines.append("")
        lines.append(summary_line)

        # [User Request] Add Consecutive Streak and Cash Weight line
        # Streak Calculation
        if daily_return_pct > 0.0001:  # float point tolerance
            if current_streak >= 0:
                current_streak += 1
            else:
                current_streak = 1
        elif daily_return_pct < -0.0001:
            if current_streak <= 0:
                current_streak -= 1
            else:
                current_streak = -1
        else:
            current_streak = 0

        streak_str = "변동 없음"
        if current_streak > 0:
            streak_str = f"{current_streak}일 연속 상승"
        elif current_streak < 0:
            streak_str = f"{abs(current_streak)}일 연속 하락"

        extra_line = f"{streak_str}"
        lines.append(extra_line)

        lines.extend(table_lines)

        for ticker_key, ts in result.ticker_timeseries.items():
            ticker_key_upper = str(ticker_key).upper()
            # DataFrame만 처리 (메타데이터 문자열 제외)
            if not isinstance(ts, pd.DataFrame):
                continue
            if target_date in ts.index:
                prev_rows_cache[ticker_key_upper] = ts.loc[target_date].copy()

    return lines


def dump_backtest_log(
    result: AccountBacktestResult,
    account_settings: dict[str, Any],
    *,
    results_dir: Path | str | None = None,
) -> Path:
    """Write a detailed backtest log to disk and return the file path."""

    account_id = result.account_id
    country_code = result.country_code

    # 계정별 폴더 생성
    if results_dir is not None:
        base_dir = Path(results_dir) / account_id / "results"
    else:
        base_dir = DEFAULT_RESULTS_DIR / account_id / "results"

    base_dir.mkdir(parents=True, exist_ok=True)

    # 파일명에 날짜 추가
    date_str = pd.Timestamp.now().strftime("%Y-%m-%d")
    path = base_dir / f"backtest_{date_str}.log"
    lines: list[str] = []

    lines.append(f"백테스트 로그 생성: {pd.Timestamp.now().isoformat(timespec='seconds')}")
    lines.append("1. ========= 일자별 성과 상세 ==========")

    daily_lines = _generate_daily_report_lines(result, account_settings)
    lines.extend(daily_lines)

    months_range_value = getattr(result, "months_range", None)
    if months_range_value is None:
        if isinstance(account_settings, dict):
            months_range_value = account_settings.get("strategy", {}).get("MONTHS_RANGE")
    if months_range_value is None:
        raise ValueError("MONTHS_RANGE 설정이 필요합니다. 계정 설정의 strategy.MONTHS_RANGE 값을 확인하세요.")
    months_range_value = int(months_range_value)

    summary_section = print_backtest_summary(
        summary=result.summary,
        account_id=account_id,
        country_code=country_code,
        test_months_range=months_range_value,
        initial_capital_krw=result.initial_capital_krw,
        portfolio_topn=result.portfolio_topn,
        ticker_summaries=getattr(result, "ticker_summaries", []),
        category_summaries=getattr(result, "category_summaries", []),
        core_start_dt=result.start_date,
        emit_to_logger=False,
        section_start_index=2,
    )
    if summary_section:
        lines.extend(summary_section)

    with path.open("w", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")
    return path


__all__ = [
    "print_backtest_summary",
    "format_period_return_with_listing_date",
    "dump_backtest_log",
]
