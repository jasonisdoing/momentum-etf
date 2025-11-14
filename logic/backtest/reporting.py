"""백테스트 결과를 표시하기 위한 유틸리티 모음."""

from __future__ import annotations

import math
import numbers
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from logic.backtest.account_runner import AccountBacktestResult
from logic.entry_point import DECISION_CONFIG
from utils.account_registry import get_account_settings
from utils.notification import build_summary_line_from_summary_data
from utils.report import format_kr_money, render_table_eaw
from utils.logger import get_app_logger
from utils.settings_loader import get_account_precision, resolve_strategy_params

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[2] / "zresults"
logger = get_app_logger()


# ---------------------------------------------------------------------------
# 콘솔 요약 출력 관련 헬퍼
# ---------------------------------------------------------------------------


def format_period_return_with_listing_date(series_summary: Dict[str, Any], core_start_dt: pd.Timestamp) -> str:
    """Format period return percentage, optionally appending listing date."""

    period_return_pct = series_summary.get("period_return_pct", 0.0)
    listing_date = series_summary.get("listing_date")

    if listing_date and core_start_dt:
        listing_dt = pd.to_datetime(listing_date)
        if listing_dt > core_start_dt:
            return f"{period_return_pct:+.2f}%({listing_date})"

    return f"{period_return_pct:+.2f}%"


def print_backtest_summary(
    *,
    summary: Dict[str, Any],
    account_id: str,
    country_code: str,
    test_months_range: int,
    initial_capital_krw: float,
    portfolio_topn: int,
    ticker_summaries: List[Dict[str, Any]],
    core_start_dt: pd.Timestamp,
    emit_to_logger: bool = True,
    section_start_index: int = 1,
) -> List[str]:
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
    precision = account_settings.get("precision", {}).get("amt_precision", 0) or account_settings.get("amt_precision", 0)
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

    initial_capital_local = float(summary.get("initial_capital_local", summary.get("initial_capital", initial_capital_krw)))
    initial_capital_krw_value = float(summary.get("initial_capital_krw", initial_capital_krw))
    final_value_local = float(summary.get("final_value_local", summary.get("final_value", 0.0)))
    final_value_krw_value = float(summary.get("final_value_krw", final_value_local))
    fx_rate_to_krw = float(summary.get("fx_rate_to_krw", 1.0) or 1.0)

    money_formatter = format_kr_money

    output_lines: List[str] = []
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

    add_section_heading("사용된 설정값")
    if "MA_PERIOD" not in merged_strategy or merged_strategy.get("MA_PERIOD") is None:
        raise ValueError(f"'{account_id}' 계정 설정에 'strategy.MA_PERIOD' 값이 필요합니다.")
    ma_period = merged_strategy["MA_PERIOD"]
    momentum_label = f"{ma_period}일"

    stop_loss_source = strategy_tuning.get("STOP_LOSS_PCT")
    try:
        holding_stop_loss_pct = float(stop_loss_source if stop_loss_source is not None else portfolio_topn)
    except (TypeError, ValueError):
        holding_stop_loss_pct = float(portfolio_topn)

    # 포트폴리오 N개 종목 중 한 종목만 N% 하락해 손절될 경우 전체 손실은 1%가 된다.
    if abs(holding_stop_loss_pct - round(holding_stop_loss_pct)) < 1e-6:
        stop_loss_label = f"{int(round(holding_stop_loss_pct))}%"
    else:
        stop_loss_label = f"{holding_stop_loss_pct:.2f}%"

    used_settings = {
        "계정": account_id.upper(),
        "시장 코드": country_code.upper(),
        "테스트 기간": f"최근 {test_months_range}개월",
        "초기 자본": money_formatter(initial_capital_local),
        "포트폴리오 종목 수 (TopN)": portfolio_topn,
        "모멘텀 스코어 MA 기간": momentum_label,
        "교체 매매 점수 임계값": replace_threshold,
        "개별 종목 손절매": stop_loss_label,
        "매도 후 재매수 금지 기간": f"{cooldown_days}일",
    }

    if currency != "KRW":
        used_settings["초기 자본 (KRW 환산)"] = format_kr_money(initial_capital_krw_value)

    if currency != "KRW" and fx_rate_to_krw != 1.0:
        used_settings["적용 환율 (KRW)"] = f"1 {currency} ≈ {format_kr_money(fx_rate_to_krw)}"

    for key, value in used_settings.items():
        add(f"| {key}: {value}")
    add_section_heading("지표 설명")
    add("  - Sharpe: 위험(변동성) 대비 수익률. 높을수록 좋음 (기준: >1 양호, >2 우수).")
    add("  - SDR (Sharpe/MDD): Sharpe를 MDD(%)로 나눈 값. 수익 대비 최대 낙폭 효율성. 높을수록 우수 (기준: >0.2 양호, >0.3 우수).")

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
            # 5칸 폭으로 오른쪽 정렬 (콤마 포함)
            man_part_padded = man_part.rjust(5)
            return f"{sign}{eok_part}억 {man_part_padded}만원"
        return text

    add_section_heading("주별 성과 요약")
    weekly_summary_rows = summary.get("weekly_summary") or []
    if isinstance(weekly_summary_rows, list) and weekly_summary_rows:
        headers = ["주차(종료일)", "평가금액", "주간 수익률", "누적 수익률"]
        table_rows = []
        for item in weekly_summary_rows:
            week_label = item.get("week_end") or "-"
            value = item.get("value")
            weekly_ret = item.get("weekly_return_pct")
            cum_ret = item.get("cumulative_return_pct")
            value_display = money_formatter(value) if _is_finite_number(value) else "-"
            value_display = _align_korean_money_for_weekly(value_display)
            table_rows.append(
                [
                    week_label,
                    value_display,
                    f"{weekly_ret:+.2f}%" if _is_finite_number(weekly_ret) else "-",
                    f"{cum_ret:+.2f}%" if _is_finite_number(cum_ret) else "-",
                ]
            )
        aligns = ["left", "right", "right", "right"]
        for line in render_table_eaw(headers, table_rows, aligns):
            add(line)
    else:
        add("| 주별 성과 데이터를 찾을 수 없습니다.")

    add_section_heading("월별 성과 요약")
    if "monthly_returns" in summary and not summary["monthly_returns"].empty:
        monthly_returns = summary["monthly_returns"]
        yearly_returns = summary["yearly_returns"]
        monthly_cum_returns = summary.get("monthly_cum_returns")

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

        cum_pivot_df = None
        if monthly_cum_returns is not None and not monthly_cum_returns.empty:
            cum_pivot_df = (
                monthly_cum_returns.mul(100)
                .to_frame("cum_return")
                .pivot_table(
                    index=monthly_cum_returns.index.year,
                    columns=monthly_cum_returns.index.month,
                    values="cum_return",
                )
            )

        headers = ["연도"] + [f"{m}월" for m in range(1, 13)] + ["연간"]
        rows_data: List[List[str]] = []
        for year, row in pivot_df.iterrows():
            monthly_row_data = [str(year)]
            for month in range(1, 13):
                val = row.get(month)
                monthly_row_data.append(f"{val:+.2f}%" if pd.notna(val) else "-")

            yearly_val = row.get("연간")
            monthly_row_data.append(f"{yearly_val:+.2f}%" if pd.notna(yearly_val) else "-")
            rows_data.append(monthly_row_data)

            if cum_pivot_df is not None and year in cum_pivot_df.index:
                cum_row = cum_pivot_df.loc[year]
                cum_row_data = ["  (누적)"]
                for month in range(1, 13):
                    cum_val = cum_row.get(month)
                    cum_row_data.append(f"{cum_val:+.2f}%" if pd.notna(cum_val) else "-")

                last_valid_month_index = cum_row.last_valid_index()
                if last_valid_month_index is not None:
                    cum_annual_val = cum_row[last_valid_month_index]
                    cum_row_data.append(f"{cum_annual_val:+.2f}%")
                else:
                    cum_row_data.append("-")
                rows_data.append(cum_row_data)

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
        table_lines = render_table_eaw(headers, rows, aligns)
        for line in table_lines:
            add(line)
    else:
        add("| 종목별 성과 데이터가 없어 표시할 수 없습니다.")

    add_section_heading("백테스트 결과 요약")
    add(f"| 기간: {summary['start_date']} ~ {summary['end_date']} ({test_months_range} 개월)")

    add(f"| 초기 자본: {money_formatter(initial_capital_local)}")
    if currency != "KRW":
        add(f"| 초기 자본 (KRW): {format_kr_money(initial_capital_krw_value)}")
    add(f"| 최종 자산: {money_formatter(final_value_local)}")
    if currency != "KRW":
        add(f"| 최종 자산 (KRW): {format_kr_money(final_value_krw_value)}")

    benchmarks_info = summary.get("benchmarks")
    if isinstance(benchmarks_info, list) and benchmarks_info:
        add("| 벤치마크 기간수익률(%)")
        for idx, bench in enumerate(benchmarks_info, start=1):
            name = str(bench.get("name") or bench.get("ticker") or "-").strip()
            ticker = str(bench.get("ticker") or "-").strip()
            cum_ret = bench.get("cumulative_return_pct")
            cum_label = f"{float(cum_ret):+.2f}%" if cum_ret is not None else "N/A"
            add(f"| {idx}. {name}({ticker}): {cum_label}")
    else:
        benchmark_name = summary.get("benchmark_name") or "S&P 500"
        bench_ret = summary.get("benchmark_cum_ret_pct")
        if bench_ret is not None:
            add(f"| 벤치마크 기간수익률(%): {benchmark_name}: {bench_ret:+.2f}%")

    if isinstance(benchmarks_info, list) and benchmarks_info:
        add("| 벤치마크 CAGR(%)")
        for idx, bench in enumerate(benchmarks_info, start=1):
            name = str(bench.get("name") or bench.get("ticker") or "-").strip()
            ticker = str(bench.get("ticker") or "-").strip()
            cagr_ret = bench.get("cagr_pct")
            cagr_label = f"{float(cagr_ret):+.2f}%" if cagr_ret is not None else "N/A"
            add(f"| {idx}. {name}({ticker}): {cagr_label}")
    else:
        benchmark_name = summary.get("benchmark_name") or "S&P 500"
        bench_cagr = summary.get("benchmark_cagr_pct")
        if bench_cagr is not None:
            add(f"| 벤치마크 CAGR(%): {benchmark_name}: {bench_cagr:+.2f}%")

    if isinstance(benchmarks_info, list) and benchmarks_info:
        add("| 벤치마크 SDR(Sharpe/MDD)")
        for idx, bench in enumerate(benchmarks_info, start=1):
            name = str(bench.get("name") or bench.get("ticker") or "-").strip()
            ticker = str(bench.get("ticker") or "-").strip()
            sdr = bench.get("sharpe_to_mdd")
            sdr_label = f"{float(sdr):.3f}" if sdr is not None else "N/A"
            add(f"| {idx}. {name}({ticker}): {sdr_label}")

    add(f"| 기간수익률(%): {summary['period_return']:+.2f}%")
    add(f"| CAGR(%): {summary['cagr']:+.2f}%")
    add(f"| MDD(%): {-summary['mdd']:.2f}%")
    add(f"| Sharpe: {summary.get('sharpe', 0.0):.2f}")
    add(f"| SDR (Sharpe/MDD): {summary.get('sharpe_to_mdd', 0.0):.3f}")

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


def _resolve_formatters(account_settings: Dict[str, Any]):
    account_id = str(account_settings.get("account") or "").strip().lower()
    try:
        precision = get_account_precision(account_id)
    except Exception:
        precision = {}

    if not isinstance(precision, dict):
        precision = {}

    currency = "KRW"
    qty_precision = int(precision.get("qty_precision", 0))
    price_precision = int(precision.get("price_precision", 0))

    digits = max(price_precision, 0)

    def _format_price(value: float) -> str:
        if not _is_finite_number(value):
            return "-"
        return f"{float(value):,.{digits}f}"

    def _krw_price(value: float) -> str:
        if not _is_finite_number(value):
            return "-"
        return f"{int(round(value)):,}"

    return currency, format_kr_money, _krw_price, qty_precision, digits


def _format_date_kor(ts: pd.Timestamp) -> str:
    ts = pd.to_datetime(ts)
    weekday_map = {0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토", 6: "일"}
    weekday = weekday_map.get(ts.weekday(), "")
    return f"{ts.year}년 {ts.month}월 {ts.day}일({weekday})"


def _build_daily_table_rows(
    *,
    result: AccountBacktestResult,
    target_date: pd.Timestamp,
    total_value: float,
    total_cash: float,
    price_formatter,
    money_formatter,
    qty_precision: int,
    buy_date_map: Dict[str, Optional[pd.Timestamp]],
    holding_days_map: Dict[str, int],
    prev_rows_cache: Dict[str, Optional[pd.Series]],
) -> List[List[str]]:
    entries: List[Tuple[Tuple[int, int, float, str], List[str]]] = []

    tickers_order: List[str] = []
    if "CASH" in result.ticker_timeseries:
        tickers_order.append("CASH")
    tickers_order.extend(
        sorted(
            [str(t) for t in result.ticker_timeseries.keys() if str(t).upper() != "CASH" and not str(t).startswith("__")],
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
        pv_val = row.get("pv")
        avg_cost_val = row.get("avg_cost")

        price = float(price_val) if pd.notna(price_val) else 0.0
        shares = float(shares_val) if pd.notna(shares_val) else 0.0
        pv = float(pv_val) if pd.notna(pv_val) else price * shares
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
        pv_display = money_formatter(pv)
        cost_basis = avg_cost * shares if _is_finite_number(avg_cost) and shares > 0 else 0.0
        eval_profit_value = 0.0 if is_cash else pv - cost_basis
        # 누적 손익 = 평가 손익 (현재 보유분만 계산, 실현 손익은 제외)
        # 이유: 백테스트 시작일 이전의 실현 손익이 포함되는 문제를 방지
        cumulative_profit_value = eval_profit_value
        evaluated_profit_display = money_formatter(eval_profit_value)
        evaluated_pct = (eval_profit_value / cost_basis * 100.0) if cost_basis > 0 and _is_finite_number(eval_profit_value) else 0.0
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
    from logic.common import filter_category_duplicates

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
    sorted_rows: List[List[str]] = []
    for idx, item in enumerate(filtered_items, 1):
        row = item["row_data"]
        row[0] = str(idx)
        sorted_rows.append(row)

    return sorted_rows


def _generate_daily_report_lines(
    result: AccountBacktestResult,
    account_settings: Dict[str, Any],
) -> List[str]:
    (
        _currency,
        money_formatter,
        price_formatter,
        qty_precision,
        _price_precision,
    ) = _resolve_formatters(account_settings)

    portfolio_df = result.portfolio_timeseries
    lines: List[str] = []

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

    buy_date_map: Dict[str, Optional[pd.Timestamp]] = {}
    holding_days_map: Dict[str, int] = {}
    prev_rows_cache: Dict[str, Optional[pd.Series]] = {}

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

        summary_data = {
            "principal": float(result.initial_capital),
            "total_equity": total_value,
            "total_holdings_value": total_holdings,
            "total_cash": total_cash,
            "daily_profit_loss": daily_profit_loss,
            "daily_return_pct": daily_return_pct,
            "eval_profit_loss": eval_profit_loss,
            "eval_return_pct": eval_return_pct,
            "cum_profit_loss": total_value - float(result.initial_capital),
            "cum_return_pct": cumulative_return_pct,
            "held_count": held_count,
            "portfolio_topn": int(result.portfolio_topn),
        }

        prefix = f"{_format_date_kor(target_date)} |"
        summary_line = build_summary_line_from_summary_data(
            summary_data,
            money_formatter,
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
        )

        table_lines = render_table_eaw(headers, rows, aligns)

        lines.append("")
        lines.append(summary_line)

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
    account_settings: Dict[str, Any],
    *,
    results_dir: Optional[Path | str] = None,
) -> Path:
    """Write a detailed backtest log to disk and return the file path."""

    account_id = result.account_id
    country_code = result.country_code

    # 계정별 폴더 생성
    if results_dir is not None:
        base_dir = Path(results_dir) / account_id
    else:
        base_dir = DEFAULT_RESULTS_DIR / account_id

    base_dir.mkdir(parents=True, exist_ok=True)

    # 파일명에 날짜 추가
    date_str = pd.Timestamp.now().strftime("%Y-%m-%d")
    path = base_dir / f"backtest_{date_str}.log"
    lines: List[str] = []

    lines.append(f"백테스트 로그 생성: {pd.Timestamp.now().isoformat(timespec='seconds')}")
    lines.append("1. ========= 기본정보 ==========")
    lines.append(f"계정: {account_id.upper()} ({country_code.upper()}) | 기간: {result.start_date:%Y-%m-%d} ~ {result.end_date:%Y-%m-%d}")
    base_line = f"초기 자본: {result.initial_capital:,.0f} {result.currency or 'KRW'}"
    if (result.currency or "KRW").upper() != "KRW":
        base_line += f" (≈ {result.initial_capital_krw:,.0f} KRW)"
    base_line += f" | 포트폴리오 TOPN: {result.portfolio_topn}"
    lines.append(base_line)
    lines.append("")
    lines.append("2. ========= 일자별 성과 상세 ==========")

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
        core_start_dt=result.start_date,
        emit_to_logger=False,
        section_start_index=3,
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
