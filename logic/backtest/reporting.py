"""Utility helpers for presenting backtest results."""

from __future__ import annotations

import io
import math
import sys
from datetime import datetime
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import pandas as pd

from logic.backtest.country_runner import CountryBacktestResult
from logic.entry_point import DECISION_CONFIG
from utils.country_registry import get_country_settings
from utils.notification import build_summary_line_from_summary_data
from utils.report import (
    format_aud_money,
    format_aud_price,
    format_kr_money,
    format_usd_money,
    format_usd_price,
    render_table_eaw,
)

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[2] / "data" / "results"


def _default_months_range() -> int:
    try:
        module = import_module("backtest")
        return int(getattr(module, "TEST_MONTHS_RANGE"))
    except (ModuleNotFoundError, AttributeError, ValueError, TypeError):
        return 12


# ---------------------------------------------------------------------------
# Console summary helpers
# ---------------------------------------------------------------------------


def format_period_return_with_listing_date(
    series_summary: Dict[str, Any], core_start_dt: pd.Timestamp
) -> str:
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
    country: str,
    test_months_range: int,
    initial_capital_krw: float,
    portfolio_topn: int,
    ticker_summaries: List[Dict[str, Any]],
    core_start_dt: pd.Timestamp,
) -> None:
    """Print a concise summary for a country backtest run."""

    # Lazy import to avoid circular dependency during settings import
    from settings import common as common_settings

    country_settings = get_country_settings(country)
    currency = country_settings.get("currency", "KRW")
    precision = country_settings.get("precision", {}).get(
        "amt_precision", 0
    ) or country_settings.get("amt_precision", 0)
    try:
        precision = int(precision)
    except (TypeError, ValueError):
        precision = 0

    strategy_cfg = country_settings.get("strategy", {}) or {}
    if not isinstance(strategy_cfg, dict):
        strategy_cfg = {}

    if "tuning" in strategy_cfg or "static" in strategy_cfg:
        strategy_tuning = (
            strategy_cfg.get("tuning") if isinstance(strategy_cfg.get("tuning"), dict) else {}
        )
        strategy_static = (
            strategy_cfg.get("static") if isinstance(strategy_cfg.get("static"), dict) else {}
        )
    else:  # 구 포맷과의 호환성
        strategy_tuning = strategy_cfg
        strategy_static = strategy_cfg

    merged_strategy = dict(strategy_static)
    merged_strategy.update(strategy_tuning)

    cooldown_days = int(
        strategy_static.get("COOLDOWN_DAYS", strategy_cfg.get("COOLDOWN_DAYS", 0)) or 0
    )
    replace_threshold = strategy_tuning.get("REPLACE_SCORE_THRESHOLD")
    if replace_threshold is None:
        replace_threshold = strategy_cfg.get("REPLACE_SCORE_THRESHOLD", 0.5)

    if currency == "AUD":
        money_formatter = format_aud_money
    elif currency == "USD":
        money_formatter = format_usd_money
    else:
        money_formatter = format_kr_money

    benchmark_name = "S&P 500"

    summary_lines = [
        "\n" + "=" * 30 + "\n 백테스트 결과 요약 ".center(30, "=") + "\n" + "=" * 30,
        f"| 기간: {summary['start_date']} ~ {summary['end_date']} ({test_months_range} 개월)",
    ]

    risk_off_periods = summary.get("risk_off_periods")
    if isinstance(risk_off_periods, pd.DataFrame):
        if not risk_off_periods.empty:
            for _, row in risk_off_periods.iterrows():
                start = pd.to_datetime(row.get("start"))
                end = pd.to_datetime(row.get("end"))
                summary_lines.append(
                    f"| 투자 중단: {start.strftime('%Y-%m-%d')} ~ {end.strftime('%Y-%m-%d')}"
                )
        else:
            summary_lines.append("| 투자 중단: N/A")
    elif risk_off_periods:
        for start, end in risk_off_periods:
            summary_lines.append(
                f"| 투자 중단: {start.strftime('%Y-%m-%d')} ~ {end.strftime('%Y-%m-%d')}"
            )
    else:
        summary_lines.append("| 투자 중단: N/A")

    summary_lines.extend(
        [
            f"| 초기 자본: {money_formatter(summary['initial_capital_krw'])}",
            f"| 최종 자산: {money_formatter(summary['final_value'])}",
            (
                f"| 누적 수익률: {summary['cumulative_return_pct']:+.2f}% "
                f"({benchmark_name}: {summary.get('benchmark_cum_ret_pct', 0.0):+.2f}%)"
            ),
            (
                f"| CAGR (연간 복리 성장률): {summary['cagr_pct']:+.2f}% "
                f"({benchmark_name}: {summary.get('benchmark_cagr_pct', 0.0):+.2f}%)"
            ),
            f"| MDD (최대 낙폭): {-summary['mdd_pct']:.2f}%",
            f"| Sharpe Ratio: {summary.get('sharpe_ratio', 0.0):.2f}",
            f"| Sortino Ratio: {summary.get('sortino_ratio', 0.0):.2f}",
            f"| Calmar Ratio: {summary.get('calmar_ratio', 0.0):.2f}",
            f"| Ulcer Index: {summary.get('ulcer_index', 0.0):.2f}",
            f"| CUI (Calmar/Ulcer): {summary.get('cui', 0.0):.2f}",
            "=" * 30,
        ]
    )

    print("\n" + "=" * 30 + "\n 사용된 설정값 ".center(30, "=") + "\n" + "=" * 30)
    if "MA_PERIOD" not in merged_strategy or merged_strategy.get("MA_PERIOD") is None:
        raise ValueError(f"'{country}' 국가 설정에 'strategy.MA_PERIOD' 값이 필요합니다.")
    ma_period = merged_strategy["MA_PERIOD"]
    momentum_label = f"{ma_period}일"

    holding_stop_loss_pct = strategy_static.get(
        "HOLDING_STOP_LOSS_PCT",
        strategy_cfg.get("HOLDING_STOP_LOSS_PCT"),
    )
    if holding_stop_loss_pct is None:
        raise ValueError("strategy 설정에 'HOLDING_STOP_LOSS_PCT' 값이 필요합니다.")
    stop_loss_label = f"{holding_stop_loss_pct}%"

    market_regime_enabled = getattr(common_settings, "MARKET_REGIME_FILTER_ENABLED", None)
    if market_regime_enabled is None:
        market_regime_enabled = merged_strategy.get("MARKET_REGIME_FILTER_ENABLED")
    if market_regime_enabled is None:
        raise ValueError("공통 또는 국가 전략 설정에 'MARKET_REGIME_FILTER_ENABLED' 값이 필요합니다.")

    used_settings = {
        "국가": country.upper(),
        "테스트 기간": f"최근 {test_months_range}개월",
        "초기 자본": money_formatter(initial_capital_krw),
        "포트폴리오 종목 수 (TopN)": portfolio_topn,
        "모멘텀 스코어 MA 기간": momentum_label,
        "교체 매매 점수 임계값": replace_threshold,
        "개별 종목 손절매": stop_loss_label,
        "매도 후 재매수 금지 기간": f"{cooldown_days}일",
        "시장 위험 필터": "활성" if market_regime_enabled else "비활성",
    }

    for key, value in used_settings.items():
        print(f"| {key}: {value}")
    print("=" * 30)

    print("\n[지표 설명]")
    print("  - Sharpe Ratio (샤프 지수): 위험(변동성) 대비 수익률. 높을수록 좋음 (기준: >1 양호, >2 우수).")
    print("  - Sortino Ratio (소티노 지수): 하락 위험 대비 수익률. 높을수록 좋음 (기준: >2 양호, >3 우수).")
    print("  - Calmar Ratio (칼마 지수): 최대 낙폭 대비 연간 수익률. 높을수록 좋음 (기준: >1 양호, >3 우수).")
    print("  - Ulcer Index (얼서 지수): 고점 대비 낙폭의 지속성과 깊이를 반영. 낮을수록 안정적.")

    if "monthly_returns" in summary and not summary["monthly_returns"].empty:
        print("\n" + "=" * 30 + "\n 월별 성과 요약 ".center(30, "=") + "\n" + "=" * 30)

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
        print("\n" + "\n".join(render_table_eaw(headers, rows_data, aligns)))

    if ticker_summaries:
        print("\n" + "=" * 30 + "\n 종목별 성과 요약 ".center(30, "=") + "\n" + "=" * 30)
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

        sorted_summaries = sorted(
            ticker_summaries, key=lambda x: x["total_contribution"], reverse=True
        )

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
        print("\n" + "\n".join(table_lines))

    for line in summary_lines:
        print(line)


# ---------------------------------------------------------------------------
# Backtest log helpers
# ---------------------------------------------------------------------------


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not math.isnan(value) and not math.isinf(value)


def _format_quantity(amount: float, precision: int) -> str:
    if not _is_finite_number(amount):
        return "-"
    if precision <= 0:
        return f"{int(round(amount)):,}"
    return f"{amount:,.{precision}f}".rstrip("0").rstrip(".")


def _resolve_formatters(country_settings: Dict[str, Any]):
    precision = country_settings.get("precision", {})
    currency = str(precision.get("currency") or country_settings.get("currency", "KRW")).upper()
    qty_precision = int(precision.get("qty_precision", 0) or 0)
    price_precision = int(precision.get("price_precision", 0) or 0)

    if currency == "AUD":
        return currency, format_aud_money, format_aud_price, qty_precision, price_precision
    if currency == "USD":
        return currency, format_usd_money, format_usd_price, qty_precision, price_precision

    def _krw_price(value: float) -> str:
        if not _is_finite_number(value):
            return "-"
        return f"{int(round(value)):,}"

    return currency, format_kr_money, _krw_price, qty_precision, price_precision


def _format_date_kor(ts: pd.Timestamp) -> str:
    ts = pd.to_datetime(ts)
    return f"{ts.year}년 {ts.month}월 {ts.day}일"


def _build_daily_table_rows(
    *,
    result: CountryBacktestResult,
    country_settings: Dict[str, Any],
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
            [str(t) for t in result.ticker_timeseries.keys() if str(t).upper() != "CASH"],
            key=lambda x: str(x).upper(),
        )
    )

    for _, ticker in enumerate(tickers_order, 1):
        ts = result.ticker_timeseries.get(ticker)
        if ts is None:
            ts = result.ticker_timeseries.get(str(ticker).upper())
        if ts is None or target_date not in ts.index:
            continue

        row = ts.loc[target_date]
        ticker_key = str(ticker).upper()
        meta = result.ticker_meta.get(ticker_key, {})
        evaluation = result.evaluated_records.get(ticker_key, {})

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
        total_value_safe = (
            total_value if _is_finite_number(total_value) and total_value > 0 else 0.0
        )
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

        price_display = (
            "1"
            if is_cash
            else price_formatter(price)
            if _is_finite_number(price) and price > 0
            else "-"
        )
        shares_display = "1" if is_cash else _format_quantity(shares, qty_precision)
        pv_display = money_formatter(pv)
        cost_basis = avg_cost * shares if _is_finite_number(avg_cost) and shares > 0 else 0.0
        eval_profit_value = 0.0 if is_cash else pv - cost_basis
        evaluated_profit = evaluation.get("realized_profit", 0.0)
        cumulative_profit_value = evaluated_profit + eval_profit_value
        evaluated_profit_display = money_formatter(eval_profit_value)
        evaluated_pct = (
            (eval_profit_value / cost_basis * 100.0)
            if cost_basis > 0 and _is_finite_number(eval_profit_value)
            else 0.0
        )
        evaluated_pct_display = f"{evaluated_pct:+.1f}%" if cost_basis > 0 else "-"
        initial_value = evaluation.get("initial_value") or cost_basis
        cumulative_pct = (
            (cumulative_profit_value / initial_value * 100.0)
            if initial_value and _is_finite_number(cumulative_profit_value)
            else 0.0
        )
        cumulative_pct_display = f"{cumulative_pct:+.1f}%" if initial_value else "-"
        score_display = f"{float(score):.1f}" if _is_finite_number(score) else "-"
        filter_display = f"{int(filter_val)}일" if _is_finite_number(filter_val) else "-"
        weight_display = f"{weight:.0f}%"
        if is_cash and total_value_safe > 0:
            cash_ratio = (total_cash / total_value_safe) if _is_finite_number(total_cash) else 0.0
            weight_display = f"{cash_ratio * 100.0:.0f}%"

        phrase = note or str(row.get("phrase", ""))

        decision_order = DECISION_CONFIG.get(decision, {}).get("order", 99)
        score_val = float(score) if _is_finite_number(score) else float("-inf")
        sort_key = (
            0 if is_cash else 1,
            decision_order,
            -score_val,
            ticker_key,
        )

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
            f"{int(filter_val)}일" if _is_finite_number(filter_val) else "-",
            phrase,
        ]
        entries.append((sort_key, row_data))

    entries.sort(key=lambda item: item[0])

    sorted_rows: List[List[str]] = []
    for idx, (_, row) in enumerate(entries, 1):
        row[0] = str(idx)
        sorted_rows.append(row)

    return sorted_rows


def _generate_daily_report_lines(
    result: CountryBacktestResult,
    country_settings: Dict[str, Any],
) -> List[str]:
    (
        _currency,
        money_formatter,
        price_formatter,
        qty_precision,
        _price_precision,
    ) = _resolve_formatters(country_settings)

    portfolio_df = result.portfolio_timeseries
    lines: List[str] = []

    headers = [
        "#",
        "티커",
        "종목명",
        "카테고리",
        "상태",
        "보유일",
        "현재가",
        "일간(%)",
        "보유수량",
        "보유금액",
        "평가손익",
        "평가(%)",
        "누적손익",
        "누적(%)",
        "비중",
        "점수",
        "지속",
        "문구",
    ]
    aligns = [
        "right",
        "left",
        "left",
        "left",
        "center",
        "right",
        "right",
        "right",
        "right",
        "right",
        "right",
        "right",
        "right",
        "right",
        "right",
        "right",
        "right",
        "left",
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
            country_settings=country_settings,
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
        lines.append("")
        lines.extend(table_lines)

        for ticker_key, ts in result.ticker_timeseries.items():
            ticker_key_upper = str(ticker_key).upper()
            if target_date in ts.index:
                prev_rows_cache[ticker_key_upper] = ts.loc[target_date].copy()

    return lines


def dump_backtest_log(
    result: CountryBacktestResult,
    country_settings: Dict[str, Any],
    *,
    country: str,
    results_dir: Optional[Path | str] = None,
) -> Path:
    """Write a detailed backtest log to disk and return the file path."""

    base_dir = Path(results_dir) if results_dir is not None else DEFAULT_RESULTS_DIR
    base_dir.mkdir(parents=True, exist_ok=True)
    path = base_dir / f"backtest_{country}.txt"
    lines: List[str] = []

    start_stamp = datetime.now().isoformat(timespec="seconds")
    legacy_header = [
        f"백테스트 결과가 다음 파일에 저장됩니다: data/results/backtest_{country}.txt",
        "",
        f"KOR 국가의 ETF 종목들을 대상으로 {getattr(result, 'months_range', _default_months_range())}개월 기간 백테스트를 실행합니다.",
        f"국가별 설정 파일(settings/country/{country}.json)을 사용하여 전략을 적용합니다.",
        f"실행 시간: {start_stamp} | 테스트 기간: {result.start_date:%Y-%m-%d} ~ {result.end_date:%Y-%m-%d}",
        "",
    ]
    lines.extend(legacy_header)

    lines.append(f"백테스트 로그 생성: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(
        f"국가: {result.country.upper()} | 기간: {result.start_date:%Y-%m-%d} ~ {result.end_date:%Y-%m-%d}"
    )
    lines.append(f"초기 자본: {result.initial_capital:,.0f} | 포트폴리오 TOPN: {result.portfolio_topn}")
    lines.append("")

    daily_lines = _generate_daily_report_lines(result, country_settings)
    lines.extend(daily_lines)

    summary_buffer = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = summary_buffer
    try:
        print_backtest_summary(
            summary=result.summary,
            country=country,
            test_months_range=getattr(result, "months_range", _default_months_range()),
            initial_capital_krw=result.initial_capital,
            portfolio_topn=result.portfolio_topn,
            ticker_summaries=getattr(result, "ticker_summaries", []),
            core_start_dt=result.start_date,
        )
    finally:
        sys.stdout = original_stdout
    summary_text = summary_buffer.getvalue()
    if summary_text:
        lines.extend(summary_text.strip("\n").splitlines())

    with path.open("w", encoding="utf-8") as fp:
        fp.write("\n".join(lines) + "\n")
    return path


__all__ = [
    "print_backtest_summary",
    "format_period_return_with_listing_date",
    "dump_backtest_log",
]
