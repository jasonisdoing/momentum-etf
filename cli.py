"""MomentumEtf 프로젝트용 국가 기반 CLI."""

from __future__ import annotations

import argparse
import io
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import math

import pandas as pd

from utils.account_registry import (
    get_country_settings,
    get_strategy_rules,
    list_available_countries,
)
from utils.notification import build_summary_line_from_summary_data
from utils.backtest_utils import print_backtest_summary
from utils.report import (
    format_aud_money,
    format_aud_price,
    format_kr_money,
    format_usd_money,
    format_usd_price,
    render_table_eaw,
)

from logic.backtest.country_runner import (
    CountryBacktestResult,
    run_country_backtest,
    DEFAULT_TEST_MONTHS_RANGE,
)
from logic.entry_point import DECISION_CONFIG

RESULTS_DIR = Path(__file__).resolve().parent / "data" / "results"


def _available_country_choices() -> list[str]:
    choices = list_available_countries()
    if not choices:
        raise SystemExit("국가 설정(JSON)이 존재하지 않습니다. data/settings/*.json 파일을 확인하세요.")
    return choices


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MomentumEtf 국가 추천 실행기",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "country",
        choices=_available_country_choices(),
        help="실행할 국가 코드",
    )
    parser.add_argument(
        "--date",
        help="YYYY-MM-DD 형식의 기준일 (미지정 시 파이프라인 기본값 사용)",
    )
    parser.add_argument("--output", help="결과 JSON 저장 경로")
    parser.add_argument(
        "--backtest",
        action="store_true",
        help="국가 백테스트 실행",
    )
    # JSON은 항상 보기 좋게 정렬되어 출력됩니다.
    return parser


def _invoke_country_pipeline(country: str, *, date_str: str | None) -> List[Dict[str, Any]]:
    """국가별 추천을 생성하고 결과를 반환합니다.

    Args:
        country: 국가 코드 (예: 'kor', 'aus')
        date_str: 기준일 (YYYY-MM-DD 형식), None인 경우 오늘 날짜 사용

    Returns:
        List[Dict[str, Any]]: 추천 결과 리스트
    """
    from logic.recommend import generate_signal_report

    signals = generate_signal_report(
        country=country,
        date_str=date_str,
    )

    if not signals:
        print(f"[WARN] {country.upper()}에 대한 추천을 생성하지 못했습니다.")
        return []

    # print(f"\n[DEBUG] Generated signals for {country}:")
    # print(f"- Signals count: {len(signals)}")
    # print("\nSample signal data (first 3 items):")
    # for i, signal in enumerate(signals[:3], 1):
    #     print(f"{i}. {signal.get('ticker')} - {signal.get('name')}")
    #     print(f"   State: {signal.get('state')}, Daily %: {signal.get('daily_pct')}")
    #     print(f"   Price: {signal.get('price')}, Score: {signal.get('score')}")

    return signals


def _dump_json(data: Any, path: Path) -> None:
    """데이터를 JSON 파일로 저장합니다. 항상 보기 좋게 정렬된 형식으로 저장됩니다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2, sort_keys=False, default=str)


def _print_run_header(country: str, *, date_str: str | None) -> None:
    banner = f"=== {country.upper()} 추천 생성 ==="
    print("\n" + banner)
    print(f"[INFO] 기준일: {date_str or 'auto (latest trading day)'}")


def _print_result_summary(
    items: List[Dict[str, Any]], country: str, date_str: str | None = None
) -> None:
    """추천 결과를 요약하여 출력합니다.

    Args:
        items: _invoke_country_pipeline()의 결과 리스트
        country: 국가 코드
        date_str: 기준일 (선택 사항)
    """
    if not items:
        print(f"[WARN] {country.upper()}에 대한 결과가 없습니다.")
        return

    # 상태별 카운트
    from collections import Counter

    state_counts = Counter(item.get("state", "UNKNOWN") for item in items)
    state_summary = ", ".join(f"{state}: {count}" for state, count in sorted(state_counts.items()))

    # 기준일 설정
    base_date = items[0].get("base_date") if items else (date_str or "N/A")

    print(f"\n=== {country.upper()} 추천 요약 (기준일: {base_date}) ===")
    print(f"[INFO] 총 {len(items)}개 항목 ({state_summary})")

    # 상위 10개 항목 미리보기
    preview_count = min(10, len(items))
    if preview_count > 0:
        print(f"\n[INFO] 상위 {preview_count}개 항목 미리보기:")
        headers = ["순위", "티커", "종목명", "카테고리", "상태", "점수", "일간수익률", "보유일"]
        aligns = ["right", "left", "left", "left", "center", "right", "right", "right"]
        rows = []

        for item in items[:preview_count]:
            holding_days = item.get("holding_days")
            if isinstance(holding_days, (int, float)):
                holding_days_str = f"{int(holding_days)}"
            else:
                holding_days_str = "-"

            rows.append(
                [
                    str(item.get("rank", "-")),
                    str(item.get("ticker", "-")),
                    str(item.get("name", "-")),
                    str(item.get("category", "-")),
                    str(item.get("state", "-")),
                    f"{item.get('score', 0):.2f}"
                    if isinstance(item.get("score"), (int, float))
                    else "-",
                    f"{item.get('daily_pct', 0):.2f}%"
                    if isinstance(item.get("daily_pct"), (int, float))
                    else "-",
                    holding_days_str,
                ]
            )

        for line in render_table_eaw(headers, rows, aligns):
            print(line)

    # 추가 정보 출력
    buy_count = sum(1 for item in items if item.get("state") == "BUY")
    print(f"\n[INFO] 매수 추천: {buy_count}개, 대기: {len(items) - buy_count}개")
    print(f"[INFO] 결과가 성공적으로 생성되었습니다. (총 {len(items)}개 항목)")


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


def _render_backtest_overview(
    result: CountryBacktestResult, country_settings: Dict[str, Any]
) -> List[str]:
    currency, money_fmt, _, _, _ = _resolve_formatters(country_settings)
    summary = result.summary
    lines = [
        "=== 백테스트 결과 요약 ===",
        f"기간: {summary['start_date']} ~ {summary['end_date']}",
        f"초기 자본: {money_fmt(summary['initial_capital'])}",
        f"최종 자산: {money_fmt(summary['final_value'])}",
        f"누적 수익률: {summary['cumulative_return_pct']:+.2f}%",
        f"평가 수익률: {summary['evaluation_return_pct']:+.2f}%",
        f"보유 종목 수: {summary['held_count']} / TOP{result.portfolio_topn}",
        f"통화: {currency}",
    ]
    return lines


def _render_legacy_header(
    result: CountryBacktestResult,
    *,
    country: str,
    months_range: int,
) -> List[str]:
    start_stamp = datetime.now().isoformat(timespec="seconds")
    file_path = f"data/results/backtest_{country}.txt"
    header_lines = [
        f"백테스트 결과가 다음 파일에 저장됩니다: {file_path}",
        "",
        f"KOR 국가의 ETF 종목들을 대상으로 {months_range}개월 기간 백테스트를 실행합니다.",
        f"국가별 설정 파일(settings/country/{country}.json)을 사용하여 전략을 적용합니다.",
        (
            f"실행 시간: {start_stamp} | 테스트 기간: {result.start_date:%Y-%m-%d} ~ {result.end_date:%Y-%m-%d}"
        ),
        "",
    ]
    return header_lines


def _render_used_settings_section(
    *,
    result: CountryBacktestResult,
    country: str,
    country_settings: Mapping[str, Any],
) -> List[str]:
    snapshot = getattr(result, "settings_snapshot", {}) or {}
    strategy_rules = snapshot.get("strategy_rules", {})
    common_settings = snapshot.get("common_settings", {})
    strategy_settings = snapshot.get("strategy_settings", {})
    _, money_fmt, _, _, _ = _resolve_formatters(country_settings)

    lines = ["", "=" * 30, " 사용된 설정값 ".center(30, "="), "=" * 30]
    lines.extend(
        [
            f"| 국가: {country.upper()}",
            f"| 테스트 기간: 최근 {getattr(result, 'months_range', DEFAULT_TEST_MONTHS_RANGE)}개월",
            f"| 초기 자본: {money_fmt(result.initial_capital)}",
            f"| 포트폴리오 종목 수 (TopN): {strategy_rules.get('portfolio_topn', result.portfolio_topn)}",
            f"| 모멘텀 스코어 MA 기간: {strategy_rules.get('ma_period', '-') }일",
            f"| 교체 매매 점수 임계값: {strategy_rules.get('replace_threshold', '-')}",
            f"| 개별 종목 손절매: {common_settings.get('HOLDING_STOP_LOSS_PCT', '-')}%",
            f"| 매도 후 재매수 금지 기간: {strategy_settings.get('COOLDOWN_DAYS', 0)}일",
            "| 시장 위험 필터: "
            + ("활성" if common_settings.get("MARKET_REGIME_FILTER_ENABLED") else "비활성"),
        ]
    )
    if common_settings.get("MARKET_REGIME_FILTER_ENABLED"):
        lines.append(f"| 시장 위험 필터 지표: {common_settings.get('MARKET_REGIME_FILTER_TICKER', '-')}")
        lines.append(
            f"| 시장 위험 필터 MA 기간: {common_settings.get('MARKET_REGIME_FILTER_MA_PERIOD', '-')}일"
        )
    lines.append("=" * 30)
    return lines


def _render_metric_descriptions_section(result: CountryBacktestResult) -> List[str]:
    summary = result.summary
    lines = ["", "[지표 설명]"]
    lines.append(f"  - Sharpe Ratio (샤프 지수): 위험 대비 수익률. 값={summary.get('sharpe_ratio', 0.0):.2f}")
    lines.append(
        f"  - Sortino Ratio (소티노 지수): 하락 위험 대비 수익률. 값={summary.get('sortino_ratio', 0.0):.2f}"
    )
    lines.append(
        f"  - Calmar Ratio (칼마 지수): 연간 수익률 대비 최대 낙폭. 값={summary.get('calmar_ratio', 0.0):.2f}"
    )
    lines.append(f"  - Ulcer Index (얼서 지수): 누적 낙폭의 깊이. 값={summary.get('ulcer_index', 0.0):.2f}")
    lines.append(f"  - CUI (Calmar/Ulcer): 칼마 대비 울서 비율. 값={summary.get('cui', 0.0):.2f}")
    return lines


def _format_period_return_with_listing_date(entry: Mapping[str, Any]) -> str:
    listing = entry.get("listing_date")
    value = entry.get("period_return_pct")
    base = (
        f"{float(value):+.1f}%"
        if isinstance(value, (int, float)) and math.isfinite(float(value))
        else "-"
    )
    if listing:
        return f"{base} (상장: {listing})"
    return base


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

    for idx, ticker in enumerate(tickers_order, 1):
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
        signal2 = row.get("signal2")
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
        hold_ret = ((price / avg_cost) - 1.0) * 100.0 if avg_cost > 0 and shares > 0 else 0.0
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

        buy_date = buy_date_map.get(ticker_key)
        buy_date_display = (
            pd.to_datetime(buy_date).strftime("%Y-%m-%d") if buy_date is not None else "-"
        )
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
        hold_ret_display = (
            f"{hold_ret:+.1f}%"
            if (not is_cash and _is_finite_number(hold_ret) and shares > 0 and avg_cost > 0)
            else "-"
        )
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
        cumulative_profit_display = money_formatter(cumulative_profit_value)
        cumulative_pct = (
            (cumulative_profit_value / initial_value * 100.0)
            if initial_value and _is_finite_number(cumulative_profit_value)
            else 0.0
        )
        cumulative_pct_display = f"{cumulative_pct:+.1f}%" if initial_value else "-"
        signal2_display = f"{float(signal2):.1f}%" if _is_finite_number(signal2) else "-"
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
            filter_display,
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

        # update prev_rows cache (복사하여 이후 비교에 사용)
        for ticker_key, ts in result.ticker_timeseries.items():
            ticker_key_upper = str(ticker_key).upper()
            if target_date in ts.index:
                prev_rows_cache[ticker_key_upper] = ts.loc[target_date].copy()

    return lines


def _dump_backtest_log(
    result: CountryBacktestResult, country_settings: Dict[str, Any], *, country: str
) -> Path:
    """백테스트 로그를 파일로 저장합니다."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"backtest_{country}.txt"
    lines: List[str] = []

    # 1. 레거시 헤더 추가
    legacy_header = _render_legacy_header(
        result,
        country=country,
        months_range=getattr(result, "months_range", DEFAULT_TEST_MONTHS_RANGE),
    )
    lines.extend(legacy_header)

    # 2. 기본 정보 추가
    lines.append(f"백테스트 로그 생성: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(
        f"국가: {result.country.upper()} | 기간: {result.start_date:%Y-%m-%d} ~ {result.end_date:%Y-%m-%d}"
    )
    lines.append(f"초기 자본: {result.initial_capital:,.0f} | 포트폴리오 TOPN: {result.portfolio_topn}")
    lines.append("")

    daily_lines = _generate_daily_report_lines(result, country_settings)
    lines.extend(daily_lines)

    # print_backtest_summary를 사용해서 모든 요약 정보 추가
    summary_buffer = io.StringIO()
    original_stdout = sys.stdout
    sys.stdout = summary_buffer
    try:
        print_backtest_summary(
            summary=result.summary,
            country=country,
            test_months_range=getattr(result, "months_range", DEFAULT_TEST_MONTHS_RANGE),
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


def main() -> None:
    # 명령줄 인자 파싱
    parser = build_parser()
    args = parser.parse_args()

    # 국가 코드 유효성 검사
    country = args.country.lower()

    try:
        # 국가 설정 로드 (유효성 검사용)
        get_country_settings(country)
        get_strategy_rules(country)
    except Exception as exc:
        parser.error(f"국가 설정을 로드하는 중 오류가 발생했습니다: {exc}")

    # 실행 헤더 출력
    _print_run_header(country, date_str=args.date)

    try:
        if args.backtest:
            result = run_country_backtest(country)
            country_settings = get_country_settings(country)
            log_path = _dump_backtest_log(result, country_settings, country=country)
            # 상세한 백테스트 결과 요약을 콘솔에 출력
            print_backtest_summary(
                summary=result.summary,
                country=country,
                test_months_range=getattr(result, "months_range", DEFAULT_TEST_MONTHS_RANGE),
                initial_capital_krw=result.initial_capital,
                portfolio_topn=result.portfolio_topn,
                ticker_summaries=getattr(result, "ticker_summaries", []),
                core_start_dt=result.start_date,
            )
            print(f"\n✅ 백테스트 로그를 '{log_path}'에 저장했습니다.")
        else:
            items = _invoke_country_pipeline(country, date_str=args.date)
            _print_result_summary(items, country, args.date)
            output_path = Path(args.output) if args.output else RESULTS_DIR / f"{country}.json"
            _dump_json(items, output_path)
            print(f"\n✅ {country.upper()} 결과를 '{output_path}'에 저장했습니다.")

    except Exception as exc:
        raise SystemExit(f"오류가 발생했습니다: {exc}") from exc


if __name__ == "__main__":
    main()
