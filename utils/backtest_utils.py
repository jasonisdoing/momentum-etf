"""백테스트 관련 유틸리티 함수들."""

from typing import Dict, List, Any
import pandas as pd
from datetime import datetime
from .report import (
    format_kr_money,
    format_aud_money,
    format_usd_money,
    render_table_eaw,
)
from .data_loader import get_aud_to_krw_rate, get_usd_to_krw_rate
from utils.account_registry import get_country_settings


def format_period_return_with_listing_date(s: Dict[str, Any], core_start_dt: pd.Timestamp) -> str:
    """기간수익률을 상장일과 함께 포맷합니다."""
    period_return_pct = s.get("period_return_pct", 0.0)
    listing_date = s.get("listing_date")

    if listing_date and core_start_dt:
        # 상장일이 테스트 시작일 이후인 경우에만 상장일 표시
        listing_dt = pd.to_datetime(listing_date)
        if listing_dt > core_start_dt:
            return f"{period_return_pct:+.2f}%({listing_date})"

    return f"{period_return_pct:+.2f}%"


def print_backtest_summary(
    summary: Dict,
    country: str,
    test_months_range: int,
    initial_capital_krw: float,
    portfolio_topn: int,
    ticker_summaries: List[Dict[str, Any]],
    core_start_dt: pd.Timestamp,
):
    from settings import common as common_settings

    """백테스트 결과 요약을 콘솔에 출력합니다."""
    # 국가 설정에서 통화 정보 가져오기
    country_settings = get_country_settings(country)
    currency = country_settings.get("currency", "KRW")
    precision = country_settings.get("precision", {}).get(
        "amt_precision", 0
    ) or country_settings.get("amt_precision", 0)
    try:
        precision = int(precision)
    except (TypeError, ValueError):
        precision = 0

    # 전략 설정에서 필요한 값들 가져오기
    strategy_settings = country_settings.get("strategy", {})
    cooldown_days = strategy_settings.get("COOLDOWN_DAYS", 0)
    replace_threshold = strategy_settings.get("REPLACE_SCORE_THRESHOLD", 0.5)

    # 통화 정보에 따라 통화 형식 함수를 가져옴
    if currency == "AUD":
        money_formatter = format_aud_money
    elif currency == "USD":
        money_formatter = format_usd_money
    else:
        money_formatter = format_kr_money
    benchmark_name = "BTC" if country == "coin" else "S&P 500"

    summary_lines = [
        "\n" + "=" * 30 + "\n 백테스트 결과 요약 ".center(30, "=") + "\n" + "=" * 30,
        f"| 기간: {summary['start_date']} ~ {summary['end_date']} ({test_months_range} 개월)",
    ]

    if summary.get("risk_off_periods"):
        for start, end in summary["risk_off_periods"]:
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
    if "MA_PERIOD" not in strategy_settings or strategy_settings.get("MA_PERIOD") is None:
        raise ValueError(f"'{country}' 국가 설정에 'strategy.MA_PERIOD' 값이 필요합니다.")
    ma_period = strategy_settings["MA_PERIOD"]
    momentum_label = f"{ma_period}일"

    holding_stop_loss_pct = getattr(common_settings, "HOLDING_STOP_LOSS_PCT", None)
    if holding_stop_loss_pct is None:
        holding_stop_loss_pct = strategy_settings.get("HOLDING_STOP_LOSS_PCT")
    if holding_stop_loss_pct is None:
        raise ValueError("공통 또는 국가 전략 설정에 'HOLDING_STOP_LOSS_PCT' 값이 필요합니다.")
    stop_loss_label = f"{holding_stop_loss_pct}%"

    market_regime_enabled = getattr(common_settings, "MARKET_REGIME_FILTER_ENABLED", None)
    if market_regime_enabled is None:
        market_regime_enabled = strategy_settings.get("MARKET_REGIME_FILTER_ENABLED")
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

    # 월별 성과 요약 테이블 출력
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
        rows_data = []
        for year, row in pivot_df.iterrows():
            # 월간 수익률 행
            monthly_row_data = [str(year)]
            for month in range(1, 13):
                val = row.get(month)
                monthly_row_data.append(f"{val:+.2f}%" if pd.notna(val) else "-")

            yearly_val = row.get("연간")
            monthly_row_data.append(f"{yearly_val:+.2f}%" if pd.notna(yearly_val) else "-")
            rows_data.append(monthly_row_data)

            # 누적 수익률 행
            if cum_pivot_df is not None and year in cum_pivot_df.index:
                cum_row = cum_pivot_df.loc[year]
                cum_row_data = ["  (누적)"]
                for month in range(1, 13):
                    cum_val = cum_row.get(month)
                    cum_row_data.append(f"{cum_val:+.2f}%" if pd.notna(cum_val) else "-")

                # 연말 누적 수익률을 찾습니다.
                last_valid_month_index = cum_row.last_valid_index()
                if last_valid_month_index is not None:
                    cum_annual_val = cum_row[last_valid_month_index]
                    cum_row_data.append(f"{cum_annual_val:+.2f}%")
                else:
                    cum_row_data.append("-")
                rows_data.append(cum_row_data)

        aligns = ["left"] + ["right"] * (len(headers) - 1)
        print("\n" + "\n".join(render_table_eaw(headers, rows_data, aligns)))

    # 종목별 성과 요약 테이블 출력
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
