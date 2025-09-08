"""
리포트 및 로그 출력을 위한 포맷팅 유틸리티 함수 모음.
"""

import re
from typing import Dict, List
from unicodedata import east_asian_width, normalize

import pandas as pd

def format_kr_money(value: float) -> str:
    """금액을 '억'과 '만원' 단위의 한글 문자열로 포맷합니다."""
    if value is None:
        return "-"
    man = int(round(value / 10_000))
    if man >= 10_000:
        uk = man // 10_000
        rem = man % 10_000
        return f"{uk}억 {rem:,}만원" if rem > 0 else f"{uk}억"
    return f"{man:,}만원"


def format_aud_money(value: float) -> str:
    """금액을 호주 달러(A$) 형식의 문자열로 포맷합니다."""
    if value is None:
        return "-"
    return f"A${value:,.2f}"


def render_table_eaw(
    headers: List[str],
    rows: List[List[str]],
    aligns: List[str]
) -> List[str]:
    """
    동아시아 문자 너비를 고려하여 리스트 데이터를 ASCII 테이블 문자열로 렌더링합니다.
    """

    _ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

    def _clean(s: str) -> str:
        if not isinstance(s, str):
            s = str(s)
        s = _ANSI_RE.sub('', s)
        s = normalize('NFKC', s)
        return s

    def _disp_width_eaw(s: str) -> int:
        """동아시아 문자를 포함한 문자열의 실제 터미널 출력 너비를 계산합니다."""
        s = _clean(s)
        w = 0
        for ch in s:
            # 박스 드로잉 문자는 터미널에서 넓게 렌더링되는 경우가 많습니다.
            if '\u2500' <= ch <= '\u257f':
                w += 2
                continue
            eaw = east_asian_width(ch)
            # 'Ambiguous'(A) 문자를 Wide로 처리하여 대부분의 터미널에서 정렬이 깨지지 않도록 합니다.
            if eaw in ('W', 'F', 'A'):
                w += 2
            else:
                w += 1
        return w

    def _pad(s: str, width: int, align: str) -> str:
        """주어진 너비와 정렬에 맞게 문자열에 패딩을 추가합니다."""
        s_str = str(s)
        s_clean = _clean(s_str)
        dw = _disp_width_eaw(s_clean)
        if dw >= width:
            return s_str
        pad = width - dw
        if align == 'right':
            return ' ' * pad + s_str
        elif align == 'center':
            left = pad // 2
            right = pad - left
            return ' ' * left + s_str + ' ' * right
        else: # left
            return s_str + ' ' * pad

    widths = [max(_disp_width_eaw(v) for v in [headers[j]] + [r[j] for r in rows]) for j in range(len(headers))]

    def _hline():
        return '+' + '+'.join('-' * (w + 2) for w in widths) + '+'

    out = [_hline()]
    header_cells = [_pad(headers[j], widths[j], 'center' if aligns[j] == 'center' else 'left') for j in range(len(headers))]
    out.append('| ' + ' | '.join(header_cells) + ' |')
    out.append(_hline())
    for r in rows:
        cells = [_pad(r[j], widths[j], aligns[j]) for j in range(len(headers))]
        out.append('| ' + ' | '.join(cells) + ' |')
    out.append(_hline())
    return out


def generate_strategy_comparison_report(all_results: List[Dict], country: str = "kor") -> str:
    """여러 전략의 백테스트 결과를 받아 비교 리포트 문자열을 생성합니다."""
    if not all_results:
        return "비교할 결과가 없습니다."

    report_lines = []
    money_formatter = format_aud_money if country == "aus" else format_kr_money

    # 1. 요약 테이블 생성
    headers = [
        "전략", "기간", "CAGR", "MDD", "Sharpe", "Sortino", "Calmar", "누적수익률", "최종자산",
    ]
    rows = [
        [
            r["strategy"],
            f"{r['start_date']}~{r['end_date']}",
            f"{r['cagr_pct']:.2f}%",
            f"-{r['mdd_pct']:.2f}%",
            f"{r.get('sharpe_ratio', 0.0):.2f}",
            f"{r.get('sortino_ratio', 0.0):.2f}",
            f"{r.get('calmar_ratio', 0.0):.2f}",
            f"{r['cumulative_return_pct']:.2f}%",
            money_formatter(r["final_value"]),
        ]
        for r in all_results
    ]
    aligns = ["left", "left", "right", "right", "right", "right", "right", "right", "right"]

    initial_capital = all_results[0]["initial_capital"]
    report_lines.append("\n" + "=" * 30 + "\n" + " 전략 비교 결과 요약 ".center(30, "=") + "\n" + "=" * 30)
    report_lines.append(f"(초기 자본: {money_formatter(initial_capital)})")
    report_lines.extend(render_table_eaw(headers, rows, aligns))

    report_lines.append("\n[지표 설명]")
    report_lines.append("  - CAGR: 연간 복리 성장률")
    report_lines.append("  - MDD: 최대 낙폭 (고점 대비 최대 하락률)")
    report_lines.append("  - Sharpe Ratio (샤프 지수): 위험(변동성) 대비 수익률. 높을수록 좋음 (기준: >1 양호, >2 우수).")
    report_lines.append("  - Sortino Ratio (소티노 지수): 하락 위험 대비 수익률. 높을수록 좋음 (기준: >2 양호, >3 우수).")
    report_lines.append("  - Calmar Ratio (칼마 지수): 최대 낙폭 대비 연간 수익률. 높을수록 좋음 (기준: >1 양호, >3 우수).")

    # 2. 월별 성과 비교
    all_monthly_returns = {
        res["strategy"]: res["monthly_returns"]
        for res in all_results
        if "monthly_returns" in res and not res["monthly_returns"].empty
    }
    all_yearly_returns = {
        res["strategy"]: res["yearly_returns"]
        for res in all_results
        if "yearly_returns" in res and not res["yearly_returns"].empty
    }
    all_monthly_cum_returns = {
        res["strategy"]: res.get("monthly_cum_returns")
        for res in all_results
    }

    if all_monthly_returns:
        monthly_df_check = pd.DataFrame(all_monthly_returns)
        if monthly_df_check.empty:
            report_lines.append("\n월별 수익률 데이터가 없어 비교를 건너뜁니다.")
        else:
            report_lines.append("\n" + "=" * 30 + "\n" + " 월별 성과 비교 ".center(30, "=") + "\n" + "=" * 30)

            monthly_df = monthly_df_check.mul(100)
            yearly_df = pd.DataFrame(all_yearly_returns).mul(100)
            monthly_cum_df = pd.DataFrame({k: v for k, v in all_monthly_cum_returns.items() if v is not None}).mul(100)
            strategy_names = list(all_monthly_returns.keys())

            all_years_set = set(monthly_df.index.year)
            if not yearly_df.empty:
                all_years_set.update(yearly_df.index.year)

            all_years = sorted(list(all_years_set))

            for year in all_years:
                report_lines.append(f"\n--- {year}년 ---")

                year_monthly_df = monthly_df[monthly_df.index.year == year]

                headers = ["월"] + strategy_names
                rows_data = []

                for month in range(1, 13):
                    month_data = [f"{month}월"]
                    cum_data = ["  (누적)"]
                    month_end_dt = pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)

                    has_data_for_month = month_end_dt in year_monthly_df.index
                    if has_data_for_month:
                        month_ret_row = year_monthly_df.loc[month_end_dt]
                        cum_ret_row = monthly_cum_df.loc[month_end_dt] if not monthly_cum_df.empty and month_end_dt in monthly_cum_df.index else None
                        for strategy in strategy_names:
                            val = month_ret_row.get(strategy)
                            month_data.append(f"{val:+.2f}%" if pd.notna(val) else "-")

                            cum_val = cum_ret_row.get(strategy) if cum_ret_row is not None else None
                            cum_data.append(f"{cum_val:+.2f}%" if pd.notna(cum_val) else "-")
                    else:
                        month_data.extend(["-"] * len(strategy_names))
                        cum_data.extend(["-"] * len(strategy_names))
                    rows_data.append(month_data)
                    if has_data_for_month:
                        rows_data.append(cum_data)

                yearly_data = ["연간"]
                year_end_dt = pd.Timestamp(year, 12, 31) + pd.offsets.YearEnd(0)
                if not yearly_df.empty and year_end_dt in yearly_df.index:
                    year_row = yearly_df.loc[year_end_dt]
                    for strategy in strategy_names:
                        val = year_row.get(strategy)
                        yearly_data.append(f"{val:+.2f}%" if pd.notna(val) else "-")
                else:
                    yearly_data.extend(["-"] * len(strategy_names))
                rows_data.append(yearly_data)

                # 연간 누적 수익률 행 추가
                cum_yearly_data = ["  (누적)"]
                year_cum_df = monthly_cum_df[monthly_cum_df.index.year == year] if not monthly_cum_df.empty else pd.DataFrame()
                if not year_cum_df.empty:
                    last_date_of_year = year_cum_df.index[-1]
                    cum_year_row = monthly_cum_df.loc[last_date_of_year]
                    for strategy in strategy_names:
                        cum_val = cum_year_row.get(strategy)
                        cum_yearly_data.append(
                            f"{cum_val:+.2f}%" if pd.notna(cum_val) else "-"
                        )
                else:
                    cum_yearly_data.extend(["-"] * len(strategy_names))
                rows_data.append(cum_yearly_data)

                aligns = ["left"] + ["right"] * len(strategy_names)
                report_lines.extend(render_table_eaw(headers, rows_data, aligns))

    return "\n".join(report_lines)