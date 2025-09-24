import os
import sys
from datetime import datetime

import streamlit as st

try:
    import pytz
except ImportError:
    pytz = None


# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    pass
except ImportError:
    st.error("yfinance 라이브러리가 설치되지 않았습니다. `pip install yfinance`로 설치해주세요.")
    yf = None
    st.stop()

from signals import get_market_regime_status_string
from utils.account_registry import (
    get_accounts_by_country,
    load_accounts,
)
from utils.db_manager import (
    get_latest_signal_report,
)
from utils.data_loader import get_aud_to_krw_rate


def main():
    """메인 대시보드를 렌더링합니다."""
    st.set_page_config(page_title="Momentum ETF", page_icon="📈", layout="wide")
    st.title("대시보드")

    hide_amounts = st.toggle("금액 숨기기", key="hide_amounts")

    status_html = get_market_regime_status_string()
    if status_html:
        # 페이지 우측 상단에 시장 상태를 표시합니다.
        st.markdown(f'<div style="text-align: right;">{status_html}</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&display=swap');
            body {
                font-family: 'Noto Sans KR', sans-serif;
            }
            .block-container {
                max-width: 100%;
                padding-top: 1rem;
                padding-left: 2rem;
                padding-right: 2rem;
            }
            /* Custom CSS to reduce sidebar width */
            [data-testid="stSidebar"] {
                width: 150px !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.spinner("계좌 및 환율 정보 로딩 중..."):
        load_accounts(force_reload=True)
        all_accounts = []
        for country_code in ["kor", "aus", "coin"]:
            accounts = get_accounts_by_country(country_code)
            if accounts:
                for acc in accounts:
                    if acc.get("is_active", True):
                        all_accounts.append(acc)

        aud_krw_rate = get_aud_to_krw_rate()

    if not all_accounts:
        st.info("활성화된 계좌가 없습니다. `country_mapping.json`에 계좌를 추가하고 `is_active: true`로 설정해주세요.")
        st.stop()

    account_summaries = []
    total_initial_capital_krw = 0.0
    total_current_equity_krw = 0.0

    for account_info in all_accounts:
        country = account_info["country"]
        account = account_info["account"]

        try:
            # signal_reports 컬렉션에서 가장 최근의 요약 데이터를 가져옵니다.
            today_dt = None
            if pytz:
                try:
                    seoul_tz = pytz.timezone("Asia/Seoul")
                    today_dt = datetime.now(seoul_tz)
                except Exception:
                    today_dt = datetime.now()
            else:
                today_dt = datetime.now()
            report_data = get_latest_signal_report(country, account, date=today_dt)
            if not report_data or "summary" not in report_data:
                continue

            summary = report_data["summary"]

            # --- KRW로 모든 값 변환 ---
            initial_capital_krw = summary.get("principal", 0.0)
            current_equity_krw = summary.get("total_equity", 0.0)
            daily_profit_loss_krw = summary.get("daily_profit_loss", 0.0)
            eval_profit_loss_krw = summary.get("eval_profit_loss", 0.0)
            cum_profit_loss_krw = summary.get("cum_profit_loss", 0.0)
            total_cash_krw = summary.get("total_cash", 0.0)

            # --- Add to totals (already in KRW) ---
            total_initial_capital_krw += initial_capital_krw
            total_current_equity_krw += current_equity_krw

            # --- Prepare summary for display (all in KRW) ---
            account_summaries.append(
                {
                    "display_name": account_info["display_name"],
                    "principal": initial_capital_krw,
                    "current_equity": current_equity_krw,
                    "total_cash": total_cash_krw,
                    "daily_profit_loss": daily_profit_loss_krw,
                    "daily_return_pct": summary.get("daily_return_pct", 0.0),
                    "eval_profit_loss": eval_profit_loss_krw,
                    "eval_return_pct": summary.get("eval_return_pct", 0.0),
                    "cum_profit_loss": cum_profit_loss_krw,
                    "cum_return_pct": summary.get("cum_return_pct", 0.0),
                    "currency": "KRW",  # Always display in KRW
                    "amt_precision": 0,  # Always display as integer KRW
                    "qty_precision": 0,
                    "order": account_info.get("order", 99),
                }
            )
        except Exception as e:
            st.warning(f"'{account_info['display_name']}' 계좌 정보를 불러오는 중 오류 발생: {e}")
            continue

    # --- 총 자산 요약 표시 ---
    st.subheader("총 자산 요약 (KRW 환산)")
    total_profit_loss_krw = total_current_equity_krw - total_initial_capital_krw
    total_cum_return_pct = (
        ((total_current_equity_krw / total_initial_capital_krw) - 1) * 100
        if total_initial_capital_krw > 0
        else 0.0
    )

    col1, col2, col3 = st.columns(3)

    initial_capital_display = "****** 원" if hide_amounts else f"{total_initial_capital_krw:,.0f} 원"
    current_equity_display = "****** 원" if hide_amounts else f"{total_current_equity_krw:,.0f} 원"
    profit_loss_display = "****** 원" if hide_amounts else f"{total_profit_loss_krw:,.0f} 원"

    col1.metric(label="총 초기자본", value=initial_capital_display)
    col2.metric(
        label="총 평가금액",
        value=current_equity_display,
        delta=profit_loss_display,
    )
    col3.metric(label="총 누적수익률", value=f"{total_cum_return_pct:.2f}%")

    if aud_krw_rate:
        st.caption(f"적용 환율: 1 AUD = {aud_krw_rate:,.2f} KRW")
    else:
        st.warning("AUD/KRW 환율 조회에 실패하여, 총 자산 요약에 호주 계좌가 정확히 반영되지 않았을 수 있습니다.")

    st.markdown("---")

    # --- 계좌별 상세 현황 표시 ---
    st.subheader("계좌별 상세 현황")

    # Display header
    header_cols = st.columns((1.5, 1.5, 1.5, 1, 1.5, 1, 1.5, 1, 1.5, 1.5))
    header_cols[0].markdown("**계좌**")
    header_cols[1].markdown(
        "<div style='text-align: right;'><b>원금</b></div>", unsafe_allow_html=True
    )
    header_cols[2].markdown(
        "<div style='text-align: right;'><b>일간손익</b></div>", unsafe_allow_html=True
    )
    header_cols[3].markdown(
        "<div style='text-align: right;'><b>일간(%)</b></div>", unsafe_allow_html=True
    )
    header_cols[4].markdown(
        "<div style='text-align: right;'><b>평가손익</b></div>", unsafe_allow_html=True
    )
    header_cols[5].markdown(
        "<div style='text-align: right;'><b>평가(%)</b></div>", unsafe_allow_html=True
    )
    header_cols[6].markdown(
        "<div style='text-align: right;'><b>누적손익</b></div>", unsafe_allow_html=True
    )
    header_cols[7].markdown(
        "<div style='text-align: right;'><b>누적(%)</b></div>", unsafe_allow_html=True
    )
    header_cols[8].markdown(
        "<div style='text-align: right;'><b>현금</b></div>", unsafe_allow_html=True
    )
    header_cols[9].markdown(
        "<div style='text-align: right;'><b>평가금액</b></div>", unsafe_allow_html=True
    )
    st.markdown("""<hr style="margin:0.5rem 0;" />""", unsafe_allow_html=True)

    for summary in sorted(account_summaries, key=lambda x: x.get("order", 99)):
        currency_symbol = "원"  # All summaries are now in KRW
        amt_precision = summary["amt_precision"]

        cols = st.columns((1.5, 1.5, 1.5, 1, 1.5, 1, 1.5, 1, 1.5, 1.5))
        cols[0].write(summary["display_name"])

        def format_amount(value):
            return f"{value:,.{amt_precision}f} {currency_symbol}"

        def format_amount_with_sign(value):
            color = "red" if value >= 0 else "blue"
            sign = "+" if value > 0 else ""
            return f"<div style='text-align: right; color: {color};'>{sign}{value:,.{amt_precision}f} {currency_symbol}</div>"

        def format_pct(value):
            color = "red" if value > 0 else "blue" if value < 0 else "black"
            return f"<div style='text-align: right; color: {color};'>{value:+.2f}%</div>"

        if hide_amounts:
            hidden_str = f"****** {currency_symbol}"
            cols[1].markdown(
                f"<div style='text-align: right;'>{hidden_str}</div>", unsafe_allow_html=True
            )
            cols[2].markdown(
                f"<div style='text-align: right;'>{hidden_str}</div>", unsafe_allow_html=True
            )
            cols[4].markdown(
                f"<div style='text-align: right;'>{hidden_str}</div>", unsafe_allow_html=True
            )
            cols[6].markdown(
                f"<div style='text-align: right;'>{hidden_str}</div>", unsafe_allow_html=True
            )
            cols[8].markdown(
                f"<div style='text-align: right;'>{hidden_str}</div>", unsafe_allow_html=True
            )
            cols[9].markdown(
                f"<div style='text-align: right;'>{hidden_str}</div>", unsafe_allow_html=True
            )
        else:
            # 원금
            cols[1].markdown(
                f"<div style='text-align: right;'>{format_amount(summary['principal'])}</div>",
                unsafe_allow_html=True,
            )
            # 일간손익
            cols[2].markdown(
                format_amount_with_sign(summary["daily_profit_loss"]), unsafe_allow_html=True
            )
            # 평가손익
            cols[4].markdown(
                format_amount_with_sign(summary["eval_profit_loss"]), unsafe_allow_html=True
            )
            # 누적손익
            cols[6].markdown(
                format_amount_with_sign(summary["cum_profit_loss"]), unsafe_allow_html=True
            )
            # 현금
            cols[8].markdown(
                f"<div style='text-align: right;'>{format_amount(summary['total_cash'])}</div>",
                unsafe_allow_html=True,
            )
            # 평가금액
            cols[9].markdown(
                f"<div style='text-align: right;'>{format_amount(summary['current_equity'])}</div>",
                unsafe_allow_html=True,
            )

        # % 값들
        cols[3].markdown(format_pct(summary["daily_return_pct"]), unsafe_allow_html=True)
        cols[5].markdown(format_pct(summary["eval_return_pct"]), unsafe_allow_html=True)
        cols[7].markdown(format_pct(summary["cum_return_pct"]), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
