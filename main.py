import os
import sys

import pandas as pd
import streamlit as st

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import yfinance as yf
except ImportError:
    st.error("yfinance 라이브러리가 설치되지 않았습니다. `pip install yfinance`로 설치해주세요.")
    yf = None
    st.stop()

from signals import get_market_regime_status_string
from utils.account_registry import (
    get_account_file_settings,
    get_accounts_by_country,
    load_accounts,
)
from utils.db_manager import get_portfolio_snapshot, get_previous_portfolio_snapshot


@st.cache_data(ttl=3600)  # 1시간 동안 환율 정보 캐시
def get_aud_to_krw_rate():
    """yfinance를 사용하여 AUD/KRW 환율을 조회합니다."""
    if not yf:
        return None
    try:
        ticker = yf.Ticker("AUDKRW=X")
        # 가장 최근 가격을 가져오기 위해 2일간의 1분 단위 데이터 시도
        data = ticker.history(period="2d", interval="1m")
        if not data.empty:
            return data["Close"].iloc[-1]
        # 1m 데이터가 없으면 일 단위 데이터로 폴백
        data = ticker.history(period="2d")
        if not data.empty:
            return data["Close"].iloc[-1]
    except Exception as e:
        print(f"AUD/KRW 환율 정보를 가져오는 데 실패했습니다: {e}")
        return None
    return None


def main():
    """메인 대시보드를 렌더링합니다."""
    st.set_page_config(page_title="main", page_icon="📈", layout="wide")
    st.title("📈 메인 대시보드")

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
            settings = get_account_file_settings(country, account)
            initial_capital = float(settings.get("initial_capital", 0.0))

            snapshot = get_portfolio_snapshot(country, account)
            if not snapshot:
                continue

            current_equity = float(snapshot.get("total_equity", 0.0))
            snapshot_date = pd.to_datetime(snapshot.get("date"))

            prev_snapshot = get_previous_portfolio_snapshot(country, snapshot_date, account)
            prev_equity = float(prev_snapshot.get("total_equity", 0.0)) if prev_snapshot else 0.0

            daily_return_pct = (
                ((current_equity / prev_equity) - 1) * 100 if prev_equity > 0 else 0.0
            )
            cum_return_pct = (
                ((current_equity / initial_capital) - 1) * 100 if initial_capital > 0 else 0.0
            )

            currency = account_info.get("currency", "KRW")
            precision = account_info.get("precision", 0)

            initial_capital_krw = initial_capital
            current_equity_krw = current_equity

            if currency == "AUD" and aud_krw_rate:
                initial_capital_krw *= aud_krw_rate
                current_equity_krw *= aud_krw_rate

            total_initial_capital_krw += initial_capital_krw
            total_current_equity_krw += current_equity_krw

            account_summaries.append(
                {
                    "display_name": account_info["display_name"],
                    "initial_capital": initial_capital,
                    "current_equity": current_equity,
                    "daily_return_pct": daily_return_pct,
                    "cum_return_pct": cum_return_pct,
                    "currency": currency,
                    "precision": precision,
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
    col1.metric(label="총 초기자본", value=f"{total_initial_capital_krw:,.0f} 원")
    col2.metric(
        label="총 평가금액",
        value=f"{total_current_equity_krw:,.0f} 원",
        delta=f"{total_profit_loss_krw:,.0f} 원",
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
    header_cols = st.columns((2, 2.2, 2.2, 2.2, 1.5, 1.5))
    header_cols[0].markdown("**계좌**")
    header_cols[1].markdown(
        "<div style='text-align: right;'><b>초기자본</b></div>", unsafe_allow_html=True
    )
    header_cols[2].markdown(
        "<div style='text-align: right;'><b>평가금액</b></div>", unsafe_allow_html=True
    )
    header_cols[3].markdown(
        "<div style='text-align: right;'><b>수익금</b></div>", unsafe_allow_html=True
    )
    header_cols[4].markdown(
        "<div style='text-align: right;'><b>일간(%)</b></div>", unsafe_allow_html=True
    )
    header_cols[5].markdown(
        "<div style='text-align: right;'><b>누적(%)</b></div>", unsafe_allow_html=True
    )
    st.markdown("""<hr style="margin:0.5rem 0;" />""", unsafe_allow_html=True)

    for summary in sorted(account_summaries, key=lambda x: x["display_name"]):
        currency_symbol = "$" if summary["currency"] == "AUD" else "원"
        precision = summary["precision"]
        profit_loss = summary["current_equity"] - summary["initial_capital"]

        cols = st.columns((2, 2.2, 2.2, 2.2, 1.5, 1.5))
        cols[0].write(summary["display_name"])

        initial_capital_str = f"{summary['initial_capital']:,.{precision}f} {currency_symbol}"
        cols[1].markdown(
            f"<div style='text-align: right;'>{initial_capital_str}</div>", unsafe_allow_html=True
        )

        current_equity_str = f"{summary['current_equity']:,.{precision}f} {currency_symbol}"
        cols[2].markdown(
            f"<div style='text-align: right;'>{current_equity_str}</div>", unsafe_allow_html=True
        )

        profit_loss_color = "red" if profit_loss >= 0 else "blue"
        profit_loss_sign = "+" if profit_loss > 0 else ""
        profit_loss_str = f"{profit_loss_sign}{profit_loss:,.{precision}f} {currency_symbol}"
        cols[3].markdown(
            f"<div style='text-align: right; color: {profit_loss_color};'>{profit_loss_str}</div>",
            unsafe_allow_html=True,
        )

        cum_return_pct = summary["cum_return_pct"]
        daily_return_pct = summary["daily_return_pct"]
        daily_ret_color = (
            "red" if daily_return_pct > 0 else "blue" if daily_return_pct < 0 else "black"
        )
        cols[4].markdown(
            f"<div style='text-align: right; color: {daily_ret_color};'>{daily_return_pct:+.2f}%</div>",
            unsafe_allow_html=True,
        )

        cum_ret_color = "red" if cum_return_pct >= 0 else "blue"
        cols[5].markdown(
            f"<div style='text-align: right; color: {cum_ret_color};'>{cum_return_pct:+.2f}%</div>",
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
