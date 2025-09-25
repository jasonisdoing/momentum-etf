import os
import sys

import streamlit as st

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.account_registry import (
    get_account_file_settings,
    get_country_file_settings,
    get_common_file_settings,
    get_accounts_by_country,
    load_accounts,
)


def render_common_settings():
    """공통 설정 UI를 렌더링합니다."""
    st.header("공통 설정 (모든 국가 공유)")

    try:
        common_settings = get_common_file_settings()
    except SystemExit as e:
        st.error(str(e))
        st.stop()

    help_text = "이 값은 `data/settings/common.py` 파일에서 수정할 수 있습니다."

    st.subheader("시장 레짐 필터 (파일에서 설정)")
    st.checkbox(
        "활성화 (MARKET_REGIME_FILTER_ENABLED)",
        value=bool(common_settings["MARKET_REGIME_FILTER_ENABLED"]),
        disabled=True,
        help=help_text,
    )
    st.text_input(
        "레짐 기준 지수 티커 (MARKET_REGIME_FILTER_TICKER)",
        value=str(common_settings["MARKET_REGIME_FILTER_TICKER"]),
        disabled=True,
        help=help_text,
    )
    st.text_input(
        "레짐 MA 기간 (MARKET_REGIME_FILTER_MA_PERIOD)",
        value=str(common_settings["MARKET_REGIME_FILTER_MA_PERIOD"]),
        disabled=True,
        help=help_text,
    )

    st.subheader("위험 관리 및 지표 (파일에서 설정)")
    st.number_input(
        "보유 손절 임계값 % (HOLDING_STOP_LOSS_PCT)",
        value=float(common_settings["HOLDING_STOP_LOSS_PCT"]),
        disabled=True,
        help=f"{help_text} (양수로 입력해도 음수로 해석됩니다)",
    )
    st.text_input(
        "쿨다운 일수 (COOLDOWN_DAYS)",
        value=str(common_settings["COOLDOWN_DAYS"]),
        disabled=True,
        help=help_text,
    )


def render_account_settings(country_code: str, account_code: str):
    """계좌별 설정 UI를 렌더링합니다."""
    try:
        account_settings = get_account_file_settings(account_code)
        country_settings = get_country_file_settings(country_code)
    except SystemExit as e:
        st.error(str(e))
        st.stop()

    st.subheader("계좌 고유 설정 (파일에서 설정)")
    account_help_text = f"이 값들은 `data/settings/accounts/{account_code}.py` 파일에서 수정할 수 있습니다."

    from utils.account_registry import get_account_info

    account_info = get_account_info(account_code)
    currency = account_info.get("currency", "KRW")
    precision = account_info.get("precision", 0)

    # 기본 정보
    currency_str = f" ({currency})"
    st.text_input(
        f"초기 자본금{currency_str}",
        value=f"{float(account_settings['initial_capital_krw']):,.{precision}f}"
        if precision > 0
        else f"{int(account_settings['initial_capital_krw']):,d}",
        disabled=True,
        help=account_help_text,
    )
    st.date_input(
        "초기 자본 기준일",
        value=account_settings["initial_date"],
        disabled=True,
        help=account_help_text,
    )
    st.markdown("---")

    # 전략 파라미터
    st.subheader("국가별 전략 파라미터 (파일에서 설정)")
    country_help_text = f"이 값들은 `data/settings/country/{country_code}.py` 파일에서 수정할 수 있습니다."
    st.text_input(
        "최대 보유 종목 수 (Top-N)",
        value=str(country_settings["portfolio_topn"]),
        disabled=True,
        help=country_help_text,
    )
    st.text_input(
        "이동평균 기간 (MA)",
        value=str(country_settings["ma_period"]),
        disabled=True,
        help=country_help_text,
    )
    st.checkbox(
        "교체 매매 사용",
        value=bool(country_settings["replace_weaker_stock"]),
        disabled=True,
        help=country_help_text,
    )
    st.text_input(
        "교체 매매 점수 임계값",
        value=f"{float(country_settings['replace_threshold']):.2f}",
        disabled=True,
        help=country_help_text,
    )


def main():
    """설정 페이지를 렌더링합니다."""
    st.title("⚙️ 설정 (Settings)")

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

    render_common_settings()

    st.markdown("---")
    st.header("계좌별 설정")

    with st.spinner("계좌 정보 로딩 중..."):
        load_accounts(force_reload=False)
        account_map = {
            "kor": get_accounts_by_country("kor"),
            "aus": get_accounts_by_country("aus"),
            "coin": get_accounts_by_country("coin"),
        }

    country_tabs = st.tabs(["한국", "호주", "코인"])

    for country_code, tab in zip(["kor", "aus", "coin"], country_tabs):
        with tab:
            accounts = account_map.get(country_code, [])
            active_accounts = [acc for acc in accounts if acc.get("is_active", True)]

            if not active_accounts:
                st.info(f"'{country_code.upper()}' 국가에 활성화된 계좌가 없습니다.")
                continue

            account_options = {
                acc["account"]: acc.get("display_name", acc["account"]) for acc in active_accounts
            }
            selected_account_code = st.selectbox(
                "설정을 변경할 계좌 선택",
                options=list(account_options.keys()),
                format_func=lambda x: account_options[x],
                key=f"account_select_{country_code}",
            )

            if selected_account_code:
                render_account_settings(country_code, selected_account_code)


if __name__ == "__main__":
    main()
