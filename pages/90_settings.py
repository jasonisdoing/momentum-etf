import os
import sys

import pandas as pd
import streamlit as st

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.account_registry import get_accounts_by_country, load_accounts
from utils.db_manager import (
    get_account_settings,
    get_common_settings,
    save_common_settings,
    save_portfolio_settings,
)


def render_common_settings():
    """공통 설정 UI를 렌더링합니다."""
    st.header("공통 설정 (모든 국가 공유)")
    common = get_common_settings() or {}
    current_enabled = bool(common.get("MARKET_REGIME_FILTER_ENABLED", False))
    current_ticker = common.get("MARKET_REGIME_FILTER_TICKER")
    current_ma = common.get("MARKET_REGIME_FILTER_MA_PERIOD")
    current_stop = common.get("HOLDING_STOP_LOSS_PCT")
    current_cooldown = common.get("COOLDOWN_DAYS")

    with st.form("common_settings_form"):
        st.subheader("시장 레짐 필터")
        new_enabled = st.checkbox("활성화 (MARKET_REGIME_FILTER_ENABLED)", value=current_enabled)
        new_ticker = st.text_input(
            "레짐 기준 지수 티커 (MARKET_REGIME_FILTER_TICKER)",
            value=str(current_ticker) if current_ticker is not None else "",
            placeholder="예: ^GSPC",
        )
        new_ma_str = st.text_input(
            "레짐 MA 기간 (MARKET_REGIME_FILTER_MA_PERIOD)",
            value=str(current_ma) if current_ma is not None else "",
            placeholder="예: 20",
        )

        st.subheader("위험 관리 및 지표")
        new_stop = st.number_input(
            "보유 손절 임계값 % (HOLDING_STOP_LOSS_PCT)",
            value=float(current_stop) if current_stop is not None else 0.0,
            step=0.1,
            format="%.2f",
            help="예: 10.0 (양수로 입력해도 음수로 저장됩니다)",
        )
        new_cooldown_str = st.text_input(
            "쿨다운 일수 (COOLDOWN_DAYS)",
            value=str(current_cooldown) if current_cooldown is not None else "",
            placeholder="예: 5",
        )

        submitted = st.form_submit_button("공통 설정 저장")
        if submitted:
            error = False
            if not new_ticker:
                st.error("시장 레짐 필터 티커를 입력해주세요.")
                error = True
            if not new_ma_str.isdigit() or int(new_ma_str) < 1:
                st.error("레짐 MA 기간은 1 이상의 정수여야 합니다.")
                error = True
            if not new_cooldown_str.isdigit() or int(new_cooldown_str) < 0:
                st.error("쿨다운 일수는 0 이상의 정수여야 합니다.")
                error = True

            if not error:
                normalized_stop = -abs(float(new_stop))
                to_save = {
                    "MARKET_REGIME_FILTER_ENABLED": bool(new_enabled),
                    "MARKET_REGIME_FILTER_TICKER": new_ticker,
                    "MARKET_REGIME_FILTER_MA_PERIOD": int(new_ma_str),
                    "HOLDING_STOP_LOSS_PCT": normalized_stop,
                    "COOLDOWN_DAYS": int(new_cooldown_str),
                }
                if save_common_settings(to_save):
                    st.success("공통 설정을 저장했습니다.")
                    st.rerun()
                else:
                    st.error("공통 설정 저장에 실패했습니다.")


def render_account_settings(country_code: str, account_code: str):
    """계좌별 설정 UI를 렌더링합니다."""
    account_prefix = f"{country_code}_{account_code}"
    db_settings = get_account_settings(account_code) or {}

    current_capital = db_settings.get("initial_capital", 0)
    current_topn = db_settings.get("portfolio_topn")
    current_ma = db_settings.get("ma_period")
    current_replace_threshold = db_settings.get("replace_threshold")
    current_replace_weaker = db_settings.get("replace_weaker_stock")
    current_date = db_settings.get("initial_date", pd.Timestamp.now() - pd.DateOffset(months=12))

    with st.form(key=f"settings_form_{account_prefix}"):
        currency_str = f" ({'AUD' if country_code == 'aus' else 'KRW'})"
        new_capital = st.number_input(
            f"초기 자본금{currency_str}",
            value=float(current_capital) if country_code == "aus" else int(current_capital),
            format="%.2f" if country_code == "aus" else "%d",
        )
        new_date = st.date_input("초기 자본 기준일", value=current_date)
        new_topn_str = st.text_input(
            "최대 보유 종목 수 (Top-N)",
            value=str(current_topn) if current_topn is not None else "",
            placeholder="예: 10",
        )

        st.markdown("---")
        st.subheader("전략 파라미터")
        new_ma_str = st.text_input(
            "이동평균 기간 (MA)", value=str(current_ma) if current_ma is not None else "75"
        )
        replace_weaker_checkbox = st.checkbox("교체 매매 사용", value=bool(current_replace_weaker))
        new_replace_threshold_str = st.text_input(
            "교체 매매 점수 임계값",
            value=f"{float(current_replace_threshold):.2f}"
            if current_replace_threshold is not None
            else "",
            placeholder="예: 1.5",
        )

        if st.form_submit_button("계좌 설정 저장하기"):
            error = False
            if not new_topn_str or not new_topn_str.isdigit() or int(new_topn_str) < 1:
                st.error("최대 보유 종목 수는 1 이상의 숫자여야 합니다.")
                error = True
            if not new_ma_str or not new_ma_str.isdigit() or int(new_ma_str) < 1:
                st.error("이동평균 기간은 1 이상의 숫자여야 합니다.")
                error = True
            try:
                _ = float(new_replace_threshold_str)
            except (ValueError, TypeError):
                st.error("교체 매매 점수 임계값은 숫자여야 합니다.")
                error = True

            if not error:
                settings_to_save = {
                    "country": country_code,
                    "initial_capital": new_capital,
                    "initial_date": pd.to_datetime(
                        new_date
                    ).to_pydatetime(),  # FIX: Use the value from the date_input
                    "portfolio_topn": int(new_topn_str),
                    "ma_period": int(new_ma_str),
                    "replace_weaker_stock": bool(replace_weaker_checkbox),
                    "replace_threshold": float(new_replace_threshold_str),
                }
                if save_portfolio_settings(country_code, settings_to_save, account=account_code):
                    st.success("설정이 성공적으로 저장되었습니다.")
                    st.rerun()
                else:
                    st.error("설정 저장에 실패했습니다.")


def main():
    """설정 페이지를 렌더링합니다."""
    st.title("⚙️ 설정 (Settings)")

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
