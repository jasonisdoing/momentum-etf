from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import pandas as pd
import streamlit as st
import streamlit_authenticator as stauth

from app_pages.account_page import render_account_page
from utils.account_registry import (
    get_icon_fallback,
    load_account_configs,
)
from utils.ui import render_recommendation_table


def _to_plain_dict(value):
    if isinstance(value, Mapping):
        return {k: _to_plain_dict(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_plain_dict(v) for v in value]
    return value


def _load_authenticator() -> stauth.Authenticate:
    raw_config = st.secrets.get("auth")
    if not raw_config:
        st.error("인증 설정(st.secrets['auth'])이 구성되지 않았습니다.")
        st.stop()

    config = _to_plain_dict(raw_config)

    credentials = config.get("credentials")
    cookie = config.get("cookie") or {}
    preauthorized = config.get("preauthorized", {})

    required_keys = {"name", "key", "expiry_days"}
    if not credentials or not cookie or not required_keys.issubset(cookie):
        st.error("인증 설정 필드가 누락되었습니다. credentials/cookie 구성을 확인하세요.")
        st.stop()

    return stauth.Authenticate(
        credentials,
        cookie.get("name"),
        cookie.get("key"),
        cookie.get("expiry_days"),
        preauthorized,
    )


def _build_account_page(page_cls: Callable[..., object], account: dict[str, Any]):
    account_id = account["account_id"]
    icon = account.get("icon") or get_icon_fallback(account.get("country_code", ""))

    def _render(account_key: str = account_id) -> None:
        render_account_page(account_key)

    return page_cls(
        _render,
        title=account["name"],
        icon=icon,
        url_path=account_id,
    )


def _build_home_page(accounts: list[dict[str, Any]]):
    def _render_home_page() -> None:
        from utils.portfolio_io import load_portfolio_master, load_real_holdings_with_recommendations

        all_holdings = []
        global_principal = 0.0
        global_cash = 0.0

        for account in accounts:
            account_id = account["account_id"]
            if not account.get("settings", {}).get("show_hold", True):
                continue

            account_name = account.get("name") or account_id.upper()

            # 원금 및 현금 로드
            m_data = load_portfolio_master(account_id)
            if m_data:
                global_principal += m_data.get("total_principal", 0.0)
                global_cash += m_data.get("cash_balance", 0.0)

            df = load_real_holdings_with_recommendations(account_id)

            if df is not None and not df.empty:
                df.insert(0, "계좌", account_name)
                all_holdings.append(df)

        if not all_holdings:
            st.info("현재 모든 계좌를 통틀어 보유 중인 종목이 없습니다.")
            return

        combined_df = pd.concat(all_holdings, ignore_index=True)

        weight_df = None

        # 요약 메트릭 계산
        if "평가금액(KRW)" in combined_df.columns and "매입금액(KRW)" in combined_df.columns:
            total_valuation = combined_df["평가금액(KRW)"].sum()  # 주식 평가금액
            total_purchase = combined_df["매입금액(KRW)"].sum()  # 주식 매입금액
            total_stock_profit = total_valuation - total_purchase  # 주식 평가손익

            total_assets = total_valuation + global_cash  # 총 자산 (주식 + 현금)
            net_profit = total_assets - global_principal  # 전체 평가손익 (자산 - 원금)
            net_profit_pct = (net_profit / global_principal) * 100 if global_principal > 0 else 0.0

            # 1. 자산 요약 관련 변수 유지 (metric용)

            # 통계용 3컬럼 테이블 데이터 생성
            stat_df = pd.DataFrame(
                [
                    {
                        "총 자산": f"{total_assets:,.0f}원",
                        "매입 금액": f"{total_purchase:,.0f}원",
                        "평가 금액": f"{total_valuation:,.0f}원",
                    }
                ]
            )

            def style_stat_df(df):
                return pd.DataFrame(
                    [
                        ["background-color: #93c47d; color: black; font-size: 16px;"] * 1
                        + ["background-color: #76a5af; color: black; font-size: 16px;"] * 1
                        + ["background-color: #6fa8dc; color: black; font-size: 16px;"] * 1
                    ],
                    index=df.index,
                    columns=df.columns,
                )

            styled_stat_df = stat_df.style.apply(style_stat_df, axis=None)

            # 2. 포트폴리오 비중 테이블 데이터 생성
            bucket_cols = ["1. 모멘텀", "2. 혁신기술", "3. 시장지수", "4. 배당방어", "5. 대체헷지"]
            bucket_totals = {}
            for col in bucket_cols:
                if "버킷" in combined_df.columns:
                    val = combined_df.loc[combined_df["버킷"] == col, "평가금액(KRW)"].sum()
                else:
                    val = 0.0
                bucket_totals[col] = val

            bucket_totals["6. 현금"] = global_cash

            weight_data = {}
            if total_assets > 0:
                for k, v in bucket_totals.items():
                    weight_data[k] = f"{(v / total_assets) * 100:.2f}%"
            else:
                for k in bucket_totals.keys():
                    weight_data[k] = "0.00%"

            weight_df = pd.DataFrame([weight_data])

        if "cache_warnings" in st.session_state and st.session_state.cache_warnings:
            # {account_id: {ticker_set}}
            warning_msg = "⚠️ **다음 계좌에서 일부 종목의 가격 데이터를 불러오지 못했습니다:**\n\n"

            # 계좌 ID를 이름으로 매핑하기 위한 맵 생성
            id_to_name = {acc["account_id"]: (acc.get("name") or acc["account_id"].upper()) for acc in accounts}

            for acc_id, tickers in sorted(st.session_state.cache_warnings.items()):
                target_name = id_to_name.get(acc_id, acc_id.upper())
                ticker_str = ", ".join(sorted(tickers))
                warning_msg += f"- **{target_name}**: {ticker_str}\n"

            st.warning(
                f"{warning_msg}\n"
                "현재가가 0원으로 표시될 수 있습니다. 해결을 위해 백그라운드 스크립트(`python scripts/update_price_cache.py`)를 "
                "실행하여 가격 정보를 갱신해 주시기 바랍니다."
            )
            # 한 번 보여준 후 다음 렌더링을 위해 초기화
            st.session_state.cache_warnings = {}

        tab_summary, tab_details = st.tabs(["📊 요약", "📋 상세"])

        with tab_summary:
            if total_assets > 0 or total_purchase > 0:
                st.subheader("총 자산 요약")
                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric(label="총 자산 (주식+현금)", value=f"{total_assets:,.0f}원")
                c2.metric(label="총 투자 원금", value=f"{global_principal:,.0f}원")
                c3.metric(label="총 평가손익", value=f"{net_profit:,.0f}원", delta=f"{net_profit_pct:,.2f}%")
                c4.metric(label="총 현금 보유량", value=f"{global_cash:,.0f}원")
                c5.metric(label="주식 평가손익", value=f"{total_stock_profit:,.0f}원")

                st.divider()

                st.subheader("포트폴리오 구성 비중")
                st.dataframe(weight_df, hide_index=True, width="stretch")

                st.subheader("통계용")
                st.dataframe(styled_stat_df, hide_index=True, width="stretch")
            else:
                st.info("평가금액 및 매입금액 데이터가 없어 요약을 표시할 수 없습니다.")

        with tab_details:
            # 정렬: 계좌순(이름에 order가 포함됨) -> 버킷순
            if "bucket" in combined_df.columns:
                combined_df = combined_df.sort_values(["계좌", "bucket"], ascending=[True, True])
            else:
                combined_df = combined_df.sort_values(["계좌"], ascending=[True])

            # Rename target column to 평가수익률(%)
            if "수익률(%)" in combined_df.columns:
                combined_df = combined_df.rename(columns={"수익률(%)": "평가수익률(%)"})

            # render_recommendation_table 호출 (컬럼 순서 제어를 위해 visible_columns 명시)
            visible_cols = [
                "계좌",
                "환종",
                "버킷",
                "티커",
                "종목명",
                "일간(%)",
                "보유일",
                "평가수익률(%)",
                "수량",
                "평균 매입가",
                "현재가",
                "매입금액(KRW)",
                "평가금액(KRW)",
                "평가손익(KRW)",
                "추세(3달)",
            ]
            # Warnings moved to the top of the tabs

            render_recommendation_table(combined_df, grouped_by_bucket=False, visible_columns=visible_cols, height=900)

    return _render_home_page


def main() -> None:
    navigation = getattr(st, "navigation", None)
    page_cls = getattr(st, "Page", None)
    if navigation is None or page_cls is None:
        st.error("현재 설치된 Streamlit 버전이 `st.navigation`을 지원하지 않습니다.")
        st.stop()

    accounts = load_account_configs()
    if not accounts:
        st.error("사용할 수 있는 계정 설정이 없습니다. `zaccounts/account` 폴더를 확인해주세요.")
        st.stop()

    default_icon = "📈"

    st.set_page_config(
        page_title="Momentum ETF",
        page_icon=default_icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Open Graph 메타 태그 추가 (링크 미리보기용)
    # 참고: Streamlit의 제약으로 st.markdown()으로 추가한 메타 태그는 <body>에 들어가므로
    # 실제로는 Nginx sub_filter를 통해 <head>에 주입해야 합니다.
    st.markdown(
        """
        <meta property="og:title" content="Momentum ETF" />
        <meta property="og:description" content="추세추종 전략 기반 ETF 투자" />
        <meta property="og:image" content="https://etf.dojason.com/static/og-image.png" />
        <meta property="og:url" content="https://etf.dojason.com/" />
        <meta property="og:type" content="website" />
        <meta property="og:site_name" content="Momentum ETF" />
        <meta name="twitter:card" content="summary_large_image" />
        <meta name="twitter:title" content="Momentum ETF" />
        <meta name="twitter:description" content="추세추종 전략 기반 ETF 투자" />
        <meta name="twitter:image" content="https://etf.dojason.com/static/og-image.png" />
        """,
        unsafe_allow_html=True,
    )

    # --- 1. 페이지 정의 (인증보다 먼저 수행하여 라우팅 정보 등록) ---
    from app_pages.transactions_page import build_transaction_page

    pages = [
        page_cls(
            _build_home_page(accounts),
            title="보유종목",
            icon="🏠",
            default=True,
        )
    ]
    pages.append(build_transaction_page(page_cls))
    for account in accounts:
        pages.append(_build_account_page(page_cls, account))

    # 네비게이션 객체 생성 (이 시점에 URL 경로가 인식됨)
    pg = navigation(pages, position="top")

    # --- 인증 로직 시작 ---
    authenticator = _load_authenticator()
    # "main_login" 키를 사용하여 로그인 상태 관리
    _, auth_status, _ = authenticator.login(location="main")

    if auth_status is False:
        st.error("이메일/사용자명 또는 비밀번호가 올바르지 않습니다.")
        st.stop()
    elif auth_status is None:
        st.warning("계속하려면 로그인하세요.")
        st.stop()

    # 로그인 성공 시 사이드바에 로그아웃 버튼 표시
    with st.sidebar:
        st.write(f"환영합니다, {st.session_state.get('name', 'User')}님!")
        authenticator.logout(button_name="로그아웃", location="sidebar")
        st.divider()
    # --- 인증 로직 끝 ---

    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 0.5rem !important;
            padding-bottom: 0.5rem !important;
            padding-left: 1.0rem !important;
            padding-right: 1.0rem !important;
        }

        .block-container h1,
        .block-container h2,
        .block-container h3 {
            margin-top: 0.5rem;
        }

        .stTabs [data-baseweb="tab-list"] {
            margin-top: 0 !important;
        }

        section[data-testid="stSidebar"][aria-expanded="true"] {
            width: 12rem !important;
            min-width: 12rem !important;
        }

        section[data-testid="stSidebar"][aria-expanded="false"] {
            width: 0 !important;
            min-width: 0 !important;
        }

        section[data-testid="stSidebar"][aria-expanded="true"] > div {
            width: 12rem !important;
        }

        section[data-testid="stSidebar"][aria-expanded="false"] > div {
            width: 0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # --- 3. 라우팅 실행 ---
    pg.run()


if __name__ == "__main__":
    main()
