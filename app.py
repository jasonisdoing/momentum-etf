from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import streamlit as st
import streamlit_authenticator as stauth

from app_pages.account_page import render_account_page
from utils.account_registry import (
    get_icon_fallback,
    load_account_configs,
)
from utils.ui import load_account_recommendations, render_recommendation_table


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
    # 보유 중인 종목: HOLD + 매도 신호가 있지만 아직 보유 중인 종목
    allowed_states = {
        "HOLD",
        "BUY",
        "BUY_REPLACE",
        "SELL_TREND",
        "SELL_RSI",
        "CUT_STOPLOSS",
        # "SELL_REPLACE",
    }

    def _render_home_page() -> None:
        for account in accounts:
            account_id = account["account_id"]
            if not account.get("settings", {}).get("show_hold", True):
                continue

            account_name = account.get("name") or account_id.upper()
            df, updated_at, country_code = load_account_recommendations(account_id)

            st.text(f"{account_name} ({account_id.upper()})")

            if df is None or df.empty:
                st.info("표시할 추천 데이터가 없습니다.")
                continue

            filtered = df[df["상태"].str.upper().isin(allowed_states)]
            if filtered.empty:
                st.info("현재 보유 중인 종목이 없습니다.")
                continue

            render_recommendation_table(filtered, country_code=country_code)

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
    pages = [
        page_cls(
            _build_home_page(accounts),
            title="보유종목",
            icon="🏠",
            default=True,
        )
    ]
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
