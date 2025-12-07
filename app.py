from __future__ import annotations

from collections.abc import Callable
from typing import Any

import streamlit as st

from app_pages.account_page import render_account_page
from utils.account_registry import (
    get_icon_fallback,
    load_account_configs,
)
from utils.ui import load_account_recommendations, render_recommendation_table


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
    # 보유 중인 종목: HOLD, HOLD_CORE + 매도 신호가 있지만 아직 보유 중인 종목
    allowed_states = {"HOLD", "HOLD_CORE", "SELL_TREND", "SELL_RSI", "CUT_STOPLOSS"}

    def _render_home_page() -> None:
        for account in accounts:
            account_id = account["account_id"]
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
        st.markdown("---")
        st.markdown(
            """
            - 본 웹사이트에서 제공되는 종목 정보 및 추천은 단순 정보 제공을 목적으로 하며, 특정 종목의 매매를 권유하는 것이 아닙니다.
            - 본 정보를 이용한 투자 판단 및 매매 결과에 대하여 웹사이트 운영자는 어떠한 책임도 지지 않습니다.
            - 투자에는 원금 손실 가능성이 있으며, 투자자는 스스로 리스크를 검토해야 합니다.
            """
        )

    return _render_home_page


def main() -> None:
    navigation = getattr(st, "navigation", None)
    page_cls = getattr(st, "Page", None)
    if navigation is None or page_cls is None:
        st.error("현재 설치된 Streamlit 버전이 `st.navigation`을 지원하지 않습니다.")
        st.stop()

    accounts = load_account_configs()
    if not accounts:
        st.error("사용할 수 있는 계정 설정이 없습니다. `zsettings/account` 폴더를 확인해주세요.")
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

    pages.append(
        page_cls(
            "app_pages/all_stocks.py",
            title="전체 종목",
            icon="📊",
            url_path="all",
        )
    )

    pages.append(
        page_cls(
            "app_pages/trade.py",
            title="[Admin] trade",
            icon="📝",
            url_path="admin",
        )
    )

    # pages.append(
    #     page_cls(
    #         "app_pages/stocks.py",
    #         title="[Admin] 종목 정보",
    #         icon="📊",
    #         url_path="stocks",
    #     )
    # )

    pages.append(
        page_cls(
            "app_pages/cache_admin.py",
            title="[Admin] 종목 캐시",
            icon="🗃️",
            url_path="cache",
        )
    )

    # pages.append(
    #     page_cls(
    #         "app_pages/migration.py",
    #         title="[Admin] 마이그레이션",
    #         icon="🛠️",
    #         url_path="migration",
    #     )
    # )

    # pages.append(
    #     page_cls(
    #         "app_pages/delete.py",
    #         title="[Admin] 계정 삭제",
    #         icon="🗑️",
    #         url_path="delete",
    #     )
    # )

    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.5rem !important;
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

    navigation(pages, position="top").run()


if __name__ == "__main__":
    main()
