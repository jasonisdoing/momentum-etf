from __future__ import annotations

from typing import Any, Callable, Dict

import streamlit as st

from app_pages.account_page import render_account_page

from utils.account_registry import (
    get_icon_fallback,
    load_account_configs,
    pick_default_account,
)


def _build_account_page(page_cls: Callable[..., object], account: Dict[str, Any]):
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


def _render_home_page() -> None:
    st.title("대시보드")
    st.caption("서비스 진입점입니다. 좌측 메뉴에서 계정을 선택하세요.")


def main() -> None:
    navigation = getattr(st, "navigation", None)
    page_cls = getattr(st, "Page", None)
    if navigation is None or page_cls is None:
        st.error("현재 설치된 Streamlit 버전이 `st.navigation`을 지원하지 않습니다.")
        st.stop()

    accounts = load_account_configs()
    if not accounts:
        st.error("사용할 수 있는 계정 설정이 없습니다. `data/settings/account` 폴더를 확인해주세요.")
        st.stop()

    default_account = pick_default_account(accounts)
    default_icon = (
        default_account.get("icon")
        or get_icon_fallback(default_account.get("country_code", ""))
        or "📈"
    )

    st.set_page_config(
        page_title=default_account.get("name") or "Momentum ETF",
        page_icon=default_icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    pages = [
        page_cls(
            _render_home_page,
            title="대시보드",
            icon="🏠",
            default=True,
        )
    ]
    for account in accounts:
        pages.append(_build_account_page(page_cls, account))

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

    navigation(pages).run()

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

        section[data-testid="stSidebar"] {
            width: 12rem !important;
            min-width: 12rem !important;
        }

        section[data-testid="stSidebar"] > div {
            width: 12rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
