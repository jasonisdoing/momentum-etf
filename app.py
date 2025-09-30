from __future__ import annotations

import streamlit as st

from utils.settings_loader import get_country_settings


def _load_country_ui_settings(country: str) -> tuple[str, str]:
    try:
        settings = get_country_settings(country)
        name = settings.get("name") or country.upper()
        icon = settings.get("icon") or ""
    except Exception:
        name = country.upper()
        icon = ""
    return name, icon


def main() -> None:
    navigation = getattr(st, "navigation", None)
    page_cls = getattr(st, "Page", None)
    if navigation is None or page_cls is None:
        st.error("현재 설치된 Streamlit 버전이 `st.navigation`을 지원하지 않습니다.")
        st.stop()

    kor_info = _load_country_ui_settings("kor")
    aus_info = _load_country_ui_settings("aus")

    initial_title = kor_info[0]
    initial_icon = kor_info[1] or "🇰🇷"

    st.set_page_config(
        page_title=initial_title,
        page_icon=initial_icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    pages = [
        page_cls(
            "app_pages/kor.py",
            title="한국",
            icon=kor_info[1] or "🇰🇷",
            default=True,
        ),
        page_cls(
            "app_pages/aus.py",
            title="호주",
            icon=aus_info[1] or "🇦🇺",
        ),
        page_cls(
            "app_pages/30_trade.py",
            title="관리자",
            icon="📝",
        ),
    ]

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
