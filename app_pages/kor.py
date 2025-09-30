from __future__ import annotations

import streamlit as st

from main import load_country_recommendations, render_recommendation_table
from utils.settings_loader import get_country_settings


try:
    settings = get_country_settings("kor")
    page_title = settings.get("name") or "한국"
    page_icon = settings.get("icon") or "🇰🇷"
except Exception:
    page_title = "한국"
    page_icon = "🇰🇷"

st.title(f"{page_icon} {page_title}")
st.caption("내부 알고리즘 기반 10종목 추천")

df, updated_at = load_country_recommendations("kor")

if df is None:
    st.error(updated_at or "데이터를 불러오지 못했습니다.")
else:
    if updated_at:
        st.caption(f"데이터 업데이트: {updated_at}")

    render_recommendation_table(df, country="kor")

    st.markdown(
        """
        <style>
            .stDataFrame thead tr th {
                text-align: center;
            }
            .stDataFrame tbody tr td {
                text-align: center;
                white-space: nowrap;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
