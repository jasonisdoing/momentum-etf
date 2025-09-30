from __future__ import annotations

import streamlit as st

from main import load_country_recommendations, render_recommendation_table
from utils.settings_loader import get_country_settings


try:
    au_settings = get_country_settings("aus")
    page_title = au_settings.get("name") or "호주"
    page_icon = au_settings.get("icon") or "🇦🇺"
except Exception:
    page_title = "호주"
    page_icon = "🇦🇺"

st.title(f"{page_icon} {page_title}")
st.caption("data/results/aus.json 기반")

df, updated_at = load_country_recommendations("aus")

if df is None:
    st.error(updated_at or "데이터를 불러오지 못했습니다.")
else:
    if updated_at:
        st.caption(f"데이터 업데이트: {updated_at}")

    render_recommendation_table(df, country="aus")

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
