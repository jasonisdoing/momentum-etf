from __future__ import annotations

from typing import Any, Dict

import streamlit as st

from main import load_country_recommendations, render_recommendation_table
from utils.account_registry import get_icon_fallback
from utils.settings_loader import CountrySettingsError, get_country_settings


_DEFAULT_CAPTIONS: Dict[str, str] = {
    "kor": "내부 알고리즘 기반 10종목 추천",
    "aus": "data/results/recommendation_aus.json 기반",
}

_DATAFRAME_CSS = """
<style>
    .stDataFrame thead tr th {
        text-align: center;
    }
    .stDataFrame tbody tr td {
        text-align: center;
        white-space: nowrap;
    }
</style>
"""


def _normalize_code(value: Any, fallback: str) -> str:
    text = str(value or "").strip().lower()
    return text or fallback


def _resolve_caption(settings: Dict[str, Any], country_code: str) -> str:
    raw = settings.get("page_caption")
    if isinstance(raw, str) and raw.strip():
        return raw.strip()
    return _DEFAULT_CAPTIONS.get(
        country_code, f"data/results/recommendation_{country_code}.json 기반"
    )


def render_country_page(account_id: str) -> None:
    """Render a recommendation page for a given account (settings file)."""

    try:
        settings = get_country_settings(account_id)
    except CountrySettingsError as exc:  # pragma: no cover - Streamlit feedback only
        st.error(f"설정을 불러오지 못했습니다: {exc}")
        st.stop()

    country_code = _normalize_code(settings.get("country_code"), account_id)

    page_title = settings.get("name") or account_id.upper()
    page_icon = settings.get("icon") or get_icon_fallback(country_code)

    title_text = page_title
    if page_icon:
        title_text = f"{page_icon} {page_title}".strip()

    st.title(title_text)

    caption_text = _resolve_caption(settings, country_code)
    if caption_text:
        st.caption(caption_text)

    df, updated_at = load_country_recommendations(country_code)

    if df is None:
        st.error(updated_at or "데이터를 불러오지 못했습니다.")
        return

    if updated_at:
        st.caption(f"데이터 업데이트: {updated_at}")

    render_recommendation_table(df, country=country_code)

    st.markdown(_DATAFRAME_CSS, unsafe_allow_html=True)


__all__ = ["render_country_page"]
