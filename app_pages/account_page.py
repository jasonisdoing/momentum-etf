from __future__ import annotations

from typing import Any, Dict

import streamlit as st

from main import load_account_recommendations, render_recommendation_table
from utils.account_registry import get_icon_fallback
from utils.settings_loader import AccountSettingsError, get_account_settings


_DEFAULT_CAPTIONS: Dict[str, str] = {
    "kor": "내부 알고리즘 기반 10종목 추천",
    "aus": "data/results/recommendation_{account}.json",
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
        country_code,
        f"data/results/recommendation_{country_code}.json 기반",
    )


def render_account_page(account_id: str) -> None:
    """주어진 계정 설정을 기반으로 추천 페이지를 렌더링합니다."""

    try:
        account_settings = get_account_settings(account_id)
    except AccountSettingsError as exc:  # pragma: no cover - Streamlit 오류 피드백 전용
        st.error(f"설정을 불러오지 못했습니다: {exc}")
        st.stop()

    country_code = _normalize_code(account_settings.get("country_code"), account_id)

    page_title = account_settings.get("name") or account_id.upper()
    page_icon = account_settings.get("icon") or get_icon_fallback(country_code)

    title_text = page_title
    if page_icon:
        title_text = f"{page_icon} {page_title}".strip()

    st.title(title_text)

    caption_text = _resolve_caption(account_settings, country_code)
    if caption_text:
        st.caption(caption_text)

    df, updated_at, loaded_country_code = load_account_recommendations(account_id)
    country_code = loaded_country_code or country_code

    if df is None:
        st.error(updated_at or "데이터를 불러오지 못했습니다.")
        return

    if updated_at:
        st.caption(f"데이터 업데이트: {updated_at}")

    render_recommendation_table(df, country_code=country_code)

    st.markdown(_DATAFRAME_CSS, unsafe_allow_html=True)


__all__ = ["render_account_page"]
