import os
import sys
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.stock_list_io import get_etfs


@st.cache_data
def get_cached_etfs(country_code: str) -> List[Dict[str, Any]]:
    """종목 마스터(data/stocks/{country}.json) 데이터를 캐시하여 반환합니다."""
    return get_etfs(country_code) or []


def render_master_etf_ui(country_code: str):
    """종목 마스터 조회 UI를 렌더링합니다."""
    if country_code == "coin":
        st.info("이곳에서 가상화폐 종목을 조회할 수 있습니다.")
    else:
        st.info("이곳에서 투자 유니버스에 포함된 종목을 조회할 수 있습니다.")

    etfs_data = get_cached_etfs(country_code)
    if not etfs_data:
        st.info("조회할 종목이 없습니다.")
        return

    for etf in etfs_data:
        if "is_active" not in etf:
            st.error(f"종목 마스터 파일의 '{etf.get('ticker')}' 종목에 'is_active' 필드가 없습니다. 파일을 확인해주세요.")
            st.stop()

    df_etfs = pd.DataFrame(etfs_data)
    df_etfs["is_active"] = df_etfs["is_active"].fillna(True)

    if "name" not in df_etfs.columns:
        df_etfs["name"] = ""
    df_etfs["name"] = df_etfs["name"].fillna("").astype(str)

    if country_code == "coin":
        if "type" not in df_etfs.columns:
            df_etfs["type"] = "crypto"
        df_etfs["type"] = df_etfs["type"].fillna("crypto")
    else:
        if "type" not in df_etfs.columns:
            df_etfs["type"] = ""

    if "last_modified" not in df_etfs.columns:
        df_etfs["last_modified"] = pd.NaT

    df_etfs["modified_sort_key"] = pd.to_datetime(df_etfs["last_modified"], errors="coerce")
    df_etfs.sort_values(by=["modified_sort_key"], ascending=True, na_position="first", inplace=True)

    display_cols = ["ticker", "name", "category", "is_active"]
    df_for_display = df_etfs.reindex(columns=display_cols)

    st.dataframe(
        df_for_display,
        width="stretch",
        hide_index=True,
        key=f"etf_viewer_{country_code}",
        column_config={
            "ticker": st.column_config.TextColumn("티커"),
            "name": st.column_config.TextColumn("종목명"),
            "is_active": st.column_config.CheckboxColumn("활성", disabled=True),
        },
    )


def main():
    """종목 관리 페이지를 렌더링합니다."""
    st.title("🗃️ 종목 관리 (Master Data)")

    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&display=swap');
            body {
                font-family: 'Noto Sans KR', sans-serif;
            }
            .block-container {
                max-width: 100%;
                padding-top: 1rem;
                padding-left: 2rem;
                padding-right: 2rem;
            }
        </style>
    """,
        unsafe_allow_html=True,
    )

    tab_kor, tab_aus, tab_coin = st.tabs(["한국", "호주", "코인"])

    with tab_kor:
        with st.spinner("한국 종목 데이터를 불러오는 중..."):
            render_master_etf_ui("kor")
    with tab_aus:
        with st.spinner("호주 종목 데이터를 불러오는 중..."):
            render_master_etf_ui("aus")
    with tab_coin:
        with st.spinner("코인 종목 데이터를 불러오는 중..."):
            render_master_etf_ui("coin")


if __name__ == "__main__":
    main()
