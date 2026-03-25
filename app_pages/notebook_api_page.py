import os

import streamlit as st

from utils.notebook_exporter import CACHE_FILE


def render_notebook_api_page():
    """
    노트북LM 및 외부 크롤러를 위한 전용 데이터 서빙 페이지.
    JSON/Markdown 원본 데이터를 텍스트로 반환합니다.
    """
    data_type = st.query_params.get("data")

    if data_type == "rank":
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, encoding="utf-8-sig") as f:
                content = f.read()
            # 원본 마크다운을 코드 블록으로 출력하여 크롤러가 텍스트를 가져가기 쉽게 함
            st.code(content, language="markdown")
            # API 용도이므로 여기서 렌더링 중단
            st.stop()
        else:
            st.error("캐시 파일이 존재하지 않습니다. 시스템 정보 페이지에서 갱신을 먼저 수행해 주세요.")
            st.stop()
    else:
        st.warning("제공되지 않는 데이터 타입입니다. (?data=rank 등을 사용하세요)")
        st.stop()
