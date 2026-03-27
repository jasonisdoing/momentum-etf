"""KIS 기반 국내 ETF 마스터 조회 페이지."""

from __future__ import annotations

import re
from datetime import datetime

import pandas as pd
import streamlit as st

from services.price_service import get_realtime_snapshot
from services.reference_data_service import get_kor_etf_master
from utils.ui import render_rank_table

EXCLUSION_KEYWORD_GROUPS = {
    "인버스": ["인버스"],
    "2X": ["2X"],
    "레버리지": ["레버리지"],
    "선물": ["선물"],
    "채권(모든종류)": ["채권", "미국채", "국채", "회사채", "단기채", "장기채"],
    "혼합": ["혼합"],
    "리츠": ["리츠"],
    "합성": ["합성"],
    "커버드콜": ["커버드콜"],
}

DEFAULT_EXCLUDED_KEYWORD_GROUPS = [
    "인버스",
    "2X",
    "레버리지",
    "선물",
    "채권(모든종류)",
    "혼합",
    "리츠",
]


@st.cache_data(ttl=300, show_spinner=False)
def _load_market_table() -> tuple[pd.DataFrame, datetime | None]:
    """캐시된 KIS 국내 ETF 마스터와 실시간 스냅샷을 결합합니다."""
    df, updated_at = get_kor_etf_master()
    tickers = [str(value).strip().upper() for value in df["티커"].tolist() if str(value).strip()]
    snapshot = get_realtime_snapshot("kor", tickers)

    result = df.copy()
    result["일간(%)"] = result["티커"].map(
        lambda ticker: _safe_float((snapshot.get(str(ticker).strip().upper()) or {}).get("changeRate"))
    )
    result["현재가"] = result["티커"].map(
        lambda ticker: _safe_float((snapshot.get(str(ticker).strip().upper()) or {}).get("nowVal"))
    )
    result["Nav"] = result["티커"].map(
        lambda ticker: _safe_float((snapshot.get(str(ticker).strip().upper()) or {}).get("nav"))
    )
    result["괴리율"] = result["티커"].map(
        lambda ticker: _safe_float((snapshot.get(str(ticker).strip().upper()) or {}).get("deviation"))
    )
    result["3달(%)"] = result["티커"].map(
        lambda ticker: _safe_float((snapshot.get(str(ticker).strip().upper()) or {}).get("threeMonthEarnRate"))
    )
    result["전일거래량(주)"] = result["전일거래량"]
    result["시가총액(억)"] = result["시가총액"]
    result = result.drop(columns=["전일거래량", "시가총액"])
    return result, updated_at


def _safe_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def render_etf_market_page() -> None:
    """ETF 마켓 페이지를 렌더링합니다."""

    st.caption("한국투자증권(KIS) 종목정보파일 기반 국내 상장 ETF 조회")

    refresh_col, info_col = st.columns([1, 4])
    with refresh_col:
        if st.button("새로고침", width="stretch"):
            _load_market_table.clear()
    with info_col:
        st.caption("기본 마스터는 KIS 일일 캐시, 실시간 시세는 가격 서비스에서 조회합니다.")

    with st.spinner("KIS 국내 ETF 목록을 불러오는 중..."):
        try:
            df, master_updated_at = _load_market_table()
        except Exception as exc:
            st.error(f"ETF 마켓 캐시를 불러오지 못했습니다: {exc}")
            return

    if df.empty:
        st.error("국내 ETF 목록을 불러오지 못했습니다.")
        return

    query_col, market_cap_col, volume_col = st.columns([2.2, 1, 1])
    with query_col:
        query = st.text_input("검색", placeholder="티커 또는 종목명")
    with market_cap_col:
        min_market_cap = st.number_input(
            "최소 시가총액(억)", min_value=0.0, value=None, step=100.0, placeholder="예: 1000"
        )
    with volume_col:
        min_prev_volume = st.number_input(
            "최소 전일 거래량(주)",
            min_value=0,
            value=None,
            step=10000,
            placeholder="예: 100000",
        )
    excluded_keyword_groups = st.pills(
        "기본 제외 키워드",
        options=list(EXCLUSION_KEYWORD_GROUPS.keys()),
        default=DEFAULT_EXCLUDED_KEYWORD_GROUPS,
        selection_mode="multi",
    )

    filtered = df.copy()

    query_text = str(query or "").strip().upper()
    if query_text:
        filtered = filtered[
            filtered["티커"].astype(str).str.upper().str.contains(query_text, na=False)
            | filtered["종목명"].astype(str).str.upper().str.contains(query_text, na=False)
        ]

    if excluded_keyword_groups:
        expanded_keywords = [
            keyword for group in excluded_keyword_groups for keyword in EXCLUSION_KEYWORD_GROUPS.get(group, [])
        ]
        keyword_pattern = "|".join(re.escape(keyword) for keyword in expanded_keywords)
        filtered = filtered[~filtered["종목명"].astype(str).str.contains(keyword_pattern, case=False, na=False)]

    if min_market_cap is not None:
        filtered = filtered[filtered["시가총액(억)"].ge(float(min_market_cap)).fillna(False)]

    if min_prev_volume is not None:
        filtered = filtered[filtered["전일거래량(주)"].ge(float(min_prev_volume)).fillna(False)]

    filtered = filtered.sort_values(["일간(%)", "티커"], ascending=[False, True], na_position="last").reset_index(
        drop=True
    )

    master_updated_text = "-"
    if master_updated_at is not None:
        master_ts = pd.Timestamp(master_updated_at)
        if master_ts.tzinfo is None:
            master_ts = master_ts.tz_localize("UTC")
        master_updated_text = master_ts.tz_convert("Asia/Seoul").strftime("%Y-%m-%d %H:%M:%S")

    st.caption(
        "총 "
        f"{len(filtered):,}개"
        f" | 전체 {len(df):,}개"
        f" | KIS 마스터 갱신 {master_updated_text}"
        f" | 조회시각 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    visible_columns = [
        "티커",
        "종목명",
        "일간(%)",
        "현재가",
        "괴리율",
        "3달(%)",
        "상장일",
        "전일거래량(주)",
        "시가총액(억)",
    ]
    render_rank_table(
        df=filtered,
        country_code="kor",
        grouped_by_bucket=False,
        visible_columns=[column for column in visible_columns if column in filtered.columns],
        height=760,
        column_config_overrides={
            "전일거래량(주)": st.column_config.NumberColumn("전일거래량(주)", width=90, format="localized"),
            "시가총액(억)": st.column_config.NumberColumn("시가총액(억)", width=90, format="localized"),
        },
    )


def build_etf_market_page(page_cls):
    """ETF 마켓 페이지 빌더."""
    return page_cls(
        render_etf_market_page,
        title="ETF 마켓",
        icon="🧭",
        url_path="etf-market",
    )
