from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache

import pandas as pd
import streamlit as st
import streamlit_authenticator as stauth

from utils.cache_utils import (
    list_cached_countries,
    list_cached_tickers,
    load_cached_frame,
)
from utils.stock_list_io import get_all_etfs


@lru_cache(maxsize=8)
def _ticker_name_map(country: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    try:
        items = get_all_etfs(country)
    except Exception:
        return mapping

    for item in items:
        ticker = str(item.get("ticker") or "").upper()
        name = str(item.get("name") or "").strip()
        if ticker:
            mapping[ticker] = name if name else ticker
    return mapping


def _to_plain_dict(value):
    if isinstance(value, Mapping):
        return {k: _to_plain_dict(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_plain_dict(v) for v in value]
    return value


def _load_authenticator() -> stauth.Authenticate:
    raw_config = st.secrets.get("auth")
    if not raw_config:
        st.error("인증 설정(st.secrets['auth'])이 구성되지 않았습니다.")
        st.stop()

    config = _to_plain_dict(raw_config)

    credentials = config.get("credentials")
    cookie = config.get("cookie") or {}
    preauthorized = config.get("preauthorized", {})

    required_keys = {"name", "key", "expiry_days"}
    if not credentials or not cookie or not required_keys.issubset(cookie):
        st.error("인증 설정 필드가 누락되었습니다. credentials/cookie 구성을 확인하세요.")
        st.stop()

    return stauth.Authenticate(
        credentials,
        cookie.get("name"),
        cookie.get("key"),
        cookie.get("expiry_days"),
        preauthorized,
    )


def render_cache_admin_page() -> None:
    st.set_page_config(page_title="[Admin] 종목 캐시", page_icon="🗃️", layout="wide")
    st.caption("종목 가격 데이터 캐시를 조회하는 관리용 페이지입니다.")

    authenticator = _load_authenticator()
    _, auth_status, _ = authenticator.login(key="cache_login", location="main")

    if not auth_status:
        st.warning("이 페이지에 접근하려면 로그인이 필요합니다.")
        return

    header_col, logout_col = st.columns([5, 1])
    with logout_col:
        authenticator.logout(button_name="로그아웃", location="main", key="cache_logout")

    countries = list_cached_countries()
    if not countries:
        st.warning("캐시 데이터가 저장된 국가가 없습니다.")
        return

    selected_country = st.selectbox("국가 선택", countries, index=0, key="cache_country_selector")

    tickers = list_cached_tickers(selected_country)
    if not tickers:
        st.warning(f"{selected_country.upper()} 국가에 대한 캐시가 없습니다.")
        return

    name_map = _ticker_name_map(selected_country)

    def _format_option(value: str) -> str:
        ticker_upper = value.upper()
        display_name = name_map.get(ticker_upper, ticker_upper)
        return f"{display_name}({ticker_upper})" if display_name and display_name != ticker_upper else ticker_upper

    selected_tkr = st.selectbox(
        "종목 선택",
        tickers,
        index=0,
        key="cache_ticker_selector",
        format_func=_format_option,
        help="선택한 종목의 캐시 데이터를 아래에 표시합니다.",
    )

    if not selected_tkr:
        st.info("종목을 선택해 주세요.")
        return

    try:
        df = load_cached_frame(selected_country, selected_tkr)
        if df is None or df.empty:
            raise RuntimeError("저장된 캐시가 없거나 비어 있습니다.")
    except Exception as exc:
        st.error(f"캐시 데이터를 불러오는 중 오류가 발생했습니다: {exc}")
        return

    display_label = _format_option(selected_tkr)
    st.markdown(f"### {display_label}")

    st.caption(f"행 개수: {len(df):,} | 열: {', '.join(df.columns.astype(str)) if not df.empty else '-'}")

    if isinstance(df.index, pd.DatetimeIndex):
        df_sorted = df.sort_index(ascending=False)
    else:
        df_sorted = df.iloc[::-1].reset_index(drop=True)

    st.dataframe(df_sorted, width="stretch", height=600)


__all__ = ["render_cache_admin_page"]


render_cache_admin_page()
