from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
import streamlit_authenticator as stauth

from utils.account_registry import (
    build_account_meta,
    load_account_configs,
    pick_default_account,
)
from utils.cache_utils import get_cached_date_range
from utils.stock_list_io import get_etfs


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


@lru_cache(maxsize=1)
def _account_registry():
    accounts = load_account_configs()
    meta = build_account_meta(accounts)
    default = pick_default_account(accounts) if accounts else None
    return accounts, meta, default


def _account_options() -> List[str]:
    accounts, _, _ = _account_registry()
    return [account["account_id"] for account in accounts]


def _account_meta() -> Dict[str, Dict[str, str]]:
    return _account_registry()[1]


def _default_account_id() -> str:
    accounts, _, default = _account_registry()
    if default:
        return default["account_id"]
    if accounts:
        return accounts[0]["account_id"]
    return "kor"


def _format_account_label(account_id: str) -> str:
    meta = _account_meta().get(account_id, {})
    icon = meta.get("icon", "")
    label = meta.get("label", account_id.upper())
    return f"{icon} {label}".strip()


def _resolve_country_code(account_id: str) -> str:
    meta = _account_meta().get(account_id, {})
    return (meta.get("country_code") or account_id).strip().lower()


def _render_cache_summary_table(country_code: str) -> None:
    try:
        etf_items = get_etfs(country_code)
    except Exception as exc:
        st.error(f"ETF 목록을 불러오지 못했습니다: {exc}")
        return

    if not etf_items:
        st.info("표시할 종목이 없습니다.")
        return

    rows: List[Dict[str, Any]] = []
    for item in etf_items:
        ticker = str(item.get("ticker") or "").strip().upper()
        cache_start = cache_end = "-"
        try:
            cache_range = get_cached_date_range(country_code, ticker)
            if cache_range:
                start, end = cache_range
                if start is not None:
                    cache_start = str(start.date())
                if end is not None:
                    cache_end = str(end.date())
        except Exception:
            cache_start = cache_end = "-"

        rows.append(
            {
                "티커": ticker,
                "종목명": item.get("name") or "-",
                "추천 사용": bool(item.get("recommend_enabled", True)),
                "상장일": item.get("listing_date") or "-",
                "1달 평균 거래량": item.get("1_month_avg_volume") or 0,
                "캐시 시작": cache_start,
                "캐시 종료": cache_end,
            }
        )

    df = pd.DataFrame(rows)
    df.sort_values(by="티커", inplace=True)
    st.dataframe(
        df,
        hide_index=True,
        width="stretch",
        column_config={
            "1달 평균 거래량": st.column_config.NumberColumn("1달 평균 거래량", format=",d"),
        },
    )


def render_stocks_admin_page() -> None:
    st.title("[Admin] 종목 정보")
    st.caption("로그인 후 계정을 선택하면 종목 기본 정보와 캐시 범위를 확인할 수 있습니다.")

    authenticator = _load_authenticator()
    _, auth_status, _ = authenticator.login(key="stocks_login", location="sidebar")

    if not auth_status:
        st.warning("이 페이지에 접근하려면 로그인이 필요합니다.")
        return

    st.sidebar.write("")
    authenticator.logout(button_name="로그아웃", location="sidebar")

    accounts = _account_options()
    if not accounts:
        st.info("사용 가능한 계정이 없습니다. `zsettings/account` 폴더를 확인하세요.")
        return

    if "stocks_selected_account" not in st.session_state:
        st.session_state["stocks_selected_account"] = _default_account_id()

    def _default_index() -> int:
        try:
            return accounts.index(st.session_state["stocks_selected_account"])
        except ValueError:
            return 0

    selected_account = st.selectbox(
        "계정을 선택하세요",
        accounts,
        index=_default_index(),
        format_func=_format_account_label,
    )
    st.session_state["stocks_selected_account"] = selected_account

    country_code = _resolve_country_code(selected_account)
    st.markdown(f"**선택된 계정:** {_format_account_label(selected_account)} (국가 코드: {country_code.upper()})")

    _render_cache_summary_table(country_code)


render_stocks_admin_page()
