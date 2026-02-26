"""모든 종목 목록 페이지 — stocks.json 메타정보 테이블."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

from utils.account_registry import load_account_configs
from utils.stock_list_io import get_etfs


def _format_volume(val: Any) -> str:
    """거래량을 읽기 쉬운 형태로 포맷."""
    if val is None or pd.isna(val):
        return "-"
    try:
        num = int(float(val))
        return f"{num:,}"
    except (TypeError, ValueError):
        return str(val)


@st.cache_data(ttl=30, show_spinner=False)
def _build_all_stocks_table(account_id: str) -> pd.DataFrame:
    """stocks.json 메타정보를 DataFrame으로 반환."""

    etfs = get_etfs(account_id)
    if not etfs:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for idx, etf in enumerate(etfs, 1):
        rows.append(
            {
                "#": idx,
                "티커": etf.get("ticker", ""),
                "종목명": etf.get("name", ""),
                "상장일": etf.get("listing_date", "-"),
                "주간거래량": etf.get("1_week_avg_volume"),
                "1주(%)": etf.get("1_week_earn_rate"),
                "2주(%)": etf.get("2_week_earn_rate"),
                "1달(%)": etf.get("1_month_earn_rate"),
                "3달(%)": etf.get("3_month_earn_rate"),
                "6달(%)": etf.get("6_month_earn_rate"),
                "12달(%)": etf.get("12_month_earn_rate"),
            }
        )

    return pd.DataFrame(rows)


def _style_dataframe(df: pd.DataFrame) -> pd.io.formats.style.Styler:
    """DataFrame에 스타일 적용 (수익률 색상)."""

    def _color_pct(val: float | str) -> str:
        if val is None or pd.isna(val):
            return ""
        try:
            num = float(val)
        except (TypeError, ValueError):
            return ""
        if num > 0:
            return "color: red"
        if num < 0:
            return "color: blue"
        return "color: black"

    styled = df.style
    pct_columns = ["1주(%)", "2주(%)", "1달(%)", "3달(%)", "6달(%)", "12달(%)"]
    for col in pct_columns:
        if col in df.columns:
            styled = styled.map(_color_pct, subset=pd.IndexSlice[:, col])

    return styled


def render_all_stocks_page() -> None:
    """모든 종목 페이지 렌더링."""

    # page config은 app.py의 main()에서 이미 설정됨

    st.caption("종목 유니버스 메타정보 (stocks.json)")

    accounts_meta = load_account_configs()
    if not accounts_meta:
        st.error("설정된 계정이 없습니다.")
        return

    # 계정 이름 매핑 (라벨 -> ID), order 순 정렬됨
    account_map = {}
    for acc in accounts_meta:
        acc_id = acc["account_id"]
        label = acc["name"]
        account_map[label] = acc_id

    # 계정 선택 (Pills 스타일)
    display_options = list(account_map.keys())

    # URL 쿼리 파라미터에서 초기값 읽기 (?account=aus)
    default_label = display_options[0] if display_options else None
    query_account = st.query_params.get("account")

    if query_account:
        for label, acc_id in account_map.items():
            if acc_id == query_account:
                default_label = label
                break

    selected_label = st.pills("계정 선택", display_options, default=default_label, key="account_selector")

    if not selected_label:
        st.info("계정을 선택해주세요.")
        return

    selected_account = account_map[selected_label]

    # 선택된 계정을 URL 파라미터에 반영 (동기화)
    if selected_account != query_account:
        st.query_params["account"] = selected_account

    with st.spinner("데이터 로딩 중..."):
        df = _build_all_stocks_table(selected_account)

    if df.empty:
        st.error("종목 데이터를 불러올 수 없습니다.")
        return

    st.caption(f"총 {len(df)}개 종목 | 업데이트: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 컬럼 설정
    column_config = {
        "#": st.column_config.TextColumn("#", width=50),
        "티커": st.column_config.TextColumn("티커", width=80),
        "종목명": st.column_config.TextColumn("종목명", width=300),
        "상장일": st.column_config.TextColumn("상장일", width=110),
        "주간거래량": st.column_config.NumberColumn("주간거래량", width=120, format="%d"),
        "1주(%)": st.column_config.NumberColumn("1주(%)", width="small", format="%.2f%%"),
        "2주(%)": st.column_config.NumberColumn("2주(%)", width="small", format="%.2f%%"),
        "1달(%)": st.column_config.NumberColumn("1달(%)", width="small", format="%.2f%%"),
        "3달(%)": st.column_config.NumberColumn("3달(%)", width="small", format="%.2f%%"),
        "6달(%)": st.column_config.NumberColumn("6달(%)", width="small", format="%.2f%%"),
        "12달(%)": st.column_config.NumberColumn("12달(%)", width="small", format="%.2f%%"),
    }

    column_order = [
        "#",
        "티커",
        "종목명",
        "상장일",
        "주간거래량",
        "1주(%)",
        "2주(%)",
        "1달(%)",
        "3달(%)",
        "6달(%)",
        "12달(%)",
    ]
    existing_columns = [col for col in column_order if col in df.columns]
    df_reordered = df[existing_columns]

    # 스타일 적용
    styled_df = _style_dataframe(df_reordered)

    # 테이블 표시 (높이 자동 조절)
    calc_height = (len(df_reordered.index) + 1) * 35 + 10
    st.dataframe(
        styled_df,
        hide_index=True,
        width="stretch",
        height=calc_height,
        column_config=column_config,
    )


if __name__ == "__main__":
    render_all_stocks_page()
