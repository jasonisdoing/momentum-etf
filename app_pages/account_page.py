from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from utils.account_registry import get_icon_fallback, load_account_configs
from utils.settings_loader import AccountSettingsError, get_account_settings, resolve_strategy_params
from utils.stock_list_io import get_etfs
from utils.ui import format_relative_time, load_account_recommendations, render_recommendation_table

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


# ---------------------------------------------------------------------------
# 종목관리 탭: stocks.json 메타정보 테이블
# ---------------------------------------------------------------------------


@st.cache_data(ttl=30, show_spinner=False)
def _build_stocks_meta_table(account_id: str) -> pd.DataFrame:
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
                "1달(%)": etf.get("1_month_earn_rate"),
                "3달(%)": etf.get("3_month_earn_rate"),
                "6달(%)": etf.get("6_month_earn_rate"),
                "12달(%)": etf.get("12_month_earn_rate"),
            }
        )
    return pd.DataFrame(rows)


def _render_stocks_meta_table(account_id: str) -> None:
    """종목관리 테이블 렌더링."""

    df = _build_stocks_meta_table(account_id)
    if df.empty:
        st.info("종목 데이터가 없습니다.")
        return

    st.caption(f"총 {len(df)}개 종목")

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

    pct_columns = ["1주(%)", "1달(%)", "3달(%)", "6달(%)", "12달(%)"]
    styled = df.style
    for col in pct_columns:
        if col in df.columns:
            styled = styled.map(_color_pct, subset=pd.IndexSlice[:, col])

    column_config = {
        "#": st.column_config.TextColumn("#", width=50),
        "티커": st.column_config.TextColumn("티커", width=80),
        "종목명": st.column_config.TextColumn("종목명", width=300),
        "상장일": st.column_config.TextColumn("상장일", width=110),
        "주간거래량": st.column_config.NumberColumn("주간거래량", width=120, format="%d"),
        "1주(%)": st.column_config.NumberColumn("1주(%)", width="small", format="%.2f%%"),
        "1달(%)": st.column_config.NumberColumn("1달(%)", width="small", format="%.2f%%"),
        "3달(%)": st.column_config.NumberColumn("3달(%)", width="small", format="%.2f%%"),
        "6달(%)": st.column_config.NumberColumn("6달(%)", width="small", format="%.2f%%"),
        "12달(%)": st.column_config.NumberColumn("12달(%)", width="small", format="%.2f%%"),
    }

    column_order = ["#", "티커", "종목명", "상장일", "주간거래량", "1주(%)", "1달(%)", "3달(%)", "6달(%)", "12달(%)"]
    existing_columns = [col for col in column_order if col in df.columns]

    st.dataframe(
        styled,
        hide_index=True,
        width="stretch",
        height=600,
        column_config=column_config,
        column_order=existing_columns,
    )


# ---------------------------------------------------------------------------
# 메인 렌더 함수
# ---------------------------------------------------------------------------


def render_account_page(account_id: str) -> None:
    """주어진 계정 설정을 기반으로 계정 페이지를 렌더링합니다 (탭 포함)."""

    try:
        account_settings = get_account_settings(account_id)
    except AccountSettingsError as exc:
        st.error(f"설정을 불러오지 못했습니다: {exc}")
        st.stop()

    country_code = _normalize_code(account_settings.get("country_code"), account_id)
    page_icon = account_settings.get("icon") or get_icon_fallback(country_code)

    # 메뉴명과 동일한 이름 사용 (PORTFOLIO_TOPN 포함)
    account_configs = load_account_configs()
    account_name = None
    for config in account_configs:
        if config["account_id"] == account_id:
            account_name = config["name"]
            break

    page_title = account_name or "Momentum ETF"
    st.set_page_config(page_title=page_title, page_icon=page_icon or "📈", layout="wide")

    # 추천 데이터 로드 (탭 밖에서 한 번만)
    df, updated_at, loaded_country_code = load_account_recommendations(account_id)
    country_code = loaded_country_code or country_code

    # --- 탭: 테이블만 다르게 ---
    tab_holdings, tab_management = st.tabs(["보유종목", "종목관리"])

    with tab_holdings:
        if df is None:
            st.error(
                updated_at
                or "추천 데이터를 불러오지 못했습니다. 먼저 `python recommend.py <account>` 명령으로 스냅샷을 생성해 주세요."
            )
        else:
            render_recommendation_table(df, country_code=country_code)

    with tab_management:
        _render_stocks_meta_table(account_id)

    # --- 공통: 업데이트 시간, 설정, 푸터 ---
    if updated_at:
        if "," in updated_at:
            parts = updated_at.split(",", 1)
            date_part = parts[0].strip()
            user_part = parts[1].strip()
            updated_at_rel = format_relative_time(date_part)
            updated_at_display = f"{date_part}{updated_at_rel}, {user_part}"
        else:
            updated_at_rel = format_relative_time(updated_at)
            updated_at_display = f"{updated_at}{updated_at_rel}"

        if country_code in ("kor", "kr"):
            from datetime import datetime

            now = datetime.now()
            now_str = now.strftime("%Y-%m-%d %H:%M:%S")
            now_rel = format_relative_time(now)

            st.caption(f"추천 데이터 업데이트: {updated_at_display}  \n가격 데이터 업데이트: {now_str}{now_rel}, Naver")
        else:
            st.caption(f"데이터 업데이트: {updated_at_display}")

        with st.expander("설정", expanded=True):
            strategy_cfg = account_settings.get("strategy", {}) or {}
            cagr = None
            mdd = None
            backtested_date = None
            strategy_tuning: dict[str, Any] = {}
            if isinstance(strategy_cfg, dict):
                cagr = strategy_cfg.get("CAGR")
                mdd = strategy_cfg.get("MDD")
                backtested_date = strategy_cfg.get("BACKTESTED_DATE")
                strategy_tuning = resolve_strategy_params(strategy_cfg)

            if strategy_tuning:
                params_to_show = {
                    "MA": strategy_tuning.get("MA_PERIOD"),
                    "MA타입": strategy_tuning.get("MA_TYPE"),
                    "TopN": strategy_tuning.get("PORTFOLIO_TOPN"),
                    "교체점수": strategy_tuning.get("REPLACE_SCORE_THRESHOLD"),
                    "과매수 지표": strategy_tuning.get("OVERBOUGHT_SELL_THRESHOLD"),
                    "쿨다운 일자": strategy_tuning.get("COOLDOWN_DAYS"),
                }
                param_strs = [f"{key}: {value}" for key, value in params_to_show.items() if value is not None]
            else:
                param_strs = []

            caption_parts: list[str] = []
            if param_strs:
                param_display = ", ".join(param_strs)
                caption_parts.append(f"설정: [{param_display}]")
            else:
                caption_parts.append("설정: N/A")

            # 슬리피지 정보 추가
            from config import BACKTEST_SLIPPAGE

            slippage_config = BACKTEST_SLIPPAGE.get(country_code, {})
            buy_slip = slippage_config.get("buy_pct")
            sell_slip = slippage_config.get("sell_pct")
            if buy_slip is not None and sell_slip is not None:
                if buy_slip == sell_slip:
                    caption_parts.append(f"슬리피지: ±{buy_slip}%")
                else:
                    caption_parts.append(f"슬리피지: 매수+{buy_slip}%/매도-{sell_slip}%")

            try:
                from logic.backtest import get_hold_states

                hold_states = get_hold_states() | {"BUY", "BUY_REPLACE"}
                if df is not None:
                    current_holdings = int(df[df["상태"].isin(hold_states)].shape[0])
                    target_topn = strategy_tuning.get("PORTFOLIO_TOPN") if isinstance(strategy_tuning, dict) else None
                    if target_topn:
                        caption_parts.append(f"보유종목 수 {current_holdings}/{target_topn}")
            except Exception:
                pass

            # 성과 지표 (CAGR, MDD) 및 백테스트 일자 추가
            if cagr is not None:
                caption_parts.append(f"**CAGR: {float(cagr):.2f}%**")
            if mdd is not None:
                caption_parts.append(f"**MDD: {float(mdd):.2f}%**")
            if backtested_date:
                caption_parts.append(f"**백테스트: {backtested_date}**")

            caption_text = ", ".join(caption_parts)
            if caption_text:
                st.caption(caption_text)
            else:
                st.caption("설정 정보를 찾을 수 없습니다.")
    else:
        st.caption("데이터를 찾을 수 없습니다.")


__all__ = ["render_account_page"]
