from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from strategies.maps.constants import DECISION_CONFIG
from utils.logger import get_app_logger
from utils.recommendation_storage import fetch_latest_recommendations
from utils.recommendations import recommendations_to_dataframe
from utils.settings_loader import get_account_settings

logger = get_app_logger()


def load_account_recommendations(
    account_id: str,
) -> tuple[pd.DataFrame | None, str | None, str]:
    account_norm = (account_id or "").strip().lower()
    if not account_norm:
        return None, "계정 ID가 필요합니다.", ""

    try:
        account_settings = get_account_settings(account_norm)
    except Exception as exc:  # pragma: no cover - Streamlit 오류 메시지 전용
        return None, f"계정 설정을 불러오지 못했습니다: {exc}", ""

    country_code = (account_settings.get("country_code") or account_norm).strip().lower()

    try:
        snapshot = fetch_latest_recommendations(account_norm)
    except Exception as exc:
        return None, f"추천 스냅샷을 불러오지 못했습니다: {exc}", country_code

    if snapshot is None:
        message = (
            "추천 스냅샷을 찾을 수 없습니다. CLI에서 "
            f"`python recommend.py {account_norm}` 명령으로 데이터를 생성해 주세요."
        )
        logger.warning("추천 스냅샷을 찾을 수 없습니다 (account=%s)", account_norm)
        return None, message, country_code

    rows = snapshot.get("recommendations") or []
    try:
        df = recommendations_to_dataframe(country_code, rows)
    except Exception as exc:
        return None, f"추천 데이터를 변환하는 중 오류가 발생했습니다: {exc}", country_code

    updated_dt = snapshot.get("updated_at") or snapshot.get("created_at")
    updated_by = snapshot.get("updated_by", "")

    if isinstance(updated_dt, datetime):
        ts = pd.Timestamp(updated_dt)
        if ts.tzinfo is None or ts.tzinfo.utcoffset(ts) is None:
            ts = ts.tz_localize("UTC").tz_convert("Asia/Seoul")
        else:
            ts = ts.tz_convert("Asia/Seoul")
        updated_at = ts.strftime("%Y-%m-%d %H:%M:%S")
    else:
        try:
            parsed = pd.to_datetime(updated_dt)
            if parsed.tzinfo is None or parsed.tzinfo.utcoffset(parsed) is None:
                parsed = parsed.tz_localize("UTC").tz_convert("Asia/Seoul")
            else:
                parsed = parsed.tz_convert("Asia/Seoul")
            updated_at = parsed.strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            updated_at = str(updated_dt) if updated_dt else None

    # updated_by 정보를 updated_at에 추가
    if updated_at and updated_by:
        updated_at = f"{updated_at}, {updated_by}"

    loaded_country_code = snapshot.get("country_code") or country_code
    return df, updated_at, str(loaded_country_code or country_code)


TABLE_VISIBLE_ROWS = 26  # 헤더 1줄 + 내용 15줄
TABLE_ROW_HEIGHT = 36
TABLE_HEIGHT = TABLE_VISIBLE_ROWS * TABLE_ROW_HEIGHT

DEFAULT_COMPACT_COLUMNS_BASE = [
    "#",
    "티커",
    "종목명",
    "상태",
    "일간(%)",
    "평가(%)",
    "현재가",
    "문구",
]


def _load_account_ui_settings(account_id: str) -> tuple[str, str]:
    try:
        settings = get_account_settings(account_id)
        name = "Momentum ETF"
        icon = settings.get("icon") or ""
    except Exception:
        name = "Momentum ETF"
        icon = ""
    return name, icon


def _resolve_row_colors(country_code: str) -> dict[str, str]:
    country_code = (country_code or "").strip().lower()
    # 기본값: DECISION_CONFIG의 background를 기반으로 구성
    base_colors = {
        key.upper(): cfg.get("background")
        for key, cfg in DECISION_CONFIG.items()
        if isinstance(cfg, dict) and cfg.get("background")
    }

    return base_colors


def _inject_responsive_styles() -> None:
    """Mobile-friendly padding and typography tweaks."""
    st.markdown(
        """
        <style>
            @media (max-width: 900px) {
                .block-container {
                    padding-top: 0.75rem;
                    padding-bottom: 2rem;
                    padding-left: 0.75rem;
                    padding-right: 0.75rem;
                }
                div[data-testid="stHorizontalBlock"] {
                    gap: 0.75rem !important;
                }
                .stDataFrame table {
                    font-size: 0.85rem;
                }
            }
            @media (max-width: 600px) {
                .block-container {
                    padding-top: 0.5rem;
                    padding-bottom: 1.5rem;
                    padding-left: 0.5rem;
                    padding-right: 0.5rem;
                }
                .stDataFrame table {
                    font-size: 0.78rem;
                }
                .stDataFrame tbody tr td {
                    white-space: normal;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _style_rows_by_state(df: pd.DataFrame, *, country_code: str) -> pd.io.formats.style.Styler:
    row_colors = _resolve_row_colors(country_code)

    def _color_row(row: pd.Series) -> list[str]:
        state = str(row.get("상태", "")).upper()
        color = row_colors.get(state)
        if color:
            return [f"background-color: {color}"] * len(row)
        return [""] * len(row)

    styled = df.style.apply(_color_row, axis=1)

    def _color_daily_pct(val: float | str) -> str:
        if val is None:
            return ""
        if isinstance(val, str):
            cleaned = val.replace("%", "").replace(",", "").strip()
            if not cleaned:
                return ""
            try:
                num = float(cleaned)
            except ValueError:
                return ""
        else:
            try:
                num = float(val)
            except (TypeError, ValueError):
                return ""
        if num > 0:
            return "color: red"
        if num < 0:
            return "color: blue"
        return "color: black"

    pct_columns = ["일간(%)", "평가(%)", "1주(%)", "2주(%)", "1달(%)", "3달(%)", "고점대비"]
    for col in pct_columns:
        if col in df.columns:
            styled = styled.map(_color_daily_pct, subset=pd.IndexSlice[:, col])

    # 가격 컬럼 포맷팅
    format_dict = {}
    price_label = "현재가"

    country_lower = (country_code or "").strip().lower()
    is_kr = country_lower in {"kr", "kor"}
    is_us = country_lower in {"us", "usa", "usd"}

    if price_label in df.columns:
        if is_kr:
            format_dict[price_label] = "{:,.0f}원"
        elif is_us:
            format_dict[price_label] = "${:,.2f}"
        else:
            format_dict[price_label] = "{:,.2f}"

    if "Nav" in df.columns:
        if is_kr:
            format_dict["Nav"] = "{:,.0f}원"
        elif is_us:
            format_dict["Nav"] = "${:,.2f}"
        else:
            format_dict["Nav"] = "{:,.2f}"

    if format_dict:
        styled = styled.format(format_dict)

    return styled


def render_recommendation_table(
    df: pd.DataFrame,
    *,
    country_code: str,
    visible_columns: list[str] | None = None,
) -> None:
    styled_df = _style_rows_by_state(df, country_code=country_code)

    price_label = "현재가"
    country_lower = (country_code or "").strip().lower()
    show_deviation = country_lower in {"kr", "kor"}

    column_config_map: dict[str, st.column_config.BaseColumn] = {
        "#": st.column_config.TextColumn("#", width=30),
        "티커": st.column_config.TextColumn("티커", width=60),
        "종목명": st.column_config.TextColumn("종목명", width="medium"),
        "카테고리": st.column_config.TextColumn("카테고리", width=100),
        "상태": st.column_config.TextColumn("상태", width=80),
        "보유일": st.column_config.NumberColumn("보유일", width=50),
        "일간(%)": st.column_config.NumberColumn("일간(%)", width="small", format="%.2f%%"),
        "평가(%)": st.column_config.NumberColumn("평가(%)", width="small", format="%.2f%%"),
        price_label: st.column_config.NumberColumn(price_label, width="small"),
        "1주(%)": st.column_config.NumberColumn("1주(%)", width="small", format="%.2f%%"),
        "2주(%)": st.column_config.NumberColumn("2주(%)", width="small", format="%.2f%%"),
        "1달(%)": st.column_config.NumberColumn("1달(%)", width="small", format="%.2f%%"),
        "3달(%)": st.column_config.NumberColumn("3달(%)", width="small", format="%.2f%%"),
        "고점대비": st.column_config.NumberColumn("고점대비", width="small", format="%.2f%%"),
        "추세(3달)": st.column_config.LineChartColumn("추세(3달)", width="small"),
        "점수": st.column_config.NumberColumn("점수", width=50, format="%.1f"),
        "RSI": st.column_config.NumberColumn("RSI", width=50, format="%.1f"),
        "지속": st.column_config.NumberColumn("지속", width=50),
        "문구": st.column_config.TextColumn("문구", width="large"),
    }
    if show_deviation and "괴리율" in df.columns:
        column_config_map["괴리율"] = st.column_config.NumberColumn("괴리율", width="small", format="%.2f%%")

    if visible_columns:
        columns = [col for col in visible_columns if col in df.columns]
        styled_df = styled_df.reindex(columns=columns)
    else:
        columns = list(df.columns)

    selected_column_config = {key: column_config_map[key] for key in columns if key in column_config_map}

    row_count = len(df.index) if len(df.index) < TABLE_VISIBLE_ROWS else TABLE_VISIBLE_ROWS - 1
    table_height = max(TABLE_ROW_HEIGHT * (row_count + 1), TABLE_ROW_HEIGHT * 6)

    st.dataframe(
        styled_df,
        hide_index=True,
        width="stretch",
        height=table_height,
        column_config=selected_column_config,
    )


__all__ = [
    "load_account_recommendations",
    "render_recommendation_table",
    "_resolve_row_colors",
]
