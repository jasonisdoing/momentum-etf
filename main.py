from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from utils.recommendations import get_recommendations_dataframe
from utils.settings_loader import get_country_settings
from utils.tuning_config import DEFAULT_ROW_COLORS, TUNING_CONFIG
from logic.strategies.maps.constants import DECISION_CONFIG


DATA_DIR = Path(__file__).resolve().parent / "data" / "results"


def load_country_recommendations(country: str) -> tuple[pd.DataFrame | None, str | None]:
    country_norm = (country or "").strip().lower()
    if not country_norm:
        return None, "국가 코드가 필요합니다."

    file_path = DATA_DIR / f"{country_norm}.json"
    if not file_path.exists():
        return None, f"데이터 파일을 찾을 수 없습니다: {file_path}"

    try:
        df = get_recommendations_dataframe(country_norm)
    except Exception as exc:  # pragma: no cover - Streamlit용 오류 메시지
        return None, f"추천 데이터를 불러오지 못했습니다: {exc}"

    updated_at = datetime.fromtimestamp(file_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    return df, updated_at


TABLE_VISIBLE_ROWS = 21  # header 1줄 + 내용 20줄
TABLE_ROW_HEIGHT = 32
TABLE_HEIGHT = TABLE_VISIBLE_ROWS * TABLE_ROW_HEIGHT


def _load_country_ui_settings(country: str) -> tuple[str, str]:
    try:
        settings = get_country_settings(country)
        name = settings.get("name") or country.upper()
        icon = settings.get("icon") or ""
    except Exception:
        name = country.upper()
        icon = ""
    return name, icon


def _resolve_row_colors(country: str) -> dict[str, str]:
    country = (country or "").strip().lower()
    # 기본값: DECISION_CONFIG의 background를 기반으로 구성
    base_colors = {
        key.upper(): cfg.get("background")
        for key, cfg in DECISION_CONFIG.items()
        if isinstance(cfg, dict) and cfg.get("background")
    }

    # 튜닝 설정에서 오버라이드할 색상 정보 로드 (없으면 DEFAULT_ROW_COLORS 사용)
    config = TUNING_CONFIG.get(country) or {}
    override_colors = config.get("ROW_COLORS") or DEFAULT_ROW_COLORS

    # copy to avoid mutating shared defaults & 병합
    merged = {str(k).upper(): str(v) for k, v in base_colors.items() if v}
    for key, value in (override_colors or {}).items():
        if value:
            merged[str(key).upper()] = str(value)

    return merged


def _style_rows_by_state(df: pd.DataFrame, *, country: str) -> pd.io.formats.style.Styler:
    row_colors = _resolve_row_colors(country)

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

    if "일간(%)" in df.columns:
        styled = styled.map(_color_daily_pct, subset=pd.IndexSlice[:, "일간(%)"])

    return styled


def render_recommendation_table(df: pd.DataFrame, *, country: str) -> None:
    styled_df = _style_rows_by_state(df, country=country)

    st.dataframe(
        styled_df,
        hide_index=True,
        width="stretch",
        height=TABLE_HEIGHT,
        column_config={
            "#": st.column_config.TextColumn("#", width="small"),
            "티커": st.column_config.TextColumn("티커", width="small"),  # medium -> small
            "종목명": st.column_config.TextColumn("종목명", width="medium"),  # large -> medium
            "카테고리": st.column_config.TextColumn("카테고리", width="small"),  # medium -> small
            "상태": st.column_config.TextColumn("상태", width="small"),
            "보유일": st.column_config.TextColumn("보유일", width="small"),
            "현재가": st.column_config.TextColumn("현재가", width="small"),
            "일간(%)": st.column_config.NumberColumn("일간(%)", width="small"),  # 이미 오른쪽 정렬
            "점수": st.column_config.NumberColumn("점수", width="small"),  # Text -> Number (오른쪽 정렬)
            "문구": st.column_config.TextColumn("문구", width="large"),  # 기본값 -> large
        },
    )


def main():
    page_title, page_icon = _load_country_ui_settings("kor")
    if not page_icon:
        page_icon = "🇰🇷"
    if not page_title:
        page_title = "한국"

    st.set_page_config(
        page_title=page_title,
        page_icon=page_icon,
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.title(f"{page_icon} {page_title}")
    st.caption("내부적일 알고리즘에 의해서 10종목을 보유할 수 있게 추천합니다.")

    df, updated_at = load_country_recommendations("kor")

    if df is None:
        st.error(updated_at or "데이터를 불러오지 못했습니다.")
        return

    if updated_at:
        st.caption(f"데이터 업데이트: {updated_at}")

    render_recommendation_table(df, country="kor")

    st.markdown(
        """
        <style>
            .stDataFrame thead tr th {
                text-align: center;
            }
            .stDataFrame tbody tr td {
                text-align: center;
                white-space: nowrap;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "load_country_recommendations",
    "render_recommendation_table",
    "_resolve_row_colors",
]
