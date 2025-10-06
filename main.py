from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from utils.logger import get_app_logger
from utils.recommendations import get_recommendations_dataframe
from utils.settings_loader import get_account_settings
from strategies.maps.constants import DECISION_CONFIG


DATA_DIR = Path(__file__).resolve().parent / "data" / "results"
logger = get_app_logger()


def _regenerate_recommendations(account_id: str, country_code: str) -> tuple[bool, str | None]:
    account_norm = (account_id or "").strip().lower()
    country_norm = (country_code or "").strip().lower()

    if not account_norm:
        return False, "계정 ID가 필요합니다."

    try:
        from logic.recommend.pipeline import (
            RecommendationReport,
            generate_account_recommendation_report,
        )
    except Exception as exc:  # pragma: no cover - Streamlit 오류 메시지 전용
        logger.error(
            "추천 파이프라인 모듈을 불러오지 못했습니다 (account=%s, country=%s): %s",
            account_norm,
            country_norm,
            exc,
            exc_info=True,
        )
        return False, f"추천 데이터를 생성하기 위한 모듈을 불러오지 못했습니다: {exc}"

    try:
        report = generate_account_recommendation_report(account_id=account_norm, date_str=None)
    except Exception as exc:  # pragma: no cover - Streamlit 오류 메시지 전용
        logger.error(
            "추천 데이터를 생성하는 중 오류가 발생했습니다 (account=%s, country=%s): %s",
            account_norm,
            country_norm,
            exc,
            exc_info=True,
        )
        return False, f"추천 데이터를 생성하는 중 오류가 발생했습니다: {exc}"

    if not isinstance(report, RecommendationReport):
        logger.error(
            "추천 결과 타입이 예상과 다릅니다 (account=%s, type=%s)",
            account_norm,
            type(report).__name__,
        )
        return False, "추천 데이터 생성 결과의 형식이 올바르지 않습니다."

    recommendations = getattr(report, "recommendations", None)
    if recommendations is None:
        logger.error(
            "추천 결과에 recommendations 항목이 없습니다 (account=%s)",
            account_norm,
        )
        return False, "추천 데이터 생성 결과에 추천 목록이 포함되어 있지 않습니다."

    try:
        from utils.recommendation_storage import save_recommendation_report

        save_recommendation_report(report)
    except Exception as exc:  # pragma: no cover - Streamlit 오류 메시지 전용
        logger.error(
            "추천 데이터를 저장하는 중 오류가 발생했습니다 (account=%s, country=%s): %s",
            account_norm,
            country_norm,
            exc,
            exc_info=True,
        )
        return False, f"추천 데이터를 저장하는 중 오류가 발생했습니다: {exc}"

    if not recommendations:
        logger.warning(
            "추천 데이터가 비어 있지만 빈 파일을 생성했습니다 (account=%s, country=%s)",
            account_norm,
            country_norm,
        )

    logger.info(
        "추천 데이터를 자동으로 생성했습니다 (account=%s, country=%s)",
        account_norm,
        country_norm,
    )
    return True, None


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

    account_file = DATA_DIR / f"recommendation_{account_norm}.json"
    country_file = DATA_DIR / f"recommendation_{country_code}.json"

    file_path: Path | None = None
    source_key = country_code

    if account_file.exists():
        file_path = account_file
        source_key = account_norm
    elif country_file.exists():
        file_path = country_file

    if file_path is None:
        logger.warning(
            "추천 파일이 존재하지 않아 자동 생성을 시도합니다 (account=%s, country=%s). 확인한 경로: %s / %s | 현재 작업 디렉터리: %s",
            account_norm,
            country_code,
            account_file,
            country_file,
            Path.cwd(),
        )
        success, error_message = _regenerate_recommendations(account_norm, country_code)
        if not success:
            message = error_message or f"데이터 파일을 찾을 수 없습니다: {account_file}"
            return None, message, country_code

        if account_file.exists():
            file_path = account_file
            source_key = account_norm
        elif country_file.exists():
            file_path = country_file

    if file_path is None or not file_path.exists():
        target_path = file_path or account_file
        return None, f"데이터 파일을 찾을 수 없습니다: {target_path}", country_code

    try:
        df = get_recommendations_dataframe(country_code, source_key=source_key)
    except Exception as exc:  # pragma: no cover - Streamlit 오류 메시지 전용
        return None, f"추천 데이터를 불러오지 못했습니다: {exc}", country_code

    updated_at = datetime.fromtimestamp(file_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    return df, updated_at, country_code


TABLE_VISIBLE_ROWS = 16  # 헤더 1줄 + 내용 15줄
TABLE_ROW_HEIGHT = 33
TABLE_HEIGHT = TABLE_VISIBLE_ROWS * TABLE_ROW_HEIGHT


def _load_account_ui_settings(account_id: str) -> tuple[str, str]:
    try:
        settings = get_account_settings(account_id)
        name = settings.get("name") or account_id.upper()
        icon = settings.get("icon") or ""
    except Exception:
        name = account_id.upper()
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

    pct_columns = ["일간(%)", "평가(%)", "1주(%)", "2주(%)", "3주(%)"]
    for col in pct_columns:
        if col in df.columns:
            styled = styled.map(_color_daily_pct, subset=pd.IndexSlice[:, col])

    return styled


def render_recommendation_table(df: pd.DataFrame, *, account_id: str, country_code: str) -> None:
    styled_df = _style_rows_by_state(df, country_code=country_code)

    st.dataframe(
        styled_df,
        hide_index=True,
        width="stretch",
        height=TABLE_HEIGHT,
        column_config={
            "#": st.column_config.TextColumn("#", width="small"),
            "티커": st.column_config.TextColumn("티커", width="small"),  # medium 크기 설정을 small로 조정
            "종목명": st.column_config.TextColumn("종목명", width="medium"),  # large 크기 설정을 medium으로 조정
            "카테고리": st.column_config.TextColumn("카테고리", width="small"),  # medium 크기 설정을 small로 조정
            "상태": st.column_config.TextColumn("상태", width="small"),
            "보유일": st.column_config.TextColumn("보유일", width="small"),
            "일간(%)": st.column_config.NumberColumn("일간(%)", width="small"),
            "평가(%)": st.column_config.NumberColumn("평가(%)", width="small"),
            "현재가": st.column_config.TextColumn("현재가", width="small"),
            "1주(%)": st.column_config.NumberColumn("1주(%)", width="small"),
            "2주(%)": st.column_config.NumberColumn("2주(%)", width="small"),
            "3주(%)": st.column_config.NumberColumn("3주(%)", width="small"),
            "점수": st.column_config.NumberColumn("점수", width="small"),  # 문자열 대신 수치형 컬럼으로 표시
            "문구": st.column_config.TextColumn("문구", width="large"),  # 기본값 -> large
        },
    )


def main():
    default_account = "kor"
    page_title, page_icon = _load_account_ui_settings(default_account)
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
    st.caption("내부 알고리즘 기반으로 계정 추천을 제공합니다.")

    df, updated_at, country_code = load_account_recommendations(default_account)

    if df is None:
        st.error(updated_at or "데이터를 불러오지 못했습니다.")
        return

    if updated_at:
        st.caption(f"데이터 업데이트: {updated_at}")

    render_recommendation_table(
        df,
        account_id=default_account,
        country_code=country_code or default_account,
    )

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
    "load_account_recommendations",
    "render_recommendation_table",
    "_resolve_row_colors",
]
