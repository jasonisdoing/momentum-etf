from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

from config import BUCKET_CONFIG, BUCKET_MAPPING, HIGH_POINT_GREEN_THRESHOLD
from utils.logger import get_app_logger

logger = get_app_logger()


class LoadingStatus:
    """화면 상단에 공통 로딩 안내를 표시합니다."""

    def __init__(self) -> None:
        self._placeholder = st.empty()

    def update(self, message: str) -> None:
        self._placeholder.info(f"⏳ 로딩 중... {message}")

    def clear(self) -> None:
        self._placeholder.empty()


def create_loading_status() -> LoadingStatus:
    """공통 로딩 안내 핸들러를 생성합니다."""
    return LoadingStatus()


def inject_global_css() -> None:
    """전역 CSS를 주입합니다. main()에서만 호출하세요."""
    st.markdown(
        """
        <style>
        /* padding-top은 상단 탭 네비게이션에서 일관되게 동작하는 값 */
        .block-container {
            padding-top: 2.5rem !important;
            padding-bottom: 0.5rem !important;
            padding-left: 1.0rem !important;
            padding-right: 1.0rem !important;
        }
        div[data-testid="stVerticalBlock"] {
            gap: 0.1rem !important;
        }
        .block-container h1,
        .block-container h2,
        .block-container h3 {
            margin-top: 0.5rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            margin-top: 0 !important;
        }
        section[data-testid="stSidebar"][aria-expanded="true"] {
            width: 12rem !important;
            min-width: 12rem !important;
        }
        section[data-testid="stSidebar"][aria-expanded="false"] {
            width: 0 !important;
            min-width: 0 !important;
        }
        section[data-testid="stSidebar"][aria-expanded="true"] > div {
            width: 12rem !important;
        }
        section[data-testid="stSidebar"][aria-expanded="false"] > div {
            width: 0 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


TABLE_VISIBLE_ROWS = 26  # 헤더 1줄 + 내용 15줄
TABLE_ROW_HEIGHT = 36
TABLE_HEIGHT = TABLE_VISIBLE_ROWS * TABLE_ROW_HEIGHT

DEFAULT_COMPACT_COLUMNS_BASE = [
    "버킷",
    "티커",
    "종목명",
    "상태",
    "일간(%)",
    "평가(%)",
    "현재가",
    "문구",
]


def format_relative_time(dt_input: datetime | pd.Timestamp | str | None) -> str:
    """
    Convert a datetime object (or string) to a relative time string (e.g. '(5분 전)').
    Returns empty string if input is invalid or None.
    """
    if not dt_input:
        return ""

    try:
        # 1. Parse/Normalize to datetime
        if isinstance(dt_input, str):
            # Handle "YYYY-MM-DD HH:MM:SS, User" format
            if "," in dt_input:
                dt_input = dt_input.split(",")[0].strip()

            # Try parsing typical formats if it's a string
            try:
                dt = pd.to_datetime(dt_input).to_pydatetime()
            except Exception:
                return ""
        elif isinstance(dt_input, pd.Timestamp):
            dt = dt_input.to_pydatetime()
        elif isinstance(dt_input, datetime):
            dt = dt_input
        else:
            return ""

        # 2. Handle Timezone
        # If dt has no timezone, assume it's already in the target timezone (e.g. KST relative to now)
        # or UTC. But here we usually deal with KST strings or aware datetimes.
        # Let's compare with a naive 'now' if dt is naive, or aware 'now' if dt is aware.

        now = datetime.now(dt.tzinfo)
        diff = now - dt

        seconds = diff.total_seconds()

        # Future check (should not happen usually but valid)
        if seconds < 0:
            return ""

        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)

        if days > 0:
            return f"({days}일 전)"
        elif hours > 0:
            return f"({hours}시간 전)"
        elif minutes > 0:
            return f"({minutes}분 전)"
        else:
            return "(방금 전)"

    except Exception:
        return ""


def _style_rows_by_state(df: pd.DataFrame, *, country_code: str) -> pd.io.formats.style.Styler:
    _ = country_code
    holding_color = "#d9ead3"
    non_positive_score_color = "#e0e0e0"

    def _color_row(row: pd.Series) -> list[str]:
        is_holding = str(row.get("보유", "")).strip() == "보유"
        color = holding_color if is_holding else None
        if color is None:
            score = row.get("추세")
            try:
                if score is not None and not pd.isna(score) and float(score) < 0:
                    color = non_positive_score_color
            except (TypeError, ValueError):
                pass
        if color:
            return [f"background-color: {color};"] * len(row)
        return [""] * len(row)

    def _style_bucket(val: Any) -> str:
        val_str = str(val or "")
        for b_id, cfg in BUCKET_CONFIG.items():
            if cfg["name"] in val_str:
                return f"background-color: {cfg['bg_color']}; color: {cfg['text_color']}; font-weight: bold; border-radius: 4px;"
        return ""

    styled = df.style.apply(_color_row, axis=1)

    if "버킷" in df.columns:
        styled = styled.map(_style_bucket, subset=["버킷"])

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

    def _deviation_style(val: Any) -> str:
        """괴리율 스타일링: 양수 빨강, 음수 파랑, ±2% 이상 볼드체."""
        if val is None:
            return ""
        try:
            if isinstance(val, str):
                cleaned = val.replace("%", "").replace(",", "").replace("원", "").replace("$", "").strip()
                num = float(cleaned)
            else:
                num = float(val)
        except (TypeError, ValueError):
            return ""

        if num == 0:
            return ""

        if num >= 2.0:
            return "color: red; font-weight: bold"
        if num <= -2.0:
            return "color: blue; font-weight: bold"

        return "color: black"

    # 전형적인 퍼센트 컬럼들
    pct_columns = [
        "일간(%)",
        "평가수익률(%)",
        "평가(%)",
        "수익률(%)",
        "1주(%)",
        "2주(%)",
        "1달(%)",
        "3달(%)",
        "6달(%)",
        "12달(%)",
    ]
    for col in pct_columns:
        if col in df.columns:
            styled = styled.map(_color_daily_pct, subset=[col])

    # 괴리율 별도 적용 (볼드 포함)
    if "괴리율" in df.columns:
        styled = styled.map(_deviation_style, subset=["괴리율"])

    def _style_trend(val: Any) -> str:
        """추세: 양수 녹색, 음수 빨간색, 항상 볼드."""
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return "font-weight: bold;"
        try:
            num = float(val)
        except (TypeError, ValueError):
            return "font-weight: bold;"
        if num > 0:
            return "color: green; font-weight: bold;"
        if num < 0:
            return "color: red; font-weight: bold;"
        return "font-weight: bold;"

    def _style_high_point(val: Any) -> str:
        """고점: 임계값 이상 녹색(고점 근처), 미만 빨간색(낙폭 큼), 항상 볼드."""
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return "font-weight: bold;"
        try:
            num = float(val)
        except (TypeError, ValueError):
            return "font-weight: bold;"
        if num >= HIGH_POINT_GREEN_THRESHOLD:
            return "color: green; font-weight: bold;"
        return "color: red; font-weight: bold;"

    if "추세" in df.columns:
        styled = styled.map(_style_trend, subset=["추세"])
    if "고점" in df.columns:
        styled = styled.map(_style_high_point, subset=["고점"])

    # 가격 컬럼 포맷팅
    def _safe_format(fmt: str):
        def _formatter(x):
            if pd.isna(x) or x is None:
                return "-"
            try:
                return fmt.format(x)
            except Exception:
                return str(x)

        return _formatter

    format_dict = {}
    price_label = "현재가"

    country_lower = (country_code or "").strip().lower()
    is_kr = country_lower in {"kr", "kor"}
    is_us = country_lower in {"us", "usa", "usd"}
    is_aus = country_lower in {"aus", "au", "aud"}

    if price_label in df.columns:
        if is_kr:
            format_dict[price_label] = _safe_format("{:,.0f}원")
        elif is_us:
            format_dict[price_label] = _safe_format("${:,.2f}")
        elif is_aus:
            format_dict[price_label] = _safe_format("A${:,.2f}")
        else:
            format_dict[price_label] = _safe_format("{:,.2f}")

    if "수량" in df.columns:
        format_dict["수량"] = _safe_format("{:,.0f}")

    if "Nav" in df.columns:
        if is_kr:
            format_dict["Nav"] = _safe_format("{:,.0f}원")
        elif is_us:
            format_dict["Nav"] = _safe_format("${:,.2f}")
        elif is_aus:
            format_dict["Nav"] = _safe_format("A${:,.2f}")
        else:
            format_dict["Nav"] = _safe_format("{:,.2f}")

    if format_dict:
        styled = styled.format(format_dict)

    # 계좌별 색상 구분 (옵션)
    if "계좌" in df.columns:

        def _style_account_col(val: Any) -> str:
            return "font-weight: bold; color: #555;"

        styled = styled.map(_style_account_col, subset=["계좌"])

    right_align_columns = [
        "평균 매입가",
        "현재가",
        "기준가",
        "전일거래량",
        "전일거래량(주)",
        "시가총액",
        "시가총액(억)",
        "매입금액",
        "평가금액",
        "평가손익",
        "매입금액(KRW)",
        "평가금액(KRW)",
        "평가손익(KRW)",
    ]
    existing_right_align_columns = [col for col in right_align_columns if col in df.columns]
    if existing_right_align_columns:
        styled = styled.map(lambda _: "text-align: right;", subset=existing_right_align_columns)

    return styled


def render_rank_table(
    df: pd.DataFrame,
    country_code: str | None = None,
    visible_columns: list[str] | None = None,
    grouped_by_bucket: bool = True,
    height: int | None = 750,
    column_config_overrides: dict[str, st.column_config.BaseColumn] | None = None,
) -> None:
    # 스타일링 준비 (전체 DF 기준)
    # 하지만 여기서는 버킷별로 쪼개서 보여줘야 하므로, 쪼갠 뒤 각각 스타일링 적용 필요
    # 기존 _style_rows_by_state는 DF 전체를 스타일링함.

    price_label = "현재가"
    country_lower = (country_code or "").strip().lower()
    show_deviation = country_lower in {"kr", "kor"} or "괴리율" in df.columns

    # 공통 컬럼 설정
    column_config_map: dict[str, st.column_config.BaseColumn] = {
        "#": st.column_config.TextColumn("#", width=60),
        "보유여부": st.column_config.TextColumn("보유여부", width=40),
        "계좌": st.column_config.TextColumn("계좌", width=100),
        "환종": st.column_config.TextColumn("환종", width=60),
        "타입": st.column_config.TextColumn("타입", width=120),
        "버킷": st.column_config.TextColumn("버킷", width=85),
        "비중": st.column_config.ProgressColumn(
            "비중",
            width="small",
            format="%.0f%%",
            min_value=0.0,
            max_value=100.0,
        ),
        "타겟비중": st.column_config.ProgressColumn(
            "타겟비중",
            width="small",
            format="%.0f%%",
            min_value=0.0,
            max_value=100.0,
        ),
        "티커": st.column_config.TextColumn("티커", width=60),
        "종목명": st.column_config.TextColumn("종목명", width=250),
        "상장일": st.column_config.TextColumn("상장일", width=95),
        "수량": st.column_config.NumberColumn("수량", width="small", format="localized"),
        "평균 매입가": st.column_config.NumberColumn("평균 매입가", width="small", format="%.2f"),
        "일간(%)": st.column_config.NumberColumn("일간(%)", width="small", format="%.2f%%"),
        "평가(%)": st.column_config.NumberColumn("평가(%)", width="small", format="%.2f%%"),
        "평가수익률(%)": st.column_config.NumberColumn("평가수익률(%)", width="small", format="%.2f%%"),
        "Nav": st.column_config.NumberColumn("Nav", width="small", format="%.0f"),
        "매입금액(KRW)": st.column_config.NumberColumn("매입금액(KRW)", width="small", format="localized"),
        "매입금액": st.column_config.TextColumn("매입금액", width="small"),
        "평가금액(KRW)": st.column_config.NumberColumn("평가금액(KRW)", width="small", format="localized"),
        "평가금액": st.column_config.TextColumn("평가금액", width="small"),
        "평가손익(KRW)": st.column_config.NumberColumn("평가손익(KRW)", width="small", format="localized"),
        "평가손익": st.column_config.TextColumn("평가손익", width="small"),
        "수익률(%)": st.column_config.NumberColumn("수익률(%)", width="small", format="%.2f%%"),
        price_label: st.column_config.NumberColumn(price_label, width="small"),
        "기준가": st.column_config.NumberColumn("기준가", width="small", format="%.0f"),
        "전일거래량": st.column_config.NumberColumn("전일거래량", width="medium", format="%.0f"),
        "전일거래량(주)": st.column_config.NumberColumn("전일거래량(주)", width="medium", format="%.0f"),
        "시가총액": st.column_config.NumberColumn("시가총액", width="medium", format="%.0f"),
        "시가총액(억)": st.column_config.NumberColumn("시가총액(억)", width="medium", format="%.0f"),
        "상태": st.column_config.TextColumn("상태", width=80),
        "보유일": st.column_config.TextColumn("보유일", width=60),
        "1주(%)": st.column_config.NumberColumn("1주(%)", width="small", format="%.2f%%"),
        "2주(%)": st.column_config.NumberColumn("2주(%)", width="small", format="%.2f%%"),
        "1달(%)": st.column_config.NumberColumn("1달(%)", width="small", format="%.2f%%"),
        "3달(%)": st.column_config.NumberColumn("3달(%)", width="small", format="%.2f%%"),
        "6달(%)": st.column_config.NumberColumn("6달(%)", width="small", format="%.2f%%"),
        "12달(%)": st.column_config.NumberColumn("12달(%)", width="small", format="%.2f%%"),
        "고점": st.column_config.NumberColumn("고점", width="small", format="%.2f%%"),
        "추세(3달)": st.column_config.LineChartColumn("추세(3달)", width="small"),
        "추세": st.column_config.NumberColumn("추세", width=70, format="%.1f"),
        "RSI": st.column_config.NumberColumn("RSI", width=50, format="%.1f"),
        "지속": st.column_config.NumberColumn("지속", width=50),
        "문구": st.column_config.TextColumn("문구", width="large"),
        "보유": st.column_config.TextColumn("보유", width=60),
    }
    if show_deviation and "괴리율" in df.columns:
        column_config_map["괴리율"] = st.column_config.NumberColumn("괴리율", width="small", format="%.2f%%")

    if column_config_overrides:
        column_config_map.update(column_config_overrides)

    if "평균 매입가" in df.columns and not pd.api.types.is_numeric_dtype(df["평균 매입가"]):
        column_config_map["평균 매입가"] = st.column_config.TextColumn("평균 매입가", width="small")
    if price_label in df.columns and not pd.api.types.is_numeric_dtype(df[price_label]):
        column_config_map[price_label] = st.column_config.TextColumn(price_label, width="small")

    # [Segmentation] 1~5 버킷 순회
    # 버킷 매핑 (app_pages.account_page와 동일하게 유지하거나, 여기서 정의)
    bucket_names = BUCKET_MAPPING

    # DataFrame에 'bucket' 컬럼이 없으면 전체를 하나로 표시 (하위 호환)
    if "bucket" not in df.columns:
        if grouped_by_bucket:
            st.warning("버킷 정보가 없습니다. 전체 목록을 표시합니다.")
        _render_single_table(df, country_code, column_config_map, visible_columns, height=height)
        return

    # [Aggregation] 전체를 하나로 표시 (로그와 포맷 일원화)
    if not grouped_by_bucket:
        # 화면에서 전달된 순서를 그대로 사용한다.
        df_sorted = df.copy()

        # 컬럼 구성 설정
        if visible_columns is None:
            visible_columns = list(df_sorted.columns)
            # '# '와 '버킷'을 맨 앞으로
            for col in reversed(["#", "버킷"]):
                if col in visible_columns:
                    visible_columns.insert(0, visible_columns.pop(visible_columns.index(col)))

        # 'bucket' (숫자) 제외
        final_columns = [c for c in visible_columns if c != "bucket"]

        _render_single_table(df_sorted, country_code, column_config_map, final_columns, height=height)
        return

    # 버킷별 렌더링
    has_data = False
    for bucket_idx in range(1, 6):
        # 해당 버킷 데이터 필터링
        sub_df = df[df["bucket"] == bucket_idx].copy()
        if sub_df.empty:
            continue

        has_data = True
        bucket_name = bucket_names.get(bucket_idx, f"Bucket {bucket_idx}")

        st.subheader(f"{bucket_name} ({len(sub_df)})")
        # 버킷 컬럼은 표시할 필요 없으므로 제외 (옵션)
        _render_single_table(sub_df, country_code, column_config_map, visible_columns, height=height)
        st.write("")  # 간격

    if not has_data:
        st.info("표시할 종목이 없습니다.")


def _render_single_table(
    df: pd.DataFrame,
    country_code: str,
    column_config_map: dict,
    visible_columns: list[str] | None = None,
    height: int | None = 750,
) -> None:
    """단일 테이블 렌더링 헬퍼"""
    styled_df = _style_rows_by_state(df, country_code=country_code)

    if visible_columns:
        columns = [col for col in visible_columns if col in df.columns]
        # visible_columns에 없는 컬럼은 제외
        # 단, 스타일링을 위해 필요한 데이터가 날아가지 않도록 주의 (style object는 df 참조함)
        # st.dataframe은 pandas Styler 객체를 받으면, columns 인자로 표시 컬럼 제어 가능?
        # -> st.dataframe(styled_df, column_order=...) 사용
        pass
    else:
        columns = list(df.columns)
        # bucket 등 내부용 컬럼 제외
        if "bucket" in columns:
            columns.remove("bucket")

    # column_order를 위해 filtered columns 준비
    column_order = []
    if "#" in columns:
        column_order.append("#")

    for col in columns:
        if col in column_config_map and col not in column_order:
            column_order.append(col)

    selected_column_config = {key: column_config_map[key] for key in column_order if key in column_config_map}

    # 높이 조절
    row_count = len(df.index)
    if height is None:
        # height가 None인 경우 전체 데이터를 보여주기 위해 높이 계산 (스크롤바 제거용)
        calc_height = (row_count + 1) * 35 + 10
    else:
        # height가 지정된 경우 해당 높이로 제한 (기본 750px)
        calc_height = min((row_count + 1) * 35 + 10, height)

    # 너무 작으면 보기 흉하므로 최소 높이 설정
    if calc_height < 150:
        calc_height = 150

    st.dataframe(
        styled_df,
        hide_index=True,
        width="stretch",
        height=calc_height,
        column_config=selected_column_config,
        column_order=column_order,
    )


__all__ = [
    "render_rank_table",
    "format_relative_time",
    "create_loading_status",
]
