from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

from strategies.maps.constants import DECISION_CONFIG
from utils.logger import get_app_logger
from utils.recommendation_storage import fetch_latest_recommendations
from utils.recommendations import recommendations_to_dataframe
from utils.settings_loader import get_account_settings

logger = get_app_logger()


@st.cache_data(ttl=30, show_spinner=False)
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

    # [KOR] 실시간 데이터 오버레이 (NAVER API)
    if country_code in ("kor", "kr"):
        try:
            from utils.data_loader import fetch_naver_etf_inav_snapshot

            tickers = [r.get("ticker") for r in rows if r.get("ticker")]
            realtime_data = fetch_naver_etf_inav_snapshot(tickers)

            if realtime_data:
                for row in rows:
                    ticker = str(row.get("ticker") or "").strip().upper()
                    if ticker in realtime_data:
                        rt = realtime_data[ticker]
                        # 1. 현재가
                        if "nowVal" in rt:
                            row["price"] = rt["nowVal"]
                        # 2. 일간 등락률
                        if "changeRate" in rt:
                            row["daily_pct"] = rt["changeRate"]
                        # 3. NAV
                        if "nav" in rt:
                            row["nav_price"] = rt["nav"]
                        # 4. 괴리율
                        if "deviation" in rt:
                            row["price_deviation"] = rt["deviation"]
                        # 5. 종목명 (선택)
                        if "itemname" in rt:
                            new_name = rt["itemname"]
                            stock_note = row.get("stock_note")
                            if stock_note:
                                new_name = f"{new_name}({stock_note})"
                            row["name"] = new_name
                        # 6. 3개월 수익률 (선택)
                        if "threeMonthEarnRate" in rt:
                            row["return_3m"] = rt["threeMonthEarnRate"]
        except Exception as e:
            logger.warning(f"실시간 데이터 오버레이 실패: {e}")

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
    "버킷",
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
    # 전형적인 퍼센트 컬럼들
    pct_columns = ["일간(%)", "평가(%)", "1주(%)", "1달(%)", "3달(%)", "6달(%)", "12달(%)", "고점대비"]
    for col in pct_columns:
        if col in df.columns:
            styled = styled.map(_color_daily_pct, subset=[col])

    # 괴리율 별도 적용 (볼드 포함)
    if "괴리율" in df.columns:
        styled = styled.map(_deviation_style, subset=["괴리율"])

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

    return styled


def render_recommendation_table(
    df: pd.DataFrame,
    country_code: str | None = None,
    visible_columns: list[str] | None = None,
    grouped_by_bucket: bool = True,
) -> None:
    # 스타일링 준비 (전체 DF 기준)
    # 하지만 여기서는 버킷별로 쪼개서 보여줘야 하므로, 쪼갠 뒤 각각 스타일링 적용 필요
    # 기존 _style_rows_by_state는 DF 전체를 스타일링함.

    price_label = "현재가"
    country_lower = (country_code or "").strip().lower()
    show_deviation = country_lower in {"kr", "kor"}

    # 공통 컬럼 설정
    column_config_map: dict[str, st.column_config.BaseColumn] = {
        "계좌": st.column_config.TextColumn("계좌", width=100),
        "버킷": st.column_config.TextColumn("버킷", width=85),
        "티커": st.column_config.TextColumn("티커", width=60),
        "종목명": st.column_config.TextColumn("종목명", width=250),
        "일간(%)": st.column_config.NumberColumn("일간(%)", width="small", format="%.2f%%"),
        "평가(%)": st.column_config.NumberColumn("평가(%)", width="small", format="%.2f%%"),
        price_label: st.column_config.NumberColumn(price_label, width="small"),
        "상태": st.column_config.TextColumn("상태", width=100),
        "보유일": st.column_config.NumberColumn("보유일", width=50),
        "1주(%)": st.column_config.NumberColumn("1주(%)", width="small", format="%.2f%%"),
        # "2주(%)": st.column_config.NumberColumn("2주(%)", width="small", format="%.2f%%"),
        "1달(%)": st.column_config.NumberColumn("1달(%)", width="small", format="%.2f%%"),
        "3달(%)": st.column_config.NumberColumn("3달(%)", width="small", format="%.2f%%"),
        "6달(%)": st.column_config.NumberColumn("6달(%)", width="small", format="%.2f%%"),
        "12달(%)": st.column_config.NumberColumn("12달(%)", width="small", format="%.2f%%"),
        "고점대비": st.column_config.NumberColumn("고점대비", width="small", format="%.2f%%"),
        "추세(3달)": st.column_config.LineChartColumn("추세(3달)", width="small"),
        "점수": st.column_config.NumberColumn("점수", width=50, format="%.1f"),
        "RSI": st.column_config.NumberColumn("RSI", width=50, format="%.1f"),
        "지속": st.column_config.NumberColumn("지속", width=50),
        "문구": st.column_config.TextColumn("문구", width="large"),
    }
    if show_deviation and "괴리율" in df.columns:
        column_config_map["괴리율"] = st.column_config.NumberColumn("괴리율", width="small", format="%.2f%%")

    # [Segmentation] 1~5 버킷 순회
    # 버킷 매핑 (app_pages.account_page와 동일하게 유지하거나, 여기서 정의)
    bucket_names = {
        1: "1. 모멘텀",
        2: "2. 혁신기술",
        3: "3. 시장지수",
        4: "4. 배당방어",
        5: "5. 대체헷지",
    }

    # DataFrame에 'bucket' 컬럼이 없으면 전체를 하나로 표시 (하위 호환)
    if "bucket" not in df.columns:
        if grouped_by_bucket:
            st.warning("버킷 정보가 없습니다. 전체 목록을 표시합니다.")
        _render_single_table(df, country_code, column_config_map, visible_columns)
        return

    # [Aggregation] 전체를 하나로 표시 (로그와 포맷 일원화)
    if not grouped_by_bucket:
        # recommend.py에서 이미 rank_order로 정렬되어 있으므로 그대로 사용하되,
        # 혹시 모르니 보장함 (단, rank_order가 없는 구시작 데이터 고려)
        df_sorted = df.copy()

        # [User Request] # 컬럼과 버킷 컬럼을 모두 사용 (추천 로그와 동일하게)
        # utils/recommendations.py에서 이미 #과 버킷 컬럼이 생성됨

        # 컬럼 구성 설정
        if visible_columns is None:
            visible_columns = list(df_sorted.columns)
            # '# '와 '버킷'을 맨 앞으로
            for col in reversed(["#", "버킷"]):
                if col in visible_columns:
                    visible_columns.insert(0, visible_columns.pop(visible_columns.index(col)))

        # 'bucket' (숫자) 제외
        final_columns = [c for c in visible_columns if c != "bucket"]

        _render_single_table(df_sorted, country_code, column_config_map, final_columns)
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
        _render_single_table(sub_df, country_code, column_config_map, visible_columns)
        st.write("")  # 간격

    if not has_data:
        st.info("표시할 추천 종목이 없습니다.")


def _render_single_table(
    df: pd.DataFrame,
    country_code: str,
    column_config_map: dict,
    visible_columns: list[str] | None = None,
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
    column_order = [col for col in columns if col in column_config_map]
    # config map에 없는 컬럼도 표시하고 싶다면 추가해야 함.
    # 여기서는 config map에 있는 것만 우선 표시

    # 나머지 컬럼도 추가 (config 정의 안된 것들)
    for col in columns:
        if col not in column_order:
            column_order.append(col)

    selected_column_config = {key: column_config_map[key] for key in column_order if key in column_config_map}

    # 높이 자동 조절
    # 행 개수에 따라 조절하되, 최대 높이 제한
    row_count = len(df.index)
    # 헤더(1) + 행(N) : 높이 계산
    # st.dataframe의 height는 픽셀 단위.
    # 대략 row당 35~40px
    calc_height = (row_count + 1) * 35 + 10
    if calc_height > 600:
        calc_height = 600
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
    "load_account_recommendations",
    "render_recommendation_table",
    "format_relative_time",
    "_resolve_row_colors",
]
