from __future__ import annotations

import subprocess
from typing import Any

import pandas as pd
import streamlit as st

from config import (
    BUCKET_CONFIG,
    BUCKET_MAPPING,
    BUCKET_OPTIONS,
    BUCKET_REVERSE_MAPPING,
)
from services.reference_data_service import get_stock_reference_info
from utils.data_loader import fetch_ohlcv
from utils.rankings import build_account_rankings, get_account_rank_defaults, get_rank_months_max
from utils.settings_loader import AccountSettingsError, get_account_settings
from utils.stock_list_io import (
    add_stock,
    check_stock_status,
    get_deleted_etfs,
    get_etfs,
    hard_remove_stock,
    remove_stock,
    update_stock,
)
from utils.ui import (
    create_loading_status,
    format_relative_time,
    render_rank_table,
)

try:
    from streamlit import fragment as _streamlit_fragment
except ImportError:
    try:
        from streamlit import experimental_fragment as _streamlit_fragment
    except ImportError:
        _streamlit_fragment = None


def fragment(*args, **kwargs):
    """streamlit fragment 호환 래퍼."""
    if _streamlit_fragment is None:
        if args and callable(args[0]) and len(args) == 1 and not kwargs:
            return args[0]

        def _decorator(func):
            return func

        return _decorator

    return _streamlit_fragment(*args, **kwargs)


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


@st.cache_data(ttl=30, show_spinner=False)
def _load_cached_rankings(account_id: str, ma_type: str, ma_months: int) -> pd.DataFrame:
    """동일 조건의 순위 결과를 짧게 재사용한다."""
    return build_account_rankings(account_id, ma_type=ma_type, ma_months=ma_months)


_DISPLAY_COLUMNS = [
    "보유여부",
    "버킷",
    "티커",
    "종목명",
    "현재가",
    "일간(%)",
    "괴리율",
    "추세",
    "고점",
    "1주(%)",
    "2주(%)",
    "1달(%)",
    "3달(%)",
    "6달(%)",
    "12달(%)",
    "RSI",
    "지속",
]
_PCT_COLUMNS = ["괴리율", "일간(%)", "1주(%)", "2주(%)", "1달(%)", "3달(%)", "6달(%)", "12달(%)", "고점"]
_SCORE_COLUMNS = ["추세", "RSI"]


def _format_rank_df_for_display(df: pd.DataFrame, country_code: str) -> pd.DataFrame:
    cols = [c for c in _DISPLAY_COLUMNS if c in df.columns]
    out = df[cols].copy()

    price_fmt = (lambda v: f"{v:,.2f}") if country_code == "au" else (lambda v: f"{int(v):,}원")
    if "현재가" in out.columns:
        out["현재가"] = out["현재가"].apply(lambda v: price_fmt(v) if pd.notna(v) and v is not None else "-")

    def _fmt_pct(v):
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return "-"
        return f"+{v:.2f}%" if v > 0 else f"{v:.2f}%"

    for col in _PCT_COLUMNS:
        if col in out.columns:
            out[col] = out[col].apply(_fmt_pct)

    for col in _SCORE_COLUMNS:
        if col in out.columns:
            out[col] = out[col].apply(lambda v: f"{v:.1f}" if pd.notna(v) and v is not None else "-")

    if "지속" in out.columns:
        out["지속"] = out["지속"].apply(lambda v: f"{int(v)}일" if pd.notna(v) and v is not None else "-")

    return out


def _render_rank_table_text(account_id: str, df: pd.DataFrame, ma_type: str, ma_months: int) -> str | None:
    """순위 DataFrame을 텍스트 테이블로 렌더링합니다."""
    from utils.report import render_table_eaw
    from utils.settings_loader import get_account_settings

    try:
        country_code = str(get_account_settings(account_id).get("country_code") or "kor").strip().lower()
    except Exception:
        country_code = "kor"

    try:
        display_df = _format_rank_df_for_display(df, country_code)

        _right_align_cols = {
            "현재가",
            "괴리율",
            "추세",
            "일간(%)",
            "1주(%)",
            "2주(%)",
            "1달(%)",
            "3달(%)",
            "6달(%)",
            "12달(%)",
            "고점",
            "RSI",
            "지속",
        }
        headers = list(display_df.columns)
        rows = [[str(v) if v is not None else "-" for v in row] for row in display_df.itertuples(index=False)]
        aligns = ["right" if h in _right_align_cols else "left" for h in headers]

        table_lines = render_table_eaw(headers, rows, aligns)

        # 계좌명 조회
        try:
            from utils.account_registry import get_account_order

            order = int(get_account_order(account_id))
            base_name = get_account_settings(account_id).get("name") or account_id.upper()
            acc_name = f"{order}. {base_name}"
        except Exception:
            acc_name = account_id

        title = f"[{acc_name}] 순위 - {ma_type} {ma_months}개월"
        return title + "\n" + "\n".join(table_lines) + "\n"
    except Exception:
        return None


def _save_rank_results_locally(account_id: str, df: pd.DataFrame, ma_type: str, ma_months: int) -> None:
    """순위 결과를 zaccounts/{account_id}/results/ 에 저장합니다."""
    from utils.settings_loader import get_account_dir

    try:
        account_dir = get_account_dir(account_id)
    except Exception:
        return

    results_dir = account_dir / "results"
    results_dir.mkdir(exist_ok=True)

    date_str = pd.Timestamp.now(tz="Asia/Seoul").strftime("%Y-%m-%d")

    content = _render_rank_table_text(account_id, df, ma_type, ma_months)
    if content:
        (results_dir / f"rank_{date_str}.log").write_text(content, encoding="utf-8")


def cleanup_old_rank_logs(max_keep: int = 10) -> None:
    """각 계좌의 rank_*.log 파일을 최신 max_keep개만 남기고 삭제합니다."""
    from utils.account_registry import _load_account_configs_impl
    from utils.settings_loader import get_account_dir

    for account in _load_account_configs_impl():
        try:
            results_dir = get_account_dir(str(account["account_id"])) / "results"
        except Exception:
            continue

        if not results_dir.is_dir():
            continue

        log_files = sorted(results_dir.glob("rank_*.log"), reverse=True)
        for old_file in log_files[max_keep:]:
            old_file.unlink(missing_ok=True)


def _normalize_account_view_mode(view_mode: str | None) -> str:
    """계좌 화면 뷰 모드를 표준 라벨로 맞춥니다."""
    clean_view = str(view_mode or "").split(".")[-1].strip()
    if clean_view in {"순위", "종목 관리", "삭제된 종목"}:
        return clean_view
    return "순위"


def _normalize_code(value: Any, fallback: str) -> str:
    text = str(value or "").strip().lower()
    return text or fallback


def _format_rank_timestamp(value: Any, *, assume_utc: bool = False) -> str | None:
    if value is None:
        return None

    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        if assume_utc:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_localize("Asia/Seoul")
    ts = ts.tz_convert("Asia/Seoul").tz_localize(None)

    ampm = "오전" if ts.hour < 12 else "오후"
    hour12 = ts.hour % 12 or 12
    absolute_text = f"{ts.year}년 {ts.month}월 {ts.day}일 {ampm} {hour12}:{ts.minute:02d}분"
    relative_text = format_relative_time(ts)
    if relative_text:
        return f"{absolute_text} {relative_text}"
    return absolute_text


def _resolve_target_country_code(target_id: str) -> str:
    target_norm = (target_id or "").strip().lower()
    try:
        settings = get_account_settings(target_norm)
        code = str(settings.get("country_code") or "").strip().lower()
        if code:
            return code
    except Exception:
        pass
    return "kor"


def _build_stocks_meta_table(account_id: str) -> pd.DataFrame:
    """계좌 종목 메타정보를 DataFrame으로 반환."""
    etfs = get_etfs(account_id)
    if not etfs:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for idx, etf in enumerate(etfs, 1):
        bucket_val = etf.get("bucket", 1)
        bucket_str = BUCKET_MAPPING.get(bucket_val, "1. 모멘텀")

        row = {
            "#": idx,
            "버킷": bucket_str,
            "티커": etf.get("ticker", ""),
            "종목명": etf.get("name", ""),
            "추가일자": etf.get("added_date", "-"),
            "상장일": etf.get("listing_date", "-"),
            "주간거래량": etf.get("1_week_avg_volume"),
            "1주(%)": etf.get("1_week_earn_rate"),
            "2주(%)": etf.get("2_week_earn_rate"),
            "1달(%)": etf.get("1_month_earn_rate"),
            "3달(%)": etf.get("3_month_earn_rate"),
            "6달(%)": etf.get("6_month_earn_rate"),
            "12달(%)": etf.get("12_month_earn_rate"),
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    if not df.empty and "1주(%)" in df.columns:
        df = df.sort_values(by=["버킷", "1주(%)"], ascending=[True, False])
    return df


@fragment
def _render_stocks_meta_table(account_id: str) -> None:
    """종목관리 테이블 렌더링. 업데이트 중일 경우 readonly 모드로 전환하여 스피너 방지."""
    # 세션 스테이트 키
    readonly = False

    df = _build_stocks_meta_table(account_id)
    df_edit = df.copy()

    if df.empty:
        st.info("종목 데이터가 없습니다. 종목을 추가하거나 삭제된 종목을 복원하세요.")
    else:
        st.caption(f"총 {len(df)}개 종목 (Source: MongoDB)")

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

        # 사용자가 요청한 '명칭 있는 체크박스' 구현을 위해 불리언 컬럼 추가
        df_edit.insert(0, "수정/삭제", False)

        # 주간거래량 데이터 타입 보장 (숫자형)
        df_edit["주간거래량"] = pd.to_numeric(df_edit["주간거래량"], errors="coerce")

    def _style_bucket(val: Any) -> str:
        val_str = str(val or "")
        for b_id, cfg in BUCKET_CONFIG.items():
            if cfg["name"] in val_str:
                return f"background-color: {cfg['bg_color']}; color: {cfg['text_color']}; font-weight: bold; border-radius: 4px;"
        return ""

    pct_columns = ["1주(%)", "2주(%)", "1달(%)", "3달(%)", "6달(%)", "12달(%)"]
    styled = df_edit.style

    if not df_edit.empty:
        if "버킷" in df_edit.columns:
            styled = styled.map(_style_bucket, subset=["버킷"])

        for col in pct_columns:
            if col in df_edit.columns:
                styled = styled.map(_color_pct, subset=col)

    st.write("")  # 간격

    # --- 종목 편집 모달 ---
    @st.dialog("종목 편집", width="small")
    def open_edit_dialog(ticker: str, current_bucket_name: str, name: str):
        st.write(f"**{name}** ({ticker})")
        st.caption(f"현재 버킷: {current_bucket_name}")

        st.subheader("버킷 변경")
        new_bucket_name = st.selectbox(
            "버킷 변경", options=BUCKET_OPTIONS, index=BUCKET_OPTIONS.index(current_bucket_name)
        )

        if st.button("💾 변경사항 저장", type="primary", width="stretch"):
            new_bucket_int = BUCKET_REVERSE_MAPPING.get(new_bucket_name, 1)
            update_fields: dict[str, Any] = {"bucket": new_bucket_int}
            if update_stock(account_id, ticker, **update_fields):
                st.toast(f"✅ {ticker} 버킷 변경 완료")
                st.rerun()

        st.divider()
        st.subheader("🗑️ 종목 삭제")
        delete_reason = st.text_input(
            "삭제 사유 (선택)", placeholder="필요하면 삭제 이유를 입력하세요", key=f"edit_del_reason_{ticker}"
        )

        # type="secondary" 속성을 부여하여 CSS 선택자가 적용되도록 함
        if st.button("🗑️ 삭제 실행", type="secondary", width="stretch"):
            if remove_stock(account_id, ticker, reason=delete_reason.strip()):
                st.toast(f"✅ {ticker} 삭제 완료")
                st.rerun()

    # --- 상단 관리 버튼 영역 ---
    # [종목 추가 / 메타데이터 업데이트 / 가격 캐시 갱신] 버튼 배치 (저장 버튼 제거)
    c_mgr1, c_mgr2, c_mgr3 = st.columns([1, 1, 1])

    with c_mgr1:
        if st.button("➕ 종목 추가", key=f"btn_add_modal_{account_id}", disabled=readonly, width="stretch"):
            st.session_state[f"show_add_modal_{account_id}"] = True
            st.rerun()

    with c_mgr2:
        if st.button("메타데이터 업데이트", key=f"btn_meta_{account_id}", disabled=readonly, width="stretch"):
            st.session_state[f"show_add_modal_{account_id}"] = False
            try:
                subprocess.Popen(["python", "scripts/stock_meta_updater.py", account_id])
                st.success(f"✅ `{account_id}` 메타데이터 업데이트를 시작했습니다. (배경에서 처리가 완료됩니다)")
            except Exception as e:
                st.error(f"⚠️ 실행 시작 오류: {e}")

    with c_mgr3:
        if st.button("가격 캐시 갱신", key=f"btn_price_{account_id}", disabled=readonly, width="stretch"):
            st.session_state[f"show_add_modal_{account_id}"] = False
            try:
                subprocess.Popen(["python", "scripts/update_price_cache.py", account_id])
                st.success(f"✅ `{account_id}` 가격 캐시 갱신을 시작했습니다. (배경에서 처리가 완료됩니다)")
            except Exception as e:
                st.error(f"⚠️ 실행 시작 오류: {e}")

    st.write("")  # 간격

    # DataFrame 표시
    column_config = {
        "수정/삭제": st.column_config.CheckboxColumn("수정/삭제", width=50, help="클릭하여 수정 또는 삭제"),
        "버킷": st.column_config.SelectboxColumn(
            "버킷",
            width=85,
            options=BUCKET_OPTIONS,
            required=True,
        ),
        "티커": st.column_config.TextColumn("티커", width=55),
        "종목명": st.column_config.TextColumn("종목명", width=300),
        "추가일자": st.column_config.TextColumn("추가일자", width=90),
        "상장일": st.column_config.TextColumn("상장일", width=80),
        "주간거래량": st.column_config.NumberColumn("주간거래량", width=80, format="localized"),
        "1주(%)": st.column_config.NumberColumn("1주(%)", width="small", format="%.2f%%"),
        "2주(%)": st.column_config.NumberColumn("2주(%)", width="small", format="%.2f%%"),
        "1달(%)": st.column_config.NumberColumn("1달(%)", width="small", format="%.2f%%"),
        "3달(%)": st.column_config.NumberColumn("3달(%)", width="small", format="%.2f%%"),
        "6달(%)": st.column_config.NumberColumn("6달(%)", width="small", format="%.2f%%"),
        "12달(%)": st.column_config.NumberColumn("12달(%)", width="small", format="%.2f%%"),
    }

    column_order = [
        "수정/삭제",
        "버킷",
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
        "추가일자",
    ]

    existing_columns = [col for col in column_order if col in df_edit.columns]

    if readonly:
        # 갱신 중일 때는 static dataframe 사용 (스피너 방지)
        calc_height = min((len(df.index) + 1) * 35 + 10, 750)
        st.dataframe(
            styled,
            hide_index=True,
            width="stretch",
            height=calc_height,
            column_config=column_config,
            column_order=existing_columns,
        )
    else:
        # 데이터 에디터 출력 (체크박스 클릭 감지를 위해)
        editor_key = f"selection_{account_id}_editor"
        calc_height = min((len(df.index) + 1) * 35 + 10, 750)

        # 모든 컬럼을 비활성화하고 '수정/삭제'만 활성화
        disabled_cols = [col for col in df_edit.columns if col != "수정/삭제"]

        st.data_editor(
            styled,
            hide_index=True,
            width="stretch",
            height=calc_height,
            column_config=column_config,
            column_order=existing_columns,
            disabled=disabled_cols,
            key=editor_key,
        )

        # 변경 사항 감지 및 모달 오픈
        # st.data_editor의 'edited_rows'를 세션 스테이트에서 직접 확인
        editor_state = st.session_state.get(editor_key, {})
        edited_rows = editor_state.get("edited_rows", {})

        if edited_rows:
            # 첫 번째 변경 행만 처리 (단일 모달)
            for idx_str, changes in edited_rows.items():
                if changes.get("수정/삭제") is True:
                    idx = int(idx_str)
                    ticker = df_edit.iloc[idx]["티커"]
                    bucket_name = df_edit.iloc[idx]["버킷"]
                    name = df_edit.iloc[idx]["종목명"]

                    # 무한 리런이나 모달 닫기 후 잔상 방지
                    # To prevent the modal from popping up again when other buttons are clicked
                    if editor_key in st.session_state:
                        del st.session_state[editor_key]

                    open_edit_dialog(ticker, bucket_name, name)
                    st.session_state[f"show_add_modal_{account_id}"] = False
                    break

    # -----------------------------------------------------------------------
    # 삭제 실행 영역 (체크된 항목이 있을 때만 하단에 표시)
    # -----------------------------------------------------------------------

    # 종목 추가 다이얼로그
    @st.dialog("종목 추가")
    def open_add_dialog():
        # 검색 상태 관리를 위한 세션 스테이트 키
        ss_key_result = f"add_stock_result_{account_id}"

        # [Fix] Widget state modification error 방지: 렌더링 전 플래그 확인하여 초기화
        if st.session_state.get(f"should_clear_add_{account_id}"):
            # Note: 렌더링 루프 중 직접 수정 시 에러가 나서, 위젯 생성 전 세션 제거 혹은 값 변경 처리
            st.session_state[f"in_ticker_{account_id}"] = ""
            st.session_state[ss_key_result] = None
            st.session_state[f"should_clear_add_{account_id}"] = False

        # 국가 코드 조회 (검색용)
        country_code = _resolve_target_country_code(account_id)
        st.write(f"대상: **{account_id.upper()}** ({country_code.upper()})")

        # 국가별 플레이스홀더 설정
        if country_code == "kor":
            placeholder_text = "예: 005930"
        elif country_code == "au":
            placeholder_text = "예: VAS"
        else:
            placeholder_text = "예: Ticker"

        c_in, c_btn = st.columns([3, 1], vertical_alignment="bottom")
        with c_in:
            d_ticker = st.text_input(
                "티커 입력", placeholder=placeholder_text, max_chars=12, key=f"in_ticker_{account_id}"
            ).strip()
        with c_btn:
            do_search = st.button("🔍 조회", key=f"btn_search_{account_id}", width="stretch")

        if do_search:
            if not d_ticker:
                st.error("티커를 입력하세요.")
                st.session_state[ss_key_result] = None
            else:
                with st.spinner("정보 조회 중..."):
                    info = get_stock_reference_info(d_ticker, country_code)
                if info and info.get("name"):
                    st.session_state[ss_key_result] = info
                    # 재진입 시 정보 유지를 위해
                else:
                    st.error("종목을 찾을 수 없습니다.")
                    st.session_state[ss_key_result] = None

        # 조회 결과 표시 및 추가 버튼
        search_result = st.session_state.get(ss_key_result)
        if search_result:
            ticker_res = search_result["ticker"]
            st.success(f"✅ 종목명: **{search_result['name']}**")
            if search_result.get("listing_date"):
                st.caption(f"상장일: {search_result['listing_date']}")

            # 상태 확인
            status = check_stock_status(account_id, ticker_res)

            if status == "ACTIVE":
                st.warning(f"⚠️ 이미 '{account_id.upper()}' 계좌에 등록된 종목입니다.")
                # 이미 등록된 경우 추가 버튼 비활성화 (요청 사항: 워닝)
                st.button("➕ 추가하기", disabled=True, key=f"btn_confirm_add_{account_id}")

            else:
                if status == "DELETED":
                    st.info("🗑️ 이전에 삭제된 종목입니다. 추가 시 복구됩니다.")

                # 버킷 선택 필드 추가
                selected_bucket_name = st.selectbox(
                    "버킷 선택", options=BUCKET_OPTIONS, index=0, key=f"sb_bucket_add_{account_id}"
                )
                bucket_int = BUCKET_REVERSE_MAPPING.get(selected_bucket_name, 1)

                # 추가 버튼 (녹색 primary)
                if st.button("➕ 추가하기", type="primary", width="stretch", key=f"btn_confirm_add_{account_id}"):
                    extra_fields: dict[str, Any] = {
                        "listing_date": search_result.get("listing_date"),
                        "bucket": bucket_int,
                    }
                    success = add_stock(
                        account_id,
                        ticker_res,
                        search_result["name"],
                        **extra_fields,
                    )
                    if success:
                        msg = "복구되었습니다" if status == "DELETED" else "추가되었습니다"

                        # [Auto-Update] 추가된 종목에 대해 메타데이터 및 가격 데이터 즉시 갱신
                        with st.spinner(f"'{search_result['name']}' 데이터(메타/가격)를 갱신 중입니다..."):
                            try:
                                # 1. 메타데이터 업데이트 (상장일 등)
                                # search_result에 이미 name/listing_date가 있지만, 확실히 하기 위해 단일 업데이트 호출
                                # stock_list_io.add_stock에서 이미 파일에 썼으므로, 다시 로드해서 업데이트하거나
                                # 그냥 단일 딕셔너리 만들어서 업데이트 함수에 넘길 수도 있음.
                                # 여기서는 간단히 listing_date가 없으면 search_result 값을 쓰기도 함.

                                # 파일에 저장된 상태를 업데이트하기 위해,
                                # 전체 로드 -> 해당 종목 찾기 -> 업데이트 -> 저장 프로세스가 필요하나,
                                # update_single_stock_metadata 함수는 dict를 인자로 받아 갱신함.
                                # 따라서 파일 I/O를 직접 하거나, 전체 update를 돌리는게 나음.
                                # 하지만 전체 update는 느리므로 단일 종목만 처리하고 싶음.
                                # -> update_single_stock_metadata는 'dict'를 수정함. 저장은 안함.
                                # -> 따라서 add_stock 내부에서 이미 저장했으니, 여기서는 가격 데이터(fetch_ohlcv)만 메인으로 돌리는게 효율적.
                                #    상장일은 add_stock 할 때 이미 들어감.

                                # 가격 데이터 갱신 (force_refresh=True)
                                fetch_ohlcv(ticker_res, country=country_code, date_range=None, force_refresh=True)
                                st.toast(f"✅ {msg}: {search_result['name']} (데이터 갱신 완료)")
                            except Exception as e:
                                st.toast(f"⚠️ {msg}: {search_result['name']} (갱신 실패: {e})")

                        # [Fix] 상태 초기화: 즉시 수정하면 에러가 나므로 플래그 설정 후 리런
                        st.session_state[f"should_clear_add_{account_id}"] = True
                        st.rerun()  # 모달 유지를 위해 상단에서 다시 호출됨
                    else:
                        st.error("추가 실패 (시스템 오류)")

        # 모달 하단: 종료 버튼
        st.write("")
        st.divider()
        if st.button("닫기", key=f"btn_close_modal_internal_{account_id}", width="stretch"):
            st.session_state[f"show_add_modal_{account_id}"] = False
            st.rerun()

    # [Continuous Add] 모달 유지 로직: 플래그가 True면 강제로 모달 오픈
    if st.session_state.get(f"show_add_modal_{account_id}"):
        open_add_dialog()


def _render_rank_tab(
    account_id: str,
    country_code: str,
    *,
    selected_ma_type: str | None = None,
    selected_ma_months: int | None = None,
    loading=None,
) -> None:
    """계좌별 추세 순위 탭을 렌더링합니다."""

    default_ma_type, default_ma_months = get_account_rank_defaults(account_id)
    max_months = get_rank_months_max()
    effective_ma_type = str(selected_ma_type or default_ma_type).strip().upper()
    effective_ma_months = int(selected_ma_months or default_ma_months)
    effective_ma_months = min(max(effective_ma_months, 1), max_months)

    visible_columns = [
        "보유여부",
        "버킷",
        "티커",
        "종목명",
        "현재가",
        "일간(%)",
        "괴리율",
        "추세",
        "고점",
        "1주(%)",
        "2주(%)",
        "1달(%)",
        "3달(%)",
        "6달(%)",
        "12달(%)",
        "추세(3달)",
        "RSI",
        "지속",
    ]
    if loading is not None:
        loading.update(f"{account_id.upper()} 순위 캐시 확인")
    df = _load_cached_rankings(account_id, effective_ma_type, effective_ma_months)
    if bool(df.attrs.get("cache_blocked")):
        latest_trading_day_text = _format_rank_timestamp(df.attrs.get("latest_trading_day"))
        cache_text = _format_rank_timestamp(df.attrs.get("cache_updated_at"))
        missing_tickers = list(df.attrs.get("missing_tickers") or [])
        stale_tickers = list(df.attrs.get("stale_tickers") or [])

        st.error("기준 종가 캐시가 최신 거래일 기준으로 준비되지 않아 순위를 표시하지 않습니다.")
        if latest_trading_day_text:
            st.caption(f"최신 거래일 기준: {latest_trading_day_text}")
        if cache_text:
            st.caption(f"현재 기준 종가 캐시: {cache_text}")
        if missing_tickers:
            st.caption(f"캐시 없음: {len(missing_tickers)}개 티커")
        if stale_tickers:
            st.caption(f"오래된 캐시: {len(stale_tickers)}개 티커")
        st.caption("시스템 페이지에서 `모든 가격 캐시 업데이트`를 실행한 뒤, 완료 후 다시 이 페이지를 열어주세요.")
        st.markdown("[시스템 페이지로 이동](/system)")
        return

    if df.empty:
        st.info("표시할 순위 종목이 없습니다.")
        return

    _save_rank_results_locally(account_id, df, effective_ma_type, effective_ma_months)

    realtime_active = bool(df.attrs.get("realtime_active"))
    render_rank_table(
        df,
        country_code=country_code,
        grouped_by_bucket=False,
        visible_columns=visible_columns,
        height=900,
    )

    st.write("")
    status_icon = "🟢" if realtime_active else "🔴"
    status_text = "실시간 가격 반영 중" if realtime_active else "장마감 캐시 사용 중"
    st.caption(f"{status_icon} {status_text}")

    realtime_text = _format_rank_timestamp(df.attrs.get("realtime_fetched_at"))
    cache_text = _format_rank_timestamp(df.attrs.get("cache_updated_at"))
    ranking_text = _format_rank_timestamp(df.attrs.get("ranking_computed_at"))

    if realtime_text:
        st.caption(f"실시간 가격: {realtime_text}")
    if cache_text:
        st.caption(f"기준 종가 캐시: {cache_text}")
    if ranking_text:
        st.caption(f"순위 계산: {ranking_text}")

    st.caption("추세·지속은 기준 종가 시계열에 장중 실시간 가격을 반영해 계산합니다.")


# ---------------------------------------------------------------------------
# 메인 렌더 함수
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 삭제된 종목 관리 탭
# ---------------------------------------------------------------------------
@fragment
def _render_deleted_stocks_tab(account_id: str) -> None:
    """삭제된 종목 목록을 표시하고 복구/완전삭제 기능을 제공합니다."""
    deleted_etfs = get_deleted_etfs(account_id)
    if not deleted_etfs:
        st.info("삭제된 종목이 없습니다.")
        return

    st.subheader(f"🗑️ 삭제된 종목 ({len(deleted_etfs)}개)")

    deleted_rows = []
    for etf in deleted_etfs:
        deleted_at = etf.get("deleted_at")
        if deleted_at:
            try:
                deleted_at_str = deleted_at.strftime("%Y-%m-%d")
            except Exception:
                deleted_at_str = str(deleted_at)[:10]
        else:
            deleted_at_str = "-"

        bucket_val = etf.get("bucket", 1)
        bucket_str = BUCKET_MAPPING.get(bucket_val, "1. 모멘텀")

        deleted_rows.append(
            {
                "복구": False,
                "버킷": bucket_str,
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
                "삭제일": deleted_at_str,
                "삭제 사유": etf.get("deleted_reason", "-"),
            }
        )

    df_deleted = pd.DataFrame(deleted_rows)
    df_deleted.sort_values(by=["버킷", "삭제일"], ascending=[True, False], inplace=True)
    df_deleted["주간거래량"] = pd.to_numeric(df_deleted["주간거래량"], errors="coerce")

    # [User Request] 버튼을 테이블 위로 이동
    editor_base_key = f"deleted_editor_{account_id}"
    editor_nonce_key = f"{editor_base_key}_nonce"
    selected_tickers_key = f"{editor_base_key}_selected_tickers"
    editor_nonce = int(st.session_state.get(editor_nonce_key, 0) or 0)
    editor_key = f"{editor_base_key}_{editor_nonce}"
    editor_state = st.session_state.get(editor_key, {})

    selected_tickers = {
        str(ticker).strip().upper()
        for ticker in st.session_state.get(selected_tickers_key, [])
        if str(ticker or "").strip()
    }

    # [Fix] Sync selection state before rendering buttons
    if editor_state and editor_state.get("edited_rows"):
        changed = False
        for idx_str, changes in editor_state["edited_rows"].items():
            if "복구" in changes:
                idx = int(idx_str)
                if 0 <= idx < len(df_deleted):
                    ticker = str(df_deleted.iloc[idx]["티커"]).strip().upper()
                    if changes["복구"]:
                        if ticker not in selected_tickers:
                            selected_tickers.add(ticker)
                            changed = True
                    else:
                        if ticker in selected_tickers:
                            selected_tickers.remove(ticker)
                            changed = True
        if changed:
            st.session_state[selected_tickers_key] = list(selected_tickers)

    # 전체 선택/해제 컨트롤
    c_all_1, c_all_2, _ = st.columns([1, 1, 3])
    with c_all_1:
        if st.button("✅ 전체 선택", key=f"btn_deleted_select_all_{account_id}", width="stretch"):
            st.session_state[selected_tickers_key] = [
                str(t).strip().upper() for t in df_deleted["티커"].tolist() if str(t or "").strip()
            ]
            st.session_state[editor_nonce_key] = editor_nonce + 1
            st.rerun()
    with c_all_2:
        if st.button("↩️ 전체 해제", key=f"btn_deleted_clear_all_{account_id}", width="stretch"):
            st.session_state[selected_tickers_key] = []
            st.session_state[editor_nonce_key] = editor_nonce + 1
            st.rerun()

    to_restore_df = (
        df_deleted[df_deleted["티커"].astype(str).str.upper().isin(selected_tickers)].copy()
        if selected_tickers
        else pd.DataFrame()
    )

    if not to_restore_df.empty:
        c_res1, c_res2 = st.columns(2)
        with c_res1:
            if st.button("♻️ 선택 종목 복구", type="primary", key=f"btn_tab_restore_{account_id}", width="stretch"):
                restored = 0
                for _, row in to_restore_df.iterrows():
                    ticker = row["티커"]
                    bucket_name = row["버킷"]
                    bucket_int = BUCKET_REVERSE_MAPPING.get(bucket_name, 1)
                    if add_stock(account_id, ticker, bucket=bucket_int):
                        restored += 1
                if restored > 0:
                    st.session_state[selected_tickers_key] = []
                    st.session_state[editor_nonce_key] = editor_nonce + 1
                    st.success(f"{restored}개 종목 복구 완료!")
                    st.rerun()
        with c_res2:
            st.markdown(
                f"""
                <style>
                div.st-key-btn_tab_hard_del_{account_id} button {{
                    background-color: #d32f2f !important;
                    color: white !important;
                    border: none !important;
                }}
                div.st-key-btn_tab_hard_del_{account_id} button:hover {{
                    background-color: #b71c1c !important;
                    color: white !important;
                }}
                </style>
                """,
                unsafe_allow_html=True,
            )
            if st.button(
                "💀 선택 종목 완전 삭제",
                type="secondary",
                key=f"btn_tab_hard_del_{account_id}",
                width="stretch",
            ):
                deleted_count = 0
                for _, row in to_restore_df.iterrows():
                    ticker = row["티커"]
                    if hard_remove_stock(account_id, ticker):
                        deleted_count += 1
                if deleted_count > 0:
                    st.session_state[selected_tickers_key] = []
                    st.session_state[editor_nonce_key] = editor_nonce + 1
                    st.success(f"{deleted_count}개 종목 영구 삭제 완료!")
                    st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.caption("복구하거나 완전 삭제할 종목을 아래 테이블에서 선택하세요.")

    df_deleted_editor = df_deleted.copy()
    df_deleted_editor["복구"] = df_deleted_editor["티커"].astype(str).str.upper().isin(selected_tickers)

    def _style_deleted_row(row: pd.Series) -> list[str]:
        ticker = str(row.get("티커") or "").strip().upper()
        if ticker in selected_tickers:
            return ["background-color: #eceff1"] * len(row)
        return [""] * len(row)

    styled_deleted_editor = df_deleted_editor.style.apply(_style_deleted_row, axis=1)

    edited_deleted = st.data_editor(
        styled_deleted_editor,
        hide_index=True,
        width="stretch",
        column_config={
            "복구": st.column_config.CheckboxColumn("복구", width=20),
            "버킷": st.column_config.SelectboxColumn("버킷", width=50, options=BUCKET_OPTIONS),
            "티커": st.column_config.TextColumn("티커", width=50),
            "종목명": st.column_config.TextColumn("종목명", width=250),
            "상장일": st.column_config.TextColumn("상장일", width=70),
            "주간거래량": st.column_config.NumberColumn("주간거래량", width=50, format="localized"),
            "1주(%)": st.column_config.NumberColumn("1주(%)", width="small", format="%.2f%%"),
            "2주(%)": st.column_config.NumberColumn("2주(%)", width="small", format="%.2f%%"),
            "1달(%)": st.column_config.NumberColumn("1달(%)", width="small", format="%.2f%%"),
            "3달(%)": st.column_config.NumberColumn("3달(%)", width="small", format="%.2f%%"),
            "6달(%)": st.column_config.NumberColumn("6달(%)", width="small", format="%.2f%%"),
            "12달(%)": st.column_config.NumberColumn("12달(%)", width="small", format="%.2f%%"),
            "삭제일": st.column_config.TextColumn("삭제일", width=90),
            "삭제 사유": st.column_config.TextColumn("삭제 사유", width=300),
        },
        column_order=[
            "복구",
            "버킷",
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
            "삭제일",
            "삭제 사유",
        ],
        disabled=[
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
            "삭제일",
            "삭제 사유",
        ],
        key=editor_key,
    )

    if (
        isinstance(edited_deleted, pd.DataFrame)
        and "복구" in edited_deleted.columns
        and "티커" in edited_deleted.columns
    ):
        # selection_now = (
        #     edited_deleted.loc[edited_deleted["복구"] == True, "티커"]  # noqa: E712
        #     .astype(str)
        #     .str.strip()
        #     .str.upper()
        #     .tolist()
        # )
        # st.session_state[selected_tickers_key] = selection_now
        pass


def render_account_page(
    account_id: str,
    view_mode: str | None = None,
    loading=None,
    *,
    rank_params: dict[str, Any] | None = None,
) -> None:
    """주어진 계정 설정을 기반으로 계정 페이지를 렌더링합니다 (탭 포함)."""
    owns_loading = loading is None
    loading = loading or create_loading_status()

    try:
        loading.update(f"{account_id.upper()} 설정 조회")
        try:
            account_settings = get_account_settings(account_id)
        except AccountSettingsError as exc:
            st.error(f"설정을 불러오지 못했습니다: {exc}")
            st.stop()

        country_code = _normalize_code(account_settings.get("country_code"), account_id)

        normalized_view_mode = _normalize_account_view_mode(view_mode)

        if view_mode is None:
            view_mode = st.segmented_control(
                "뷰",
                ["순위", "종목 관리", "삭제된 종목"],
                default="순위",
                key=f"view_{account_id}",
                label_visibility="collapsed",
            )
            normalized_view_mode = _normalize_account_view_mode(view_mode)

        if normalized_view_mode == "종목 관리":
            loading.update(f"{account_id.upper()} 종목 관리 테이블 준비")
            _render_stocks_meta_table(account_id)
        elif normalized_view_mode == "삭제된 종목":
            loading.update(f"{account_id.upper()} 삭제 종목 테이블 준비")
            _render_deleted_stocks_tab(account_id)
        else:
            rank_params = rank_params or {}
            _render_rank_tab(
                account_id,
                country_code,
                selected_ma_type=rank_params.get("ma_type"),
                selected_ma_months=rank_params.get("ma_months"),
                loading=loading,
            )
    finally:
        if owns_loading:
            loading.clear()


def render_account_setup_page(account_id: str) -> None:
    """계좌/풀 공용 종목 관리 뷰를 렌더링한다."""
    _render_stocks_meta_table(account_id)


def render_account_deleted_page(account_id: str) -> None:
    """계좌/풀 공용 삭제된 종목 뷰를 렌더링한다."""
    _render_deleted_stocks_tab(account_id)


__all__ = [
    "render_account_page",
    "render_account_setup_page",
    "render_account_deleted_page",
]
