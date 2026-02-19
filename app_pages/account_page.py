from __future__ import annotations

import glob
import os
import subprocess
import time
from typing import Any

import pandas as pd
import streamlit as st

from config import (
    BUCKET_CONFIG,
    BUCKET_MAPPING,
    BUCKET_OPTIONS,
    BUCKET_REVERSE_MAPPING,
)
from scripts.update_price_cache import refresh_cache_for_target
from utils.account_registry import get_icon_fallback, load_account_configs
from utils.data_loader import fetch_ohlcv
from utils.settings_loader import AccountSettingsError, get_account_settings, resolve_strategy_params
from utils.stock_list_io import (
    add_stock,
    check_stock_status,
    get_deleted_etfs,
    get_etfs,
    remove_stock,
    update_stock,
)
from utils.stock_meta_updater import fetch_stock_info, update_account_metadata
from utils.ui import format_relative_time, load_account_recommendations, render_recommendation_table

try:
    from streamlit import fragment
except ImportError:
    try:
        from streamlit import experimental_fragment as fragment
    except ImportError:

        def fragment(func):
            return func


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
# 스타일 및 설정
# ---------------------------------------------------------------------------


def _build_stocks_meta_table(account_id: str) -> pd.DataFrame:
    """stocks.json 메타정보를 DataFrame으로 반환."""
    etfs = get_etfs(account_id)
    if not etfs:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for idx, etf in enumerate(etfs, 1):
        bucket_val = etf.get("bucket", 1)
        bucket_str = BUCKET_MAPPING.get(bucket_val, "1. 모멘텀")

        rows.append(
            {
                "#": idx,
                "버킷": bucket_str,
                "티커": etf.get("ticker", ""),
                "종목명": etf.get("name", ""),
                "추가일자": etf.get("added_date", "-"),
                "상장일": etf.get("listing_date", "-"),
                "주간거래량": etf.get("1_week_avg_volume"),
                "1주(%)": etf.get("1_week_earn_rate"),
                "1달(%)": etf.get("1_month_earn_rate"),
                "3달(%)": etf.get("3_month_earn_rate"),
                "6달(%)": etf.get("6_month_earn_rate"),
                "12달(%)": etf.get("12_month_earn_rate"),
            }
        )
    df = pd.DataFrame(rows)
    if not df.empty and "1주(%)" in df.columns:
        df = df.sort_values(by=["버킷", "1주(%)"], ascending=[True, False])
    return df


@fragment
def _render_stocks_meta_table(account_id: str) -> None:
    """종목관리 테이블 렌더링. 업데이트 중일 경우 readonly 모드로 전환하여 스피너 방지."""

    # 세션 스테이트 키
    key_meta = f"updating_meta_{account_id}"
    key_price = f"updating_price_{account_id}"

    is_updating_meta = st.session_state.get(key_meta, False)
    is_updating_price = st.session_state.get(key_price, False)
    is_updating = is_updating_meta or is_updating_price

    # 상단 컨트롤: 이제 관리 모드는 상시 활성화 (사용자 요청)
    # 상단 컨트롤: 이제 관리 모드는 상시 활성화 (사용자 요청)
    readonly = is_updating

    df = _build_stocks_meta_table(account_id)

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

        df_edit = df.copy()
        # 사용자가 요청한 '명칭 있는 체크박스' 구현을 위해 불리언 컬럼 추가
        df_edit.insert(0, "수정/삭제", False)

        def _style_bucket(val: Any) -> str:
            val_str = str(val or "")
            for b_id, cfg in BUCKET_CONFIG.items():
                if cfg["name"] in val_str:
                    return f"background-color: {cfg['bg_color']}; color: {cfg['text_color']}; font-weight: bold; border-radius: 4px;"
            return ""

        pct_columns = ["1주(%)", "1달(%)", "3달(%)", "6달(%)", "12달(%)"]
        styled = df_edit.style

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

        new_bucket_name = st.selectbox(
            "버킷 변경", options=BUCKET_OPTIONS, index=BUCKET_OPTIONS.index(current_bucket_name)
        )

        st.divider()
        st.subheader("🗑️ 종목 삭제")
        delete_reason = st.text_input(
            "삭제 사유 (필수)", placeholder="삭제 이유를 입력하세요", key=f"edit_del_reason_{ticker}"
        )

        c1, c2 = st.columns(2)
        with c1:
            if st.button("💾 변경사항 저장", type="primary", use_container_width=True):
                new_bucket_int = BUCKET_REVERSE_MAPPING.get(new_bucket_name, 1)
                if update_stock(account_id, ticker, bucket=new_bucket_int):
                    st.toast(f"✅ {ticker} 버킷 변경 완료")
                    st.rerun()
        with c2:
            if st.button("🗑️ 삭제 실행", type="secondary", use_container_width=True):
                if not delete_reason or not delete_reason.strip():
                    st.error("삭제 사유를 입력해야 합니다.")
                elif remove_stock(account_id, ticker, reason=delete_reason.strip()):
                    st.toast(f"✅ {ticker} 삭제 완료")
                    st.rerun()

    # --- 상단 관리 버튼 영역 ---
    # [종목 추가 / 메타데이터 업데이트 / 가격 캐시 갱신] 버튼 배치 (저장 버튼 제거)
    c_mgr1, c_mgr2, c_mgr3 = st.columns([1, 1, 1])

    with c_mgr1:
        if st.button("➕ 종목 추가", key=f"btn_add_modal_{account_id}", disabled=readonly, use_container_width=True):
            st.session_state[f"show_add_modal_{account_id}"] = True
            st.rerun()

    with c_mgr2:
        if st.button("메타데이터 업데이트", key=f"btn_meta_{account_id}", disabled=readonly, use_container_width=True):
            st.session_state[key_meta] = True
            st.rerun()

    with c_mgr3:
        if st.button("가격 캐시 갱신", key=f"btn_price_{account_id}", disabled=readonly, use_container_width=True):
            st.session_state[key_price] = True
            st.rerun()

    st.write("")  # 간격

    # DataFrame 표시
    column_config = {
        "수정/삭제": st.column_config.CheckboxColumn("수정/삭제", width=90, help="클릭하여 수정 또는 삭제"),
        "버킷": st.column_config.SelectboxColumn(
            "버킷",
            width=150,
            options=BUCKET_OPTIONS,
            required=True,
        ),
        "티커": st.column_config.TextColumn("티커", width=80),
        "종목명": st.column_config.TextColumn("종목명", width=300),
        "추가일자": st.column_config.TextColumn("추가일자", width=100),
        "상장일": st.column_config.TextColumn("상장일", width=110),
        "주간거래량": st.column_config.NumberColumn("주간거래량", width=120, format="%d"),
        "1주(%)": st.column_config.NumberColumn("1주(%)", width="small", format="%.2f%%"),
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
        "1달(%)",
        "3달(%)",
        "6달(%)",
        "12달(%)",
        "추가일자",
    ]

    existing_columns = [col for col in column_order if col in df_edit.columns]

    if readonly:
        # 갱신 중일 때는 static dataframe 사용 (스피너 방지)
        calc_height = min((len(df.index) + 1) * 35 + 10, 400)
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
        calc_height = min((len(df.index) + 1) * 35 + 10, 400)

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

                    # [중요] 모달이 열리기 전 세션 스테이트에서 체크박스 상태를 리셋하여
                    # 무한 리런이나 모달 닫기 후 잔상 방지
                    # 하지만 직접 수정이 안되므로 모달 내부에서 리런을 유도함
                    open_edit_dialog(ticker, bucket_name, name)
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
        try:
            settings = get_account_settings(account_id)
            country_code = settings.get("country_code", "kor")
        except Exception:
            country_code = "kor"

        st.write(f"계좌: **{account_id.upper()}** ({country_code.upper()})")

        # 국가별 플레이스홀더 설정
        if country_code == "kor":
            placeholder_text = "예: 005930"
        elif country_code in ["us", "usa"]:
            placeholder_text = "예: SPY"
        elif country_code in ["au", "aus"]:
            placeholder_text = "예: VAS"
        else:
            placeholder_text = "예: Ticker"

        c_in, c_btn = st.columns([3, 1], vertical_alignment="bottom")
        with c_in:
            d_ticker = st.text_input(
                "티커 입력", placeholder=placeholder_text, max_chars=12, key=f"in_ticker_{account_id}"
            ).strip()
        with c_btn:
            do_search = st.button("🔍 조회", key=f"btn_search_{account_id}", use_container_width=True)

        if do_search:
            if not d_ticker:
                st.error("티커를 입력하세요.")
                st.session_state[ss_key_result] = None
            else:
                with st.spinner("정보 조회 중..."):
                    info = fetch_stock_info(d_ticker, country_code)
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
                if st.button(
                    "➕ 추가하기", type="primary", use_container_width=True, key=f"btn_confirm_add_{account_id}"
                ):
                    success = add_stock(
                        account_id,
                        ticker_res,
                        search_result["name"],
                        listing_date=search_result.get("listing_date"),
                        bucket=bucket_int,
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
        if st.button("닫기", key=f"btn_close_modal_internal_{account_id}", use_container_width=True):
            st.session_state[f"show_add_modal_{account_id}"] = False
            st.rerun()

    # [Continuous Add] 모달 유지 로직: 플래그가 True면 강제로 모달 오픈
    if st.session_state.get(f"show_add_modal_{account_id}"):
        open_add_dialog()

    # 삭제된 종목 표시 (접이식 Expander로 변경)
    if not readonly:
        deleted_etfs = get_deleted_etfs(account_id)
        if deleted_etfs:
            st.markdown("---")
            with st.expander(f"🗑️ 삭제된 종목 ({len(deleted_etfs)}개)", expanded=False):
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
                            "삭제일": deleted_at_str,
                            "삭제 사유": etf.get("deleted_reason", "-"),
                        }
                    )
                df_deleted = pd.DataFrame(deleted_rows)
                df_deleted.sort_values(by=["버킷", "삭제일"], ascending=[True, False], inplace=True)

                edited_deleted = st.data_editor(
                    df_deleted.style.map(lambda _: "background-color: #ffe0e6"),
                    hide_index=True,
                    width="stretch",
                    column_config={
                        "복구": st.column_config.CheckboxColumn("복구", width="small"),
                        "버킷": st.column_config.SelectboxColumn("버킷", width=150, options=BUCKET_OPTIONS),
                        "티커": st.column_config.TextColumn("티커", width=80),
                        "종목명": st.column_config.TextColumn("종목명", width=250),
                        "삭제일": st.column_config.TextColumn("삭제일", width=110),
                        "삭제 사유": st.column_config.TextColumn("삭제 사유", width=300),
                    },
                    disabled=["티커", "종목명", "삭제일", "삭제 사유"],
                    key=f"deleted_editor_{account_id}",
                )

                to_restore_df = edited_deleted[edited_deleted["복구"]]
                if not to_restore_df.empty:
                    st.info(f"선택한 {len(to_restore_df)}개 종목을 복구합니다.")
                    if st.button("♻️ 선택 종목 복구", type="primary", key=f"btn_restore_{account_id}"):
                        restored = 0
                        for _, row in to_restore_df.iterrows():
                            ticker = row["티커"]
                            bucket_name = row["버킷"]
                            bucket_int = BUCKET_REVERSE_MAPPING.get(bucket_name, 1)

                            if add_stock(account_id, ticker, bucket=bucket_int):
                                restored += 1
                        st.success(f"{restored}개 종목 복구 완료!")
                        st.rerun()
        else:
            st.info("삭제된 종목이 없습니다.")

    # -----------------------------------------------------------------------
    # 업데이트 실행 로직 (readonly 모드일 때 실행됨)
    # -----------------------------------------------------------------------
    if is_updating_meta:
        st.divider()
        # [User Request] 스피너 아이콘 제거를 위해 st.status 대신 st.empty 사용
        status_area = st.empty()
        p_bar = st.progress(0)

        status_area.info("메타데이터 업데이트 준비 중...")

        def on_progress(curr, total, ticker):
            pct = min(curr / total, 1.0)
            p_bar.progress(pct)
            status_area.info(f"메타데이터 획득 중: {curr}/{total} - {ticker}")

        try:
            update_account_metadata(account_id, progress_callback=on_progress)
            status_area.success("메타데이터 업데이트 완료!")
            time.sleep(1.0)
        except Exception as e:
            status_area.error(f"실패: {e}")
            time.sleep(3.0)

        # 상태 해제 및 리런
        del st.session_state[key_meta]
        st.rerun()

    if is_updating_price:
        st.divider()
        status_area = st.empty()
        p_bar = st.progress(0)

        status_area.info("가격 캐시 갱신 준비 중...")

        def on_progress(curr, total, ticker):
            pct = min(curr / total, 1.0)
            p_bar.progress(pct)
            status_area.info(f"가격 캐시 갱신 중: {curr}/{total} - {ticker}")

        try:
            refresh_cache_for_target(account_id, None, progress_callback=on_progress)
            status_area.success("가격 캐시 갱신 완료!")
            time.sleep(1.0)
        except Exception as e:
            status_area.error(f"실패: {e}")
            time.sleep(3.0)

        del st.session_state[key_price]
        st.rerun()


def _get_latest_log_content(account_id: str) -> tuple[str | None, str | None]:
    """
    Get the content of the latest recommend_*.log file for the given account.
    Returns (filename, content).
    """
    log_dir = os.path.join("zaccounts", account_id, "results")
    search_pattern = os.path.join(log_dir, "recommend_*.log")
    files = glob.glob(search_pattern)

    if not files:
        return None, None

    latest_file = max(files, key=os.path.getmtime)
    try:
        with open(latest_file, encoding="utf-8") as f:
            content = f.read()
        return os.path.basename(latest_file), content
    except Exception:
        return os.path.basename(latest_file), "파일을 읽는 중 오류가 발생했습니다."


@fragment
def _render_run_recommendation(account_id: str) -> None:
    """추천 실행 화면 렌더링"""

    st.caption("이 기능은 백그라운드에서 추천 스크립트(`recommend.py`)를 즉시 실행합니다.")

    # 세션 스테이트 초기화
    if "admin_console_log" not in st.session_state:
        st.session_state["admin_console_log"] = ""

    # 1. 실행 버튼
    if st.button("🚀 추천 시스템 즉시 실행", type="primary", key=f"btn_run_rec_{account_id}"):
        status_area = st.empty()
        status_area.info(f"🚀 `{account_id}` 계정 추천 실행 중...")

        try:
            # logs reset before run
            st.session_state["admin_console_log"] = ""

            result = subprocess.run(
                ["python", "recommend.py", account_id],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )

            # 실행 결과 저장
            st.session_state["admin_console_log"] = result.stdout

            if result.returncode == 0:
                status_area.success(f"✅ `{account_id}` 추천 실행 완료!")
                time.sleep(1)
                st.rerun()
            else:
                # 에러 로그 파싱 (마지막 줄 또는 [ERROR] 포함 라인)
                lines = result.stdout.strip().splitlines()
                error_msg = f"❌ 실행 실패 (Exit Code: {result.returncode})"
                # 뒤에서부터 탐색하여 [ERROR]가 있는 가장 마지막 줄 찾기
                for line in reversed(lines):
                    if "[ERROR]" in line:
                        error_msg = f"❌ {line.strip()}"
                        break
                else:
                    # [ERROR]를 못 찾았으면 그냥 마지막 줄 표시
                    if lines:
                        error_msg = f"❌ {lines[-1].strip()}"

                status_area.error(error_msg)

        except Exception as e:
            status_area.error(f"실행 중 예외 발생: {str(e)}")
            st.session_state["admin_console_log"] += f"\n[System Error] {str(e)}"

    st.divider()

    # 2. 콘솔 로그
    with st.expander("콘솔 로그", expanded=False):
        log_content = st.session_state.get("admin_console_log", "")
        if log_content:
            st.code(log_content)
        else:
            st.info("실행 이력이 없습니다.")

    # 3. 파일 결과 (항상 최신 파일 로드)
    file_name, file_content = _get_latest_log_content(account_id)

    expander_title = f"📁 최신 결과 파일 ({file_name})" if file_name else "📁 최신 결과 파일 (없음)"
    with st.expander(expander_title, expanded=True):
        if file_content:
            st.code(file_content, language="text")
        else:
            st.warning("표시할 결과 파일이 존재하지 않습니다.")


def _get_active_holdings(df: pd.DataFrame) -> pd.DataFrame:
    """보유 중인 종목만 필터링합니다."""
    try:
        from logic.backtest import get_hold_states

        hold_states = get_hold_states() | {"BUY", "BUY_REPLACE"}
        return df[df["상태"].isin(hold_states)].copy()
    except Exception:
        return df


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

    # 메뉴명과 동일한 이름 사용 (BUCKET_TOPN 포함)
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

    view_mode = st.pills(
        "뷰",
        ["1. 추천 결과", "2. 종목 관리", "3. 추천 실행"],
        default="1. 추천 결과",
        key=f"view_{account_id}",
        label_visibility="collapsed",
    )

    if view_mode == "2. 종목 관리":
        _render_stocks_meta_table(account_id)
    elif view_mode == "3. 추천 실행":
        _render_run_recommendation(account_id)
    else:  # "1. 추천 결과" (Default)
        if df is None:
            st.error(
                updated_at
                or "추천 데이터를 불러오지 못했습니다. 먼저 `python recommend.py <account>` 명령으로 스냅샷을 생성해 주세요."
            )
        else:
            # 보유 종목만 필터링
            df_held = _get_active_holdings(df)
            if df_held.empty:
                st.info("현재 보유 중인 종목이 없습니다.")
            else:
                render_recommendation_table(
                    df_held,
                    country_code=country_code,
                    grouped_by_bucket=False,
                    # customize_columns={"#": ("버킷", 120)} # This will be implemented in utils/ui.py
                )

    # --- 공통: 업데이트 시간, 설정, 푸터 (보유종목/종목추세 탭에서만 표시) ---
    if view_mode in ("1. 추천 결과", "2. 종목 추세") and updated_at:
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

                params_to_show = {}
                if strategy_tuning.get("MA_MONTH"):
                    params_to_show["MA개월"] = strategy_tuning.get("MA_MONTH")

                from config import OPTIMIZATION_METRIC, REBALANCE_MODE

                params_to_show.update(
                    {
                        "MA타입": strategy_tuning.get("MA_TYPE"),
                        "리밸런스 주기": REBALANCE_MODE,
                        "최적화 지표": OPTIMIZATION_METRIC,
                    }
                )

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
                from core.backtest.portfolio import get_hold_states

                hold_states = get_hold_states() | {"BUY", "BUY_REPLACE"}
                if df is not None:
                    current_holdings = int(df[df["상태"].isin(hold_states)].shape[0])
                    target_topn = strategy_tuning.get("BUCKET_TOPN") if isinstance(strategy_tuning, dict) else None
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
    elif view_mode in ("1. 보유 종목", "2. 종목 추세"):
        st.caption("데이터를 찾을 수 없습니다.")


__all__ = ["render_account_page"]
