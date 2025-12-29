from __future__ import annotations

import glob
import os
import subprocess
from datetime import datetime

import pandas as pd
import streamlit as st

from utils.recommendation_storage import fetch_latest_recommendations
from utils.settings_loader import list_available_accounts
from utils.ui import format_relative_time


def _get_db_time_info(account_id: str) -> str | None:
    """
    Fetch the last updated time from MongoDB and format it with relative time.
    Format: 'YYYY-MM-DD HH:MM:SS(X분 전), User'
    """
    try:
        snapshot = fetch_latest_recommendations(account_id)
        if not snapshot:
            return None

        updated_dt = snapshot.get("updated_at") or snapshot.get("created_at")
        updated_by = snapshot.get("updated_by", "")

        # 1. Datetime 객체 변환 (Asia/Seoul)
        ts_seoul = None
        if isinstance(updated_dt, datetime):
            ts = pd.Timestamp(updated_dt)
            if ts.tzinfo is None or ts.tzinfo.utcoffset(ts) is None:
                ts_seoul = ts.tz_localize("UTC").tz_convert("Asia/Seoul")
            else:
                ts_seoul = ts.tz_convert("Asia/Seoul")
        else:
            try:
                parsed = pd.to_datetime(updated_dt)
                if parsed.tzinfo is None or parsed.tzinfo.utcoffset(parsed) is None:
                    ts_seoul = parsed.tz_localize("UTC").tz_convert("Asia/Seoul")
                else:
                    ts_seoul = parsed.tz_convert("Asia/Seoul")
            except Exception:
                pass

        if ts_seoul is None:
            return str(updated_dt)

        # 2. 절대 시간 포맷팅
        time_str = ts_seoul.strftime("%Y-%m-%d %H:%M:%S")

        # 3. 상대 시간 사용
        rel_str = format_relative_time(ts_seoul)

        final_str = f"{time_str}{rel_str}"
        if updated_by:
            final_str = f"{final_str}, {updated_by}"

        return final_str

    except Exception:
        return None


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


def render_admin_page() -> None:
    st.set_page_config(page_title="[Admin] 관리자", page_icon="⚙️", layout="wide")
    st.title("⚙️ 관리자 페이지")

    # 1. 세션 스테이트 초기화
    if "admin_console_log" not in st.session_state:
        st.session_state["admin_console_log"] = ""
    if "admin_last_account" not in st.session_state:
        st.session_state["admin_last_account"] = None

    # 2. 계정 선택
    accounts = list_available_accounts()
    if not accounts:
        st.error("사용 가능한 계정이 없습니다.")
        return

    selected_account = st.selectbox(
        "계정 선택", accounts, index=0, key="admin_account_selector", help="추천을 실행할 계정을 선택하세요."
    )

    # 계정이 변경되었으면 콘솔 로그 초기화
    if selected_account != st.session_state["admin_last_account"]:
        st.session_state["admin_last_account"] = selected_account
        st.session_state["admin_console_log"] = ""

    st.markdown("---")
    st.subheader("💡 추천 실행")

    # DB 업데이트 시간 조회
    time_info = _get_db_time_info(selected_account)

    if time_info:
        # 볼드체로 표시
        st.markdown(f"**최근 실행: {time_info}**")
    else:
        st.info("아직 추천 데이터가 없습니다.")

    # 3. 추천 실행 버튼
    if st.button("추천 실행", type="primary", key="btn_run_recommend"):
        if not selected_account:
            st.warning("계정을 선택해주세요.")
            return

        status_area = st.empty()
        status_area.info(f"🚀 `{selected_account}` 계정 추천 실행 중...")

        try:
            # logs reset before run
            st.session_state["admin_console_log"] = ""

            result = subprocess.run(
                ["python", "recommend.py", selected_account],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )

            # 실행 결과 저장
            st.session_state["admin_console_log"] = result.stdout

            if result.returncode == 0:
                status_area.success(f"✅ `{selected_account}` 추천 실행 완료!")
                # Rerun to update the "Last Run" time and file content
                st.rerun()
            else:
                status_area.error(f"❌ 실행 실패 (Exit Code: {result.returncode})")

        except Exception as e:
            status_area.error(f"실행 중 예외 발생: {str(e)}")
            st.session_state["admin_console_log"] += f"\n[System Error] {str(e)}"

    # 4. 결과 표시 (항상 표시)
    st.markdown("---")

    # 4-1. 콘솔 로그
    with st.expander("콘솔 로그", expanded=False):
        log_content = st.session_state.get("admin_console_log", "")
        if log_content:
            st.code(log_content)
        else:
            st.info("실행 이력이 없습니다.")

    # 4-2. 파일 결과 (항상 최신 파일 로드)
    # 파일 정보 조회
    file_name, file_content = _get_latest_log_content(selected_account)

    expander_title = f"파일 결과 ({file_name})" if file_name else "파일 결과 (파일 없음)"
    with st.expander(expander_title, expanded=True):
        if file_content:
            st.code(file_content, language="text")
        else:
            st.warning("표시할 결과 파일이 존재하지 않습니다.")


__all__ = ["render_admin_page"]

if __name__ == "__main__":
    render_admin_page()
