from __future__ import annotations

from datetime import datetime

import pandas as pd
import streamlit as st

from utils.recommendation_storage import fetch_latest_recommendations
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


def render_admin_page() -> None:
    st.set_page_config(page_title="관리자", page_icon="⚙️", layout="wide")

    # 2. 계정 선택
    from utils.account_registry import load_account_configs

    account_configs = load_account_configs()
    accounts = [cfg["account_id"] for cfg in account_configs]

    if not accounts:
        st.error("사용 가능한 계정이 없습니다.")
        return

    st.title("⚙️ 관리자 페이지")

    st.info("추천 실행 기능은 각 계좌 페이지의 '추천실행' 탭으로 이동되었습니다.")

    st.markdown("---")
    st.subheader("📊 계정 상태 요약")

    for account in accounts:
        time_info = _get_db_time_info(account)
        st.write(f"- **{account}**: {time_info if time_info else '데이터 없음'}")


__all__ = ["render_admin_page"]

if __name__ == "__main__":
    render_admin_page()
