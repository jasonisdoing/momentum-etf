from __future__ import annotations

import subprocess

import pandas as pd
import streamlit as st

from utils.account_registry import load_account_configs
from utils.pool_registry import load_pool_configs


def _run_background(command: list[str], success_message: str) -> None:
    try:
        subprocess.Popen(command)
        st.success(success_message)
    except Exception as exc:
        st.error(f"⚠️ 실행 시작 오류: {exc}")


def render_system_page() -> None:
    st.subheader("🛠️ 시스템 정보")

    account_ids = [item["account_id"] for item in load_account_configs()]
    pool_ids = [item["pool_id"] for item in load_pool_configs()]

    summary_rows = [
        {"구분": "계좌", "개수": len(account_ids), "대상": ", ".join(account_ids) if account_ids else "-"},
        {"구분": "종목풀", "개수": len(pool_ids), "대상": ", ".join(pool_ids) if pool_ids else "-"},
    ]
    st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)

    schedule_rows = [
        {
            "작업": "종목 메타데이터 업데이트",
            "대상": "모든 계좌 + 모든 종목풀",
            "자동 주기": "매일 09:00 KST",
            "실행 명령": "python scripts/stock_meta_updater.py",
        },
        {
            "작업": "가격 캐시 업데이트",
            "대상": "모든 계좌 + 모든 종목풀",
            "자동 주기": "매시 정각 KST",
            "실행 명령": "python scripts/update_price_cache.py",
        },
        {
            "작업": "종목풀 랭킹 생성",
            "대상": "모든 종목풀",
            "자동 주기": "수동 실행",
            "실행 명령": "python scripts/run_all_pool_ranks.py",
        },
        {
            "작업": "계좌 추천 생성",
            "대상": "모든 계좌",
            "자동 주기": "수동 실행",
            "실행 명령": "python scripts/run_all_account_recommendations.py",
        },
        {
            "작업": "전체 자산 요약 알림",
            "대상": "전체 계좌 요약",
            "자동 주기": "매일 11:00, 18:00, 23:00, 06:00 KST",
            "실행 명령": "python scripts/slack_asset_summary.py",
        },
    ]
    st.dataframe(pd.DataFrame(schedule_rows), width="stretch", hide_index=True)
    st.caption("자동 주기는 현재 `.github/workflows` 기준입니다.")

    st.write("")
    st.subheader("🚀 전체 수동 실행")

    row1_col1, row1_col2 = st.columns(2)
    with row1_col1:
        if st.button("모든 메타데이터 업데이트", width="stretch", key="btn_system_meta_all"):
            _run_background(
                ["python", "scripts/stock_meta_updater.py"],
                "✅ 전체 메타데이터 업데이트를 시작했습니다.",
            )
    with row1_col2:
        if st.button("모든 가격 캐시 업데이트", width="stretch", key="btn_system_cache_all"):
            _run_background(
                ["python", "scripts/update_price_cache.py"],
                "✅ 전체 가격 캐시 업데이트를 시작했습니다.",
            )

    row2_col1, row2_col2 = st.columns(2)
    with row2_col1:
        if st.button("모든 종목풀 랭킹 생성", type="primary", width="stretch", key="btn_system_rank_all"):
            _run_background(
                ["python", "scripts/run_all_pool_ranks.py"],
                "✅ 전체 종목풀 랭킹 생성을 시작했습니다.",
            )
    with row2_col2:
        if st.button("모든 계좌 추천 생성", type="primary", width="stretch", key="btn_system_rec_all"):
            _run_background(
                ["python", "scripts/run_all_account_recommendations.py"],
                "✅ 전체 계좌 추천 생성을 시작했습니다.",
            )

    if st.button("전체 자산 요약 알림 전송", width="stretch", key="btn_system_asset_summary"):
        _run_background(
            ["python", "scripts/slack_asset_summary.py"],
            "✅ 전체 자산 요약 알림 전송을 시작했습니다.",
        )
