import os
import threading
import time

import pandas as pd
import streamlit as st

from utils.account_registry import load_account_configs
from utils.portfolio_io import load_real_holdings_table
from utils.rankings import build_account_rankings, get_account_rank_defaults

CACHE_FILE = "data/notebook_rank_cache.md"
UPDATE_LOCK = threading.Lock()


def _df_to_markdown_simple(df: pd.DataFrame) -> str:
    """tabulate 의존성 없이 DataFrame을 Markdown 표로 변환 (공공용)"""
    if df.empty:
        return ""
    cols = df.columns.tolist()
    header = "| " + " | ".join(map(str, cols)) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df.iterrows():
        values = []
        for x in row.values:
            val = "" if pd.isna(x) else str(x)
            val = val.replace("\n", " ").replace("|", "\\|")
            values.append(val)
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join([header, sep] + rows)


def _generate_notebook_rank_markdown() -> str:
    """모든 계좌의 랭킹 정보를 Markdown 형식으로 생성 (캐싱 없이 직접 실행)"""
    accounts = load_account_configs()
    md_output = ""

    for account in accounts:
        account_id = account["account_id"]
        account_name = account.get("name") or account_id.upper()

        ma_type, ma_months = get_account_rank_defaults(account_id)

        # 실시간 시세 없이 캐시 데이터 기반으로 렌더링
        df_rank = build_account_rankings(account_id, ma_type=ma_type, ma_months=ma_months)

        md_output += f"# {account_name}\n\n"
        md_output += f"MA Type: {ma_type}, Months: {ma_months}\n\n"

        if df_rank.empty:
            md_output += "순위 데이터가 없습니다.\n\n"
        else:
            target_df = df_rank.copy()

            # 보유 정보 계산
            try:
                df_holdings = load_real_holdings_table(account_id)
                held_tickers = set(df_holdings["티커"].astype(str).tolist())
            except Exception:
                held_tickers = set()

            if "티커" in target_df.columns:
                target_df["보유"] = target_df["티커"].apply(lambda x: "보유" if str(x) in held_tickers else "")
            else:
                target_df["보유"] = ""

            # 현재가 포맷팅 (국가별)
            if "현재가" in target_df.columns:
                if "kor" in account_id.lower():
                    target_df["현재가"] = target_df["현재가"].apply(lambda x: f"{int(x):,}원" if pd.notna(x) else "")
                elif "aus" in account_id.lower():
                    target_df["현재가"] = target_df["현재가"].apply(lambda x: f"A${x:,.2f}" if pd.notna(x) else "")
                elif "us" in account_id.lower():
                    target_df["현재가"] = target_df["현재가"].apply(lambda x: f"${x:,.2f}" if pd.notna(x) else "")
                else:
                    target_df["현재가"] = target_df["현재가"].apply(lambda x: f"{x:,.2f}" if pd.notna(x) else "")

            # 주요 컬럼만 선택하여 깔끔하게 제공 (보유 컬럼을 맨 앞으로)
            cols = [
                "보유",
                "버킷",
                "티커",
                "종목명",
                "현재가",
                "점수",
                "일간(%)",
                "1달(%)",
                "3달(%)",
                "6달(%)",
                "12달(%)",
            ]
            actual_cols = [c for c in cols if c in target_df.columns]
            md_output += _df_to_markdown_simple(target_df[actual_cols]) + "\n\n"

        md_output += "---\n\n"

    return md_output


def _background_update_task():
    """백그라운드에서 캐시 파일을 갱신하는 태스크"""
    if not UPDATE_LOCK.acquire(blocking=False):
        return  # 이미 다른 스레드에서 갱신 중이면 종료

    try:
        new_md = _generate_notebook_rank_markdown()
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            f.write(new_md)
    finally:
        UPDATE_LOCK.release()


def render_public_notebook_rank() -> None:
    """노트북LM용 퍼블릭 랭킹 정보 출력 (SWR: Stale-While-Revalidate 적용)"""
    # 불필요한 Streamlit UI 요소 숨기기 (노트북LM 파싱 방해 금지)
    st.markdown(
        """
        <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            [data-testid="stSidebar"] {display: none;}
        </style>
    """,
        unsafe_allow_html=True,
    )

    st.title("계좌별 ETF 추세 정보 및 보유여부")
    st.caption(
        "이 페이지는 접속 즉시 캐시된 정보를 보여주며, 데이터가 1시간 이상 경과한 경우 백그라운드에서 자동으로 갱신됩니다."
    )

    # 1. 기존 캐시 파일 읽기
    md_content = ""
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, encoding="utf-8") as f:
            md_content = f.read()

    # 2. 데이터가 없으면 즉시 생성 (최초 1회만 블로킹)
    if not md_content:
        with st.spinner("최초 데이터를 생성 중입니다..."):
            md_content = _generate_notebook_rank_markdown()
            os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                f.write(md_content)

    # 3. 화면에 즉시 출력
    st.markdown(md_content)

    # 4. 백그라운드 갱신 여부 판단 (1시간 주기)
    last_mod = os.path.getmtime(CACHE_FILE)
    if (time.time() - last_mod) > 3600:
        threading.Thread(target=_background_update_task, daemon=True).start()
