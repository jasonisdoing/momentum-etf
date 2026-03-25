import os
import threading
import time

import pandas as pd

from utils.account_registry import load_account_configs
from utils.portfolio_io import load_real_holdings_table
from utils.rankings import build_account_rankings, get_account_rank_defaults

CACHE_FILE = "static/notebook_rank.md"
UPDATE_LOCK = threading.Lock()


def _df_to_markdown_simple(df: pd.DataFrame) -> str:
    """tabulate 의존성 없이 DataFrame을 Markdown 표로 변환"""
    if df.empty:
        return ""

    # NaN 처리
    df = df.fillna("")

    # 헤더
    header = "| " + " | ".join(df.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(df.columns)) + " |"

    # 바디
    rows = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join([str(val) for val in row]) + " |")

    return "\n".join([header, separator] + rows)


def generate_notebook_rank_markdown(progress_bar=None, status_callback=None) -> str:
    """노트북LM용 전체 계좌 랭킹 정보를 마크다운 형식으로 생성"""
    accounts = load_account_configs()
    md_output = []
    total_accounts = len(accounts)

    for idx, account in enumerate(accounts, start=1):
        account_id = str(account["account_id"])
        account_name = str(account.get("name") or account_id)
        country_code = str(account.get("country_code") or "").strip().lower()

        if status_callback:
            status_callback(f"데이터 추출 중: {account_name} ({idx}/{total_accounts})")
        if progress_bar:
            progress_bar.progress(idx / total_accounts)

        ma_type, ma_months = get_account_rank_defaults(account_id)

        # 1. 순위 데이터 (Rankings)
        rank_df = build_account_rankings(account_id, ma_type=ma_type, ma_months=ma_months)
        if rank_df.empty:
            continue

        # 2. 보유 종목 데이터 (Holdings)
        holdings_df = load_real_holdings_table(account_id)
        holding_tickers = set()
        if not holdings_df.empty:
            holding_tickers = set(holdings_df["티커"].astype(str).str.strip().str.upper().unique())

        # 3. 데이터 결합 및 포맷팅
        display_df = rank_df.copy()

        # '보유' 컬럼 추가 (가장 왼쪽)
        display_df.insert(
            0, "보유", display_df["티커"].apply(lambda x: "보유" if str(x).strip().upper() in holding_tickers else "")
        )

        # 필요한 컬럼만 선택
        cols = ["보유", "버킷", "티커", "종목명", "현재가", "점수", "일간(%)", "1달(%)", "3달(%)", "6달(%)", "12달(%)"]
        display_df = display_df[cols]

        # 통화 포맷팅
        def format_price(val):
            try:
                if pd.isna(val) or val == "":
                    return ""
                num = float(val)
                if country_code == "kor":
                    return f"{int(num):,}원"
                elif country_code == "aus":
                    return f"A${num:,.2f}"
                else:  # us 등
                    return f"${num:,.2f}"
            except Exception:
                return str(val)

        display_df["현재가"] = display_df["현재가"].apply(format_price)

        # 마크다운 작성
        md_output.append(f"# {idx}. {account_name}\n")
        md_output.append(f"MA Type: {ma_type}, Months: {ma_months}\n")
        md_output.append(_df_to_markdown_simple(display_df))
        md_output.append("\n---\n")

    return "\n".join(md_output)


def update_notebook_rank_cache(force=False, progress_bar=None, status_callback=None) -> bool:
    """
    캐시 파일을 업데이트합니다.
    - force=True 이면 즉시 갱신
    - force=False 이면 1시간 경과 시에만 갱신
    """
    if not force and os.path.exists(CACHE_FILE):
        mtime = os.path.getmtime(CACHE_FILE)
        if time.time() - mtime < 3600:  # 1시간 미만
            return False

    if not UPDATE_LOCK.acquire(blocking=False):
        return False

    try:
        new_md = generate_notebook_rank_markdown(progress_bar=progress_bar, status_callback=status_callback)
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            f.write(new_md)
        return True
    finally:
        UPDATE_LOCK.release()


def update_cache_in_background():
    """백그라운드 스레드에서 캐시 업데이트 (비유도형)"""
    thread = threading.Thread(target=update_notebook_rank_cache, args=(False,), daemon=True)
    thread.start()
