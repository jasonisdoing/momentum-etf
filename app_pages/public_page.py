import pandas as pd
import streamlit as st

from utils.account_registry import load_account_configs
from utils.portfolio_io import load_real_holdings_table
from utils.rankings import build_account_rankings, get_account_rank_defaults


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


def render_public_notebook_rank() -> None:
    """노트북LM용 퍼블릭 랭킹 정보 출력 (인증 없음)"""
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
    st.caption("이 페이지는 노트북LM용으로 최적화된 최신 랭킹 정보를 제공합니다.")

    accounts = load_account_configs()
    md_output = ""

    for account in accounts:
        account_id = account["account_id"]
        account_name = account.get("name") or account_id.upper()

        ma_type, ma_months = get_account_rank_defaults(account_id)

        # 실시간 시세 없이 캐시 데이터 기반으로 빠르게 렌더링 (퍼블릭용)
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

    st.markdown(md_output)
