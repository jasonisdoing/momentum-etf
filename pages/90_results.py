from pathlib import Path
from typing import List, Optional, Dict, Any

import streamlit as st
import pandas as pd

from signals import calculate_benchmark_comparison
from utils.account_registry import get_account_info

# 페이지 전체 폭 사용
st.set_page_config(page_title="시그널 결과 뷰어", layout="wide")
st.markdown(
    """
    <style>
      .block-container { max-width: 100% !important; padding-left: 2rem; padding-right: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# 앱 루트 기준 경로들
APP_ROOT = Path(__file__).resolve().parent.parent
RESULT_DIR = APP_ROOT / "results"


def list_result_files(account: str | None) -> List[Path]:
    files: List[Path] = []
    if RESULT_DIR.exists():
        pattern = f"signal_{account}_*.log" if account else "signal_*.log"
        for p in sorted(RESULT_DIR.glob(pattern)):
            files.append(p)
    return files


def read_text(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            return p.read_text(encoding="cp949")
        except Exception:
            return p.read_text(errors="ignore")
    except Exception as e:
        return f"[오류] 파일을 읽는 중 문제가 발생했습니다: {e}"


def main():
    st.title("🧾 시그널 결과 뷰어")

    account = st.query_params.get("account")

    files = list_result_files(account)
    if not files:
        if account:
            st.info(f"results/ 에 '{account}' 계좌의 signal_*.log 파일이 없습니다.")
        else:
            st.info("results/ 에 signal_*.log 파일이 없습니다.")
        return

    names = [p.name for p in files]
    default_idx = 0
    selected = st.selectbox("파일 선택", names, index=default_idx)

    target = next((p for p in files if p.name == selected), None)

    if target is None or not target.exists():
        st.error("선택한 파일을 찾을 수 없습니다.")
        return

    st.caption(str(target))

    text = read_text(target)

    # 줄 수 제한 제거 요청에 따라 전체 내용을 표시합니다.

    # 고정폭 폰트 적용
    st.markdown(
        """
        <style>
        code, pre {
            font-family: 'D2Coding', 'NanumGothic Coding', 'Consolas', 'Courier New', monospace !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # 렌더링: 원문 그대로 보여줍니다.
    st.code(text or "(빈 파일)", language="text")

    # --- 벤치마크 비교 섹션 ---
    try:
        # 파일명에서 account, date 추출: signal_{account}_{YYYY-MM-DD}.log
        parts = selected.rsplit(".", 1)[0].split("_")
        acct_from_file = parts[1] if len(parts) >= 3 else None
        date_from_file = parts[2] if len(parts) >= 3 else None

        account_code = acct_from_file or st.query_params.get("account")
        date_str = date_from_file

        if account_code and date_str:
            info = get_account_info(account_code) or {}
            country = str(info.get("country") or "").strip()
            if country:
                bm_results: Optional[List[Dict[str, Any]]] = calculate_benchmark_comparison(
                    country, account_code, date_str
                )
                if bm_results:
                    st.markdown("---")
                    st.subheader("벤치마크 비교")
                    data_for_df = []
                    for res in bm_results:
                        row_data = {
                            "티커": res.get("ticker", "-"),
                            "벤치마크": res.get("name", "N/A"),
                            "누적수익률": res.get("cum_ret_pct")
                            if not res.get("error")
                            else res.get("error"),
                            "초과수익률": res.get("excess_return_pct") if not res.get("error") else "-",
                        }
                        data_for_df.append(row_data)
                    st.dataframe(
                        pd.DataFrame(data_for_df),
                        hide_index=True,
                        width="stretch",
                        column_config={
                            "누적수익률": st.column_config.NumberColumn(format="%.2f%%"),
                            "초과수익률": st.column_config.NumberColumn(format="%+.2f%%"),
                        },
                    )
    except Exception:
        pass


if __name__ == "__main__":
    main()
