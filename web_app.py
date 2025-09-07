import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

from typing import Dict, Tuple, Optional, List

import pandas as pd
import streamlit as st

import settings
from utils.data_loader import read_holdings_file, read_tickers_file


def main() -> None:
    st.set_page_config(page_title="MomentumPilot 포트폴리오", layout="wide")
    st.title("MomentumPilot 포트폴리오 (웹 UI)")

    # Persisted settings store (data.json), including evaluation amount
    import json, os
    DATA_PATH = 'data/data.json'
    LEGACY_PATH = 'initial_capital.json'

    def _load_data() -> dict:
        # Try new path
        try:
            if os.path.exists(DATA_PATH):
                with open(DATA_PATH, 'r', encoding='utf-8') as f:
                    obj = json.load(f)
                    if isinstance(obj, dict):
                        return obj
        except Exception:
            pass
        # Migrate from legacy file if present
        try:
            if os.path.exists(LEGACY_PATH):
                with open(LEGACY_PATH, 'r', encoding='utf-8') as f:
                    obj = json.load(f)
                    if isinstance(obj, dict):
                        # Write to new path and return
                        with open(DATA_PATH, 'w', encoding='utf-8') as out:
                            json.dump(obj, out, ensure_ascii=False)
                        return obj
        except Exception:
            pass
        return {}

    def _save_data(d: dict) -> None:
        """Save data.json with pretty formatting (UTF-8, indent, sorted keys)."""
        try:
            with open(DATA_PATH, 'w', encoding='utf-8') as f:
                json.dump(
                    d,
                    f,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                f.write('\n')  # final newline for POSIX-friendly files
        except Exception:
            pass

    # Small helper: show auto-disappearing success toast, fallback to success box
    def _notify_saved(msg: str):
        try:
            if hasattr(st, 'toast'):
                st.toast(msg, icon='✅')
            else:
                st.success(msg)
        except Exception:
            st.success(msg)

    # Top input: evaluation amount
    data_store = _load_data()
    try:
        saved_cap = int(float(data_store.get('initial_capital', getattr(settings, 'INITIAL_CAPITAL', 100_000_000))))
    except Exception:
        saved_cap = int(getattr(settings, 'INITIAL_CAPITAL', 100_000_000))
    eval_amount = st.number_input(
        "평가금액 (원)", min_value=0, value=int(saved_cap), step=1_000_000, format="%d", key='eval_amount_top'
    )
    if eval_amount != saved_cap:
        data_store['initial_capital'] = int(eval_amount)
        _save_data(data_store)
        _notify_saved('평가금액이 저장되었습니다')

    # Prefill from local holdings file if present
    try:
        init_pos = read_holdings_file('data/holdings.csv')
    except Exception:
        init_pos = None

    # Tickers from tickers.txt
    pairs: List[Tuple[str, str]] = read_tickers_file('data/tickers.txt')
    # tickers.txt가 없어도 수동입력은 가능하므로 강제 종료하지 않음

    # Manual holdings editor
    st.markdown("## 보유 입력(수동)")
    st.caption("티커 입력 → 명칭 자동 채우기(tickers.txt 참조). 수량은 정수, 단가는 원 단위.")

    name_map = {t: n for t, n in pairs if n}
    def _name_for(tkr: str) -> str:
        if not tkr:
            return ''
        # Prefer target.txt provided name
        if tkr in name_map and str(name_map[tkr]).strip():
            return str(name_map[tkr]).strip()
        # No external lookup; leave blank if not found
        return ''

    # Build/restore editable DataFrame from session state to allow auto-fill on blur
    if 'ed_df' not in st.session_state:
        if init_pos:
            rows_ed = []
            for tkr, v in init_pos.items():
                rows_ed.append({
                    '티커': tkr,
                    '이름': _name_for(tkr),
                    '수량': int(v.get('shares') or 0),
                    '매수단가': float(v.get('avg_cost')) if v.get('avg_cost') is not None else 0.0,
                })
            st.session_state.ed_df = pd.DataFrame(rows_ed)
        else:
            st.session_state.ed_df = pd.DataFrame([{ '티커':'', '이름':'', '수량':0, '매수단가':0.0 } for _ in range(5)])

    # Controls aligned above the table (right: 저장)
    _, right_col = st.columns([3,1])
    with right_col:
        if st.button('저장 (data/holdings.csv)', key='btn_save_holdings'):
            # sanitize and save using session state
            src_df = st.session_state.get('ed_df', pd.DataFrame()).copy()
            if not src_df.empty:
                src_df['티커'] = src_df['티커'].astype(str).str.strip()
                # Robust numeric parsing (allow commas/strings)
                def _to_float(v):
                    try:
                        if v is None:
                            return 0.0
                        s = str(v).replace(',', '').strip()
                        return float(s) if s != '' else 0.0
                    except Exception:
                        return 0.0
                def _to_int(v):
                    try:
                        return int(round(_to_float(v)))
                    except Exception:
                        return 0
                src_df['_shares_num'] = src_df['수량'].apply(_to_float)
                src_df = src_df[(src_df['티커'] != '') & (src_df['_shares_num'] > 0)]
            try:
                import csv
                with open('data/holdings.csv','w', encoding='utf-8', newline='') as f:
                    w = csv.writer(f)
                    for _, r in src_df.iterrows():
                        tkr = str(r.get('티커') or '').strip()
                        sh = _to_int(r.get('수량'))
                        ac = _to_float(r.get('매수단가'))
                        nm = str(r.get('이름') or '').strip()
                        if not tkr or sh <= 0:
                            continue
                        # Save order: 티커, 명칭, 수량, 금액(매수단가)
                        w.writerow([tkr, nm, sh, int(round(ac))])
                _notify_saved(f"data/holdings.csv 저장 완료 ({len(src_df)}개)")
            except Exception as e:
                st.error(f"저장 실패: {e}")

    df_ed_before = st.session_state.ed_df.copy()
    df_ed = st.data_editor(
        st.session_state.ed_df,
        key='ed_holdings',
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            '티커': st.column_config.TextColumn(help='6자리 숫자 티커'),
            '이름': st.column_config.TextColumn(disabled=True),
            '수량': st.column_config.NumberColumn(min_value=0, step=1),
            '매수단가': st.column_config.NumberColumn(min_value=0.0, step=1.0),
        }
    )

    # Show current saved line count in holdings.csv just below the table
    def _count_saved_rows(path: str = 'data/holdings.csv') -> int:
        import os, csv
        if not os.path.exists(path):
            return 0
        try:
            with open(path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                cnt = 0
                for row in reader:
                    if not row:
                        continue
                    # Schema: ticker, name, shares, amount
                    tkr = (row[0] or '').strip()
                    shares_s = (row[2] if len(row) > 2 else '').strip().replace(',', '')
                    try:
                        sh = int(float(shares_s)) if shares_s != '' else 0
                    except Exception:
                        # Fallback for old schema: ticker, shares
                        shares_s = (row[1] if len(row) > 1 else '').strip().replace(',', '')
                        sh = 0
                    if tkr and sh > 0:
                        cnt += 1
                return cnt
        except Exception:
            return 0

    saved_rows = _count_saved_rows()
    st.caption(f"현재 data/holdings.csv 저장 행 수: {saved_rows}개")

    # Auto-fill names on ticker edit (blur causes rerun)
    auto_filled = False
    # Align indexes
    df_ed = df_ed.reset_index(drop=True)
    df_ed_before = df_ed_before.reset_index(drop=True)
    # Ensure required columns exist
    for col in ['티커','이름','수량','매수단가']:
        if col not in df_ed.columns:
            df_ed[col] = '' if col in ('티커','이름') else 0
    # Detect new/changed tickers or missing names
    for i in range(len(df_ed)):
        tkr = str(df_ed.at[i, '티커']).strip() if '티커' in df_ed.columns else ''
        name_now = str(df_ed.at[i, '이름']).strip() if '이름' in df_ed.columns else ''
        prev_tkr = str(df_ed_before.at[i, '티커']).strip() if i < len(df_ed_before) and '티커' in df_ed_before.columns else ''
        if tkr and ((not name_now) or (tkr != prev_tkr)):
            nm = _name_for(tkr)
            if nm:
                df_ed.at[i, '이름'] = nm
                auto_filled = True
    if auto_filled:
        st.session_state.ed_df = df_ed
        try:
            st.rerun()
        except Exception:
            # For older/newer Streamlit versions
            if hasattr(st, 'experimental_rerun'):
                st.experimental_rerun()


if __name__ == "__main__":
    main()
