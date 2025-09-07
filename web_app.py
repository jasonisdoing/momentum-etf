import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import json, os
import glob
from typing import Dict, Tuple, Optional, List

import pandas as pd
import streamlit as st


import settings
from utils.data_loader import read_tickers_file

try:
    from pykrx import stock as _stock
except Exception:
    _stock = None


def main() -> None:
    st.set_page_config(page_title="MomentumPilot 포트폴리오", layout="wide")
    st.title("MomentumPilot 포트폴리오 (웹 UI)")

    # Small helper: show auto-disappearing success toast, fallback to success box
    def _notify_saved(msg: str):
        try:
            if hasattr(st, 'toast'):
                st.toast(msg, icon='✅')
            else:
                st.success(msg)
        except Exception:
            st.success(msg)

    def get_latest_trading_day(ref_date: pd.Timestamp, ref_ticker: str = "005930") -> Optional[pd.Timestamp]:
        """주어진 기준일 또는 그 이전의 가장 가까운 거래일을 찾습니다."""
        if _stock is None:
            st.warning("pykrx가 설치되지 않아 정확한 거래일 확인이 어렵습니다. 오늘 날짜를 기준으로 진행합니다.")
            return ref_date

        current_date = ref_date
        for _ in range(30):  # 최대 30일 전까지 탐색
            try:
                date_str = current_date.strftime('%Y%m%d')
                df = _stock.get_market_ohlcv_by_date(date_str, date_str, ref_ticker)
                if not df.empty:
                    return current_date
            except Exception:
                pass
            current_date -= pd.Timedelta(days=1)
        return None

    def load_portfolio_data(portfolio_path: Optional[str] = None, data_dir: str = 'data') -> Optional[Dict]:
        """
        지정된 포트폴리오 스냅샷 파일 또는 최신 파일을 로드합니다.
        파일을 성공적으로 로드하면 'total_equity', 'holdings' 등이 포함된 딕셔너리를 반환합니다.
        """
        filepath_to_load = None
        if portfolio_path:
            if os.path.exists(portfolio_path):
                filepath_to_load = portfolio_path
            else:
                st.warning(f"지정된 포트폴리오 파일 '{portfolio_path}'를 찾을 수 없습니다.")
                return None
        else:
            try:
                portfolio_files = glob.glob(os.path.join(data_dir, 'portfolio_*.json'))
                if not portfolio_files: return None
                latest_file = max(portfolio_files, key=os.path.getmtime)
                filepath_to_load = latest_file
            except Exception:
                return None

        if not filepath_to_load: return None

        try:
            with open(filepath_to_load, 'r', encoding='utf-8') as f:
                data = json.load(f)
            holdings_list = data.get('holdings', [])
            holdings_dict = {
                item['ticker']: {
                    'name': item.get('name', ''),
                    'shares': item.get('shares', 0),
                    'avg_cost': item.get('avg_cost', 0.0)
                } for item in holdings_list if item.get('ticker')
            }
            return {
                'date': data.get('date'), 'total_equity': data.get('total_equity'),
                'holdings': holdings_dict, 'filepath': filepath_to_load
            }
        except Exception as e:
            st.error(f"포트폴리오 파일 '{filepath_to_load}' 로드 중 오류 발생: {e}")
            return None

    # Load initial state from the latest portfolio file, with fallback to old method
    portfolio_data = load_portfolio_data()

    if portfolio_data:
        st.info(f"최신 포트폴리오 파일 '{os.path.basename(portfolio_data['filepath'])}'의 데이터로 초기화되었습니다.")
        saved_cap = int(portfolio_data.get('total_equity', 0))
        init_pos = portfolio_data.get('holdings', {})
    else:
        st.warning("포트폴리오 파일(portfolio_*.json)을 찾을 수 없습니다. 새 포트폴리오를 생성하려면 아래 정보를 입력하고 저장하세요.")
        saved_cap = int(getattr(settings, 'INITIAL_CAPITAL', 100_000_000))
        init_pos = {}

    # Top input: evaluation amount
    eval_amount = st.number_input(
        "평가금액 (원)", min_value=0, value=int(saved_cap), step=1_000_000, format="%d", key='eval_amount_top'
    )

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
        if st.button('포트폴리오 스냅샷 저장', key='btn_save_portfolio'):
            today = pd.Timestamp.now().normalize()
            archive_date = get_latest_trading_day(today)
            if not archive_date:
                st.error("오류: 최근 거래일을 찾을 수 없습니다. pykrx가 설치되었는지 확인해주세요.")
            else:
                archive_date_str = archive_date.strftime('%Y-%m-%d')
                src_df = st.session_state.get('ed_df', pd.DataFrame()).copy()
                
                archived_holdings = []
                if not src_df.empty:
                    for _, r in src_df.iterrows():
                        try:
                            ticker = str(r.get('티커') or '').strip()
                            if not ticker: continue

                            shares = int(float(str(r.get('수량', 0)).replace(',', '')))
                            if shares <= 0: continue

                            avg_cost = int(float(str(r.get('매수단가', 0)).replace(',', '')))
                            name = str(r.get('이름') or '').strip()

                            archived_holdings.append({
                                "ticker": ticker,
                                "name": name,
                                "shares": shares,
                                "avg_cost": avg_cost
                            })
                        except (ValueError, TypeError):
                            continue

                portfolio_archive = {
                    "date": archive_date_str,
                    "total_equity": int(eval_amount),
                    "holdings": archived_holdings
                }

                output_filename = f"portfolio_{archive_date_str}.json"
                output_path = os.path.join("data", output_filename)

                try:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(portfolio_archive, f, ensure_ascii=False, indent=2)
                        f.write('\n')
                    _notify_saved(f"성공: 포트폴리오를 '{output_path}' 파일에 저장했습니다.")
                    st.rerun()
                except Exception as e:
                    st.error(f"오류: 파일 저장 중 문제가 발생했습니다 - {e}")

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
