import os
import sys
import warnings

from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd
import threading
import streamlit as st

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")



import settings as global_settings
from test import TEST_MONTHS_RANGE
from logic.settings import SECTOR_COUNTRY_OPTIONS
from logic import settings as logic_settings
from status import (
    generate_status_report,
    get_market_regime_status_string,
    get_benchmark_status_string,
    _maybe_notify_basic_status,
    _maybe_notify_signal_summary,
    _maybe_notify_coin_detailed,
)
from utils.data_loader import fetch_yfinance_name, get_trading_days, fetch_pykrx_name, fetch_ohlcv_for_tickers
from utils.db_manager import (
    get_available_snapshot_dates, get_portfolio_snapshot, get_all_trades, get_all_daily_equities, 
    get_sectors, save_sectors, get_sector_stock_counts, delete_sectors_and_reset_stocks,
    get_status_report_from_db, save_status_report_to_db, delete_trade_by_id, save_sector_changes,
    get_stocks, save_stocks, save_daily_equity, save_trade, 
    get_app_settings, save_app_settings, get_common_settings, save_common_settings,
)

try:
    from pykrx import stock as _stock
except Exception:
    _stock = None

COUNTRY_CODE_MAP = {"kor": "한국", "aus": "호주", "coin": "가상화폐"}


# --- Functions ---

@st.cache_data(ttl=600)
def get_cached_benchmark_status(country: str) -> Optional[str]:
    """벤치마크 비교 문자열을 캐시하여 반환합니다."""
    # status.py에서 직접 임포트하면 순환 참조 가능성이 있으므로, 함수 내에서 호출합니다.
    return get_benchmark_status_string(country)



def get_cached_status_report(country: str, date_str: str, force_recalculate: bool = False, prefetched_data: Optional[Dict[str, pd.DataFrame]] = None):
    """
    MongoDB를 사용하여 현황 데이터를 캐시합니다.
    force_recalculate=True일 경우, 캐시를 무시하고 다시 계산합니다.
    """
    try:
        report_date = pd.to_datetime(date_str).to_pydatetime()
    except (ValueError, TypeError):
        st.error(f"잘못된 날짜 형식입니다: {date_str}")
        return None

    if not force_recalculate:
        # 1. DB에서 먼저 찾아봅니다.
        report_from_db = get_status_report_from_db(country, report_date)
        if report_from_db:
            # DB에 저장된 형식은 딕셔너리, 반환 형식은 튜플이어야 합니다.
            return (
                report_from_db.get("header_line"),
                report_from_db.get("headers"),
                report_from_db.get("rows")
            )

    # 2. DB에 없거나, 강제로 다시 계산해야 하는 경우
    try:
        # 코인: 재계산 시 최신 /accounts → trades 동기화 및 평가 스냅샷 수행
        if country == 'coin':
            try:
                from scripts.sync_bithumb_accounts_to_trades import main as _sync_trades
                _sync_trades()
            except Exception:
                pass
            try:
                from scripts.snapshot_bithumb_balances import main as _snapshot_equity
                _snapshot_equity()
            except Exception:
                pass
        new_report = generate_status_report(country=country, date_str=date_str, prefetched_data=prefetched_data)
        if new_report:
            # 3. 계산된 결과를 DB에 저장합니다.
            save_status_report_to_db(country, report_date, new_report)
        return new_report
    except Exception as e:
        # 계산 중 발생한 오류를 사용자에게 알리고, 디버깅을 위해 콘솔에 전체 오류를 출력합니다.
        import traceback
        st.error(f"'{date_str}' 현황 계산 중 오류가 발생했습니다. 자세한 내용은 콘솔 로그를 확인해주세요.")
        print(f"--- 현황 계산 오류: {country}/{date_str} ---")
        traceback.print_exc()
        print("------------------------------------")
        return None


def check_password():
    """
    사용자가 올바른 비밀번호를 입력했는지 확인합니다.
    비밀번호가 맞지 않으면 입력을 요청하고 False를 반환합니다.
    """
    # 비밀번호를 환경 변수 또는 settings.py에서 가져옵니다.
    # 배포 환경에서는 환경 변수(Secrets) 사용을 권장합니다.
    correct_password = os.environ.get("WEBAPP_PASSWORD") or getattr(
        global_settings, "WEBAPP_PASSWORD", None
    )

    # 비밀번호가 설정되지 않은 경우, 바로 접근을 허용합니다.
    if not correct_password:
        return True

    # st.session_state를 사용하여 로그인 상태를 유지합니다.
    if "password_correct" not in st.session_state:
        st.session_state["password_correct"] = False

    if st.session_state["password_correct"]:
        return True

    # 비밀번호 입력 폼
    with st.form("password_form"):
        st.title("MomentumPilot")
        st.header("비밀번호를 입력하세요")
        password = st.text_input("Password", type="password", label_visibility="collapsed")
        submitted = st.form_submit_button("로그인")

        if submitted:
            if password == correct_password:
                st.session_state["password_correct"] = True
                st.rerun()
            else:
                st.error("비밀번호가 올바르지 않습니다.")
    return False


def style_returns(val) -> str:
    """수익률 값(숫자)에 대해 양수는 빨간색, 음수는 파란색으로 스타일을 적용합니다."""
    color = ""
    if isinstance(val, (int, float)):
        if val > 0:
            color = "red"
        elif val < 0:
            color = "blue"
    return f"color: {color}"


def _display_status_report_df(df: pd.DataFrame, country_code: str):
    """
    현황 리포트 DataFrame에 종목 메타데이터(이름, 국가, 업종)를 실시간으로 병합하고 스타일을 적용하여 표시합니다.
    """
    # 1. 종목 메타데이터 로드
    stocks_data = get_stocks(country_code)
    if not stocks_data:
        meta_df = pd.DataFrame(columns=['ticker', '이름', '국가', '업종'])
    else:
        meta_df = pd.DataFrame(stocks_data)
        required_cols = ['ticker', 'name', 'country', 'sector']
        for col in required_cols:
            if col not in meta_df.columns:
                meta_df[col] = None
        
        meta_df = meta_df[required_cols]
        meta_df.rename(columns={'name': '이름', 'country': '국가', 'sector': '업종'}, inplace=True)

    # 2. 메타데이터 병합
    df_merged = pd.merge(df, meta_df, left_on='티커', right_on='ticker', how='left')
    
    # 메타데이터 병합 후 컬럼이 없을 경우를 대비
    if '이름' in df_merged.columns:
        df_merged['이름'] = df_merged['이름'].fillna(df_merged['티커'])
    else:
        df_merged['이름'] = df_merged['티커']

    # 국가 및 업종 필드 처리
    country_fill = '글로벌' if country_code == 'coin' else '-'
    if '국가' in df_merged.columns:
        df_merged['국가'] = df_merged['국가'].fillna(country_fill)
    else:
        df_merged['국가'] = country_fill

    sector_fill = '가상화폐' if country_code == 'coin' else '-'
    if '업종' in df_merged.columns:
        df_merged['업종'] = df_merged['업종'].fillna(sector_fill)
    else:
        df_merged['업종'] = sector_fill

    # 3. 컬럼 순서 재정렬
    # '이름' 컬럼은 아래 column_config에서 '종목명'으로 표시됩니다.
    final_cols = [
        '#', '티커', '이름', '상태', '매수일자', '보유일', '현재가', 
        '일간수익률', '보유수량', '금액', '누적수익률', '비중', '고점대비', '점수', '지속', '국가', '업종', '문구'
    ]
    
    existing_cols = [col for col in final_cols if col in df_merged.columns]
    df_display = df_merged[existing_cols].copy()

    # 숫자형으로 변환해야 하는 컬럼 목록
    numeric_cols = ["현재가", "일간수익률", "보유수량", "금액", "누적수익률", "비중", "점수"]
    for col in numeric_cols:
        if col in df_display.columns:
            # errors='coerce'를 사용하여 숫자로 변환할 수 없는 값은 NaN으로 만듭니다.
            df_display[col] = pd.to_numeric(df_display[col], errors='coerce')

    # 4. 스타일 적용 및 표시
    if "#" in df_display.columns:
        df_display = df_display.set_index("#")

    style_cols = ["일간수익률", "누적수익률"]
    styler = df_display.style
    for col in style_cols:
        if col in df_display.columns:
            styler = styler.map(style_returns, subset=[col])
    
    formats = {
        "일간수익률": "{:+.2f}%",
        "누적수익률": "{:+.2f}%",
        "비중": "{:.1f}%",
        "점수": "{:.2f}",
    }

    # 국가별로 통화 형식 지정
    if country_code in ["kor", "coin"]:
        formats["현재가"] = "{:,.0f}"
        formats["금액"] = "{:,.0f}"
    if country_code in ["aus"]:
        formats["현재가"] = "{:,.2f}"
        formats["금액"] = "{:,.2f}"

    # 코인은 보유수량을 소수점 8자리로 표시
    if country_code == "coin" and "보유수량" in df_display.columns:
        formats["보유수량"] = "{:.8f}"
    styler = styler.format(formats, na_rep="-")

    num_rows_to_display = min(len(df_display), 15)
    height = (num_rows_to_display + 1) * 35 + 3

    shares_format_str = "%.8f" if country_code == "coin" else "%d"

    st.dataframe(
        styler,
        width='stretch',
        height=height,
        column_config={
            "이름": st.column_config.TextColumn("종목명", width="medium"),
            "상태": st.column_config.TextColumn(width="small"),
            "보유일": st.column_config.TextColumn(width="small"),
            "보유수량": st.column_config.NumberColumn(format=shares_format_str),
            "일간수익률": st.column_config.TextColumn(width="small"),
            "누적수익률": st.column_config.TextColumn(width="small"),
            "비중": st.column_config.TextColumn(width="small"),
            "지속": st.column_config.TextColumn(width="small"),
            "국가": st.column_config.TextColumn(width="small"),
            "업종": st.column_config.TextColumn(width="medium"),
            "문구": st.column_config.TextColumn("문구", width="large"),
        }
    )

@st.dialog("BUY")
def show_buy_dialog(country_code: str):
    """매수(BUY) 거래 입력을 위한 모달 다이얼로그를 표시합니다."""
    
    currency_str = f" ({'AUD' if country_code == 'aus' else 'KRW'})"
    message_key = f"buy_message_{country_code}"

    def on_buy_submit():
        # st.session_state에서 폼 데이터 가져오기
        trade_date = st.session_state[f"buy_date_{country_code}"]
        ticker = st.session_state[f"buy_ticker_{country_code}"].strip()
        shares = st.session_state[f"buy_shares_{country_code}"]
        price = st.session_state[f"buy_price_{country_code}"]
        
        if not ticker or not shares > 0 or not price > 0:
            st.session_state[message_key] = (
                "error",
                "종목코드, 수량, 가격을 모두 올바르게 입력해주세요.",
            )
            return

        stock_name = ""
        if country_code == "kor" and _stock:
            # pykrx는 주식과 ETF의 이름 조회 함수가 다르므로,
            # 두 함수 모두 시도하여 유효한 문자열 결과를 찾습니다.
            try:
                # 1. ETF 이름 조회 시도
                name_candidate = _stock.get_etf_ticker_name(ticker)
                if isinstance(name_candidate, str) and name_candidate:
                    stock_name = name_candidate
            except Exception:
                pass  # 실패하면 다음으로 넘어감

            if not stock_name:
                try:
                    # 2. 주식 이름 조회 시도
                    name_candidate = _stock.get_market_ticker_name(ticker)
                    if isinstance(name_candidate, str) and name_candidate:
                        stock_name = name_candidate
                except Exception:
                    pass  # 최종 실패
        elif country_code == "aus":
            stock_name = fetch_yfinance_name(ticker)

        trade_data = {
            "country": country_code,
            "date": pd.to_datetime(trade_date).to_pydatetime(),
            "ticker": ticker.upper(),
            "name": stock_name,
            "action": "BUY",
            "shares": float(shares),
            "price": float(price),
            "note": "Manual input from web app"
        }
        
        if save_trade(trade_data):
            st.session_state[message_key] = ("success", "거래가 성공적으로 저장되었습니다.")
        else:
            st.session_state[message_key] = (
                "error",
                "거래 저장에 실패했습니다. 콘솔 로그를 확인해주세요.",
            )

    # 다이얼로그 내에서 오류 메시지만 표시합니다. 성공 메시지는 메인 화면에서 토스트로 표시됩니다.
    if message_key in st.session_state:
        msg_type, msg_text = st.session_state[message_key]
        if msg_type == "success":
            st.error(msg_text)
            # 오류 메시지는 한 번만 표시되도록 세션에서 제거합니다.
            del st.session_state[message_key]

    with st.form(f"trade_form_{country_code}"):
        st.date_input("거래일", value="today", key=f"buy_date_{country_code}")
        st.text_input("종목코드 (티커)", key=f"buy_ticker_{country_code}")
        shares_format_str = "%.8f" if country_code == "coin" else "%d"
        st.number_input("수량", min_value=0.00000001, step=0.00000001, format=shares_format_str, key=f"buy_shares_{country_code}")
        st.number_input(
            f"매수 단가{currency_str}", 
            min_value=0.0, 
            format="%.4f" if country_code == "aus" else ("%d" if country_code in ["kor", "coin"] else "%d"),
            key=f"buy_price_{country_code}"
        )
        st.form_submit_button("거래 저장", on_click=on_buy_submit)


@st.dialog("SELL", width="large")
def show_sell_dialog(country_code: str):
    """보유 종목 매도를 위한 모달 다이얼로그를 표시합니다."""
    currency_str = f" ({'AUD' if country_code == 'aus' else 'KRW'})"
    message_key = f"sell_message_{country_code}"

    from utils.data_loader import fetch_naver_realtime_price, fetch_ohlcv
    
    latest_date_str = get_available_snapshot_dates(country_code)[0] if get_available_snapshot_dates(country_code) else None
    if not latest_date_str:
        st.warning("보유 종목이 없어 매도할 수 없습니다.")
        return

    snapshot = get_portfolio_snapshot(country_code, date_str=latest_date_str)
    if not snapshot or not snapshot.get("holdings"):
        st.warning("보유 종목이 없어 매도할 수 없습니다.")
        return
        
    holdings = snapshot.get("holdings", [])
    
    holdings_with_prices = []
    with st.spinner("보유 종목의 현재가를 조회하는 중..."):
        for h in holdings:
            price = None
            if country_code == "kor":
                price = fetch_naver_realtime_price(h['ticker'])
                if not price:
                    df = fetch_ohlcv(h['ticker'], country='kor', months_back=1)
                    if df is not None and not df.empty:
                        if isinstance(df.columns, pd.MultiIndex):
                            df.columns = df.columns.get_level_values(0)
                            df = df.loc[:, ~df.columns.duplicated()]
                        price = df['Close'].iloc[-1]
            elif country_code == "aus":
                df = fetch_ohlcv(h['ticker'], country='aus', months_back=1)
                if df is not None and not df.empty:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                        df = df.loc[:, ~df.columns.duplicated()]
                    price = df['Close'].iloc[-1]
            
            # 불리언 평가 및 계산 전에 가격을 스칼라로 보장합니다.
            # 예: 중복 컬럼 등으로 인해 함수가 Series를 반환하는 경우를 처리합니다.
            price_val = price.item() if isinstance(price, pd.Series) else price

            if price_val and pd.notna(price_val):
                value = h['shares'] * price_val
                return_pct = (price_val / h['avg_cost'] - 1) * 100 if h.get('avg_cost', 0) > 0 else 0.0
                holdings_with_prices.append({
                    'ticker': h['ticker'], 'name': h['name'], 'shares': h['shares'],
                    'price': price_val, 'value': value, 'return_pct': return_pct
                })

    if not holdings_with_prices:
        st.error("보유 종목의 현재가를 조회할 수 없습니다.")
        return

    df_holdings = pd.DataFrame(holdings_with_prices)

    def on_sell_submit():
        # st.session_state에서 폼 데이터 가져오기
        sell_date = st.session_state[f"sell_date_{country_code}"]
        editor_state = st.session_state[f"sell_editor_{country_code}"]

        # data_editor에서 선택된 행의 인덱스를 찾습니다.
        selected_indices = [
            idx for idx, edit in editor_state.get("edited_rows", {}).items() 
            if edit.get("선택")
        ]

        if not selected_indices:
            st.session_state[message_key] = ("warning", "매도할 종목을 선택해주세요.")
            return

        selected_rows = df_holdings.loc[selected_indices]
        
        success_count = 0
        for _, row in selected_rows.iterrows():
            trade_data = {
                "country": country_code,
                "date": pd.to_datetime(sell_date).to_pydatetime(),
                "ticker": row['ticker'], "name": row['name'], "action": "SELL",
                "shares": row['shares'], "price": row['price'],
                "note": "Manual sell from web app"
            }
            if save_trade(trade_data):
                success_count += 1
        
        if success_count == len(selected_rows):
            st.session_state[message_key] = (
                "success",
                f"{success_count}개 종목의 매도 거래가 성공적으로 저장되었습니다.",
            )
        else:
            st.session_state[message_key] = ("error", "일부 거래 저장에 실패했습니다. 콘솔 로그를 확인해주세요.")

    # 다이얼로그 내에서 오류/경고 메시지만 표시합니다.
    if message_key in st.session_state:
        msg_type, msg_text = st.session_state[message_key]
        if msg_type == "success":
            if msg_type == "warning":
                st.warning(msg_text)
            else:
                st.error(msg_text)
            # 메시지는 한 번만 표시되도록 세션에서 제거합니다.
            del st.session_state[message_key]

    with st.form(f"sell_form_{country_code}"):
        st.subheader("매도할 종목을 선택하세요 (전체 매도)")
        st.date_input("매도일", value="today", key=f"sell_date_{country_code}")
        
        df_holdings['선택'] = False
        # 정렬이 필요한 컬럼은 숫자형으로 유지하고, column_config에서 포맷팅합니다.
        # 이렇게 하면 '평가금액' 등에서 문자열이 아닌 숫자 기준으로 올바르게 정렬됩니다.
        df_display = df_holdings[['선택', 'name', 'ticker', 'shares', 'return_pct', 'value', 'price']].copy()
        value_col_name = f'평가금액{currency_str}'
        price_col_name = f'현재가{currency_str}'
        df_display.rename(columns={
            'name': '종목명', 'ticker': '티커', 'shares': '보유수량', 'return_pct': '수익률', 
            'value': value_col_name, 'price': price_col_name
        }, inplace=True)
        
        st.data_editor(
            df_display, hide_index=True, width='stretch',
            key=f"sell_editor_{country_code}",
            disabled=['종목명', '티커', '보유수량', value_col_name, '수익률', price_col_name],
            column_config={
                "선택": st.column_config.CheckboxColumn("삭제", required=True),
                "보유수량": st.column_config.NumberColumn(format="%.8f"),
                "수익률": st.column_config.NumberColumn(format="%.2f%%"),
                value_col_name: st.column_config.NumberColumn(
                    # 쉼표(,)를 포맷에 추가하여 3자리마다 구분자를 표시합니다.
                    format="%,.0f" if country_code == 'kor' else "%,.2f"
                ),
                price_col_name: st.column_config.NumberColumn(
                    format="%.4f" if country_code == 'aus' else "%d"
                ),
            }
        )
        
        st.form_submit_button("선택 종목 매도", on_click=on_sell_submit)


@st.dialog("업종 추가")
def show_add_sector_dialog(country_code: str):
    """업종 추가를 위한 모달 다이얼로그"""
    with st.form(f"add_sector_form_{country_code}"):
        sector_name = st.text_input("업종명")
        added_date = st.date_input("추가일", value="today")
        submitted = st.form_submit_button("저장")

        if submitted:
            if not sector_name:
                st.error("업종명을 입력해주세요.")
                return

            with st.spinner("업종을 추가하는 중..."):
                current_sectors_data = get_sectors()
                current_sector_names = {s.get("name") for s in current_sectors_data}
                if sector_name in current_sector_names:
                    st.error(f"업종 '{sector_name}'은(는) 이미 존재합니다.")
                    return
                
                new_sector = {
                    "name": sector_name,
                    "added_date": pd.to_datetime(added_date).to_pydatetime()
                }
                current_sectors_data.append(new_sector)
                
                if save_sectors(current_sectors_data):
                    st.success(f"업종 '{sector_name}'이(가) 추가되었습니다.")
                    st.rerun()
                else:
                    st.error("업종 추가에 실패했습니다.")


@st.dialog("종목 추가")
def show_add_stock_dialog(country_code: str, sectors: List[Dict]):
    """종목 추가를 위한 모달 다이얼로그"""
    # no external validation required for coin tickers

    with st.form(f"add_stock_form_{country_code}"):
        ticker = st.text_input("티커")
        
        stock_name_input = ""
        market_country = ""
        stock_type = ""
        sector = ""

        if country_code == "coin":
            stock_name_input = st.text_input("종목명")
            market_country = "글로벌"
            stock_type = "crypto"
            sector = "가상화폐"
        else:
            country_options = [""] + SECTOR_COUNTRY_OPTIONS
            market_country = st.selectbox("소속 국가(시장)", options=country_options, index=0, format_func=lambda x: "선택 안 함" if x == "" else x)
            stock_type = st.selectbox("타입", options=["etf", "stock"])
            
            sector_names = sorted([s.get("name") for s in sectors if s.get("name")])
            sector_options = [""] + sector_names
            sector = st.selectbox("업종", options=sector_options, index=0, format_func=lambda x: "선택 안 함" if x == "" else x)
        
        submitted = st.form_submit_button("저장")

        if submitted:
            if not ticker:
                st.error("티커는 필수 항목입니다.")
                return

            stock_name = ""
            if country_code == "coin":
                if not stock_name_input:
                    st.error("종목명은 필수 항목입니다.")
                    return
                stock_name = stock_name_input
            else:
                if not stock_type:
                    st.error("타입은 필수 항목입니다.")
                    return

            with st.spinner("종목 정보를 조회하고 추가하는 중..."):
                if country_code != "coin":
                    # 코인이 아닌 경우에도 종목명을 조회합니다.
                    fetched_name = ""
                    if country_code == "kor" and _stock:
                        try:
                            name_candidate = _stock.get_etf_ticker_name(ticker)
                            if isinstance(name_candidate, str) and name_candidate:
                                fetched_name = name_candidate
                        except Exception: pass
                        if not fetched_name:
                            try:
                                name_candidate = _stock.get_market_ticker_name(ticker)
                                if isinstance(name_candidate, str) and name_candidate:
                                    fetched_name = name_candidate
                            except Exception: pass
                    elif country_code == "aus":
                        fetched_name = fetch_yfinance_name(ticker)
                    
                    if not fetched_name:
                        st.warning(f"티커 '{ticker.upper()}'의 종목명을 찾을 수 없습니다. 종목명 없이 추가됩니다.")
                    stock_name = fetched_name # 조회한 종목명을 사용

                current_stocks = get_stocks(country_code)
                if ticker.upper() in [s['ticker'] for s in current_stocks]:
                    st.error(f"티커 '{ticker.upper()}'는 현재 포트폴리오에 이미 존재합니다.")
                    return

                new_stock = {
                    "country": market_country,
                    "ticker": ticker.upper(),
                    "name": stock_name,
                    "type": stock_type,
                    "sector": sector
                }
                current_stocks.append(new_stock)
                
                if save_stocks(country_code, current_stocks):
                    success_msg = f"종목 '{ticker.upper()}'이(가) 추가되었습니다."
                    if stock_name:
                        success_msg = f"종목 '{ticker.upper()} ({stock_name})'이(가) 추가되었습니다."
                    st.success(success_msg)
                    st.rerun()
                else:
                    st.error("종목 추가에 실패했습니다.")


def render_master_stock_ui(country_code: str, sectors: List[Dict]):
    """종목 마스터 관리 UI를 렌더링합니다."""
    # from utils.data_loader import fetch_crypto_name # 사용자 요청에 따라 제거됨
    
    country_name = COUNTRY_CODE_MAP.get(country_code, "기타")

    if country_code == "coin":
        st.info("이곳에서 가상화폐 종목을 관리할 수 있습니다. 티커와 종목명만 직접 입력합니다.")
    else:
        st.info("이곳에서 투자 유니버스에 포함될 종목을 관리할 수 있습니다.")

    col1, col2, _ = st.columns([1.2, 1.8, 7.0])
    with col1:
        if st.button("종목 추가", key=f"add_stock_btn_{country_code}"):
            show_add_stock_dialog(country_code, sectors)
    
    if country_code != "coin":
        with col2:
            if st.button("종목명 일괄 업데이트", key=f"update_names_btn_{country_code}"):
                with st.spinner("모든 종목명을 업데이트하는 중..."):
                    stocks_to_update = get_stocks(country_code)
                    updated_count = 0
                    
                    for stock in stocks_to_update:
                        # 이름이 없거나, 티커와 이름이 같은 경우에 업데이트 시도
                        if not stock.get("name") or stock.get("name") == stock.get("ticker"):
                            new_name = ""
                            ticker = stock.get("ticker")
                            if not ticker:
                                continue

                            if country_code == "kor":
                                new_name = fetch_pykrx_name(ticker)
                            elif country_code == "aus":
                                new_name = fetch_yfinance_name(ticker)
                            
                            # 새로운 이름이 있고, 기존 이름과 다를 경우에만 업데이트
                            if new_name and new_name != stock.get("name"):
                                stock['name'] = new_name
                                updated_count += 1
                    
                    if updated_count > 0:
                        if save_stocks(country_code, stocks_to_update):
                            st.success(f"{updated_count}개 종목의 이름이 업데이트되었습니다.")
                            st.rerun()
                        else:
                            st.error("종목명 업데이트 저장에 실패했습니다.")
                    else:
                        st.info("업데이트할 종목명이 없습니다.")

    with st.spinner("종목 마스터 데이터를 불러오는 중..."):
        # 사용 가능한 업종 목록을 이름 리스트로 준비합니다.
        sector_names = sorted([s.get("name") for s in sectors if s.get("name")])
        sector_options = [""] + sector_names
        default_sector = ""

        stocks_data = get_stocks(country_code)
        if not stocks_data:
            st.info("관리할 종목이 없습니다. '종목 추가' 버튼을 눌러 종목을 추가해주세요.")
            return

        df_stocks = pd.DataFrame(stocks_data)

        # 데이터 정합성을 위한 처리
        if country_code == "coin":
            for col, default_val in [("country", "글로벌"), ("sector", "가상화폐"), ("type", "crypto")]:
                if col not in df_stocks.columns:
                    df_stocks[col] = default_val
                df_stocks[col] = df_stocks[col].fillna(default_val)
        else:
            if 'country' not in df_stocks.columns:
                df_stocks['country'] = ""
            df_stocks['country'] = df_stocks['country'].apply(lambda c: c if c in SECTOR_COUNTRY_OPTIONS else "")
            
            if 'sector' not in df_stocks.columns:
                df_stocks['sector'] = ""
            df_stocks['sector'] = df_stocks['sector'].apply(lambda s: s if s in sector_options else default_sector)

            if 'type' not in df_stocks.columns:
                df_stocks['type'] = ""

        # 'last_modified' 컬럼이 없는 구버전 데이터와의 호환성을 위해 추가
        if 'last_modified' not in df_stocks.columns:
            df_stocks['last_modified'] = pd.NaT


        # 정렬 로직: 1. 업종 미지정 우선, 2. 오래된 수정일자 우선
        df_stocks['sector_sort_key'] = df_stocks['sector'].apply(lambda s: 0 if s == default_sector else 1)
        df_stocks['modified_sort_key'] = pd.to_datetime(df_stocks['last_modified'], errors='coerce')

        df_stocks.sort_values(
            by=['sector_sort_key', 'modified_sort_key'],
            ascending=[True, True],
            na_position='first',  # 수정일자가 없는 가장 오래된 데이터부터 표시
            inplace=True
        )
        
        # 정렬에 사용된 임시 컬럼들을 삭제합니다.
        df_for_display = df_stocks.drop(columns=['sector_sort_key', 'modified_sort_key', 'last_modified'], errors='ignore')

        # 컬럼 순서 조정
        if country_code == "coin":
            display_cols = ['삭제', 'ticker', 'name', 'country', 'sector']
        else:
            display_cols = ['삭제', 'ticker', 'country', 'name', 'type', 'sector']
        df_for_display = df_stocks.reindex(columns=display_cols)
        df_for_display['삭제'] = False


        st.info("아래 표에서 종목 정보를 수정하거나, 삭제할 종목을 선택 후 '변경사항 저장' 버튼을 눌러주세요.")
        
        column_config_dict = {
            "삭제": st.column_config.CheckboxColumn("삭제", required=True),
            "ticker": st.column_config.TextColumn("티커", disabled=True),
            "name": st.column_config.TextColumn("종목명", disabled=False),
        }

        if country_code == "coin":
            column_config_dict.update({
                "country": st.column_config.TextColumn("국가", disabled=True),
                "sector": st.column_config.TextColumn("업종", disabled=True),
            })
        else:
            column_config_dict.update({
                "country": st.column_config.SelectboxColumn(
                    "국가",
                    options=[""] + SECTOR_COUNTRY_OPTIONS,
                    required=False,
                ),
                "type": st.column_config.TextColumn("타입", disabled=True),
                "sector": st.column_config.SelectboxColumn(
                    "업종",
                    options=sector_options,
                    required=False,
                ),
            })

        edited_df = st.data_editor(
            df_for_display,
            width='stretch',
            hide_index=True,
            key=f"stock_editor_{country_code}",
            column_config=column_config_dict,
        )

        if st.button("변경사항 저장", key=f"save_stock_changes_{country_code}"):
            with st.spinner("종목 정보를 업데이트하는 중..."):
                # 사용자가 수정한 행을 식별하여 타임스탬프를 업데이트합니다.
                editor_state = st.session_state[f"stock_editor_{country_code}"]
                edited_rows_indices = editor_state.get("edited_rows", {}).keys()
                
                if edited_rows_indices:
                    edited_tickers = df_stocks.iloc[list(edited_rows_indices)]['ticker'].tolist()
                else:
                    edited_tickers = []

                # 삭제로 표시된 행을 제외하고 저장할 데이터 준비
                stocks_to_keep_df = edited_df[~edited_df['삭제']]
                
                # 코인의 경우 'country', 'type', 'sector' 컬럼이 없을 수 있으므로 조건부 드롭
                cols_to_drop = ['삭제']
                if country_code == "coin":
                    # 코인 데이터프레임에는 이 컬럼들이 없을 수 있으므로, 존재하는 경우에만 드롭
                    cols_to_drop.extend([col for col in ['country', 'type', 'sector'] if col in stocks_to_keep_df.columns])
                
                stocks_to_save = stocks_to_keep_df.drop(columns=cols_to_drop, errors='ignore').to_dict('records')

                if save_stocks(country_code, stocks_to_save, edited_tickers):
                    st.success("변경사항이 성공적으로 저장되었습니다.")
                    st.rerun()
                else:
                    st.error("종목 정보 업데이트에 실패했습니다.")


def render_master_sector_ui():
    """전역 업종 마스터 관리 UI를 렌더링합니다."""
    st.info("이곳에서 모든 포트폴리오가 공통으로 사용하는 업종 리스트를 관리할 수 있습니다.")

    with st.spinner("업종 데이터를 불러오는 중..."):
        if st.button("업종 추가", key="add_sector_btn_global"):
            # show_add_sector_dialog는 country_code를 사용하지 않으므로 빈 문자열을 전달해도 무방합니다.
            show_add_sector_dialog("")

        sectors = get_sectors()
        if not sectors:
            # 초기 데이터 시딩
            initial_sectors = [
                "지수", "AI·반도체", "IT·소프트웨어", "금융", "고배당",
                "인컴·부동산", "에너지·원자재", "산업재·인프라", "의약·헬스케어",
                "소비재 (생활·필수재)", "농업·식량", "레저·서비스",
                "미디어·엔터테인먼트", "자동차·운송·물류", "친환경·신재생에너지",
                "방산·우주항공"
            ]
            sectors_to_save = [{"name": s, "added_date": datetime.now()} for s in initial_sectors]
            if save_sectors(sectors_to_save):
                st.success("초기 업종 데이터가 생성되었습니다.")
                st.rerun()
            else:
                st.error("초기 업종 데이터 생성에 실패했습니다.")

        # 데이터 에디터를 위한 데이터프레임 생성
        sector_counts_by_country = get_sector_stock_counts()
        data_for_df = []
        for s in sectors:
            sector_name = s.get("name")
            counts = sector_counts_by_country.get(sector_name, {})
            data_for_df.append({
                "업종명": sector_name,
                "종목 수(한국)": counts.get("kor", 0),
                "종목 수(호주)": counts.get("aus", 0),
                "추가된 날짜": pd.to_datetime(s.get("added_date")) if s.get("added_date") else pd.NaT
            })

        # 총 종목 수를 기준으로 정렬
        sorted_data = sorted(
            data_for_df,
            key=lambda x: (-(x["종목 수(한국)"] + x["종목 수(호주)"]), x["추가된 날짜"] if pd.notna(x["추가된 날짜"]) else pd.Timestamp.max)
        )

        for item in sorted_data:
            if pd.notna(item["추가된 날짜"]):
                item["추가된 날짜"] = item["추가된 날짜"].strftime('%Y-%m-%d')
            else:
                item["추가된 날짜"] = "-"

        df_for_display = pd.DataFrame(sorted_data)
        original_df = df_for_display.copy()

        df_for_display['삭제'] = False
        df_for_display = df_for_display[['삭제', '업종명', '종목 수(한국)', '종목 수(호주)', '추가된 날짜']]

        st.info("아래 표에서 '업종명'을 수정하거나, 삭제할 업종을 선택 후 버튼을 눌러주세요.")

        edited_df = st.data_editor(
            df_for_display,
            width='stretch',
            hide_index=True,
            key="sector_editor_global",
            column_config={
                "삭제": st.column_config.CheckboxColumn("삭제", required=True),
                "업종명": st.column_config.TextColumn("업종명", required=True),
                "종목 수(한국)": st.column_config.NumberColumn("종목 수(한국)", disabled=True),
                "종목 수(호주)": st.column_config.NumberColumn("종목 수(호주)", disabled=True),
                "추가된 날짜": st.column_config.TextColumn("추가된 날짜", disabled=True)
            }
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("업종명 변경사항 저장", key="save_sector_changes_global"):
                name_changes = {}
                final_sectors = []
                
                edited_names = [row['업종명'] for _, row in edited_df.iterrows()]
                if len(set(edited_names)) != len(edited_df):
                    st.error("업종명은 고유해야 합니다.")
                    st.stop()

                for index in original_df.index:
                    old_name = original_df.loc[index, '업종명']
                    edited_row = edited_df.loc[index]
                    new_name = edited_row['업종명']

                    if not new_name:
                        st.error("업종명은 비워둘 수 없습니다.")
                        st.stop()

                    if old_name != new_name:
                        name_changes[old_name] = new_name
                    
                    original_sector_data = next((s for s in data_for_df if s['업종명'] == old_name), None)
                    added_date = pd.to_datetime(original_sector_data['추가된 날짜']) if original_sector_data and original_sector_data['추가된 날짜'] != '-' else datetime.now()

                    final_sectors.append({
                        "name": new_name, "added_date": added_date
                    })

                if not name_changes and original_df.equals(edited_df.drop(columns=['삭제'])):
                    st.info("변경사항이 없습니다.")
                else:
                    with st.spinner("업종 정보를 업데이트하는 중..."):
                        if save_sector_changes(final_sectors, name_changes):
                            st.success("업종 정보가 성공적으로 업데이트되었습니다.")
                            st.rerun()
                        else:
                            st.error("업종 정보 업데이트에 실패했습니다.")
        
        with col2:
            if st.button("선택한 업종 삭제", key="delete_sector_btn_global", type="primary"):
                sectors_to_delete_df = edited_df[edited_df['삭제']]
                if sectors_to_delete_df.empty:
                    st.warning("삭제할 업종을 선택해주세요.")
                else:
                    names_to_delete = sectors_to_delete_df['업종명'].tolist()
                    
                    if "글로벌" in names_to_delete:
                        st.error("'글로벌' 업종은 삭제할 수 없습니다.")
                    else:
                        with st.spinner(f"{len(names_to_delete)}개 업종을 삭제하는 중..."):
                            if delete_sectors_and_reset_stocks(names_to_delete):
                                st.success(f"{len(names_to_delete)}개 업종이 성공적으로 삭제되었습니다.")
                                st.rerun()
                            else:
                                st.error("업종 삭제에 실패했습니다.")

def _display_success_toast(country_code: str):
    """
    세션 상태에서 성공 메시지를 확인하고 토스트로 표시합니다.
    주로 다이얼로그가 닫힌 후 피드백을 주기 위해 사용됩니다.
    """
    keys_to_check = [
        f"buy_message_{country_code}",
        f"sell_message_{country_code}",
    ]
    for key in keys_to_check:
        if key in st.session_state:
            message = st.session_state[key]
            # 메시지가 (type, text) 튜플이고, type이 'success'인 경우에만 처리
            if isinstance(message, tuple) and len(message) == 2 and message[0] == "success":
                _, msg_text = st.session_state.pop(key)
                st.toast(msg_text)


def render_country_tab(country_code: str):
    """지정된 국가에 대한 탭의 전체 UI를 렌더링합니다."""
    _display_success_toast(country_code)

    sub_tab_names = ["현황", "히스토리", "트레이드", "종목 관리", "설정"]
    sub_tab_status, sub_tab_history, sub_tab_trades, sub_tab_stock_management, sub_tab_settings = st.tabs(sub_tab_names)

    # --- 공통 데이터 로딩 ---
    sorted_dates = get_available_snapshot_dates(country_code)

    # 오늘/다음 거래일을 목록에 반영
    today = pd.Timestamp.now().normalize()
    today_str = today.strftime("%Y-%m-%d")
    if country_code != 'coin':
        # 한국/호주: 실제 거래일 캘린더로 오늘/다음 거래일을 판단 (실패 시 월~금 폴백)
        next_td_str_fallback = None
        try:
            lookahead_end = (today + pd.Timedelta(days=14)).strftime("%Y-%m-%d")
            upcoming_days = get_trading_days(today_str, lookahead_end, country_code)
            is_trading_today = any(d.date() == today.date() for d in upcoming_days)
            if is_trading_today:
                if today_str not in sorted_dates:
                    sorted_dates.insert(0, today_str)
            else:
                # 다음 거래일을 찾아 추가 (예: 토요일이면 다음 월요일)
                next_td = next((d for d in upcoming_days if d.date() >= today.date()), None)
                if next_td is not None:
                    next_td_str = pd.Timestamp(next_td).strftime("%Y-%m-%d")
                    if next_td_str not in sorted_dates:
                        sorted_dates.insert(0, next_td_str)
        except Exception:
            # 무시하고 폴백 계산 수행
            pass
        # 캘린더 조회 실패 또는 주말일 때의 폴백: 다음 월~금
        if today.weekday() >= 5:  # 토/일
            delta = 7 - today.weekday()
            next_bday = (today + pd.Timedelta(days=delta))
            next_td_str_fallback = next_bday.strftime('%Y-%m-%d')
            if next_td_str_fallback not in sorted_dates:
                sorted_dates.insert(0, next_td_str_fallback)

    # --- 1. 현황 탭 (최신 날짜) ---
    with sub_tab_status:
        if not sorted_dates:
            st.warning(f"[{country_code.upper()}] 국가의 포트폴리오 데이터를 DB에서 찾을 수 없습니다.")
            st.info("먼저 '거래 입력' 버튼을 통해 거래 내역을 추가해주세요.")
        else:
            latest_date_str = sorted_dates[0]

            col1, col2 = st.columns([1, 4])

            # 스피너 문구용 표시 날짜 계산 (주말/비거래일에는 다음 거래일을 표시)
            display_date_str = latest_date_str
            try:
                if country_code != 'coin':
                    today = pd.Timestamp.now().normalize()
                    today_str = today.strftime('%Y-%m-%d')
                    lookahead_end = (today + pd.Timedelta(days=14)).strftime('%Y-%m-%d')
                    upcoming_days = get_trading_days(today_str, lookahead_end, country_code)
                    # 다음 거래일 (오늘 포함)
                    next_td = next((d for d in upcoming_days if d.date() >= today.date()), None)
                    if next_td is not None:
                        next_td_str = pd.Timestamp(next_td).strftime('%Y-%m-%d')
                        if pd.to_datetime(latest_date_str) < pd.to_datetime(next_td_str):
                            display_date_str = next_td_str
                    else:
                        # 폴백: 주말이면 다음 월요일
                        if today.weekday() >= 5:
                            delta = 7 - today.weekday()
                            next_bday = (today + pd.Timedelta(days=delta))
                            next_bday_str = next_bday.strftime('%Y-%m-%d')
                            if pd.to_datetime(latest_date_str) < pd.to_datetime(next_bday_str):
                                display_date_str = next_bday_str
            except Exception:
                # 폴백: 주말이면 다음 월요일
                if country_code != 'coin' and today.weekday() >= 5:
                    delta = 7 - today.weekday()
                    next_bday = (today + pd.Timedelta(days=delta))
                    next_bday_str = next_bday.strftime('%Y-%m-%d')
                    if pd.to_datetime(latest_date_str) < pd.to_datetime(next_bday_str):
                        display_date_str = next_bday_str

            # 백그라운드 재계산 지원
            target_date_str = display_date_str

            # 재계산 버튼: 백그라운드 스레드에서 실행하여 UI 블로킹을 피합니다.
            recalc_key = f"{country_code}_{target_date_str}"
            if st.button("다시 계산", key=f"recalc_status_{recalc_key}"):
                if "recalc_running" not in st.session_state:
                    st.session_state["recalc_running"] = {}
                if "recalc_started_at" not in st.session_state:
                    st.session_state["recalc_started_at"] = {}
                if not st.session_state["recalc_running"].get(recalc_key):
                    st.session_state["recalc_running"][recalc_key] = True
                    st.session_state["recalc_started_at"][recalc_key] = datetime.now().isoformat()

                def _recalc_task():
                    # 동기화는 스케줄러에서 수행. 여기서는 현황만 재계산/저장합니다.
                    get_cached_status_report(country=country_code, date_str=target_date_str, force_recalculate=True)

                threading.Thread(target=_recalc_task, daemon=True).start()

            # 재계산 상태 안내 및 자동 업데이트
            running = (st.session_state.get("recalc_running", {}).get(recalc_key) is True)
            if running:
                st.info(f"'{target_date_str}' 기준 현황을 백그라운드에서 다시 계산 중입니다. 다른 탭을 사용해도 됩니다.")
                # DB에 저장이 끝났는지 확인하고, 끝났으면 페이지를 새로고침합니다.
                report_ready = get_status_report_from_db(country_code, pd.to_datetime(target_date_str).to_pydatetime())
                if report_ready:
                    st.session_state["recalc_running"][recalc_key] = False
                    st.success("재계산이 완료되었습니다. 화면을 갱신합니다.")
                    st.rerun()
                # 타임아웃/지연 대비 수동 초기화 옵션
                started_at_iso = st.session_state.get("recalc_started_at", {}).get(recalc_key)
                if started_at_iso:
                    try:
                        elapsed_sec = (datetime.now() - pd.to_datetime(started_at_iso).to_pydatetime()).total_seconds()
                        if elapsed_sec > 180:
                            st.warning("재계산이 지연되는 것 같습니다. 아래 버튼으로 상태를 초기화할 수 있습니다.")
                            if st.button("재계산 상태 초기화", key=f"reset_recalc_{recalc_key}"):
                                st.session_state["recalc_running"][recalc_key] = False
                                st.rerun()
                    except Exception:
                        pass

            # 캐시된(또는 계산 완료된) 결과를 표시
            result = get_cached_status_report(
                country=country_code, date_str=target_date_str, force_recalculate=False
            )
            if result:
                header_line, headers, rows = result
                # 헤더(요약)과 경고를 분리합니다. '<br>' 이전은 요약, 이후는 경고 영역입니다.
                header_main = header_line
                warning_html = None
                if isinstance(header_line, str) and "<br>" in header_line:
                    parts = header_line.split("<br>", 1)
                    header_main = parts[0]
                    warning_html = parts[1]

                # 테이블 상단에 요약 헤더를 표시합니다.
                if header_main:
                    st.markdown(f":information_source: {header_main}", unsafe_allow_html=True)

                # 데이터와 헤더의 컬럼 수가 일치하는지 확인하여 앱 충돌 방지
                if rows and headers and len(rows[0]) != len(headers):
                    st.error(f"데이터 형식 오류: 현황 리포트의 컬럼 수({len(headers)})와 데이터 수({len(rows[0])})가 일치하지 않습니다. '다시 계산'을 시도해주세요.")
                    st.write("- 헤더:", headers)
                    st.write("- 첫 번째 행 데이터:", rows[0])
                else:
                    df = pd.DataFrame(rows, columns=headers)
                    _display_status_report_df(df, country_code)
                    # 테이블 아래에 경고(평가금액 대체 안내)가 있으면 표시합니다.
                    if warning_html:
                        st.markdown(warning_html, unsafe_allow_html=True)
                    # 테이블 아래에 벤치마크 대비 초과성과를 표시합니다.
                    benchmark_str = get_cached_benchmark_status(country_code)
                    if benchmark_str:
                        st.markdown(
                            f'<div style="text-align: left; padding-top: 0.5rem;">{benchmark_str}</div>', unsafe_allow_html=True
                        )
            else:
                st.error(f"'{latest_date_str}' 기준 ({country_code.upper()}) 현황을 생성하는 데 실패했습니다.")

    with sub_tab_history:
        history_sub_tab_names = ["현황", "평가금액"]
        history_status_tab, history_equity_tab = st.tabs(history_sub_tab_names)

        with history_status_tab:
            # Drop today from history explicitly
            past_dates = [d for d in sorted_dates if d != pd.Timestamp.now().strftime('%Y-%m-%d')]
            # 코인: 시작일부터 어제까지 모든 날짜로 탭을 생성하고, 중복을 제거합니다.
            if country_code == "coin" and past_dates:
                try:
                    # Apply INITIAL_DATE floor
                    coin_settings = get_app_settings(country_code) or {}
                    initial_dt = None
                    if coin_settings.get("initial_date"):
                        try:
                            initial_dt = pd.to_datetime(coin_settings.get("initial_date")).normalize()
                        except Exception:
                            initial_dt = None
                    oldest = pd.to_datetime(past_dates[-1]).normalize()
                    start_dt = max(oldest, initial_dt) if initial_dt is not None else oldest
                    yesterday = (pd.Timestamp.now().normalize() - pd.Timedelta(days=1))
                    full_range = pd.date_range(start=start_dt, end=yesterday, freq='D')
                    # 최신이 먼저 오도록 내림차순 정렬 후 문자열로 변환
                    past_dates = [d.strftime('%Y-%m-%d') for d in full_range[::-1]]
                except Exception:
                    # 폴백: 중복 제거만 수행
                    seen = set()
                    uniq = []
                    # Also filter below INITIAL_DATE in fallback path
                    init_str = None
                    try:
                        if coin_settings.get("initial_date"):
                            init_str = pd.to_datetime(coin_settings.get("initial_date")).strftime('%Y-%m-%d')
                    except Exception:
                        init_str = None
                    for d in past_dates:
                        if init_str and d < init_str:
                            continue
                        if d not in seen:
                            seen.add(d)
                            uniq.append(d)
                    past_dates = uniq
            # 한국/호주: 히스토리의 첫 탭은 항상 '마지막 거래일'이 되도록 보정합니다.
            if country_code in ("kor", "aus") and past_dates:
                try:
                    today = pd.Timestamp.now().normalize()
                    lookback_start = (today - pd.Timedelta(days=21)).strftime('%Y-%m-%d')
                    today_str = today.strftime('%Y-%m-%d')
                    trading_days = get_trading_days(lookback_start, today_str, country_code)
                    if trading_days:
                        last_td = max(d for d in trading_days if d.date() <= today.date())
                        last_td_str = pd.Timestamp(last_td).strftime('%Y-%m-%d')
                        # 히스토리 탭 목록 맨 앞에 마지막 거래일이 오도록 정렬 보정
                        if last_td_str in past_dates:
                            past_dates = [last_td_str] + [d for d in past_dates if d != last_td_str]
                        else:
                            past_dates = [last_td_str] + past_dates
                except Exception:
                    pass
            if not past_dates:
                st.info("과거 현황 데이터가 없습니다.")
            else:
                # 코인 탭에서는 '과거 전체 다시계산' 기능을 제공하지 않습니다.
                if country_code != 'coin' and st.button("과거 전체 다시계산", key=f"recalc_all_hist_{country_code}"):
                    # 1. 재계산에 필요한 모든 종목과 전체 기간을 결정합니다.
                    stocks_from_db = get_stocks(country_code)
                    tickers = [s['ticker'] for s in stocks_from_db]
                    
                    oldest_date = pd.to_datetime(past_dates[-1])
                    newest_date = pd.to_datetime(past_dates[0])
                    
                    # 웜업 기간 계산에 필요한 파라미터를 DB에서 읽어옵니다.
                    app_settings_db = get_app_settings(country_code)
                    common_settings = get_common_settings()
                    if (not app_settings_db
                        or "ma_period_etf" not in app_settings_db
                        or "ma_period_stock" not in app_settings_db):
                        st.error("오류: DB에 MA 기간 설정이 없습니다. 각 국가 탭의 '설정'에서 값을 저장해주세요.")
                        return
                    if (not common_settings
                        or "ATR_PERIOD_FOR_NORMALIZATION" not in common_settings):
                        st.error("오류: DB 공통 설정에 ATR 기간이 없습니다. '설정' 탭의 공통 설정에서 값을 저장해주세요.")
                        return
                    try:
                        max_ma_period = max(int(app_settings_db["ma_period_etf"]), int(app_settings_db["ma_period_stock"]))
                        atr_period_norm = int(common_settings["ATR_PERIOD_FOR_NORMALIZATION"])
                    except (ValueError, TypeError):
                        st.error("오류: DB 설정값 형식이 올바르지 않습니다. 숫자 여부를 확인해주세요.")
                        return
                    warmup_days = int(max(max_ma_period, atr_period_norm) * 1.5)

                    # 2. 모든 기간의 데이터를 한 번에 병렬로 가져옵니다.
                    prefetched_data = fetch_ohlcv_for_tickers(
                        tickers, 
                        country=country_code, 
                        date_range=[oldest_date.strftime('%Y-%m-%d'), newest_date.strftime('%Y-%m-%d')],
                        warmup_days=warmup_days
                    )

                    # 3. 미리 가져온 데이터를 사용하여 각 날짜를 순차적으로 재계산합니다.
                    progress_text = "과거 현황 데이터를 다시 계산하는 중..."
                    progress_bar = st.progress(0, text=progress_text)
                    total_dates = len(past_dates)
                    for i, date_str in enumerate(past_dates):
                        get_cached_status_report(
                            country=country_code, date_str=date_str, force_recalculate=True, prefetched_data=prefetched_data
                        )
                        progress_bar.progress((i + 1) / total_dates, text=f"{progress_text} ({i+1}/{total_dates})")
                    progress_bar.empty()
                    st.success("모든 과거 현황 데이터가 다시 계산되었습니다.")
                    st.rerun()

                history_date_tabs = st.tabs(past_dates)
                for i, date_str in enumerate(past_dates):
                    with history_date_tabs[i]:
                        # 과거 데이터는 기본적으로 DB에서 표시. 헤더의 기준일이 탭 날짜와 불일치하면 자동 재계산하여 교정.
                        want_date = pd.to_datetime(date_str).to_pydatetime()
                        report_from_db = get_status_report_from_db(country_code, want_date)
                        needs_recalc = False
                        if report_from_db:
                            header_line = str(report_from_db.get("header_line") or "")
                            # 기대 접두부: "기준일: YYYY-MM-DD(" (요일은 다를 수 있으므로 괄호 전까지만 비교)
                            expected_prefix = f"기준일: {date_str}("
                            if not header_line.startswith(expected_prefix):
                                needs_recalc = True
                        if not report_from_db or needs_recalc:
                            # 캐시 불일치 또는 헤더 교정 필요: 재계산 후 저장/표시
                            new_report = get_cached_status_report(country_code, date_str, force_recalculate=True)
                            if new_report:
                                header_line, headers, rows = new_report
                                st.markdown(f":information_source: {header_line}", unsafe_allow_html=True)
                                if rows and headers and len(rows[0]) == len(headers):
                                    df = pd.DataFrame(rows, columns=headers)
                                    _display_status_report_df(df, country_code)
                                else:
                                    st.error("재계산된 데이터 형식이 올바르지 않습니다.")
                            else:
                                st.info(f"'{date_str}' 기준 현황 데이터를 생성할 수 없습니다.")
                        else:
                            headers = report_from_db.get("headers")
                            rows = report_from_db.get("rows")
                            st.markdown(f":information_source: {report_from_db.get('header_line')}", unsafe_allow_html=True)
                            if rows and headers and len(rows[0]) != len(headers):
                                if country_code != 'coin':
                                    st.error(f"데이터 형식 오류: 현황 리포트의 컬럼 수({len(headers)})와 데이터 수({len(rows[0])})가 일치하지 않습니다. '과거 전체 다시계산'을 시도해주세요.")
                                else:
                                    st.error(f"데이터 형식 오류: 현황 리포트의 컬럼 수({len(headers)})와 데이터 수({len(rows[0])})가 일치하지 않습니다. 해당 날짜를 '다시 계산'해 주세요.")
                                st.write("- 헤더:", headers)
                                st.write("- 첫 번째 행 데이터:", rows[0])
                            else:
                                df = pd.DataFrame(rows, columns=headers)
                                _display_status_report_df(df, country_code)
                        # 수동 재계산 버튼
                        if st.button("이 날짜 다시 계산하기", key=f"recalc_hist_{country_code}_{date_str}_{i}"):
                            new_report = get_cached_status_report(country_code, date_str, force_recalculate=True)
                            if new_report:
                                st.success("재계산 완료")
                                st.rerun()
                                with st.spinner(f"'{date_str}' 기준 현황 데이터를 계산/저장 중..."):
                                    calc_result = get_cached_status_report(
                                        country=country_code, date_str=date_str, force_recalculate=True
                                    )
                                if calc_result:
                                    st.success("계산/저장 완료!")
                                    st.rerun()
        
        with history_equity_tab:
            app_settings = get_app_settings(country_code)
            initial_date = (app_settings.get("initial_date") if app_settings else None) or (datetime.now() - pd.DateOffset(months=3))

            currency_str = f" ({'AUD' if country_code == 'aus' else 'KRW'})"

            start_date_str = initial_date.strftime("%Y-%m-%d")
            end_date_str = datetime.now().strftime("%Y-%m-%d")

            with st.spinner("거래일 및 평가금액 데이터를 불러오는 중..."):
                all_trading_days = get_trading_days(start_date_str, end_date_str, country_code)
                if not all_trading_days and country_code == 'kor': # 호주는 yfinance가 주말을 건너뛰므로 거래일 조회가 필수는 아님
                    st.warning("거래일을 조회할 수 없습니다.")
                else:
                    start_dt_obj = pd.to_datetime(start_date_str).to_pydatetime()
                    end_dt_obj = pd.to_datetime(end_date_str).to_pydatetime()
                    existing_equities = get_all_daily_equities(country_code, start_dt_obj, end_dt_obj)
                    equity_data_map = {pd.to_datetime(e['date']).normalize(): e for e in existing_equities}

                    # 거래일 조회가 실패한 경우(예: 호주), DB에 있는 날짜만 사용
                    if not all_trading_days:
                        all_trading_days = sorted(list(equity_data_map.keys()))

                    data_for_editor = []
                    for trade_date in all_trading_days:
                        existing_data = equity_data_map.get(trade_date, {})
                        row = {"date": trade_date, "total_equity": existing_data.get("total_equity", 0.0)}
                        if country_code == "aus":
                            is_data = existing_data.get("international_shares", {})
                            row["is_value"] = is_data.get("value", 0.0)
                            row["is_change_pct"] = is_data.get("change_pct", 0.0)
                        data_for_editor.append(row)

                    df_to_edit = pd.DataFrame(data_for_editor)

                    column_config = {
                        "date": st.column_config.DateColumn("일자", format="YYYY-MM-DD", disabled=True),
                        "total_equity": st.column_config.NumberColumn(f"총 평가금액{currency_str}", format="%.2f" if country_code == "aus" else "%d", required=True),
                    }
                    if country_code == "aus":
                        column_config["is_value"] = st.column_config.NumberColumn(f"해외주식 평가액{currency_str}", format="%.2f")
                        column_config["is_change_pct"] = st.column_config.NumberColumn("해외주식 수익률(%)", format="%.2f", help="수익률(%)만 입력합니다. 예: 5.5")

                    st.info("총 평가금액을 수정한 후 아래 '저장하기' 버튼을 눌러주세요.")
                    
                    edited_df = st.data_editor(df_to_edit, key=f"equity_editor_{country_code}", width='stretch', hide_index=True, column_config=column_config)

                    if st.button("평가금액 저장하기", key=f"save_all_equities_{country_code}"):
                        with st.spinner("변경된 평가금액을 저장하는 중..."):
                            saved_count = 0
                            for _, row in edited_df.iterrows():
                                date_to_save = row['date'].to_pydatetime()
                                equity_to_save = row['total_equity']
                                is_data_to_save = None
                                if country_code == 'aus':
                                    is_data_to_save = {"value": row['is_value'], "change_pct": row['is_change_pct']}
                                
                                if save_daily_equity(country_code, date_to_save, equity_to_save, is_data_to_save):
                                    saved_count += 1
                            
                            st.success(f"{saved_count}개 날짜의 평가금액을 저장/업데이트했습니다.")
                            st.rerun()

    with sub_tab_trades:
        # 코인 탭: 거래 입력 대신 보유 현황/데이터 편집만 제공 (동기화 버튼 제거)
        if country_code == 'coin':
            pass

        if country_code != 'coin':
            col1, col2, _ = st.columns([1, 1, 8])
            with col1:
                if st.button("BUY", key=f"add_buy_btn_{country_code}"):
                    show_buy_dialog(country_code)
            with col2:
                if st.button("SELL", key=f"add_sell_btn_{country_code}"):
                    show_sell_dialog(country_code)
        
        all_trades = get_all_trades(country_code)
        if not all_trades:
            st.info("거래 내역이 없습니다.")
        else:
            df_trades = pd.DataFrame(all_trades)
            # 코인 전용: 티커 필터(ALL 포함)
            if country_code == 'coin' and 'ticker' in df_trades.columns:
                unique_tickers = sorted({str(t).upper() for t in df_trades['ticker'].dropna().tolist()})
                options = ['ALL'] + unique_tickers
                selected = st.selectbox(
                    '티커 필터', options, index=0, key=f"coin_trades_filter_{country_code}"
                )
                if selected != 'ALL':
                    df_trades = df_trades[df_trades['ticker'].str.upper() == selected]
            
            # 금액(수량*가격) 계산: 정수, 천단위 콤마
            try:
                amt = (pd.to_numeric(df_trades.get('shares'), errors='coerce').fillna(0.0) *
                       pd.to_numeric(df_trades.get('price'), errors='coerce').fillna(0.0))
                df_trades['amount'] = amt.round(0).astype('Int64').fillna(0).astype(object).apply(
                    lambda x: f"{int(x):,}" if pd.notna(x) else "0"
                )
            except Exception:
                df_trades['amount'] = "0"

            # 삭제 선택을 위한 컬럼 추가
            df_trades['delete'] = False
            
            # 표시할 컬럼 순서 정의
            # 기록시간 대신 거래시간(빗썸 시간, 'date')을 우선 표시합니다.
            cols_to_show = ['delete', 'date', 'action', 'ticker', 'name', 'shares', 'price', 'amount', 'note', 'id']
            # reindex를 사용하여 이전 데이터에 'created_at'이 없어도 오류가 발생하지 않도록 합니다.
            df_display = df_trades.reindex(columns=cols_to_show).copy()
            
            # 날짜 및 시간 포맷팅
            df_display['date'] = pd.to_datetime(df_display['date'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')

            edited_df = st.data_editor(
                df_display,
                key=f"trades_editor_{country_code}",
                hide_index=True,
                width='stretch',
                column_config={
                    "delete": st.column_config.CheckboxColumn("삭제", required=True),
                    "id": None, # ID 컬럼은 숨김
                    "date": st.column_config.TextColumn("거래시간"),
                    "action": st.column_config.TextColumn("종류"),
                    "ticker": st.column_config.TextColumn("티커"),
                    "name": st.column_config.TextColumn("종목명", width="medium"),
                    "shares": st.column_config.NumberColumn("수량", format="%.8f" if country_code in ["coin"] else "%.0f"),
                    "price": st.column_config.NumberColumn("가격", format="%.4f" if country_code == "aus" else "%d"),
                    "amount": st.column_config.NumberColumn("금액", format="%.0f"),
                    "note": st.column_config.TextColumn("비고", width="large"),
                },
                disabled=['date', 'action', 'ticker', 'name', 'shares', 'price', 'amount', 'note']
            )

            if st.button("선택한 거래 삭제", key=f"delete_trade_btn_{country_code}", type="primary"):
                trades_to_delete = edited_df[edited_df['delete']]
                if not trades_to_delete.empty:
                    with st.spinner(f"{len(trades_to_delete)}개의 거래를 삭제하는 중..."):
                        deleted_count = 0
                        for trade_id in trades_to_delete['id']:
                            if delete_trade_by_id(trade_id):
                                deleted_count += 1
                        
                        st.success(f"{deleted_count}개의 거래를 성공적으로 삭제했습니다.")
                        st.rerun()
                else:
                    st.warning("삭제할 거래를 선택해주세요.")

    with sub_tab_stock_management:
        with st.spinner("종목 마스터 데이터를 불러오는 중..."):
            available_sectors = get_sectors()
            if not available_sectors:
                st.warning("업종이 등록되지 않았습니다. '마스터 정보' 탭에서 먼저 업종을 추가해주세요.")
            else:
                render_master_stock_ui(country_code, available_sectors)
                    
    with sub_tab_settings:
        # 1. DB에서 현재 설정값 로드
        db_settings = get_app_settings(country_code)
        current_capital = db_settings.get("initial_capital", 0) if db_settings else 0
        current_topn = db_settings.get("portfolio_topn") if db_settings else None
        current_ma_etf = db_settings.get("ma_period_etf") if db_settings else None
        current_ma_stock = db_settings.get("ma_period_stock") if db_settings else None
        current_replace_threshold = db_settings.get("replace_threshold") if db_settings else None
        current_replace_weaker = db_settings.get("replace_weaker_stock") if db_settings else None
        current_max_replacements = db_settings.get("max_replacements_per_day") if db_settings else None

        # TEST_MONTHS_RANGE를 기반으로 기본 날짜를 계산합니다. (필수 설정)
        test_months_range = TEST_MONTHS_RANGE
        default_date = pd.Timestamp.now() - pd.DateOffset(months=test_months_range)
        current_date = db_settings.get("initial_date", default_date) if db_settings else default_date

        with st.form(key=f"settings_form_{country_code}"):
            currency_str = f" ({'AUD' if country_code == 'aus' else 'KRW'})"

            new_capital = st.number_input(
                f"초기 자본금 (INITIAL_CAPITAL){currency_str}",
                value=float(current_capital) if country_code == "aus" else int(current_capital),
                format="%.2f" if country_code == "aus" else "%d",
                help="포트폴리오의 시작 자본금을 설정합니다. 누적 수익률 계산의 기준이 됩니다."
            )

            new_date = st.date_input(
                "초기 자본 기준일 (INITIAL_DATE)",
                value=current_date,
                help="초기 자본금이 투입된 날짜를 설정합니다."
            )

            if current_topn is None:
                st.warning("최대 보유 종목 수(PORTFOLIO_TOPN)를 설정해주세요.")

            new_topn_str = st.text_input(
                "최대 보유 종목 수 (PORTFOLIO_TOPN)",
                value=str(current_topn) if current_topn is not None else "",
                placeholder="예: 10",
                help="포트폴리오에서 최대로 보유할 종목의 개수를 설정합니다."
            )

            st.markdown("---")
            st.subheader("전략 파라미터")

            if current_ma_etf is None:
                st.warning("ETF 이동평균 기간(MA_PERIOD_FOR_ETF)을 설정해주세요.")
            new_ma_etf_str = st.text_input(
                "ETF 이동평균 기간 (MA_PERIOD_FOR_ETF)",
                value=str(current_ma_etf) if current_ma_etf is not None else "15",
                placeholder="예: 15",
                help="ETF 종목의 추세 판단에 사용될 이동평균 기간입니다."
            )

            if current_ma_stock is None:
                st.warning("개별종목 이동평균 기간(MA_PERIOD_FOR_STOCK)을 설정해주세요.")
            new_ma_stock_str = st.text_input(
                "개별종목 이동평균 기간 (MA_PERIOD_FOR_STOCK)",
                value=str(current_ma_stock) if current_ma_stock is not None else "75",
                placeholder="예: 75",
                help="개별 주식/코인 종목의 추세 판단에 사용될 이동평균 기간입니다."
            )

            # 교체 매매 사용 여부 (bool)
            replace_weaker_checkbox = st.checkbox(
                "교체 매매 사용 (REPLACE_WEAKER_STOCK)",
                value=bool(current_replace_weaker) if current_replace_weaker is not None else False,
                help="포트폴리오가 가득 찼을 때, 더 강한 후보가 있을 경우 약한 보유종목을 교체할지 여부"
            )

            # 하루 최대 교체 수
            max_replacements_str = st.text_input(
                "하루 최대 교체 수 (MAX_REPLACEMENTS_PER_DAY)",
                value=str(current_max_replacements) if current_max_replacements is not None else "",
                placeholder="예: 5",
                help="하루에 실행할 수 있는 교체 매매의 최대 종목 수"
            )

            # 교체 매매 임계값 설정 (DB에서 관리)
            new_replace_threshold_str = st.text_input(
                "교체 매매 점수 임계값 (REPLACE_SCORE_THRESHOLD)",
                value=("{:.2f}".format(float(current_replace_threshold)) if current_replace_threshold is not None else ""),
                placeholder="예: 1.5",
                help="교체 매매 실행 조건: 새 후보 점수가 기존 보유 점수보다 이 값만큼 높을 때 교체.",
            )

            # 코인 전용 임포트 기간 설정 제거됨 (트레이드 동기화 폐지)

            save_settings_submitted = st.form_submit_button("설정 저장하기")

            if save_settings_submitted:
                error = False
                if not new_topn_str or not new_topn_str.isdigit() or int(new_topn_str) < 1:
                    st.error("최대 보유 종목 수는 1 이상의 숫자여야 합니다.")
                    error = True
                if not new_ma_etf_str or not new_ma_etf_str.isdigit() or int(new_ma_etf_str) < 1:
                    st.error("ETF 이동평균 기간은 1 이상의 숫자여야 합니다.")
                    error = True
                if not new_ma_stock_str or not new_ma_stock_str.isdigit() or int(new_ma_stock_str) < 1:
                    st.error("개별종목 이동평균 기간은 1 이상의 숫자여야 합니다.")
                    error = True
                # max_replacements_per_day 검증 (정수 >= 0)
                if not max_replacements_str or not max_replacements_str.isdigit() or int(max_replacements_str) < 0:
                    st.error("하루 최대 교체 수는 0 이상의 정수여야 합니다.")
                    error = True
                # replace_threshold 검증 (float 가능 여부)
                try:
                    _ = float(new_replace_threshold_str)
                except Exception:
                    st.error("교체 매매 점수 임계값은 숫자여야 합니다.")
                    error = True
                
                if not error:
                    new_topn = int(new_topn_str)
                    new_ma_etf = int(new_ma_etf_str)
                    new_ma_stock = int(new_ma_stock_str)
                    new_max_replacements = int(max_replacements_str)
                    new_replace_threshold = float(new_replace_threshold_str)
                    settings_to_save = {
                        "country": country_code,
                        "initial_capital": new_capital,
                        "initial_date": pd.to_datetime(new_date).to_pydatetime(),
                        "portfolio_topn": new_topn,
                        "ma_period_etf": new_ma_etf,
                        "ma_period_stock": new_ma_stock,
                        "replace_weaker_stock": bool(replace_weaker_checkbox),
                        "max_replacements_per_day": new_max_replacements,
                        "replace_threshold": new_replace_threshold,
                    }
                    # 코인용 빗썸 임포트 기간 설정은 더 이상 사용하지 않습니다.
                    if save_app_settings(country_code, settings_to_save):
                        st.success("설정이 성공적으로 저장되었습니다.")
                        st.rerun()
                    else:
                        st.error("설정 저장에 실패했습니다.")


def main():
    """MomentumPilot 오늘의 현황 웹 UI를 렌더링합니다."""
    st.set_page_config(page_title="MomentumPilot Status", layout="wide")

    # 페이지 상단 여백을 줄이기 위한 CSS 주입
    st.markdown("""
        <style>
            .block-container {
                padding-top: 1rem;
            }
        </style>
    """, unsafe_allow_html=True)

    if not check_password():
        st.stop()  # 비밀번호가 맞지 않으면 앱의 나머지 부분을 렌더링하지 않습니다.

    # 제목과 시장 상태를 한 줄에 표시
    # "최근 중단" 기간이 길어지면서 줄바꿈되는 현상을 방지하기 위해
    # 오른쪽 컬럼의 너비를 늘립니다. (3:1 -> 2.5:1.5)
    col1, col2 = st.columns([2.5, 1.5])
    with col1:
        st.title("Momentum. Pilot.")
    with col2:
        # 시장 상태는 한 번만 계산하여 10분간 캐시합니다.
        @st.cache_data(ttl=600)
        def _get_cached_market_status():
            return get_market_regime_status_string()

        market_status_str = _get_cached_market_status()
        if market_status_str:
            # st.markdown을 사용하여 오른쪽 정렬 및 상단 패딩을 적용합니다.
            st.markdown(
                f'<div style="text-align: right; padding-top: 1.5rem; font-size: 1.1rem;">{market_status_str}</div>', unsafe_allow_html=True
            )

    tab_names = ["코인", "한국", "호주", "마스터", "설정", "텔레그램"]
    tab_coin, tab_kor, tab_aus, tab_master, tab_settings, tab_telegram = st.tabs(tab_names)

    with tab_coin:
        render_country_tab("coin")
        
    with tab_kor:
        render_country_tab("kor")

    with tab_aus:
        render_country_tab("aus")


    with tab_master:
        sector_management_tab, = st.tabs(["업종"])
        with sector_management_tab:
            render_master_sector_ui()

    with tab_settings:
        st.header("공통 설정 (모든 국가 공유)")
        common = get_common_settings() or {}
        current_enabled = bool(common.get("MARKET_REGIME_FILTER_ENABLED")) if "MARKET_REGIME_FILTER_ENABLED" in common else False
        current_ticker = common.get("MARKET_REGIME_FILTER_TICKER")
        current_ma = common.get("MARKET_REGIME_FILTER_MA_PERIOD")
        current_stop = common.get("HOLDING_STOP_LOSS_PCT")
        current_cooldown = common.get("COOLDOWN_DAYS")
        current_atr = common.get("ATR_PERIOD_FOR_NORMALIZATION")

        with st.form("common_settings_form"):
            st.subheader("시장 레짐 필터")
            new_enabled = st.checkbox("활성화 (MARKET_REGIME_FILTER_ENABLED)", value=current_enabled)
            new_ticker = st.text_input("레짐 기준 지수 티커 (MARKET_REGIME_FILTER_TICKER)", value=str(current_ticker) if current_ticker is not None else "", placeholder="예: ^GSPC")
            new_ma_str = st.text_input("레짐 MA 기간 (MARKET_REGIME_FILTER_MA_PERIOD)", value=str(current_ma) if current_ma is not None else "", placeholder="예: 20")

            st.subheader("위험 관리 및 지표")
            new_stop = st.number_input("보유 손절 임계값 % (HOLDING_STOP_LOSS_PCT)", value=float(current_stop) if current_stop is not None else 0.0, step=0.1, format="%.2f", help="예: -10.0")
            new_cooldown_str = st.text_input("쿨다운 일수 (COOLDOWN_DAYS)", value=str(current_cooldown) if current_cooldown is not None else "", placeholder="예: 5")
            new_atr_str = st.text_input("ATR 기간 (ATR_PERIOD_FOR_NORMALIZATION)", value=str(current_atr) if current_atr is not None else "", placeholder="예: 14")

            # (스케줄러 주기는 텔레그램 탭으로 이동)

            submitted = st.form_submit_button("공통 설정 저장")
            if submitted:
                error = False
                # Required validations
                if not new_ticker:
                    st.error("시장 레짐 필터 티커를 입력해주세요.")
                    error = True
                if not new_ma_str.isdigit() or int(new_ma_str) < 1:
                    st.error("레짐 MA 기간은 1 이상의 정수여야 합니다.")
                    error = True
                if not new_cooldown_str.isdigit() or int(new_cooldown_str) < 0:
                    st.error("쿨다운 일수는 0 이상의 정수여야 합니다.")
                    error = True
                if not new_atr_str.isdigit() or int(new_atr_str) < 1:
                    st.error("ATR 기간은 1 이상의 정수여야 합니다.")
                    error = True

                if not error:
                    # Normalize stop loss: interpret positive value as negative threshold
                    normalized_stop = -abs(float(new_stop))
                    to_save = {
                        "MARKET_REGIME_FILTER_ENABLED": bool(new_enabled),
                        "MARKET_REGIME_FILTER_TICKER": new_ticker,
                        "MARKET_REGIME_FILTER_MA_PERIOD": int(new_ma_str),
                        "HOLDING_STOP_LOSS_PCT": normalized_stop,
                        "COOLDOWN_DAYS": int(new_cooldown_str),
                        "ATR_PERIOD_FOR_NORMALIZATION": int(new_atr_str),
                    }
                    if save_common_settings(to_save):
                        st.success("공통 설정을 저장했습니다.")
                        st.rerun()
                    else:
                        st.error("공통 설정 저장에 실패했습니다.")

    with tab_telegram:
        st.header("텔레그램 설정")
        common = get_common_settings() or {}
        tg_enabled = bool(common.get("TELEGRAM_ENABLED", False))
        tg_token = common.get("TELEGRAM_BOT_TOKEN") or ""
        tg_chat = str(common.get("TELEGRAM_CHAT_ID") or "")

        with st.form("telegram_settings_form"):
            st.subheader("알림")
            new_tg_enabled = st.checkbox("텔레그램 알림 사용 (TELEGRAM_ENABLED)", value=tg_enabled, help="교체매매/현황 메시지를 텔레그램으로 전송")
            new_tg_token = st.text_input("봇 토큰 (TELEGRAM_BOT_TOKEN)", value=tg_token, type="password", placeholder="예: 123456:ABC-DEF...", help="@BotFather 로 생성")
            new_tg_chat = st.text_input("채팅 ID (TELEGRAM_CHAT_ID)", value=tg_chat, placeholder="예: 123456789", help="@userinfobot 등으로 확인")

            st.subheader("전송 주기")
            # 스케줄 주기(시간) – 정수 입력 (0 입력 시 크론 스케줄 사용)
            cur_kor = common.get("SCHEDULE_EVERY_HOURS_KOR")
            cur_aus = common.get("SCHEDULE_EVERY_HOURS_AUS")
            cur_coin = common.get("SCHEDULE_EVERY_HOURS_COIN")
            try:
                defv_kor = int(cur_kor) if cur_kor is not None else 0
            except Exception:
                defv_kor = 0
            try:
                defv_aus = int(cur_aus) if cur_aus is not None else 0
            except Exception:
                defv_aus = 0
            try:
                defv_coin = int(cur_coin) if cur_coin is not None else 0
            except Exception:
                defv_coin = 0
            sel_kor = st.number_input("한국 텔레그램 전송 주기(시간)", min_value=0, step=1, value=defv_kor, help="N시간마다 실행. 0 입력 시 환경변수 크론 스케줄 사용")
            sel_aus = st.number_input("호주 텔레그램 전송 주기(시간)", min_value=0, step=1, value=defv_aus)
            sel_coin = st.number_input("코인 텔레그램 전송 주기(시간)", min_value=0, step=1, value=defv_coin)

            st.caption("테스트는 스케줄과 무관하게 1회 계산 후 텔레그램으로 전송합니다. 한국/호주는 거래일에만 전송됩니다.")
            test_country_label = st.selectbox("테스트 국가", options=["한국", "호주", "코인"], index=2)
            _country_map = {"한국": "kor", "호주": "aus", "코인": "coin"}
            test_country = _country_map[test_country_label]

            cols = st.columns(3)
            with cols[0]:
                tg_save = st.form_submit_button("텔레그램 설정 저장")
            with cols[1]:
                tg_test = st.form_submit_button("텔레그램 테스트 전송")
            with cols[2]:
                tg_ping = st.form_submit_button("간단 텍스트 전송(직접)")

        if 'tg_save' in locals() and tg_save:
            error = False
            to_save = {}
            if new_tg_enabled:
                if not new_tg_token or not new_tg_chat:
                    st.error("텔레그램 알림을 사용하려면 봇 토큰과 채팅 ID가 필요합니다.")
                    error = True
                else:
                    to_save.update({
                        "TELEGRAM_ENABLED": True,
                        "TELEGRAM_BOT_TOKEN": new_tg_token.strip(),
                        "TELEGRAM_CHAT_ID": new_tg_chat.strip(),
                    })
            else:
                to_save.update({"TELEGRAM_ENABLED": False})

            # 스케줄 전송 주기 저장
            to_save.update({
                "SCHEDULE_EVERY_HOURS_KOR": int(sel_kor),
                "SCHEDULE_EVERY_HOURS_AUS": int(sel_aus),
                "SCHEDULE_EVERY_HOURS_COIN": int(sel_coin),
            })

            if not error:
                if save_common_settings(to_save):
                    st.success("텔레그램 설정을 저장했습니다.")
                    st.rerun()
                else:
                    st.error("텔레그램 설정 저장에 실패했습니다.")

        if 'tg_test' in locals() and tg_test:
            # 테스트 전송: 입력값 우선 저장 후 실행
            error = False
            if not new_tg_enabled:
                st.warning("텔레그램 알림이 비활성화되어 있어 전송되지 않을 수 있습니다. 활성화 후 테스트하세요.")
            if not new_tg_token or not new_tg_chat:
                st.error("봇 토큰과 채팅 ID를 입력해주세요.")
                error = True
            if not error:
                if save_common_settings({
                    "TELEGRAM_ENABLED": True,
                    "TELEGRAM_BOT_TOKEN": new_tg_token.strip(),
                    "TELEGRAM_CHAT_ID": new_tg_chat.strip(),
                }):
                    try:
                        res = generate_status_report(country=test_country, date_str=None, prefetched_data=None)
                        if not res:
                            st.error("현황 계산 실패로 테스트 전송을 건너뜁니다.")
                        else:
                            header_line, headers, rows_sorted = res
                            if test_country == 'coin':
                                sent = _maybe_notify_coin_detailed(test_country, header_line, headers, rows_sorted, force=True)
                            else:
                                sent_basic = _maybe_notify_basic_status(test_country, header_line, force=True)
                                sent_signals = _maybe_notify_signal_summary(test_country, headers, rows_sorted, force=True)
                                sent = sent_basic or sent_signals
                            if sent:
                                st.success("텔레그램 테스트 전송 완료(강제 전송). 채팅에서 확인하세요.")
                            else:
                                from utils.notify import get_last_error
                                err = get_last_error()
                                if err:
                                    st.warning(f"전송 시도는 했지만 응답이 없었습니다. 상세: {err}")
                                else:
                                    st.warning("전송 시도는 했지만 응답이 없었습니다. 토큰/채팅 ID를 확인하세요.")
                    except Exception as e:
                        st.error(f"테스트 전송 중 오류: {e}")
                else:
                    st.error("설정 저장 실패로 테스트 전송을 건너뜁니다.")

        if 'tg_ping' in locals() and tg_ping:
            # 상태 계산 없이, 토큰/채팅ID만으로 전송 테스트
            error = False
            if not new_tg_token or not new_tg_chat:
                st.error("봇 토큰과 채팅 ID를 입력해주세요.")
                error = True
            if not error:
                if save_common_settings({
                    "TELEGRAM_ENABLED": True,
                    "TELEGRAM_BOT_TOKEN": new_tg_token.strip(),
                    "TELEGRAM_CHAT_ID": new_tg_chat.strip(),
                }):
                    try:
                        from utils.notify import send_telegram_message, get_last_error
                        ok = send_telegram_message(f"[테스트] MomentumPilot 핑 메시지 ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
                        if ok:
                            st.success("간단 텍스트 전송 성공. 텔레그램에서 확인하세요.")
                        else:
                            err = get_last_error()
                            st.warning(f"전송 실패: {err or '상세 사유 없음'}")
                    except Exception as e:
                        st.error(f"전송 중 오류: {e}")
                else:
                    st.error("설정 저장 실패로 전송을 건너뜁니다.")


if __name__ == "__main__":
    main()
