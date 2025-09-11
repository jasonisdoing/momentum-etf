import os
import sys

from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd
import streamlit as st

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import settings as global_settings
from logic.settings import TEST_DATE_RANGE, SECTOR_COUNTRY_OPTIONS
from status import generate_status_report
from utils.data_loader import fetch_yfinance_name, get_trading_days, fetch_pykrx_name, fetch_ohlcv_for_tickers
from utils.report import format_kr_money
from utils.db_manager import (
    get_available_snapshot_dates, get_portfolio_snapshot, get_all_trades, get_all_daily_equities, 
    get_sectors, save_sectors, is_sector_in_use, get_sector_stock_counts, delete_sectors_and_reset_stocks,
    get_status_report_from_db, save_status_report_to_db, delete_trade_by_id, save_sector_changes,
    get_stocks, save_stocks, save_daily_equity, save_trade, 
    get_app_settings, save_app_settings,
)

try:
    from pykrx import stock as _stock
except Exception:
    _stock = None

COUNTRY_CODE_MAP = {"kor": "한국", "aus": "호주"}


# --- Functions ---


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


@st.dialog("BUY")
def show_buy_dialog(country_code: str):
    """매수(BUY) 거래 입력을 위한 모달 다이얼로그를 표시합니다."""

    # on_click 콜백은 위젯이 렌더링되기 전에 실행됩니다.
    # 따라서 모든 로직을 콜백 함수 내에서 처리하고, st.rerun()으로 다이얼로그를 닫습니다.
    def on_buy_submit():
        # st.session_state에서 폼 데이터 가져오기
        trade_date = st.session_state[f"buy_date_{country_code}"]
        ticker = st.session_state[f"buy_ticker_{country_code}"]
        shares = st.session_state[f"buy_shares_{country_code}"]
        price = st.session_state[f"buy_price_{country_code}"]

        if not ticker or not shares > 0 or not price > 0:
            st.session_state[f"buy_message"] = ("error", "종목코드, 수량, 가격을 모두 올바르게 입력해주세요.")
            return  # st.rerun()을 호출하지 않아 다이얼로그가 닫히지 않고 오류 메시지 표시

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
            "shares": int(shares),
            "price": float(price),
            "note": "Manual input from web app"
        }
        
        if save_trade(trade_data):
            # 성공 메시지를 세션에 저장하고, st.rerun()으로 다이얼로그를 닫고 앱을 새로고침합니다.
            st.session_state[f"buy_message"] = ("success", "거래가 성공적으로 저장되었습니다.")
            st.rerun()
        else:
            st.session_state[f"buy_message"] = ("error", "거래 저장에 실패했습니다. 콘솔 로그를 확인해주세요.")

    # 이전 실행에서 저장된 메시지가 있다면 표시합니다.
    if f"buy_message" in st.session_state:
        msg_type, msg_text = st.session_state.pop(f"buy_message")
        if msg_type == "success":
            st.toast(msg_text)
        else:
            st.error(msg_text)

    with st.form(f"trade_form_{country_code}"):
        st.date_input("거래일", value="today", key=f"buy_date_{country_code}")
        st.text_input("종목코드 (티커)", key=f"buy_ticker_{country_code}")
        st.number_input("수량", min_value=1, step=1, key=f"buy_shares_{country_code}")
        st.number_input(
            "매수 단가", 
            min_value=0.0, 
            format="%.4f" if country_code == "aus" else "%d",
            key=f"buy_price_{country_code}"
        )
        st.form_submit_button("거래 저장", on_click=on_buy_submit)


@st.dialog("SELL", width="large")
def show_sell_dialog(country_code: str):
    """보유 종목 매도를 위한 모달 다이얼로그를 표시합니다."""
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
            
            # Ensure price is a scalar before boolean evaluation and calculations.
            # This handles cases where a function returns a Series (e.g., due to duplicate columns).
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
            st.session_state[f"sell_message"] = ("warning", "매도할 종목을 선택해주세요.")
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
            st.session_state[f"sell_message"] = ("success", f"{success_count}개 종목의 매도 거래가 성공적으로 저장되었습니다.")
        else:
            st.session_state[f"sell_message"] = ("error", "일부 거래 저장에 실패했습니다. 콘솔 로그를 확인해주세요.")
        
        # st.rerun()을 호출하여 다이얼로그를 닫고 앱을 새로고침합니다.
        st.rerun()

    # 이전 실행에서 저장된 메시지가 있다면 표시합니다.
    if f"sell_message" in st.session_state:
        msg_type, msg_text = st.session_state.pop(f"sell_message")
        if msg_type == "success":
            st.toast(msg_text)
        elif msg_type == "warning":
            st.warning(msg_text)
        else:
            st.error(msg_text)

    with st.form(f"sell_form_{country_code}"):
        st.subheader("매도할 종목을 선택하세요 (전체 매도)")
        st.date_input("매도일", value="today", key=f"sell_date_{country_code}")
        
        df_holdings['선택'] = False
        # 정렬이 필요한 컬럼은 숫자형으로 유지하고, column_config에서 포맷팅합니다.
        # 이렇게 하면 '평가금액' 등에서 문자열이 아닌 숫자 기준으로 올바르게 정렬됩니다.
        df_display = df_holdings[['선택', 'name', 'ticker', 'shares', 'return_pct', 'value', 'price']].copy()
        df_display.rename(columns={
            'name': '종목명', 'ticker': '티커', 'shares': '보유수량',
            'return_pct': '수익률', 'value': '평가금액', 'price': '현재가'
        }, inplace=True)
        
        st.data_editor(
            df_display, hide_index=True, width='stretch',
            key=f"sell_editor_{country_code}",
            disabled=['종목명', '티커', '보유수량', '평가금액', '수익률', '현재가'],
            column_config={
                "선택": st.column_config.CheckboxColumn(required=True),
                "보유수량": st.column_config.NumberColumn(format="%d"),
                "수익률": st.column_config.NumberColumn(format="%.2f%%"),
                "평가금액": st.column_config.NumberColumn(
                    # 쉼표(,)를 포맷에 추가하여 3자리마다 구분자를 표시합니다.
                    format="%,.0f" if country_code == 'kor' else "%,.2f"
                ),
                "현재가": st.column_config.NumberColumn(
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
    with st.form(f"add_stock_form_{country_code}"):
        ticker = st.text_input("티커")
        # 종목의 실제 소속 국가(시장)를 선택합니다.
        country_options = [""] + SECTOR_COUNTRY_OPTIONS
        market_country = st.selectbox("소속 국가(시장)", options=country_options, index=0, format_func=lambda x: "선택 안 함" if x == "" else x)
        stock_type = st.selectbox("타입", options=["stock", "etf"])
        
        sector_names = sorted([s.get("name") for s in sectors if s.get("name")])
        sector_options = [""] + sector_names
        sector = st.selectbox("업종", options=sector_options, index=0, format_func=lambda x: "선택 안 함" if x == "" else x)
        
        submitted = st.form_submit_button("저장")

        if submitted:
            # 업종(sector)은 더 이상 필수 항목이 아닙니다.
            if not ticker or not stock_type:
                st.error("티커와 타입은 필수 항목입니다.")
                return

            with st.spinner("종목 정보를 조회하고 추가하는 중..."):
                # 중복 확인
                current_stocks = get_stocks(country_code) # 현재 포트폴리오(탭)의 종목 목록
                if ticker.upper() in [s['ticker'] for s in current_stocks]:
                    st.error(f"티커 '{ticker.upper()}'는 현재 포트폴리오에 이미 존재합니다.")
                    return

                # 종목명 조회
                stock_name = ""
                if country_code == "kor" and _stock:
                    try:
                        name_candidate = _stock.get_etf_ticker_name(ticker)
                        if isinstance(name_candidate, str) and name_candidate:
                            stock_name = name_candidate
                    except Exception:
                        pass
                    if not stock_name:
                        try:
                            name_candidate = _stock.get_market_ticker_name(ticker)
                            if isinstance(name_candidate, str) and name_candidate:
                                stock_name = name_candidate
                        except Exception:
                            pass
                elif country_code == "aus":
                    stock_name = fetch_yfinance_name(ticker)
                
                if not stock_name:
                    st.warning(f"티커 '{ticker.upper()}'의 종목명을 찾을 수 없습니다. 종목명 없이 추가됩니다.")

                # 새 종목 추가
                new_stock = {
                    "country": market_country, # 종목의 실제 소속 국가
                    "ticker": ticker.upper(),
                    "name": stock_name,
                    "type": stock_type,
                    "sector": sector
                }
                current_stocks.append(new_stock)
                
                # 전체 목록 저장 (save_stocks는 모든 국가의 종목을 다룸)
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
    
    country_name = COUNTRY_CODE_MAP.get(country_code, "기타")

    st.info("이곳에서 투자 유니버스에 포함될 종목을 관리할 수 있습니다.")

    col1, col2, _ = st.columns([1.2, 1.8, 7.0])
    with col1:
        if st.button("종목 추가", key=f"add_stock_btn_{country_code}"):
            show_add_stock_dialog(country_code, sectors)
    with col2:
        if st.button("종목명 일괄 업데이트", key=f"update_names_btn_{country_code}"):
            with st.spinner("모든 종목명을 업데이트하는 중..."):
                stocks_to_update = get_stocks(country_code)
                updated_count = 0
                for stock in stocks_to_update:
                    if not stock.get("name"): # 이름이 없는 경우
                        new_name = ""
                        if country_code == "kor":
                            new_name = fetch_pykrx_name(stock['ticker'])
                        elif country_code == "aus":
                            new_name = fetch_yfinance_name(stock['ticker'])
                        
                        if new_name:
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
        df_stocks = pd.DataFrame(stocks_data, columns=["ticker", "name", "type", "sector", "country", "last_modified"])

        # 데이터 정합성을 위한 처리
        df_stocks['country'] = df_stocks['country'].apply(lambda c: c if c in SECTOR_COUNTRY_OPTIONS else "")
        df_stocks['sector'] = df_stocks['sector'].apply(
            lambda s: s if s in sector_options else default_sector
        )

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

        # 삭제 UI를 위해 '삭제' 컬럼을 추가하고 컬럼 순서를 조정합니다.
        df_for_display['삭제'] = False
        df_for_display = df_for_display[['삭제', 'ticker', 'country', 'name', 'type', 'sector']]

        st.info("아래 표에서 종목 정보를 수정하거나, 삭제할 종목을 선택 후 '변경사항 저장' 버튼을 눌러주세요.")
        
        edited_df = st.data_editor(
            df_for_display,
            width='stretch',
            hide_index=True,
            key=f"stock_editor_{country_code}",
            column_config={
                "삭제": st.column_config.CheckboxColumn("삭제", required=True),
                "ticker": st.column_config.TextColumn("티커", disabled=True),
                "country": st.column_config.SelectboxColumn(
                    "국가",
                    options=[""] + SECTOR_COUNTRY_OPTIONS,
                    required=False,
                ),
                "name": st.column_config.TextColumn("종목명", disabled=True),
                "type": st.column_config.TextColumn("타입", disabled=True),
                "sector": st.column_config.SelectboxColumn(
                    "업종",
                    options=sector_options,
                    required=False,
                ),
            },
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
                stocks_to_save = stocks_to_keep_df.drop(columns=['삭제']).to_dict('records')

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


def render_country_tab(country_code: str):
    """지정된 국가에 대한 탭의 전체 UI를 렌더링합니다."""
    sub_tab_names = ["현황", "히스토리", "트레이드", "종목 관리", "설정"]
    sub_tab_status, sub_tab_history, sub_tab_trades, sub_tab_stock_management, sub_tab_settings = st.tabs(sub_tab_names)

    # --- 공통 데이터 로딩 ---
    sorted_dates = get_available_snapshot_dates(country_code)

    # 오늘 날짜가 거래일이고, 아직 데이터가 없는 경우 날짜 목록에 추가
    today = pd.Timestamp.now().normalize()
    today_str = today.strftime("%Y-%m-%d")

    # 오늘이 평일(월~금)이고, DB에 오늘 날짜 데이터가 아직 없는 경우
    if today.weekday() < 5 and today_str not in sorted_dates:
        sorted_dates.insert(0, today_str)

    # --- 1. 현황 탭 (최신 날짜) ---
    with sub_tab_status:
        if not sorted_dates:
            st.warning(f"[{country_code.upper()}] 국가의 포트폴리오 데이터를 DB에서 찾을 수 없습니다.")
            st.info("먼저 '거래 입력' 버튼을 통해 거래 내역을 추가해주세요.")
        else:
            latest_date_str = sorted_dates[0]

            # 다시 계산 버튼
            force_recalc = st.button("다시 계산", key=f"recalc_status_{country_code}_{latest_date_str}")

            spinner_message = f"'{latest_date_str}' 기준 현황 데이터를 계산하고 있습니다..."
            with st.spinner(spinner_message):
                result = get_cached_status_report(
                    country=country_code, date_str=latest_date_str, force_recalculate=force_recalc
                )
            if result:
                header_line, headers, rows = result
                st.markdown(f":information_source: {header_line}", unsafe_allow_html=True)
                df = pd.DataFrame(rows, columns=headers)
                if "#" in df.columns:
                    df = df.set_index("#")
                style_cols = ["일간수익률", "누적수익률"]
                styler = df.style
                for col in style_cols:
                    if col in df.columns:
                        styler = styler.map(style_returns, subset=[col])
                formats = {
                    "일간수익률": "{:+.2f}%",
                    "누적수익률": "{:+.2f}%",
                    "비중": "{:.1f}%",
                    "점수": "{:.2f}",
                }
                styler = styler.format(formats, na_rep="-")

                num_rows_to_display = min(len(df), 15)
                height = (num_rows_to_display + 1) * 35 + 3

                st.dataframe(
                    styler,                    
                    width='stretch',
                    height=height,
                    column_config={
                        "상태": st.column_config.TextColumn(width="small"),
                        "보유일": st.column_config.TextColumn(width="small"),
                        "보유수량": st.column_config.TextColumn(width="small"),
                        "일간수익률": st.column_config.TextColumn(width="small"),
                        "누적수익률": st.column_config.TextColumn(width="small"),
                        "비중": st.column_config.TextColumn(width="small"),
                        "문구": st.column_config.TextColumn("문구", width="large"),
                    }
                )
            else:
                st.error(f"'{latest_date_str}' 기준 ({country_code.upper()}) 현황을 생성하는 데 실패했습니다.")

    with sub_tab_history:
        history_sub_tab_names = ["현황", "평가금액"]
        history_status_tab, history_equity_tab = st.tabs(history_sub_tab_names)

        with history_status_tab:
            past_dates = sorted_dates[1:]
            if not past_dates:
                st.info("과거 현황 데이터가 없습니다.")
            else:
                if st.button("과거 전체 다시계산", key=f"recalc_all_hist_{country_code}"):
                    # 1. 재계산에 필요한 모든 종목과 전체 기간을 결정합니다.
                    stocks_from_db = get_stocks(country_code)
                    tickers = [s['ticker'] for s in stocks_from_db]
                    
                    oldest_date = pd.to_datetime(past_dates[-1])
                    newest_date = pd.to_datetime(past_dates[0])
                    
                    from logic import settings
                    max_ma_period = max(getattr(settings, "MA_PERIOD_FOR_ETF", 0), getattr(settings, "MA_PERIOD_FOR_STOCK", 0))
                    atr_period_norm = int(getattr(settings, "ATR_PERIOD_FOR_NORMALIZATION", 20))
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

                        spinner_message = f"'{date_str}' 기준 현황 데이터를 계산하고 있습니다..."
                        with st.spinner(spinner_message):
                            result = get_cached_status_report(
                                country=country_code, date_str=date_str, force_recalculate=False
                            )
                        if result:
                            header_line, headers, rows = result
                            st.markdown(f":information_source: {header_line}", unsafe_allow_html=True)
                            df = pd.DataFrame(rows, columns=headers)
                            if "#" in df.columns:
                                df = df.set_index("#")
                            style_cols = ["일간수익률", "누적수익률"]
                            styler = df.style
                            for col in style_cols:
                                if col in df.columns:
                                    styler = styler.map(style_returns, subset=[col])
                            formats = {
                                "일간수익률": "{:+.2f}%",
                                "누적수익률": "{:+.2f}%",
                                "비중": "{:.1f}%",
                            "점수": "{:.2f}",
                            }
                            styler = styler.format(formats, na_rep="-")

                            num_rows_to_display = min(len(df), 15)
                            height = (num_rows_to_display + 1) * 35 + 3

                            st.dataframe(
                                styler,                                
                                width='stretch',
                                height=height,
                                column_config={
                                    "상태": st.column_config.TextColumn(width="small"),
                                    "보유일": st.column_config.TextColumn(width="small"),
                                    "보유수량": st.column_config.TextColumn(width="small"),
                                    "일간수익률": st.column_config.TextColumn(width="small"),
                                    "누적수익률": st.column_config.TextColumn(width="small"),
                                    "비중": st.column_config.TextColumn(width="small"),
                                    "문구": st.column_config.TextColumn("문구", width="large"),
                                }
                            )
                        else:
                            st.error(f"'{date_str}' 기준 ({country_code.upper()}) 현황을 생성하는 데 실패했습니다.")
        
        with history_equity_tab:
            app_settings = get_app_settings(country_code)
            initial_date = (app_settings.get("initial_date") if app_settings else None) or (datetime.now() - pd.DateOffset(months=3))
            
            start_date_str = initial_date.strftime("%Y-%m-%d")
            end_date_str = datetime.now().strftime("%Y-%m-%d")

            with st.spinner("거래일 및 평가금액 데이터를 불러오는 중..."):
                all_trading_days = get_trading_days(start_date_str, end_date_str, country_code)
                if not all_trading_days:
                    st.warning("거래일을 조회할 수 없습니다.")
                else:
                    start_dt_obj = pd.to_datetime(start_date_str).to_pydatetime()
                    end_dt_obj = pd.to_datetime(end_date_str).to_pydatetime()
                    existing_equities = get_all_daily_equities(country_code, start_dt_obj, end_dt_obj)
                    equity_data_map = {pd.to_datetime(e['date']).normalize(): e for e in existing_equities}

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
                        "total_equity": st.column_config.NumberColumn("총 평가금액", format="%.2f" if country_code == "aus" else "%d", required=True),
                    }
                    if country_code == "aus":
                        column_config["is_value"] = st.column_config.NumberColumn("해외주식 평가액", format="%.2f")
                        column_config["is_change_pct"] = st.column_config.NumberColumn("해외주식 수익률(%)", format="%.2f")

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
            
            # 삭제 선택을 위한 컬럼 추가
            df_trades['delete'] = False
            
            # 표시할 컬럼 순서 정의
            # 'created_at' 컬럼을 추가합니다.
            cols_to_show = ['delete', 'created_at', 'date', 'action', 'ticker', 'name', 'shares', 'price', 'note', 'id']
            # reindex를 사용하여 이전 데이터에 'created_at'이 없어도 오류가 발생하지 않도록 합니다.
            df_display = df_trades.reindex(columns=cols_to_show).copy()
            
            # 날짜 및 시간 포맷팅
            df_display['date'] = pd.to_datetime(df_display['date']).dt.strftime('%Y-%m-%d')
            # 'created_at'이 있는 경우에만 포맷팅을 적용합니다.
            df_display['created_at'] = pd.to_datetime(df_display['created_at'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
            df_display['created_at'] = df_display['created_at'].fillna('-') # NaT를 '-'로 바꿉니다.

            edited_df = st.data_editor(
                df_display,
                key=f"trades_editor_{country_code}",
                hide_index=True,
                width='stretch',
                column_config={
                    "delete": st.column_config.CheckboxColumn("삭제", required=True),
                    "id": None, # ID 컬럼은 숨김
                    "created_at": st.column_config.TextColumn("기록시간"),
                    "date": st.column_config.TextColumn("거래일"),
                    "action": st.column_config.TextColumn("종류"),
                    "ticker": st.column_config.TextColumn("티커"),
                    "name": st.column_config.TextColumn("종목명", width="medium"),
                    "shares": st.column_config.NumberColumn("수량", format="%d"),
                    "price": st.column_config.NumberColumn("가격", format="%.4f" if country_code == "aus" else "%d"),
                    "note": st.column_config.TextColumn("비고", width="large"),
                },
                disabled=['created_at', 'date', 'action', 'ticker', 'name', 'shares', 'price', 'note']
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
        # 1. DB에서 현재 설정값 로드, 없으면 0을 기본값으로 사용
        db_settings = get_app_settings(country_code)
        current_capital = db_settings.get("initial_capital", 0) if db_settings else 0

        # TEST_DATE_RANGE의 시작일을 기본 날짜로 사용
        test_date_range = TEST_DATE_RANGE
        default_date_str = (test_date_range or [""])[0]
        default_date = pd.to_datetime(default_date_str) if default_date_str else datetime.now()
        current_date = db_settings.get("initial_date", default_date) if db_settings else default_date

        with st.form(key=f"settings_form_{country_code}"):
            new_capital = st.number_input(
                "초기 자본금 (INITIAL_CAPITAL)",
                value=float(current_capital) if country_code == "aus" else int(current_capital),
                format="%.2f" if country_code == "aus" else "%d",
                help="포트폴리오의 시작 자본금을 설정합니다. 누적 수익률 계산의 기준이 됩니다."
            )

            new_date = st.date_input(
                "초기 자본 기준일 (INITIAL_DATE)",
                value=current_date,
                help="초기 자본금이 투입된 날짜를 설정합니다."
            )

            save_settings_submitted = st.form_submit_button("설정 저장하기")

            if save_settings_submitted:
                settings_to_save = {
                    "country": country_code,
                    "initial_capital": new_capital,
                    "initial_date": pd.to_datetime(new_date).to_pydatetime()
                }
                if save_app_settings(country_code, settings_to_save):
                    st.success("설정이 성공적으로 저장되었습니다.")
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

    st.title("Momentum. Pilot.")

    tab_names = ["한국주식", "호주주식", "마스터 정보", "설정"]
    tab_kor, tab_aus, tab_master, tab_settings = st.tabs(tab_names)

    with tab_kor:
        render_country_tab("kor")

    with tab_aus:
        render_country_tab("aus")

    with tab_master:
        sector_management_tab, = st.tabs(["업종"])
        with sector_management_tab:
            render_master_sector_ui()

    with tab_settings:
        st.header("설정")
        st.write("이곳에 웹앱 관련 설정을 변경하는 UI를 추가할 수 있습니다.")
        st.info("예: 기본 전략 변경, 알림 설정 등")


if __name__ == "__main__":
    main()
