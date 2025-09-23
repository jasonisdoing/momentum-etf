import os
import sys
import time
import warnings
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# .env 파일이 있다면 로드합니다.
load_dotenv()

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings(
    "ignore",
    message=r"\['break_start', 'break_end'\] are discontinued",
    category=UserWarning,
    module=r"^pandas_market_calendars\.",
)

# FIX: Add missing imports and remove unnecessary ones
from utils.account_registry import load_accounts
from signals import get_market_regime_status_string
from utils.data_loader import get_trading_days
from utils.db_manager import get_db_connection


def main():
    """MomentumETF 대시보드 메인 페이지를 렌더링합니다."""
    st.set_page_config(page_title="MomentumETF", layout="wide")

    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;500;700&display=swap');
            body {
                font-family: 'Noto Sans KR', sans-serif;
            }
            .block-container {
                max-width: 100%;
                padding-top: 1rem;
                padding-left: 2rem;
                padding-right: 2rem;
            }
        </style>
    """,
        unsafe_allow_html=True,
    )

    # --- 초기 로딩 단계별 시간 측정 (콘솔 출력) ---
    print("\n[MAIN] 1/4: 데이터베이스 연결 확인 시작...")
    start_time = time.time()
    with st.spinner("데이터베이스 연결 확인 중..."):
        if get_db_connection() is None:
            st.error(
                """
            **데이터베이스 연결 실패**

            MongoDB 데이터베이스에 연결할 수 없습니다. 다음 사항을 확인해주세요:

            1.  **환경 변수**: `MONGO_DB_CONNECTION_STRING` 환경 변수가 올바르게 설정되었는지 확인하세요.
            2.  **IP 접근 목록**: 현재 서비스의 IP 주소가 MongoDB Atlas의 'IP Access List'에 추가되었는지 확인하세요.
            3.  **클러스터 상태**: MongoDB Atlas 클러스터가 정상적으로 실행 중인지 확인하세요.
            """
            )
            st.stop()  # DB 연결 실패 시 앱 실행 중단
    duration = time.time() - start_time
    print(f"[MAIN] 1/4: 데이터베이스 연결 확인 완료 ({duration:.2f}초)")

    print("[MAIN] 2/4: 거래일 캘린더 데이터 확인 시작...")
    start_time = time.time()
    with st.spinner("거래일 캘린더 데이터 확인 중..."):
        try:
            import pandas_market_calendars as mcal  # noqa: F401
        except ImportError as e:
            st.error(
                "거래일 캘린더 라이브러리(pandas-market-calendars)를 불러올 수 없습니다.\n"
                "다음 명령으로 설치 후 다시 시도하세요: pip install pandas-market-calendars\n"
                f"상세: {e}"
            )
            st.stop()

        try:
            today = pd.Timestamp.now().normalize()
            start = (today - pd.DateOffset(days=30)).strftime("%Y-%m-%d")
            end = (today + pd.DateOffset(days=7)).strftime("%Y-%m-%d")
            problems = []
            for c in ("kor", "aus"):
                days = get_trading_days(start, end, c)
                if not days:
                    problems.append(c)
            if problems:
                st.error(
                    "거래일 캘린더를 조회하지 못했습니다: "
                    + ", ".join({"kor": "한국", "aus": "호주"}[p] for p in problems)
                    + "\nKOSPI/ASX 캘린더를 사용할 수 있는지 확인해주세요."
                )
                st.stop()
        except Exception as e:
            st.error(f"거래일 캘린더 초기화 중 오류가 발생했습니다: {e}")
            st.stop()
    duration = time.time() - start_time
    print(f"[MAIN] 2/4: 거래일 캘린더 데이터 확인 완료 ({duration:.2f}초)")

    # 제목과 시장 상태를 한 줄에 표시
    col1, col2 = st.columns([2.5, 1.5])
    with col1:
        st.title("Momentum. ETF.")
    with col2:
        print("[MAIN] 3/4: 시장 레짐 상태 분석 시작...")
        start_time = time.time()
        with st.spinner("시장 레짐 상태 분석 중..."):
            # 시장 상태는 한 번만 계산하여 10분간 캐시합니다.
            @st.cache_data(ttl=600)
            def _get_cached_market_status():
                return get_market_regime_status_string()

            market_status_str = _get_cached_market_status()
        duration = time.time() - start_time
        print(f"[MAIN] 3/4: 시장 레짐 상태 분석 완료 ({duration:.2f}초)")

        if market_status_str:
            st.markdown(
                f'<div style="text-align: right; padding-top: 1.5rem; font-size: 1.1rem;">{market_status_str}</div>',
                unsafe_allow_html=True,
            )

    print("[MAIN] 4/4: 계좌 정보 로딩 시작...")
    start_time = time.time()
    with st.spinner("계좌 정보 로딩 중..."):
        # FIX: load_accounts is called to populate the registry if needed by other pages.
        load_accounts(force_reload=False)
    duration = time.time() - start_time
    print(f"[MAIN] 4/4: 계좌 정보 로딩 완료 ({duration:.2f}초)")

    # FIX: Remove old tab logic and provide guidance for the new multi-page structure.
    st.markdown("---")
    st.header("🚀 시작하기")
    st.info(
        """
        왼쪽 사이드바 메뉴를 사용하여 각 기능 페이지로 이동할 수 있습니다.

        - **assets**: 계좌별 자산(평가금액) 및 거래 내역을 관리합니다.
        - **signal**: 날짜별 매매 신호를 확인합니다.
        - **master_data**: 투자 유니버스에 포함된 종목을 조회합니다.
        - **settings**: 앱의 공통 설정 및 계좌별 전략 파라미터를 관리합니다.
        """
    )
    st.success("모든 페이지가 준비되었습니다. 왼쪽 사이드바에서 원하시는 메뉴를 선택하세요.")


if __name__ == "__main__":
    main()
