import os
import sys
import warnings

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

from utils.logger import get_app_logger

"""
find.py

pykrx 라이브러리를 사용하여 지정된 등락률 이상 상승한 종목들을
섹터별로 분류하여 보여주는 스크립트입니다.

[사용법]
python scripts/find.py
python scripts/find.py --min-change 10.0
"""

import argparse
from datetime import datetime, timedelta

import pandas as pd
from pykrx import stock

# --- 설정 ---
# 이름에 아래 단어가 포함된 종목은 결과에서 제외합니다.
EXCLUDE_KEYWORDS = ["레버리지", "채권", "커버드콜", "인버스", "ETN"]


def get_latest_trading_day() -> str:
    """
    오늘 또는 가장 가까운 과거의 거래일을 'YYYYMMDD' 형식으로 반환합니다.
    pykrx의 get_market_ohlcv_by_date가 휴일에는 빈 DataFrame을 반환하는 것을 이용합니다.
    """
    dt = datetime.now()
    for i in range(10):  # 최대 10일 전까지 탐색
        date_str = (dt - timedelta(days=i)).strftime("%Y%m%d")
        # KOSPI 대표 종목(005930)으로 해당 날짜의 거래 여부 확인
        df = stock.get_market_ohlcv_by_date(date_str, date_str, "005930")
        if not df.empty:
            return date_str
    # 탐색 실패 시 오늘 날짜 반환 (오류 발생 가능성 있음)
    return datetime.now().strftime("%Y%m%d")


def get_previous_trading_day(date_str: str) -> str:
    """
    주어진 날짜('YYYYMMDD')의 이전 거래일을 'YYYYMMDD' 형식으로 반환합니다.
    """
    dt = datetime.strptime(date_str, "%Y%m%d")
    # 시작 날짜 바로 전날부터 탐색
    for i in range(1, 15):  # 최대 15일 전까지 탐색
        prev_dt = dt - timedelta(days=i)
        prev_date_str = prev_dt.strftime("%Y%m%d")
        # KOSPI 대표 종목(005930)으로 해당 날짜의 거래 여부 확인
        df = stock.get_market_ohlcv_by_date(prev_date_str, prev_date_str, "005930")
        if not df.empty:
            return prev_date_str
    # 탐색 실패 시, 입력일의 하루 전을 반환 (정확하지 않을 수 있음)
    return (dt - timedelta(days=1)).strftime("%Y%m%d")


def find_top_gainers(min_change_pct: float = 5.0, asset_type: str = "etf"):
    """
    지정된 등락률 이상 상승한 종목들을 보여줍니다.
    """
    logger = get_app_logger()
    try:
        latest_day = get_latest_trading_day()
        type_str = f" ({asset_type.upper()})"
        logger.info(
            "기준일: %s-%s-%s%s",
            latest_day[:4],
            latest_day[4:6],
            latest_day[6:],
            type_str,
        )

        df_change = pd.DataFrame()

        # 1. ETF 데이터 가져오기
        if asset_type == "etf":
            logger.info("ETF의 가격 변동 정보를 가져오는 중입니다...")
            try:
                # 등락률 계산을 위해 이전 거래일이 필요합니다.
                prev_day = get_previous_trading_day(latest_day)

                # get_etf_ohlcv_by_ticker는 특정일의 모든 ETF OHLCV를 반환합니다.
                df_today = stock.get_etf_ohlcv_by_ticker(latest_day)
                df_yest = stock.get_etf_ohlcv_by_ticker(prev_day)

                if not df_today.empty and not df_yest.empty:
                    # '종가' 컬럼만 사용하여 데이터프레임을 합칩니다.
                    df_merged = pd.merge(
                        df_today[["종가"]].rename(columns={"종가": "price_today"}),
                        df_yest[["종가"]].rename(columns={"종가": "price_yest"}),
                        left_index=True,  # 인덱스가 티커입니다.
                        right_index=True,
                        how="inner",  # 양일 모두 거래된 ETF만 대상으로 합니다.
                    )

                    # 등락률을 계산합니다. 0으로 나누는 경우를 방지합니다.
                    df_merged["등락률"] = (((df_merged["price_today"] / df_merged["price_yest"]) - 1) * 100).where(df_merged["price_yest"] > 0, 0)

                    # 필요한 컬럼만 선택하여 df_change에 추가합니다.
                    df_etf_filtered = df_merged[["등락률"]].reset_index()  # 인덱스를 '티커' 컬럼으로 변환
                    df_change = pd.concat([df_change, df_etf_filtered], ignore_index=True)
            except Exception as e:
                logger.warning("ETF 정보 조회 중 오류가 발생했습니다: %s", e)

        # 2. 일반 주식 데이터 가져오기
        if asset_type == "stock":
            logger.info("일반 주식의 가격 변동 정보를 가져오는 중입니다...")
            try:
                # get_market_price_change_by_ticker는 '등락률' 컬럼을 포함합니다.
                df_stock = stock.get_market_price_change_by_ticker(latest_day, latest_day, market="ALL")
                # 필요한 컬럼만 선택하여 df_change에 추가합니다.
                df_stock_filtered = df_stock[["등락률"]].reset_index()  # 인덱스를 '티커' 컬럼으로 변환
                df_change = pd.concat([df_change, df_stock_filtered], ignore_index=True)
            except Exception as e:
                logger.warning("일반 주식 정보 조회 중 오류가 발생했습니다: %s", e)

        if df_change.empty:
            logger.info("조회할 가격 정보가 없습니다.")
            return

        top_gainers = df_change[df_change["등락률"] >= min_change_pct].copy()

        if top_gainers.empty:
            logger.info("등락률 %.2f%% 이상 상승한 종목이 없습니다.", min_change_pct)
            return

        logger.info(
            "등락률 %.2f%% 이상 상승한 종목 %d개를 찾았습니다.",
            min_change_pct,
            len(top_gainers),
        )

        # 종목명 및 섹터 정보 처리
        etf_ticker_list = set(stock.get_etf_ticker_list(latest_day))

        def get_name(ticker):
            """티커 종류(주식/ETF)에 따라 올바른 이름 조회 함수를 호출합니다."""
            if ticker in etf_ticker_list:
                return stock.get_etf_ticker_name(ticker)
            else:
                return stock.get_market_ticker_name(ticker)

        top_gainers["종목명"] = top_gainers["티커"].apply(get_name)

        # 키워드 기반 필터링
        if EXCLUDE_KEYWORDS:
            initial_count = len(top_gainers)
            # '|'로 키워드를 연결하여 정규식 OR 조건 생성
            exclude_pattern = "|".join(EXCLUDE_KEYWORDS)
            # '종목명'에 키워드가 포함되지 않은 행만 남김
            top_gainers = top_gainers[~top_gainers["종목명"].str.contains(exclude_pattern, na=False)]
            filtered_count = initial_count - len(top_gainers)
            if filtered_count > 0:
                logger.info(
                    "제외 키워드(%s)에 따라 %d개 종목을 제외했습니다.",
                    ", ".join(EXCLUDE_KEYWORDS),
                    filtered_count,
                )

        # 등락률 순으로 정렬
        sorted_gainers = top_gainers.sort_values(by="등락률", ascending=False)

        logger.info("\n--- 급등주 목록 ---")
        for _, row in sorted_gainers.iterrows():
            logger.info("  - %s (%s): +%.2f%%", row["종목명"], row["티커"], row["등락률"])

    except Exception as e:
        logger.error("오류가 발생했습니다: %s", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="금일 급등주를 보여줍니다.")
    parser.add_argument("--min-change", type=float, default=3.0, help="검색할 최소 등락률 (기본값: 5.0)")
    parser.add_argument(
        "--type",
        type=str,
        choices=["stock", "etf"],
        default="etf",
        help="검색할 종목 유형 (stock: 일반 주식, etf: ETF (기본값))",
    )
    args = parser.parse_args()
    find_top_gainers(min_change_pct=args.min_change, asset_type=args.type)
