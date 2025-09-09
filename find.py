#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
find.py

pykrx 라이브러리를 사용하여 지정된 등락률 이상 상승한 종목들을
섹터별로 분류하여 보여주는 스크립트입니다.

[사용법]
python find.py
python find.py --min-change 10.0
"""

import argparse
from datetime import datetime, timedelta

import pandas as pd
from pykrx import stock


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


def create_ticker_sector_map(date: str) -> dict:
    """
    KOSPI와 KOSDAQ의 모든 종목에 대해 {티커: 섹터명} 맵을 생성합니다.
    pykrx의 WICS 섹터 분류를 사용합니다.
    """
    print("섹터 정보를 구성하는 중입니다... (시간이 소요될 수 있습니다)")
    ticker_sector_map = {}

    # KOSPI 섹터 정보 (WICS 기준)
    kospi_indices = stock.get_index_ticker_list(market="KOSPI")
    for index_code in kospi_indices:
        index_name = stock.get_index_ticker_name(index_code)
        # '코스피 XXX' 형태의 WICS 섹터 지수만 필터링
        if index_name.startswith("코스피 ") and not any(
            s in index_name for s in ["200", "100", "50", " KRX"]
        ):
            sector_name = index_name.replace("코스피 ", "")
            tickers_in_sector = stock.get_index_ticker_list(date, index_code)
            for ticker in tickers_in_sector:
                ticker_sector_map[ticker] = sector_name

    # KOSDAQ 섹터 정보 (WICS 기준)
    kosdaq_indices = stock.get_index_ticker_list(market="KOSDAQ")
    for index_code in kosdaq_indices:
        index_name = stock.get_index_ticker_name(index_code)
        # '코스닥 XXX' 형태의 WICS 섹터 지수만 필터링
        if index_name.startswith("코스닥 ") and not any(
            s in index_name for s in ["150", "글로벌", "프리미어"]
        ):
            sector_name = index_name.replace("코스닥 ", "")
            tickers_in_sector = stock.get_index_ticker_list(date, index_code)
            for ticker in tickers_in_sector:
                # KOSPI 정보가 이미 있으면 덮어쓰지 않음 (더 상위 시장)
                if ticker not in ticker_sector_map:
                    ticker_sector_map[ticker] = sector_name

    print("섹터 정보 구성 완료.")
    return ticker_sector_map


def find_top_gainers_by_sector(min_change_pct: float = 5.0, asset_type: str = None):
    """
    지정된 등락률 이상 상승한 종목들을 섹터별로 분류하여 보여줍니다.
    """
    try:
        latest_day = get_latest_trading_day()
        type_str = (
            f" ({asset_type.upper()})" if asset_type else " (전체)"
        )
        print(f"기준일: {latest_day[:4]}-{latest_day[4:6]}-{latest_day[6:]}{type_str}\n")

        df_change = pd.DataFrame()

        # 1. ETF 데이터 가져오기
        if asset_type == "etf" or asset_type is None:
            print("ETF의 가격 변동 정보를 가져오는 중입니다...")
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
                    df_merged["등락률"] = (
                        ((df_merged["price_today"] / df_merged["price_yest"]) - 1) * 100
                    ).where(df_merged["price_yest"] > 0, 0)

                    # 필요한 컬럼만 선택하여 df_change에 추가합니다.
                    df_etf_filtered = df_merged[["등락률"]].reset_index()  # 인덱스를 '티커' 컬럼으로 변환
                    df_change = pd.concat([df_change, df_etf_filtered], ignore_index=True)
            except Exception as e:
                print(f"경고: ETF 정보 조회 중 오류가 발생했습니다: {e}")

        # 2. 일반 주식 데이터 가져오기
        if asset_type == "stock" or asset_type is None:
            print("일반 주식의 가격 변동 정보를 가져오는 중입니다...")
            try:
                # get_market_price_change_by_ticker는 '등락률' 컬럼을 포함합니다.
                df_stock = stock.get_market_price_change_by_ticker(latest_day, latest_day, market="ALL")
                # 필요한 컬럼만 선택하여 df_change에 추가합니다.
                df_stock_filtered = df_stock[['등락률']].reset_index()  # 인덱스를 '티커' 컬럼으로 변환
                df_change = pd.concat([df_change, df_stock_filtered], ignore_index=True)
            except Exception as e:
                print(f"경고: 일반 주식 정보 조회 중 오류가 발생했습니다: {e}")

        if df_change.empty:
            print("조회할 가격 정보가 없습니다.")
            return

        top_gainers = df_change[df_change["등락률"] >= min_change_pct].copy()

        if top_gainers.empty:
            print(f"등락률 {min_change_pct}% 이상 상승한 종목이 없습니다.")
            return

        print(f"등락률 {min_change_pct}% 이상 상승한 종목 {len(top_gainers)}개를 찾았습니다.")

        # ETF만 볼 때는 섹터 분류가 의미 없으므로 건너뜁니다.
        if asset_type == "etf":
            ticker_to_sector = {}
        else:
            ticker_to_sector = create_ticker_sector_map(latest_day)

        top_gainers["종목명"] = top_gainers["티커"].apply(stock.get_market_ticker_name)
        top_gainers["섹터"] = top_gainers["티커"].map(ticker_to_sector).fillna("기타")

        # ETF만 조회 시, 모든 종목을 'ETF' 섹터로 그룹화합니다.
        if asset_type == "etf":
            top_gainers["섹터"] = "ETF"

        grouped = sorted(top_gainers.groupby("섹터"), key=lambda x: x[0])

        print("\n--- 섹터별 급등주 목록 ---")
        for sector_name, group in grouped:
            print(f"\n# {sector_name}")
            sorted_group = group.sort_values(by="등락률", ascending=False)
            for _, row in sorted_group.iterrows():
                print(f"  - {row['종목명']} ({row['티커']}): +{row['등락률']:.2f}%")

    except Exception as e:
        print(f"오류가 발생했습니다: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="금일 급등주를 섹터별로 분류하여 보여줍니다.")
    parser.add_argument(
        "--min-change", type=float, default=5.0, help="검색할 최소 등락률 (기본값: 5.0)"
    )
    parser.add_argument(
        "--type",
        type=str,
        choices=["stock", "etf"],
        default=None,
        help="검색할 종목 유형 (stock: 일반 주식, etf: ETF, 미지정: 전체)",
    )
    args = parser.parse_args()
    find_top_gainers_by_sector(min_change_pct=args.min_change, asset_type=args.type)