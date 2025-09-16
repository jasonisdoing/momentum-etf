#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
호주 ETF 중 카테고리가 'TBD'인 종목의 연간 수익률을 계산하고 순위를 매기는 스크립트.

data/aus/etf.json 파일에서 카테고리가 'TBD'인 ETF 목록을 읽어, 각 ETF의 최근 1년 수익률을
계산하고 수익률이 높은 순서대로 정렬하여 출력합니다.

[사용법]
python scripts/rank_aus_etf.py
"""

import os
import sys
import time
from datetime import datetime

import pandas as pd

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import fetch_ohlcv, fetch_yfinance_name
from utils.stock_list_io import get_etfs


def calculate_annual_return(ticker: str) -> tuple[float | None, str | None]:
    """
    주어진 티커에 대해 yfinance를 사용하여 데이터를 가져오고 1년 수익률을 계산합니다.
    성공 시 (수익률, 종목명) 튜플을, 실패 시 (None, None)을 반환합니다.
    """
    # yfinance API의 과도한 호출을 방지하기 위해 약간의 지연을 추가합니다.
    time.sleep(0.2)

    # 1년 전 데이터부터 조회
    end_date = datetime.now()
    start_date = end_date - pd.DateOffset(years=1)

    date_range = [start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")]

    # 데이터 로더를 사용하여 OHLCV 데이터 조회
    df = fetch_ohlcv(ticker, country="aus", date_range=date_range)

    if df is None or df.empty or len(df) < 2:
        return None, None

    # 날짜순으로 정렬
    df = df.sort_index()

    # 실제 데이터 기간이 1년에 근접하는지 확인 (약 11개월 미만이면 건너뛰기)
    if (df.index[-1] - df.index[0]).days < 330:
        return None, None

    start_price = df["Close"].iloc[0]
    end_price = df["Close"].iloc[-1]

    if pd.isna(start_price) or pd.isna(end_price) or start_price == 0:
        return None, None

    # 연간 수익률 계산 (%)
    annual_return = ((end_price / start_price) - 1) * 100

    # 종목명 조회
    name = fetch_yfinance_name(ticker)

    return annual_return, name


def main():
    """메인 실행 함수"""
    print("data/aus/ 폴더의 종목 목록에서 'TBD' 카테고리의 ETF를 읽어옵니다...")
    all_aus_etfs = get_etfs("aus")
    if not all_aus_etfs:
        print("오류: 'data/aus/' 폴더에서 호주 종목 목록을 찾을 수 없습니다.")
        return

    # 'TBD' 카테고리에 속하는 ETF 티커만 필터링합니다.
    tickers = [
        s["ticker"] for s in all_aus_etfs if s.get("category") == "TBD" and s.get("type") == "etf"
    ]

    if not tickers:
        print("분석할 'TBD' 카테고리의 ETF가 없습니다.")
        return

    print(f"총 {len(tickers)}개 ETF의 1년 수익률을 계산합니다...")

    results = []
    # 순차적으로 각 티커의 데이터를 조회합니다.
    for i, ticker in enumerate(tickers):
        print(f"  - 처리 중 ({i + 1}/{len(tickers)}): {ticker}")
        try:
            annual_return, name = calculate_annual_return(ticker)
            if annual_return is not None and name is not None:
                results.append(
                    {
                        "ticker": f"ASX:{ticker}",
                        "name": name,
                        "AnnualReturn": annual_return,
                    }
                )
        except Exception as exc:
            print(f"    - {ticker} 처리 중 오류 발생: {exc}")

    if not results:
        print("\n수익률을 계산할 수 있는 ETF가 없습니다.")
        return

    # AnnualReturn 기준으로 내림차순 정렬
    sorted_results = sorted(results, key=lambda x: x["AnnualReturn"], reverse=True)

    print("\n" + "=" * 50)
    print(">>> 호주 ETF 1년 수익률 순위 <<<")
    print("=" * 50)
    for i, item in enumerate(sorted_results, 1):
        print(
            f"{i:2d}. {item['name']} ({item['ticker']})\n"
            f"    - 1년 수익률: {item['AnnualReturn']:.2f}%"
        )
    print("=" * 50)


if __name__ == "__main__":
    main()
