#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
etf_categorized.csv 파일에서 ETF 목록을 읽어 최근 N개월 수익률과 모멘텀 스코어를 계산하고,
최종 결과를 수익률 순으로 3_etf_processed.csv 파일에 저장합니다.
수익률 계산에 실패한 종목은 콘솔에 출력됩니다.

[사용법]
python scripts/process_etf.py <country>
예: python scripts/process_etf.py aus
예: python scripts/process_etf.py kor
"""

import argparse
import os
import sys
import time
from datetime import datetime

import pandas as pd

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import fetch_ohlcv
from utils.momentum_scorer import momentum_score_krx, momentum_score_yf

# --- 설정 ---
# 수익률을 계산할 최근 개월 수
MONTHS_TO_CALCULATE = 12


def get_etf_return(full_ticker: str, country: str, months: int) -> tuple[float | None, str | None]:
    """
    주어진 전체 티커(예: ASX:VHY)에 대해 최근 N개월 수익률을 계산합니다.
    성공 시 (수익률, None)을, 실패 시 (None, 실패사유)를 반환합니다.
    """
    # yfinance API의 과도한 호출을 방지하기 위해 약간의 지연을 추가합니다.
    time.sleep(0.2)

    # "ASX:VHY" 형식에서 "VHY" 부분만 추출
    if country == "aus":
        if ":" not in full_ticker:
            return None, "INVALID_TICKER_FORMAT"
        ticker_part = full_ticker.split(":")[-1]
    else:  # kor
        ticker_part = full_ticker

    # N개월 전 데이터부터 조회
    end_date = datetime.now()
    start_date = end_date - pd.DateOffset(months=months)
    date_range = [start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")]

    # 데이터 로더를 사용하여 OHLCV 데이터 조회
    df = fetch_ohlcv(ticker_part, country=country, date_range=date_range)

    if df is None or df.empty:
        return None, "NO_DATA"

    # 날짜순으로 정렬
    df = df.sort_index()

    if len(df) < 2:
        return None, "INSUFFICIENT_DATA"

    # 실제 데이터 기간이 요청 기간에 근접하는지 확인 (요청 개월 수의 90% 미만이면 건너뛰기)
    required_days = months * 30 * 0.9
    if (df.index[-1] - df.index[0]).days < required_days:
        return None, "INSUFFICIENT_DATA"

    start_price = df["Close"].iloc[0]
    end_price = df["Close"].iloc[-1]

    if pd.isna(start_price) or pd.isna(end_price) or start_price == 0:
        return None, "INSUFFICIENT_DATA"

    # 연간 수익률 계산 (%)
    return_pct = ((end_price / start_price) - 1) * 100

    return return_pct, None


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="ETF 수익률을 계산하여 최종 CSV 파일을 생성합니다.")
    parser.add_argument("country", choices=["kor", "aus"], help="처리할 국가 코드 (kor 또는 aus)")
    args = parser.parse_args()
    country_code = args.country
    months_to_calculate = MONTHS_TO_CALCULATE

    # 입력 파일: {country}/2_etf_categorized.csv
    categorized_csv_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", country_code, "2_etf_categorized.csv"
    )
    # 출력 파일: {country}/3_etf_processed.csv
    final_csv_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", country_code, "3_etf_processed.csv"
    )

    if not os.path.exists(categorized_csv_path):
        print(f"오류: '{categorized_csv_path}' 파일을 찾을 수 없습니다.")
        print(f"'scripts/categorize_etf.py {country_code}'를 먼저 실행하여 파일을 생성해주세요.")
        return

    # 1. 분류된 CSV 파일 읽기
    df_categorized = pd.read_csv(categorized_csv_path, dtype={"ticker": str})
    if df_categorized.empty or "ticker" not in df_categorized.columns:
        print("분석할 ETF가 없습니다.")
        return

    # 2. 각 티커의 수익률을 계산합니다.
    print(f"\n총 {len(df_categorized)}개 ETF의 최근 {months_to_calculate}개월 수익률을 계산합니다...")
    processed_results = []
    failed_tickers = []
    for i, row in df_categorized.iterrows():
        ticker = row["ticker"]

        print(f"  - 처리 중 ({i + 1}/{len(df_categorized)}): {ticker}")
        try:
            return_pct, reason = get_etf_return(ticker, country_code, months_to_calculate)

            # 모멘텀 스코어 계산
            momentum_score = None
            if country_code == "kor":
                momentum_score = momentum_score_krx(ticker)
            elif country_code == "aus":
                # yfinance 티커 형식으로 변환 (e.g., ASX:VHY -> VHY.AX)
                ticker_part = ticker.split(":")[-1]
                yfinance_ticker = f"{ticker_part}.AX"
                print(f"    - yfinance 티커: {yfinance_ticker}")
                momentum_score = momentum_score_yf(yfinance_ticker)

            if return_pct is not None:
                processed_results.append(
                    {
                        "ticker": ticker,
                        "name": row["name"],
                        "category": row["category"],
                        "return_pct": return_pct,
                        "momentum_score": momentum_score,
                    }
                )
            else:
                failed_tickers.append({"ticker": ticker, "reason": reason})
        except Exception as exc:
            print(f"    - {ticker} 처리 중 오류 발생: {exc}")
            failed_tickers.append({"ticker": ticker, "reason": "EXCEPTION"})

    # 3. 성공적으로 처리된 결과를 CSV 파일에 덮어씁니다.
    if processed_results:
        # DataFrame으로 변환
        df = pd.DataFrame(processed_results)

        # 모멘텀 스코어가 없는 종목은 필터링
        df.dropna(subset=["momentum_score"], inplace=True)

        if not df.empty:
            # 최종 결과를 return_pct 순으로 정렬
            df_results = df.sort_values(by="return_pct", ascending=False)

            try:
                df_results.to_csv(final_csv_path, index=False, encoding="utf-8-sig")
                print(f"\n'{final_csv_path}' 파일에 {len(df_results)}개 ETF 정보를 수익률 순으로 덮어썼습니다.")
            except IOError as e:
                print(f"\nCSV 파일 쓰기 중 오류 발생: {e}")
        else:
            print("\n모멘텀 스코어를 계산할 수 있는 ETF가 없어 CSV 파일을 생성하지 않습니다.")
    else:
        print("\n성공적으로 처리된 ETF가 없어 CSV 파일을 생성하지 않습니다.")

    # 4. 처리 실패한 티커를 콘솔에 출력합니다.
    if failed_tickers:
        print("\n" + "=" * 30)
        print("수익률 계산 실패 티커 목록")
        print("=" * 30)

        no_data_tickers = sorted(
            [item["ticker"] for item in failed_tickers if item["reason"] == "NO_DATA"]
        )
        insufficient_data_tickers = sorted(
            [item["ticker"] for item in failed_tickers if item["reason"] == "INSUFFICIENT_DATA"]
        )

        if no_data_tickers:
            print(
                f"다음 {len(no_data_tickers)}개 티커는 데이터를 전혀 가져올 수 없습니다. 티커가 정확한지 확인 후 'data/{country_code}/1_etf_raw.txt'에서 제거하는 것을 권장합니다."
            )
            for ticker in no_data_tickers:
                print(f"- {ticker}")
            print("-" * 30)

        if insufficient_data_tickers:
            print(f"다음 {len(insufficient_data_tickers)}개 티커는 데이터 기간이 충분하지 않습니다. (상장 초기 종목 등)")
            for ticker in insufficient_data_tickers:
                print(f"- {ticker}")
            print("-" * 30)


if __name__ == "__main__":
    main()
