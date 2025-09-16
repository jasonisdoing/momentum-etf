#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ETF_raw.txt 파일(aus, kor)에서 ETF 티커를 읽어 중복을 제거하고, AI를 사용하여
카테고리를 분류한 후 aus_etf_categorized.csv 파일로 저장합니다.

[사전 준비]
1. pip install google-generativeai
2. Google AI Studio에서 API 키를 발급받아 'GOOGLE_API_KEY' 환경 변수로 설정합니다.

[사용법]
python scripts/categorize_etf.py <country>
예: python scripts/categorize_etf.py aus
예: python scripts/categorize_etf.py kor
"""

import argparse
import os
import sys

import pandas as pd

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import fetch_pykrx_name, fetch_yfinance_name
from utils.etf_categorizer import SECTORS, classify_etf_with_ai

try:
    from dotenv import load_dotenv
except ImportError:
    print("오류: 'python-dotenv' 라이브러리가 설치되지 않았습니다.")
    print("pip install python-dotenv 명령으로 설치해주세요.")
    sys.exit(1)

try:
    from google.api_core import exceptions as google_exceptions
except ImportError:
    print("오류: 'google-api-core' 라이브러리가 설치되지 않았습니다.")
    print("pip install google-api-core 명령으로 설치해주세요.")
    sys.exit(1)

try:
    import google.generativeai as genai
except ImportError:
    print("오류: 'google-generativeai' 라이브러리가 설치되지 않았습니다.")
    print("pip install google-generativeai 명령으로 설치해주세요.")
    sys.exit(1)


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="AI를 사용하여 ETF를 섹터별로 분류합니다.")
    parser.add_argument("country", choices=["kor", "aus"], help="분류할 국가 코드 (kor 또는 aus)")
    args = parser.parse_args()
    country_code = args.country

    # .env 파일에서 환경 변수를 로드합니다.
    load_dotenv()

    # 1. API 키 확인
    google_api_key = os.environ.get("GOOGLE_API_KEY")

    google_model = None
    if google_api_key:
        genai.configure(api_key=google_api_key)
        google_model = genai.GenerativeModel("gemini-2.0-flash")
    else:
        print("오류: 'GOOGLE_API_KEY' 환경 변수가 설정되지 않았습니다.")
        return

    # 2. raw 파일에서 중복을 제거한 티커 목록을 읽습니다.
    raw_file_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", country_code, "1_etf_raw.txt"
    )
    if not os.path.exists(raw_file_path):
        print(f"오류: '{raw_file_path}' 파일을 찾을 수 없습니다.")
        return

    unique_tickers = set()
    with open(raw_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                ticker_candidate = line.split()[0]
                if country_code == "aus" and ":" not in ticker_candidate:
                    continue
                unique_tickers.add(ticker_candidate.upper())

    if not unique_tickers:
        print("파일에 분석할 티커가 없습니다.")
        return

    sorted_unique_tickers = sorted(list(unique_tickers))

    # 3. raw 파일을 중복 제거된 티커 목록으로 덮어씁니다.
    try:
        with open(raw_file_path, "w", encoding="utf-8") as f:
            for ticker in sorted_unique_tickers:
                f.write(f"{ticker}\n")
        print(f"'{raw_file_path}' 파일의 중복을 제거하고 정렬하여 다시 썼습니다.")
    except IOError as e:
        print(f"\n'{raw_file_path}' 파일 쓰기 중 오류 발생: {e}")
        return

    # 4. 기존에 처리된 결과를 불러오고, 처리할 티커 목록을 결정합니다.
    categorized_csv_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "data", country_code, "2_etf_categorized.csv"
    )
    processed_results = []
    already_processed_tickers = set()
    if os.path.exists(categorized_csv_path):
        try:
            df_existing = pd.read_csv(categorized_csv_path, dtype={"ticker": str})
            if "ticker" in df_existing.columns:
                already_processed_tickers = set(df_existing["ticker"].astype(str))
                processed_results = df_existing.to_dict("records")
                print(f"\n기존에 처리된 {len(processed_results)}개 ETF 정보를 불러왔습니다.")
        except Exception as e:
            print(f"경고: 기존 분류 파일을 읽는 중 오류 발생: {e}")

    tickers_to_process = [t for t in sorted_unique_tickers if t not in already_processed_tickers]

    if not tickers_to_process:
        print("\n새롭게 분류할 ETF가 없습니다. 모든 티커가 이미 처리되었습니다.")
        return

    # 5. 각 티커의 이름과 카테고리를 처리합니다.
    print(f"\n총 {len(tickers_to_process)}개 신규 ETF의 이름과 카테고리를 AI로 분류합니다...")
    failed_tickers = []

    for i, ticker in enumerate(tickers_to_process):
        if country_code == "aus":
            ticker_part = ticker.split(":")[-1]
            name = fetch_yfinance_name(ticker_part)
        elif country_code == "kor":
            name = fetch_pykrx_name(ticker)
        else:
            name = None

        if not name:
            print(f"    - 이름 조회 실패: {ticker}")
            failed_tickers.append(ticker)
            continue
        try:
            print(f"  - 처리 중 ({i + 1}/{len(tickers_to_process)}): {ticker} (using Google AI)")
            category = classify_etf_with_ai(name, google_model)

            # 1. 새로운 결과를 메모리 리스트에 추가
            processed_results.append({"ticker": ticker, "name": name, "category": category})

            # 2. 현재까지의 모든 결과를 DataFrame으로 변환하여 파일에 즉시 저장
            df_to_save = pd.DataFrame(processed_results)
            df_to_save.sort_values(by="ticker", inplace=True)
            df_to_save.to_csv(categorized_csv_path, index=False, encoding="utf-8-sig")
            print(f"    -> '{ticker}' 처리 완료. CSV 파일 업데이트됨.")

            # Google AI API의 무료 티어는 분당 15회, 일일 50회 요청으로 제한될 수 있습니다.
            # 속도 제한 오류(429)를 피하기 위해 요청 사이에 충분한 지연을 둡니다.
            # time.sleep(5)

        except google_exceptions.ResourceExhausted:
            print("\n" + "!" * 50)
            print("오류: Google AI API 일일 사용량 한도(Free Tier)를 초과했습니다.")
            print("오늘 처리된 내역까지 저장하고 프로그램을 종료합니다.")
            print("내일 다시 실행하면 중단된 부분부터 이어서 처리됩니다.")
            return
        except Exception as e:
            print(f"    - AI 분류 중 예상치 못한 오류 발생: {e}")
            failed_tickers.append(ticker)

    # 6. 최종 요약 출력
    if processed_results:
        df_final = pd.DataFrame(processed_results)
        print(
            f"\n분류가 완료되었습니다. '{categorized_csv_path}' 파일에 총 {len(df_final)}개 ETF 정보가 저장되었습니다."
        )
        print("\n[분류 결과 요약]")
        print(df_final["category"].value_counts())
    else:
        print("\n처리된 ETF가 없어 CSV 파일을 업데이트하지 않았습니다.")

    # 7. 처리 실패한 티커를 콘솔에 출력합니다.
    if failed_tickers:
        print("\n" + "=" * 30)
        print("처리 실패 티커 목록")
        print("=" * 30)
        print(
            f"다음 {len(failed_tickers)}개 티커를 처리하지 못했습니다. 'data/{country_code}/1_etf_raw.txt'에서 확인 후 제거하는 것을 권장합니다."
        )
        for ticker in sorted(failed_tickers):
            print(f"- {ticker}")
        print("=" * 30)


if __name__ == "__main__":
    main()
