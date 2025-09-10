#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
clean_etf_list.py

data/kor/tickers_etf.txt 파일에 있는 ETF 목록을 정리하는 스크립트입니다.
etfcheck.co.kr의 분류를 기준으로 유사 테마 ETF를 그룹화하고,
각 그룹에서 거래대금이 가장 많은
대표 ETF 1개만 남깁니다.

[사용법]
python clean_etf_list.py
"""

import os
import re
import json
import shutil
from datetime import datetime, timedelta

import pandas as pd
from pykrx import stock

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("오류: 'requests'와 'beautifulsoup4' 라이브러리가 필요합니다.")
    print("pip install requests beautifulsoup4")
    requests = None
    BeautifulSoup = None

# --- 설정 ---
ETF_TICKER_FILE = "data/kor/tickers_etf.txt"
ETF_TICKER_BACKUP_FILE = "data/kor/tickers_etf.txt.bak"

def get_latest_trading_day() -> str:
    """가장 가까운 과거의 거래일을 'YYYYMMDD' 형식으로 반환합니다."""
    dt = datetime.now()
    for i in range(10):
        date_str = (dt - timedelta(days=i)).strftime("%Y%m%d")
        df = stock.get_market_ohlcv_by_date(date_str, date_str, "005930")
        if not df.empty:
            return date_str
    return datetime.now().strftime("%Y%m%d")

def build_etf_theme_map() -> dict:
    """
    KRX 정보데이터시스템(etfcheck.co.kr)을 스크레이핑하여 {티커: 테마} 맵을 생성합니다.
    """
    if not requests or not BeautifulSoup:
        return {}

    base_url = "https://www.etfcheck.co.kr"
    list_url = f"{base_url}/mobile/krx/etpctg/etpList.do"
    main_page_url = f"{base_url}/mobile/krx/etpctg/0101?etpType=ETF"

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    ticker_to_theme = {}
    print("etfcheck.co.kr에서 ETF 분류 정보를 가져옵니다...")

    try:
        # 1. 메인 페이지에서 그룹 목록 가져오기
        response = requests.get(main_page_url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        script_tags = soup.find_all('script')
        idx_list_str = None
        for script in script_tags:
            script_content = script.get_text()
            if 'var idxList' in script_content:
                match = re.search(r'var\s+idxList\s*=\s*(\[.*\]);', script_content, re.DOTALL)
                if match:
                    idx_list_str = match.group(1)
                    break

        if not idx_list_str:
            print("오류: ETF 그룹 목록(idxList)을 찾을 수 없습니다.")
            return {}

        idx_list = json.loads(idx_list_str)

        # 2. 각 그룹/서브그룹별로 ETF 목록 스크레이핑
        for main_group in idx_list:
            main_group_id = main_group['val']
            main_group_name = main_group['text']

            sub_list = main_group.get('subList', [])
            if not sub_list:
                # 서브그룹이 없는 경우 (예: 파생상품)
                sub_list = [{'val': '', 'text': main_group_name}]

            for sub_group in sub_list:
                sub_group_id = sub_group['val']
                sub_group_name = sub_group['text']

                payload = {'etpType': 'ETF', 'idxId': main_group_id, 'idxIndMidclss': sub_group_id}
                resp = requests.post(list_url, data=payload, headers=headers, timeout=10)
                resp.raise_for_status()
                sub_soup = BeautifulSoup(resp.text, 'html.parser')

                for row in sub_soup.select('tbody > tr'):
                    ticker = row.get('data-id')
                    if ticker:
                        theme_name = f"{main_group_name} > {sub_group_name}" if sub_group_id else main_group_name
                        ticker_to_theme[ticker] = theme_name

    except requests.exceptions.RequestException as e:
        print(f"오류: 웹사이트에 연결할 수 없습니다: {e}")
    except Exception as e:
        print(f"오류: ETF 테마 맵 생성 중 오류 발생: {e}")

    print(f"총 {len(ticker_to_theme)}개 ETF의 분류 정보를 가져왔습니다.")
    return ticker_to_theme

def clean_etf_list():
    """tickers_etf.txt 파일을 정리하는 메인 함수"""
    if not os.path.exists(ETF_TICKER_FILE):
        print(f"오류: '{ETF_TICKER_FILE}' 파일을 찾을 수 없습니다.")
        return

    print(f"'{ETF_TICKER_FILE}' 파일의 ETF 목록을 정리합니다.")

    # 1. 파일에서 티커 목록 읽기
    with open(ETF_TICKER_FILE, "r", encoding="utf-8") as f:
        tickers = [line.split()[0] for line in f if line.strip() and not line.startswith("#")]

    if not tickers:
        print("파일에 정리할 ETF가 없습니다.")
        return

    print(f"총 {len(tickers)}개의 ETF를 분석합니다...")

    # 2. KRX 정보데이터시스템에서 테마 정보 가져오기
    ticker_to_theme = build_etf_theme_map()
    if not ticker_to_theme:
        print("오류: ETF 테마 정보를 가져오지 못해 작업을 중단합니다.")
        return

    # 3. ETF 정보 및 거래대금 데이터 가져오기
    try:
        latest_day = get_latest_trading_day()
        print(f"기준일: {latest_day}")
        df_ohlcv = stock.get_etf_ohlcv_by_ticker(latest_day)
    except Exception as e:
        print(f"오류: pykrx로 ETF 정보를 가져오는 중 문제가 발생했습니다: {e}")
        return

    etf_details = []
    for ticker in tickers:
        try:
            name = stock.get_etf_ticker_name(ticker)
            # 웹에서 가져온 테마 정보 사용, 없으면 '기타'로 분류
            theme = ticker_to_theme.get(ticker, "기타")
            trading_value = df_ohlcv.loc[ticker, "거래대금"]
            etf_details.append({"ticker": ticker, "name": name, "theme": theme, "value": int(trading_value)})
        except KeyError:
            print(f"경고: 티커 '{ticker}'의 정보를 찾을 수 없습니다. 건너뜁니다.")
        except Exception as e:
            print(f"경고: 티커 '{ticker}' 처리 중 오류 발생: {e}")

    # 4. 테마별로 그룹화하고 대표 ETF 선정 (거래대금 기준)
    df = pd.DataFrame(etf_details)
    grouped = df.groupby("theme")
    selected_etfs = []
    print("\n--- 테마별 ETF 그룹 및 대표 선정 결과 ---")
    sorted_themes = sorted(list(grouped.groups.keys()))

    for theme in sorted_themes:
        group = grouped.get_group(theme).sort_values(by="value", ascending=False)
        representative_etf = group.iloc[0]
        selected_etfs.append(representative_etf)
        print(f"\n# 테마: {theme} (총 {len(group)}개)")
        for _, etf in group.iterrows():
            is_rep = "  -> [대표 선정]" if etf['ticker'] == representative_etf['ticker'] else ""
            print(f"  - {etf['name']} ({etf['ticker']}) | 거래대금: {etf['value']:,}{is_rep}")

    # 4. 사용자 확인 및 파일 쓰기
    print(f"\n--- 최종 정리 결과: {len(selected_etfs)}개 ---")
    final_df = pd.DataFrame(selected_etfs).sort_values(by="name").reset_index(drop=True)
    if input(f"\n위와 같이 '{ETF_TICKER_FILE}' 파일을 정리하시겠습니까? (y/n): ").lower() == 'y':
        shutil.copy(ETF_TICKER_FILE, ETF_TICKER_BACKUP_FILE)
        print(f"\n원본 파일을 '{ETF_TICKER_BACKUP_FILE}'(으)로 백업했습니다.")
        with open(ETF_TICKER_FILE, "w", encoding="utf-8") as f:
            f.write(f"# ETF 목록 (유사 테마 그룹별 거래대금 1위, {datetime.now().strftime('%Y-%m-%d')} 정리)\n\n")
            for _, etf in final_df.iterrows():
                f.write(f"{etf['ticker']} {etf['name']}\n")
        print(f"'{ETF_TICKER_FILE}' 파일이 성공적으로 업데이트되었습니다.")
    else:
        print("\n작업이 취소되었습니다. 파일은 변경되지 않았습니다.")

if __name__ == "__main__":
    clean_etf_list()