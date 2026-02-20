"""
find_us.py

Barchart ETF 데이터를 수동으로 입력받아 파싱합니다.
"""

import os
import sys

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import os
import sys
import time
from datetime import datetime

import requests
from bs4 import BeautifulSoup

# --- 설정 ---
# 최소 등락률 (%)
MIN_CHANGE_PCT = 3.0
# 이름에 아래 단어가 포함된 종목은 결과에서 제외합니다.
EXCLUDE_KEYWORDS = [
    "Korea",
    "Income",
    "Yieldmax",
    "Weeklypay",
    "Month",
    "2X",
    "Long",
    "Bitcoin",
    "Ethereum",
    "Ether",
    "Xrp",
    "Solana",
    "Platinum",
    "Cannabis",
    "Copper",
    "Uranium",
    "XRP",
    "Staking",
    "Gas",
    "Oil",
    "Energy",
    "Canary",
    "coin",
    "Doge",
    "Covered",
    "Call",
    "Shipping",
    "gold",  # 이미 충분함
    "silver",  # 이미 충분함
    "2x",
    "3x",
    "YieldBOOST",
    "VIX",
]
# 이름에 아래 단어 중 하나라도 포함된 종목만 포함합니다 (빈 배열이면 모든 종목 포함).
INCLUDE_KEYWORDS = []
# 최소 거래량 (0이면 필터링 안 함)
MIN_VOLUME = 10000


def fetch_finviz_etf_data(min_change_pct):
    """
    Finviz에서 ETF 상승률 상위 데이터를 가져옵니다.
    등락률이 min_change_pct 미만으로 떨어지거나, 너무 많은 페이지를 검색하면 중단합니다.
    """
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    base_url = "https://finviz.com/screener.ashx?v=111&f=ind_exchangetradedfund&o=-change"

    etfs = []
    page = 1

    while True:
        # 페이지당 20개, 1, 21, 41...
        r_param = (page - 1) * 20 + 1
        url = f"{base_url}&r={r_param}"

        try:
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            table = soup.find("table", {"class": "styled-table-new"})
            if not table:
                print(f"⚠️ {page}페이지에서 테이블 데이터를 찾을 수 없습니다.")
                break

            rows = table.find_all("tr")
            if len(rows) <= 1:
                break

            last_change = 0.0
            added_in_page = 0

            for r in rows[1:]:
                cols = [c.text.strip() for c in r.find_all("td")]
                if len(cols) >= 11:
                    ticker = cols[1]
                    name = cols[2]

                    price_str = cols[8]
                    price = float(price_str) if price_str != "-" else 0.0

                    change_pct_str = cols[9]
                    change_pct = (
                        float(change_pct_str.replace("%", "").replace("+", "").replace(",", ""))
                        if change_pct_str != "-"
                        else 0.0
                    )
                    last_change = change_pct

                    volume_str = cols[10]
                    volume = int(volume_str.replace(",", "")) if volume_str != "-" else 0

                    etfs.append(
                        {
                            "ticker": ticker,
                            "name": name,
                            "price": price,
                            "change_pct": change_pct,
                            "volume": volume,
                        }
                    )
                    added_in_page += 1

            if added_in_page == 0:
                break

            # 최소 등락률 밑으로 떨어졌으면 탐색 종료
            if last_change < min_change_pct:
                break

        except Exception as e:
            print(f"⚠️ {page}페이지 데이터를 가져오는 중 오류 발생: {e}")
            break

        page += 1
        time.sleep(1)  # 사이트 부하 방지

        # 안전장치: 최대 10페이지 (200개 종목)까지만 탐색
        if page > 10:
            break

    return etfs


def main():
    print("=" * 70)
    print("🔗 미국 ETF 급등 종목 스캔 (Finviz)")
    print("=" * 70)
    print("\n🔍 데이터를 가져오는 중입니다...")

    # 데이터 수집
    etfs = fetch_finviz_etf_data(MIN_CHANGE_PCT)

    if not etfs:
        print("\n❌ ETF 데이터를 찾을 수 없습니다.")
        return

    # 등락률 순으로 정렬
    etfs = sorted(etfs, key=lambda x: x["change_pct"], reverse=True)

    # 최소 등락률 필터링
    initial_count = len(etfs)
    etfs = [etf for etf in etfs if etf["change_pct"] >= MIN_CHANGE_PCT]
    min_change_filtered = initial_count - len(etfs)
    if min_change_filtered > 0:
        print(f"최소 등락률({MIN_CHANGE_PCT:.2f}%)에 따라 {min_change_filtered}개 종목을 제외했습니다.")

    # INCLUDE_KEYWORDS 필터링 (OR 조건: 하나라도 포함되면 포함)
    if INCLUDE_KEYWORDS:
        before_include = len(etfs)
        include_lower = [kw.lower() for kw in INCLUDE_KEYWORDS]
        etfs = [etf for etf in etfs if any(kw in etf["name"].lower() for kw in include_lower)]
        include_filtered = before_include - len(etfs)
        if include_filtered > 0:
            print(f"포함 키워드({', '.join(INCLUDE_KEYWORDS)})에 따라 {include_filtered}개 종목을 제외했습니다.")

    # EXCLUDE_KEYWORDS 필터링
    if EXCLUDE_KEYWORDS:
        before_exclude = len(etfs)
        exclude_lower = [kw.lower() for kw in EXCLUDE_KEYWORDS]
        etfs = [etf for etf in etfs if not any(kw in etf["name"].lower() for kw in exclude_lower)]
        exclude_filtered = before_exclude - len(etfs)
        if exclude_filtered > 0:
            print(f"제외 키워드({', '.join(EXCLUDE_KEYWORDS)})에 따라 {exclude_filtered}개 종목을 제외했습니다.")

    # 거래량 필터링
    if MIN_VOLUME > 0:
        before_volume = len(etfs)
        etfs = [etf for etf in etfs if etf["volume"] >= MIN_VOLUME]
        volume_filtered = before_volume - len(etfs)
        if volume_filtered > 0:
            print(f"최소 거래량({MIN_VOLUME:,})에 따라 {volume_filtered}개 종목을 제외했습니다.")

    # 필터링 후 결과가 있는지 확인
    if not etfs:
        print("\n제외 키워드 필터링 후 남은 종목이 없습니다.")
        return

    # 최종 결과 메시지
    print(f"등락률 {MIN_CHANGE_PCT:.2f}% 이상 상승한 종목 {len(etfs)}개를 찾았습니다.")

    # 결과 출력
    print(f"\n✅ {len(etfs)}개 ETF 데이터를 찾았습니다.")
    print()
    print("=" * 70)
    print(f"📅 조회 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    print()
    # 기존 종목 로드 (MongoDB)
    from collections import defaultdict

    from utils.stock_list_io import get_deleted_etfs, get_etfs

    target_accounts = ["us"]  # 미국 계좌만 확인

    existing_tickers_map = defaultdict(list)  # ticker -> list of account_ids
    deleted_tickers_map = defaultdict(list)  # ticker -> list of {account_id, deleted_at, deleted_reason}

    for account in target_accounts:
        try:
            # 활성 종목
            existing_etfs = get_etfs(account)
            for item in existing_etfs:
                existing_tickers_map[item["ticker"]].append(account)

            # 삭제된 종목
            deleted_list = get_deleted_etfs(account)
            for item in deleted_list:
                t = item.get("ticker")
                if t:
                    info = item.copy()
                    info["account_id"] = account
                    deleted_tickers_map[t].append(info)

        except Exception as e:
            print(f"⚠️ {account} 종목 로드 중 오류 발생: {e}")

    # 분류
    my_universe_list = []
    deleted_list = []
    new_discovery_list = []

    for item in etfs:
        ticker = item["ticker"]

        # 딕셔너리 키 통일 (find_kor와 맞춤)
        item["티커"] = item["ticker"]
        item["종목명"] = item["name"]
        item["등락률"] = item["change_pct"]
        item["거래량"] = item["volume"]
        item["현재가"] = item["price"]
        item["괴리율"] = None  # US는 괴리율 정보 없음
        item["3개월수익률"] = None  # US는 3개월 수익률 정보 없음

        if ticker in existing_tickers_map:
            # 계좌 정보 추가
            item["accounts"] = existing_tickers_map[ticker]
            my_universe_list.append(item)
        elif ticker in deleted_tickers_map:
            # 삭제 정보 추가
            item["deleted_infos"] = deleted_tickers_map[ticker]
            deleted_list.append(item)
        else:
            new_discovery_list.append(item)

    # 출력 헬퍼
    def print_item(item, is_deleted=False):
        ticker = item["티커"]
        name = item["종목명"]
        change_rate = item["등락률"]
        price = item["현재가"]
        volume = item.get("거래량", 0)
        volume_str = f"{volume:,}" if volume else "N/A"

        # 이름이 너무 길면 자르기
        if len(name) > 45:
            name = name[:42] + "..."

        # 계좌 표시
        accounts_str = ""
        if "accounts" in item:
            accounts_str = f"[{', '.join(item['accounts'])}] "

        base_msg = f"  - {accounts_str}{name} ({ticker}): 금일수익률: +{change_rate:.2f}%, 현재가: ${price:.2f}, 거래량: {volume_str}"

        if is_deleted:
            deleted_infos = item.get("deleted_infos", [])
            del_msg_parts = []
            for info in deleted_infos:
                acc = info.get("account_id", "?")
                d_date = info.get("deleted_at")
                d_reason = info.get("deleted_reason") or "사유없음"

                date_str = ""
                if d_date:
                    if hasattr(d_date, "strftime"):
                        date_str = d_date.strftime("%Y-%m-%d")
                    else:
                        date_str = str(d_date)[:10]
                del_msg_parts.append(f"[{acc}] {date_str} ({d_reason})")

            del_msg = " | ".join(del_msg_parts)
            print(f"{base_msg} | 🗑️ 삭제: {del_msg}")
        else:
            print(base_msg)

    # 1. 내 유니버스
    if my_universe_list:
        print()
        print("--- 내 유니버스 ETF 목록 ---")
        for item in my_universe_list:
            print_item(item)

    # 2. 삭제된 목록
    if deleted_list:
        print()
        print("--- 삭제된 ETF 목록 ---")
        for item in deleted_list:
            print_item(item, is_deleted=True)

    # 3. 신규 발견
    if new_discovery_list:
        print()
        print("--- 신규 발견 종목 ---")
        for item in new_discovery_list:
            print_item(item)
    else:
        print("\n✅ 신규로 발견된 종목이 없습니다 (모두 등록됨 혹은 삭제됨).")


if __name__ == "__main__":
    main()
