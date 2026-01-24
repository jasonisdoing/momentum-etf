"""
find_us.py

Barchart ETF 데이터를 수동으로 입력받아 파싱합니다.
"""

from datetime import datetime

# --- 설정 ---
# 최소 등락률 (%)
MIN_CHANGE_PCT = 3.0
# 이름에 아래 단어가 포함된 종목은 결과에서 제외합니다.
EXCLUDE_KEYWORDS = [
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
]
# 이름에 아래 단어 중 하나라도 포함된 종목만 포함합니다 (빈 배열이면 모든 종목 포함).
INCLUDE_KEYWORDS = []
# 최소 거래량 (0이면 필터링 안 함)
MIN_VOLUME = 500000


def parse_barchart_data(text):
    """
    Barchart에서 복사한 텍스트 데이터를 파싱합니다.

    Barchart 형식: 10줄씩 하나의 ETF
    1. 티커
    2. 종목명
    3. 현재가
    4. 변동금액
    5. 변동률
    6. 고가
    7. 저가
    8. 거래량
    9. 달러거래량
    10. 날짜
    """
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    etfs = []

    # 10줄씩 묶어서 처리
    for i in range(0, len(lines), 10):
        if i + 4 >= len(lines):  # 최소한 티커, 이름, 가격, 변동, 변동률이 있어야 함
            break

        try:
            ticker = lines[i].strip()
            name = lines[i + 1].strip()
            price_str = lines[i + 2].strip()
            change_pct_str = lines[i + 4].strip()

            # 변동률 파싱 (+4.53% 형태)
            if "%" in change_pct_str:
                change_pct = float(change_pct_str.replace("%", "").replace("+", "").replace(",", ""))
            else:
                continue

            # 가격 파싱
            try:
                price = float(price_str.replace(",", ""))
            except ValueError:  # bare except 수정
                price = 0.0

            # 거래량 파싱 (8번째 줄)
            volume = 0
            if i + 7 < len(lines):
                volume_str = lines[i + 7].strip()
                try:
                    volume = int(volume_str.replace(",", ""))
                except ValueError:  # bare except 수정
                    volume = 0

            etfs.append(
                {
                    "ticker": ticker,
                    "name": name,
                    "price": price,
                    "change_pct": change_pct,
                    "volume": volume,
                }
            )

        except (ValueError, IndexError):
            continue

    return etfs


def main():
    print("=" * 70)
    print("🔗 미국 ETF Top 100 (등락률 순)")
    print("=" * 70)
    print()
    print("1. 아래 링크를 브라우저에서 여세요:")
    print()
    print(
        "   https://www.barchart.com/etfs-funds/performance/percent-change/advances?orderBy=percentChange&orderDir=desc"
    )
    print()
    print("2. 페이지가 로드되면 테이블 데이터를 선택하여 복사하세요")
    print("   (티커, 이름, 가격, 등락률 등이 포함된 행들)")
    print()
    print("3. 아래에 복사한 텍스트를 붙여넣고 Enter를 두 번 누르세요:")
    print()
    print("-" * 70)

    # 멀티라인 입력 받기
    lines = []
    print("(텍스트 붙여넣기 후 빈 줄에서 Enter를 두 번 누르세요)")
    empty_count = 0
    while True:
        try:
            line = input()
            if line.strip() == "":
                empty_count += 1
                if empty_count >= 2:
                    break
            else:
                empty_count = 0
            lines.append(line)
        except EOFError:
            break

    text = "\n".join(lines)

    if not text.strip():
        print("\n❌ 입력된 데이터가 없습니다.")
        return

    print("\n" + "=" * 70)
    print("🔍 데이터 파싱 중...")
    print("=" * 70)

    # 데이터 파싱
    etfs = parse_barchart_data(text)

    if not etfs:
        print("\n❌ ETF 데이터를 찾을 수 없습니다.")
        print("\n💡 팁: 테이블 전체를 선택하여 복사하세요 (헤더 포함)")
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
        etfs = [etf for etf in etfs if any(kw in etf["name"] for kw in INCLUDE_KEYWORDS)]
        include_filtered = before_include - len(etfs)
        if include_filtered > 0:
            print(f"포함 키워드({', '.join(INCLUDE_KEYWORDS)})에 따라 {include_filtered}개 종목을 제외했습니다.")

    # EXCLUDE_KEYWORDS 필터링
    if EXCLUDE_KEYWORDS:
        before_exclude = len(etfs)
        etfs = [etf for etf in etfs if not any(kw in etf["name"] for kw in EXCLUDE_KEYWORDS)]
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
    print("--- 상승중인 ETF 목록 ---")
    print()

    for etf in etfs:
        ticker = etf["ticker"]
        name = etf["name"]
        change_pct = etf["change_pct"]
        price = etf["price"]
        volume = etf["volume"]

        # 이름이 너무 길면 자르기
        if len(name) > 45:
            name = name[:42] + "..."

        volume_str = f"{volume:,}" if volume > 0 else "N/A"

        print(f"  - {name} ({ticker}): 금일수익률: +{change_pct:.2f}%, 현재가: ${price:.2f}, 거래량: {volume_str}")

    print()
    print("=" * 70)

    # 기존 stocks.json 로드 및 비교
    import json
    import os

    existing_tickers = set()
    stocks_json_path = os.path.join("zaccounts", "us", "stocks.json")

    try:
        if os.path.exists(stocks_json_path):
            with open(stocks_json_path, encoding="utf-8") as f:
                data = json.load(f)
                for category in data:
                    for item in category.get("tickers", []):
                        existing_tickers.add(item.get("ticker"))
    except Exception as e:
        print(f"\n⚠️ stocks.json 로드 중 오류 발생: {e}")

    new_tickers = [etf for etf in etfs if etf["ticker"] not in existing_tickers]

    if new_tickers:
        print()
        print("--- 신규 발견 종목 ---")
        print()
        for etf in new_tickers:
            ticker = etf["ticker"]
            name = etf["name"]
            change_pct = etf["change_pct"]
            price = etf["price"]
            volume = etf["volume"]

            # 이름이 너무 길면 자르기
            if len(name) > 45:
                name = name[:42] + "..."

            volume_str = f"{volume:,}" if volume > 0 else "N/A"
            print(f"  - {name} ({ticker}): 금일수익률: +{change_pct:.2f}%, 현재가: ${price:.2f}, 거래량: {volume_str}")
        print()
        print("+" * 70)
    else:
        print("\n✅ 발견된 모든 종목이 이미 stocks.json에 존재합니다.")


if __name__ == "__main__":
    main()
