import json
import os
import sys
from typing import Optional

import pandas as pd

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import fetch_yfinance_name
from utils.report import format_aud_money, format_kr_money

try:
    from pykrx import stock as _stock
except Exception:
    _stock = None


def get_latest_trading_day(
    ref_date: pd.Timestamp, ref_ticker: str = "005930"
) -> Optional[pd.Timestamp]:
    """주어진 날짜 또는 그 이전의 가장 가까운 거래일을 찾습니다."""
    if _stock is None:
        print(
            "경고: pykrx가 설치되지 않아 정확한 거래일 확인이 어렵습니다. 오늘 날짜를 기준으로 진행합니다."
        )
        return ref_date

    current_date = ref_date
    for _ in range(30):  # 최대 30일 전까지 탐색
        try:
            date_str = current_date.strftime("%Y%m%d")
            df = _stock.get_market_ohlcv_by_date(date_str, date_str, ref_ticker)
            if not df.empty:
                return current_date
        except Exception:
            pass
        current_date -= pd.Timedelta(days=1)
    return None


def _convert_for_country(country: str):
    """지정된 단일 국가에 대해 포트폴리오 변환을 수행합니다."""
    data_dir = f"data/{country}"
    raw_path = os.path.join(data_dir, "portfolio_raw.txt")

    if not os.path.exists(raw_path):
        print(f"오류: '{raw_path}' 파일이 없습니다.")
        sample_content = """# 보유종목 (티커 이름 수량 매수단가)
# 각 종목은 한 줄에 하나씩 공백으로 구분하여 입력합니다.
# 이름에 공백이 포함될 수 있으며, 가격에 쉼표(,)나 '원'이 포함되어도 괜찮습니다.
# 예시:
# 005930 삼성전자 10 75000
# 091160 KODEX 반도체 250 37,980원
"""
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(sample_content)
        print(f"-> 예시 파일 '{raw_path}'을 생성했습니다. 내용을 수정 후 다시 실행해주세요.")
        return

    holdings = []

    with open(raw_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        try:
            if country == "aus":
                parts = line.strip().split("\t")
                if len(parts) != 4:
                    print(f"경고: 호주 형식이 잘못된 행입니다 (탭으로 4개 요소 필요). 건너뜁니다: '{line}'")
                    continue
                ticker, name, shares_str, avg_cost_str = parts
                shares = int(shares_str.replace(",", ""))
                avg_cost = float(avg_cost_str.replace("A$", "").replace(",", ""))
            else:  # kor
                parts = line.split()
                if len(parts) < 3:
                    print(f"경고: 한국 형식이 잘못된 행입니다. 건너뜁니다: '{line}'")
                    continue
                ticker = parts[0]
                avg_cost_str = parts[-1]
                shares_str = parts[-2]
                if len(parts) >= 4:
                    name = " ".join(parts[1:-2])
                else:
                    name = ""
                shares = int(shares_str.replace(",", ""))
                avg_cost = int(float(avg_cost_str.replace("원", "").replace(",", "")))

            if not name and country == "aus":
                name = fetch_yfinance_name(ticker)

            # 국가별 티커 형식 검증
            if country == "kor":
                if not (len(ticker) == 6 and ticker.isalnum()):
                    print(f"경고: 한국 시장의 잘못된 티커 형식입니다. 건너뜁니다: '{line}'")
                    continue

            holdings.append(
                {"ticker": ticker, "name": name, "shares": shares, "avg_cost": avg_cost}
            )
        except (ValueError, IndexError):
            print(f"경고: 잘못된 형식의 행입니다. 건너뜁니다: '{line}'")
            continue

    total_holding_value = sum(h["shares"] * h["avg_cost"] for h in holdings)

    holdings.sort(key=lambda x: x["ticker"])
    today = pd.Timestamp.now().normalize()
    archive_date = get_latest_trading_day(today) or today
    archive_date_str = archive_date.strftime("%Y-%m-%d")

    output_filename = f"portfolio_{archive_date_str}.json"
    output_path = os.path.join(data_dir, output_filename)

    # 기존 파일이 있으면 값을 보존하고, 없으면 0으로 초기화합니다.
    total_equity = 0
    had_equity_before = False
    international_shares = {"value": 0.0, "change_pct": 0.0}
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            # 기존 파일에 total_equity 값이 있고, 숫자 형식인 경우에만 값을 읽어옵니다.
            if "total_equity" in existing_data and isinstance(
                existing_data["total_equity"], (int, float)
            ):
                total_equity = float(existing_data["total_equity"])
                had_equity_before = True
                equity_str = (
                    format_aud_money(total_equity)
                    if country == "aus"
                    else format_kr_money(total_equity)
                )
                print(
                    f"-> 기존 파일 '{output_path}'에서 총평가액({equity_str})을 보존합니다."
                )
            # 호주 포트폴리오의 경우 international_shares 정보도 보존합니다.
            if country == "aus" and "international_shares" in existing_data:
                international_shares = existing_data["international_shares"]

        except (json.JSONDecodeError, ValueError, TypeError) as e:
            # 기존 파일 파싱에 실패하면 경고 후 0으로 초기화합니다.
            print(
                f"-> 경고: 기존 파일 '{output_path}'에서 총평가액을 읽는 중 오류 발생. 0으로 설정합니다. ({e})"
            )
            total_equity = 0
            had_equity_before = False

    # 국가별로 total_equity 데이터 타입 조정
    if country == "kor":
        final_total_equity = int(total_equity)
    else:
        final_total_equity = total_equity

    portfolio_archive = {"date": archive_date_str, "total_equity": final_total_equity}

    # 호주 포트폴리오에만 international_shares 필드를 추가합니다.
    if country == "aus":
        portfolio_archive["international_shares"] = international_shares

    portfolio_archive["holdings"] = holdings

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(portfolio_archive, f, ensure_ascii=False, indent=2)
        f.write("\n")

    # 국가별로 다른 포맷터 사용
    money_formatter = format_aud_money if country == "aus" else format_kr_money

    print(f"성공: '{raw_path}' -> '{output_path}' 변환 완료.")
    print(f" - 보유종목 수: {len(holdings)}")
    print(f" - 보유금액: {money_formatter(total_holding_value)}")
    if had_equity_before:
        print(
            f"-> 확인: 현재 총평가액은 {money_formatter(total_equity)} 입니다. 변경이 필요하면 '{output_path}' 파일을 직접 수정해주세요."
        )
    else:
        print(f"-> 중요: 생성된 '{output_path}' 파일을 열어 'total_equity' 값을 직접 수정해주세요.")


def main():
    """data/ 폴더 아래의 모든 국가 디렉토리에 대해 포트폴리오 변환을 실행합니다."""
    data_dir = "data"
    try:
        countries = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    except FileNotFoundError:
        print(f"오류: '{data_dir}' 디렉토리를 찾을 수 없습니다.")
        return

    if not countries:
        print(f"오류: '{data_dir}' 디렉토리에서 국가 폴더(kor, aus 등)를 찾을 수 없습니다.")
        return

    print(f"발견된 국가: {', '.join(countries)}. 모든 국가의 포트폴리오를 변환합니다.")

    for country in sorted(countries):
        print(f"\n--- [{country.upper()}] 포트폴리오 변환 시작 ---")
        _convert_for_country(country)

if __name__ == "__main__":
    main()
