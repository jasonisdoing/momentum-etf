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
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict

import pandas as pd
import requests
from pykrx import stock

# --- 설정 ---
# 이름에 아래 단어가 포함된 종목은 결과에서 제외합니다.
EXCLUDE_KEYWORDS = ["레버리지", "선물", "채권", "커버드콜", "인버스", "ETN"]


def fetch_naver_etf_data(min_change_pct: float) -> Optional[pd.DataFrame]:
    """
    네이버 금융 API에서 ETF 데이터를 가져옵니다.
    실패 시 None을 반환합니다.
    """
    from config import NAVER_FINANCE_ETF_API_URL, NAVER_FINANCE_HEADERS

    logger = get_app_logger()
    url = NAVER_FINANCE_ETF_API_URL
    headers = NAVER_FINANCE_HEADERS

    try:
        logger.info("네이버 API에서 ETF 데이터를 가져오는 중입니다...")
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()

        data = response.json()
        items = data.get("result", {}).get("etfItemList")

        if not isinstance(items, list) or not items:
            logger.warning("네이버 API 응답에 ETF 데이터가 없습니다.")
            logger.warning(f"응답 구조: {list(data.keys()) if isinstance(data, dict) else type(data).__name__}")
            if isinstance(data, dict) and "result" in data:
                logger.warning(
                    f"result 내부 키: {list(data['result'].keys()) if isinstance(data['result'], dict) else type(data['result']).__name__}"
                )
            return None

        # DataFrame 생성
        gainers_list = []
        for item in items:
            if not isinstance(item, dict):
                continue

            ticker = str(item.get("itemcode", "")).strip()
            name = str(item.get("itemname", "")).strip()
            change_rate = item.get("changeRate", 0)
            volume = item.get("quant", 0)  # 거래량
            risefall_rate = item.get("risefallRate")  # 괴리율 (None 허용)
            three_month_rate = item.get("threeMonthEarnRate")  # 3개월 수익률 (None 허용)
            now_val = item.get("nowVal")  # 현재가
            nav = item.get("nav")  # NAV

            # 등락률이 기준 이상인 종목만 추가
            try:
                change_rate_float = float(change_rate)
                volume_int = int(volume) if volume else 0

                # 괴리율: API에서 제공하면 사용, 없으면 nowVal/nav로 계산
                risefall_float = None
                if risefall_rate is not None:
                    risefall_float = float(risefall_rate)
                elif now_val is not None and nav is not None:
                    try:
                        now_val_float = float(now_val)
                        nav_float = float(nav)
                        if nav_float > 0:
                            risefall_float = ((now_val_float / nav_float) - 1.0) * 100.0
                    except (TypeError, ValueError):
                        pass

                three_month_float = float(three_month_rate) if three_month_rate is not None else None

                if change_rate_float >= min_change_pct:
                    gainers_list.append(
                        {
                            "티커": ticker,
                            "종목명": name,
                            "등락률": change_rate_float,
                            "거래량": volume_int,
                            "괴리율": risefall_float,
                            "3개월수익률": three_month_float,
                        }
                    )
            except (TypeError, ValueError):
                continue

        if not gainers_list:
            logger.warning(f"등락률 {min_change_pct:.2f}% 이상인 종목이 없습니다. (전체 ETF 수: {len(items)}개)")
            return pd.DataFrame(columns=["티커", "종목명", "등락률", "거래량", "괴리율"])

        df = pd.DataFrame(gainers_list)
        logger.info(f"네이버 API에서 {len(df)}개 종목 데이터를 가져왔습니다. (전체 ETF 수: {len(items)}개)")
        return df

    except requests.exceptions.Timeout as e:
        logger.error(f"네이버 API 타임아웃 (5초 초과): {e}")
        return None
    except requests.exceptions.HTTPError as e:
        logger.error(f"네이버 API HTTP 에러 (상태 코드: {response.status_code}): {e}")
        logger.error(f"응답 내용 (처음 500자): {response.text[:500]}")
        return None
    except requests.exceptions.ConnectionError as e:
        logger.error(f"네이버 API 연결 실패 (네트워크 확인 필요): {e}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"네이버 API 요청 실패: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"네이버 API 응답이 JSON 형식이 아닙니다: {e}")
        logger.error(f"응답 내용 (처음 500자): {response.text[:500]}")
        return None
    except Exception as e:
        logger.error(f"네이버 API 데이터 처리 중 예상치 못한 오류: {type(e).__name__}: {e}")
        import traceback

        logger.error(f"상세 오류:\n{traceback.format_exc()}")
        return None


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
    네이버 API 우선, 실패 시 pykrx 폴백 방식 사용.
    """
    logger = get_app_logger()
    try:
        latest_day = get_latest_trading_day()
        type_str = f" ({asset_type.upper()})"
        print(f"기준일: {latest_day[:4]}-{latest_day[4:6]}-{latest_day[6:]}{type_str}")

        df_change = pd.DataFrame()
        top_gainers = pd.DataFrame()

        # 1. ETF 데이터 가져오기
        if asset_type == "etf":
            # 네이버 API 시도 (빠름)
            naver_df = fetch_naver_etf_data(min_change_pct)

            if naver_df is None:
                # 네이버 API 실패 시 종료 (None 반환)
                logger.error("❌ 네이버 API 실패. 스크립트를 종료합니다.")
                print("❌ 네이버 API에서 데이터를 가져올 수 없습니다.")
                return

            # 네이버 API 성공 (빈 DataFrame도 성공)
            top_gainers = naver_df
            if not naver_df.empty:
                print(f"✅ 네이버 API 사용 (빠른 조회 성공)")

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

        if top_gainers.empty:
            print(f"등락률 {min_change_pct:.2f}% 이상 상승한 종목이 없습니다.")
            return

        # 키워드 기반 필터링
        if EXCLUDE_KEYWORDS:
            initial_count = len(top_gainers)
            # '|'로 키워드를 연결하여 정규식 OR 조건 생성
            exclude_pattern = "|".join(EXCLUDE_KEYWORDS)
            # '종목명'에 키워드가 포함되지 않은 행만 남김
            top_gainers = top_gainers[~top_gainers["종목명"].str.contains(exclude_pattern, na=False)]
            filtered_count = initial_count - len(top_gainers)
            if filtered_count > 0:
                print(f"제외 키워드({", ".join(EXCLUDE_KEYWORDS)})에 따라 {filtered_count}개 종목을 제외했습니다.")

        print(f"등락률 {min_change_pct:.2f}% 이상 상승한 종목 {len(top_gainers)}개를 찾았습니다.")

        # 필터링 후 결과 확인
        if top_gainers.empty:
            print(f"\n제외 키워드 필터링 후 남은 종목이 없습니다.")
            return

        # 등락률 순으로 정렬
        sorted_gainers = top_gainers.sort_values(by="등락률", ascending=False)

        print("\n--- 상승중인 ETF 목록 ---")
        for _, row in sorted_gainers.iterrows():
            ticker = row["티커"]
            name = row["종목명"]
            change_rate = row["등락률"]

            # 추가 정보 가져오기
            volume = row.get("거래량", 0)
            risefall = row.get("괴리율", None)

            # 3개월 수익률: 네이버 API만 사용
            three_month_rate = row.get("3개월수익률", None)

            # 거래량 포맷팅 (천 단위 구분)
            volume_str = f"{volume:,}" if volume else "N/A"

            # 3개월 수익률 포맷팅
            if three_month_rate is not None and pd.notna(three_month_rate):
                three_month_str = f"{three_month_rate:+.2f}%"
            else:
                three_month_str = "아직없음"

            # 괴리율 포맷팅
            risefall_str = f"{risefall:+.2f}%" if risefall is not None else "N/A"

            print(f"  - {name} ({ticker}): 금일수익률: +{change_rate:.2f}%, 3개월: {three_month_str}, 거래량: {volume_str}, 괴리율: {risefall_str}")

    except Exception as e:
        logger.error("오류가 발생했습니다: %s", e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="금일 상승중인 ETF를 보여줍니다.")
    parser.add_argument("--min-change", type=float, default=3.0, help="검색할 최소 등락률 (기본값: 3.0)")
    parser.add_argument(
        "--type",
        type=str,
        choices=["stock", "etf"],
        default="etf",
        help="검색할 종목 유형 (stock: 일반 주식, etf: ETF (기본값))",
    )
    args = parser.parse_args()
    find_top_gainers(min_change_pct=args.min_change, asset_type=args.type)
