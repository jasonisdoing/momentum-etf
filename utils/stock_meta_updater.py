"""
종목 파일(stocks/*.json)에 메타데이터(상장일, 주간 평균 거래량/거래대금)를 업데이트합니다.
"""

import json
import os
import time
from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET

import pandas as pd
import requests
import yfinance as yf

from utils.logger import get_app_logger


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STOCKS_DIR = PROJECT_ROOT / "data" / "stocks"


def _fetch_naver_listing_date(ticker: str) -> Optional[str]:
    """
    네이버 차트 API에서 한국 ETF의 실제 상장일을 가져옵니다.

    Args:
        ticker: 종목 코드 (예: 379800)

    Returns:
        상장일 문자열 (YYYY-MM-DD) 또는 None
    """
    from config import NAVER_FINANCE_CHART_API_URL

    logger = get_app_logger()
    url = f"{NAVER_FINANCE_CHART_API_URL}?symbol={ticker}&timeframe=day&count=1&requestType=0"

    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()

        # XML 파싱
        root = ET.fromstring(response.content)
        chartdata = root.find("chartdata")

        if chartdata is not None:
            origintime = chartdata.get("origintime")
            if origintime and len(origintime) == 8:  # YYYYMMDD 형식
                # YYYYMMDD -> YYYY-MM-DD 변환
                listing_date = f"{origintime[:4]}-{origintime[4:6]}-{origintime[6:8]}"
                logger.debug(f"[네이버 API] {ticker} 상장일: {listing_date}")
                return listing_date

    except Exception as e:
        logger.debug(f"[네이버 API] {ticker} 상장일 조회 실패: {e}")

    return None


def _get_cache_start_date() -> Optional[pd.Timestamp]:
    """config.py에서 캐시 시작일을 불러옵니다."""
    try:
        from utils.settings_loader import load_common_settings

        common_settings = load_common_settings()
        raw = common_settings.get("CACHE_START_DATE")
    except Exception:
        return None

    if not raw:
        return None

    try:
        ts = pd.to_datetime(raw).normalize()
    except Exception:
        return None
    if isinstance(ts, pd.DatetimeIndex):
        ts = ts[0]
    if isinstance(ts, pd.Timestamp):
        if ts.tzinfo is not None:
            ts = ts.tz_localize(None)
        return ts.normalize()
    return None


def _update_metadata_for_country(country_code: str):
    """지정된 국가의 모든 종목에 대한 메타데이터를 업데이트합니다."""
    logger = get_app_logger()
    stock_file = STOCKS_DIR / f"{country_code}.json"

    if not stock_file.exists():
        logger.warning(f"'{stock_file}'을 찾을 수 없어 메타데이터 업데이트를 건너<binary data, 1 bytes>니다.")
        return

    try:
        with stock_file.open("r", encoding="utf-8") as f:
            stock_data = json.load(f)
    except Exception as e:
        logger.error(f"'{stock_file}' 파일 로딩 실패: {e}")
        return

    updated_count = 0
    for category in stock_data:
        for stock in category.get("tickers", []):
            ticker = stock.get("ticker")
            if not ticker:
                continue

            # 이미 상장일이 있는지 확인
            existing_listing_date = stock.get("listing_date")
            has_listing_date = bool(existing_listing_date)

            yfinance_ticker = ticker
            if country_code == "aus":
                if ticker.upper().startswith("ASX:"):
                    yfinance_ticker = f"{ticker[4:]}.AX"
                elif not ticker.endswith(".AX"):
                    yfinance_ticker = f"{ticker}.AX"
            elif country_code == "kor":
                yfinance_ticker = f"{ticker}.KS"

            try:
                listing_date_str = None
                data = None  # 각 종목마다 data 변수 초기화

                # 이미 상장일이 있으면 스킵 (거래량/거래대금만 업데이트)
                if has_listing_date:
                    listing_date_str = existing_listing_date
                    logger.debug(f"[{country_code.upper()}/{ticker}] 상장일 이미 존재: {listing_date_str}, 스킵")
                else:
                    # 한국 ETF의 경우 네이버 API 우선 시도
                    if country_code == "kor":
                        listing_date_str = _fetch_naver_listing_date(ticker)
                        if listing_date_str:
                            logger.info(f"[{country_code.upper()}/{ticker}] 네이버 API에서 상장일 획득: {listing_date_str}")

                # 네이버 API 실패 시 또는 호주 ETF인 경우 yfinance 폴백
                if not listing_date_str:
                    logger.debug(f"[{country_code.upper()}/{ticker}] yfinance로 폴백하여 상장일 조회")
                    # yfinance를 통해 전체 기간 데이터 다운로드
                    data = yf.download(yfinance_ticker, period="max", progress=False, auto_adjust=False)
                    if data.empty:
                        logger.warning(f"[{country_code.upper()}/{ticker}] 데이터를 가져올 수 없습니다.")
                        continue

                    # yfinance가 MultiIndex 컬럼을 반환하는 경우 단일 레벨로 정리
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.get_level_values(0)
                        data = data.loc[:, ~data.columns.duplicated()]

                    # 중복된 인덱스가 있을 경우 마지막 항목만 남김
                    if not data.index.is_unique:
                        data = data[~data.index.duplicated(keep="last")]

                    # 1. 상장일 업데이트 (실제 상장일 저장)
                    first_trading_ts = pd.Timestamp(data.index.min()).normalize()
                    listing_date_str = first_trading_ts.strftime("%Y-%m-%d")
                    logger.info(f"[{country_code.upper()}/{ticker}] yfinance에서 상장일 획득: {listing_date_str}")

                # 상장일 저장
                stock["listing_date"] = listing_date_str

                # 주간 평균 거래량/거래대금 및 3개월 수익률은 항상 업데이트 (yfinance 데이터 필요)
                if data is None:
                    # 상장일이 이미 있어서 data가 없는 경우, yfinance로 3개월 데이터 조회
                    data = yf.download(yfinance_ticker, period="3mo", progress=False, auto_adjust=False)
                    if data.empty:
                        logger.warning(f"[{country_code.upper()}/{ticker}] 거래량/수익률 데이터를 가져올 수 없습니다.")
                    else:
                        if isinstance(data.columns, pd.MultiIndex):
                            data.columns = data.columns.get_level_values(0)
                            data = data.loc[:, ~data.columns.duplicated()]
                        if not data.index.is_unique:
                            data = data[~data.index.duplicated(keep="last")]

                # 2. 주간 평균 거래량/거래대금 업데이트
                if data is not None and not data.empty and len(data) >= 1:
                    # 거래대금 컬럼 추가
                    data["Turnover"] = data["Close"] * data["Volume"]
                    last_7_days = data.tail(7)

                    avg_volume = last_7_days["Volume"].mean()
                    avg_turnover = last_7_days["Turnover"].mean()

                    stock["1_week_avg_volume"] = int(avg_volume) if pd.notna(avg_volume) else None
                    stock["1_week_avg_turnover"] = int(avg_turnover) if pd.notna(avg_turnover) else None

                    # 3. 3개월 수익률 계산 (yfinance 데이터 사용)
                    if len(data) >= 2 and "Close" in data.columns:
                        price_start = data.iloc[0]["Close"]
                        price_end = data.iloc[-1]["Close"]
                        if pd.notna(price_start) and pd.notna(price_end) and price_start > 0:
                            earn_rate = ((price_end - price_start) / price_start) * 100
                            stock["3_month_earn_rate"] = round(earn_rate, 4)
                        else:
                            stock["3_month_earn_rate"] = None
                    else:
                        stock["3_month_earn_rate"] = None
                else:
                    stock["1_week_avg_volume"] = None
                    stock["1_week_avg_turnover"] = None
                    stock["3_month_earn_rate"] = None
                logger.info(f"[{country_code.upper()}/{ticker}] 메타데이터 획득")
                updated_count += 1
                time.sleep(0.2)  # API 호출 속도 조절

            except Exception as e:
                logger.error(f"[{country_code.upper()}/{ticker}] 메타데이터 업데이트 실패: {e}")

    if updated_count > 0:
        try:
            with stock_file.open("w", encoding="utf-8") as f:
                json.dump(stock_data, f, ensure_ascii=False, indent=4)
            logger.info(f"✅ [{country_code.upper()}] {updated_count}개 종목의 메타데이터 업데이트 완료.")
        except Exception as e:
            logger.error(f"'{stock_file}' 파일 저장 실패: {e}")


def update_stock_metadata():
    """지원하는 모든 국가의 종목 메타데이터를 업데이트합니다."""
    logger = get_app_logger()
    logger.info("종목 메타데이터 업데이트를 시작합니다...")

    supported_countries = ["aus", "kor", "us"]  # 필요에 따라 국가 추가
    for country in supported_countries:
        stock_file = STOCKS_DIR / f"{country}.json"
        if stock_file.exists():
            logger.info(f"[{country.upper()}] 메타데이터 작업 시작")
            _update_metadata_for_country(country)

    logger.info("모든 종목 메타데이터 업데이트 완료.")


if __name__ == "__main__":
    update_stock_metadata()
