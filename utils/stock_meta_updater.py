"""
종목 파일(stocks/*.json)에 메타데이터(상장일, 주간 평균 거래량/거래대금)를 업데이트합니다.
"""

import json
import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

from utils.logger import get_app_logger


PROJECT_ROOT = Path(__file__).resolve().parent.parent
STOCKS_DIR = PROJECT_ROOT / "data" / "stocks"


def _get_cache_start_date() -> Optional[pd.Timestamp]:
    """환경 변수에서 캐시 시작일을 불러오거나 기본값을 반환합니다."""
    raw = os.environ.get("CACHE_START_DATE")
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

    cache_start_ts = _get_cache_start_date()
    updated_count = 0
    for category in stock_data:
        for stock in category.get("tickers", []):
            ticker = stock.get("ticker")
            if not ticker:
                continue

            yfinance_ticker = ticker
            if country_code == "aus":
                if ticker.upper().startswith("ASX:"):
                    yfinance_ticker = f"{ticker[4:]}.AX"
                elif not ticker.endswith(".AX"):
                    yfinance_ticker = f"{ticker}.AX"
            elif country_code == "kor":
                # pykrx는 상장일 정보를 직접 제공하지 않으므로 yfinance로 조회
                yfinance_ticker = f"{ticker}.KS"

            try:
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

                # 1. 상장일 업데이트
                first_trading_ts = pd.Timestamp(data.index.min()).normalize()
                listing_target_ts = first_trading_ts

                existing_listing_raw = stock.get("listing_date")
                existing_listing_ts: Optional[pd.Timestamp] = None
                if existing_listing_raw:
                    try:
                        existing_listing_ts = pd.to_datetime(existing_listing_raw).normalize()
                    except Exception:
                        existing_listing_ts = None

                if existing_listing_ts is not None and existing_listing_ts < listing_target_ts:
                    listing_target_ts = existing_listing_ts

                if cache_start_ts is not None and listing_target_ts < cache_start_ts:
                    listing_target_ts = cache_start_ts
                stock["listing_date"] = listing_target_ts.strftime("%Y-%m-%d")

                # 2. 주간 평균 거래량/거래대금 업데이트
                if len(data) >= 7:
                    # 거래대금 컬럼 추가
                    data["Turnover"] = data["Close"] * data["Volume"]
                    last_7_days = data.tail(7)

                    avg_volume = last_7_days["Volume"].mean()
                    avg_turnover = last_7_days["Turnover"].mean()

                    stock["1_week_avg_volume"] = int(avg_volume) if pd.notna(avg_volume) else None
                    stock["1_week_avg_turnover"] = int(avg_turnover) if pd.notna(avg_turnover) else None
                else:
                    stock["1_week_avg_volume"] = None
                    stock["1_week_avg_turnover"] = None
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
    from utils.logger import setup_file_logger

    setup_file_logger()
    update_stock_metadata()
