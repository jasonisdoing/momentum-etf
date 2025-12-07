"""
종목 파일(stocks/*.json)에 메타데이터(상장일, 주간 평균 거래량/거래대금)를 업데이트합니다.
"""

import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yfinance as yf

from utils.logger import get_app_logger

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STOCKS_DIR = PROJECT_ROOT / "zsettings" / "stocks"


def _fetch_naver_listing_date(ticker: str) -> str | None:
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

        # 네이버 차트 API는 EUC-KR 기반 XML을 반환하므로 명시적으로 디코딩
        try:
            text = response.content.decode("euc-kr")
        except Exception:
            text = response.text

        # XML 파싱
        root = ET.fromstring(text)
        chartdata = root.find("chartdata")

        if chartdata is not None:
            origintime = chartdata.get("origintime")
            if origintime and len(origintime) == 8:  # YYYYMMDD 형식
                # YYYYMMDD -> YYYY-MM-DD 변환
                listing_date = f"{origintime[:4]}-{origintime[4:6]}-{origintime[6:8]}"
                logger.debug(f"[네이버 API] {ticker} 상장일: {listing_date}")
                return listing_date

    except Exception as e:
        logger.info(f"[네이버 API] {ticker} 상장일 조회 실패: {e}")

    return None


def _fetch_naver_etf_snapshot() -> dict[str, dict[str, Any]]:
    """
    네이버 ETF 리스트 API에서 한국 ETF 메타데이터를 한 번에 가져옵니다.

    Returns:
        {ticker: raw_payload_dict} 형태의 딕셔너리. 실패 시 빈 dict.
    """
    logger = get_app_logger()

    if not requests:
        logger.debug("requests 라이브러리가 없어 네이버 ETF 스냅샷 조회를 건너뜁니다.")
        return {}

    try:
        from config import NAVER_FINANCE_ETF_API_URL, NAVER_FINANCE_HEADERS
    except Exception:
        logger.debug("네이버 ETF API 설정을 불러오지 못했습니다.")
        return {}

    try:
        response = requests.get(NAVER_FINANCE_ETF_API_URL, headers=NAVER_FINANCE_HEADERS, timeout=5)
        response.raise_for_status()
    except Exception as exc:
        logger.warning("네이버 ETF 스냅샷 조회 실패: %s", exc)
        return {}

    try:
        payload = response.json()
    except Exception as exc:
        logger.warning("네이버 ETF 스냅샷 파싱 실패: %s", exc)
        return {}

    items = payload.get("result", {}).get("etfItemList")
    if not isinstance(items, list):
        return {}

    snapshot: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        ticker = str(item.get("itemcode") or "").strip().upper()
        if not ticker:
            continue
        snapshot[ticker] = item
    return snapshot


def _try_parse_float(value: Any) -> float | None:
    """문자열/숫자 값을 안전하게 float로 변환합니다."""
    if value is None:
        return None
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _get_cache_start_date() -> pd.Timestamp | None:
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
    ticker_entries: list[dict[str, Any]] = []
    for category in stock_data:
        for stock in category.get("tickers", []):
            ticker_entries.append(stock)

    total_count = len(ticker_entries)
    for idx, stock in enumerate(ticker_entries, start=1):
        ticker = stock.get("ticker")
        if not ticker:
            continue

        if country_code == "kor":
            yfinance_ticker = f"{ticker}.KS"
        else:
            yfinance_ticker = ticker

        try:
            listing_date_str = None
            data = None  # 각 종목마다 data 변수 초기화

            # 한국 주식인 경우에만 네이버 API로 상장일 조회 시도
            if country_code == "kor":
                listing_date_str = _fetch_naver_listing_date(ticker)
                if listing_date_str:
                    logger.debug(f"[{country_code.upper()}/{ticker}] 네이버 API에서 상장일 획득: {listing_date_str}")
            elif country_code == "us":
                # 미국 주식은 yfinance를 통해 메타데이터를 가져옵니다.
                try:
                    t = yf.Ticker(ticker)

                    # [UPDATE] 종목명이 없는 경우 자동 채우기
                    if not stock.get("name"):
                        try:
                            # info는 네트워크 요청이 발생하므로 다소 느릴 수 있음
                            info = t.info
                            fetched_name = info.get("longName") or info.get("shortName")
                            if fetched_name:
                                stock["name"] = fetched_name
                                logger.debug(f"[{country_code.upper()}/{ticker}] 종목명 업데이트: {fetched_name}")
                        except Exception as e:
                            logger.warning(f"[{country_code.upper()}/{ticker}] 종목명 조회 실패: {e}")

                    # history_metadata를 통해 firstTradeDate 탐색
                    # firstTradeDate 정보가 없을 수 있으므로 예외 처리 필요
                    # yfinance 버전 및 데이터 상황에 따라 다를 수 있음
                    # 대안: start="1970-01-01"로 download 해보고 첫 날짜 확인
                    # 여기서는 간단히 info 접근 시도 (느릴 수 있음) or history로 첫 날짜 확인

                    hist = t.history(period="max")
                    if not hist.empty:
                        first_date = hist.index.min()
                        listing_date_str = first_date.strftime("%Y-%m-%d")
                        logger.debug(f"[{country_code.upper()}/{ticker}] yfinance에서 상장일 획득: {listing_date_str}")
                    else:
                        logger.warning(f"[{country_code.upper()}/{ticker}] yfinance 데이터가 비어있습니다.")
                except Exception as e:
                    logger.warning(f"[{country_code.upper()}/{ticker}] yfinance 메타데이터 조회 조회 실패: {e}")

            if not listing_date_str:
                # 기존 파일의 listing_date 유지
                listing_date_str = stock.get("listing_date")

            if not listing_date_str:
                logger.warning(f"[{country_code.upper()}/{ticker}] 상장일을 가져오지 못해 스킵합니다.")
                continue

            # 상장일 저장
            stock["listing_date"] = listing_date_str

            # 주간 평균 거래량/거래대금 및 3개월 수익률은 항상 업데이트 (yfinance 데이터 필요)
            if data is None:
                # 상장일이 이미 있어서 data가 없는 경우, yfinance로 2년치 데이터 조회 (1년 수익률 계산 확보용)
                data = yf.download(yfinance_ticker, period="2y", progress=False, auto_adjust=True)
                if data.empty:
                    logger.warning(f"[{country_code.upper()}/{ticker}] 거래량/수익률 데이터를 가져올 수 없습니다.")
                else:
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.get_level_values(0)
                        data = data.loc[:, ~data.columns.duplicated()]
                    if not data.index.is_unique:
                        data = data[~data.index.duplicated(keep="last")]

            # 구버전 필드 정리
            stock.pop("1_month_avg_volume", None)  # 1달 평균 거래량 제거 -> 1주 평균으로 변경
            stock.pop("1_week_avg_turnover", None)
            stock.pop("1_month_avg_turnover", None)

            # 신규 필드 초기화
            stock["1_week_avg_volume"] = None
            stock["1_month_earn_rate"] = None
            stock["3_month_earn_rate"] = None
            stock["6_month_earn_rate"] = None
            stock["12_month_earn_rate"] = None

            # yfinance 기반 지표 계산
            if data is not None and not data.empty and len(data) >= 1:
                # 1. 1주 평균 거래량 (최근 5거래일 기준)
                last_week = data.tail(5)
                avg_volume = last_week["Volume"].mean()
                if pd.notna(avg_volume):
                    stock["1_week_avg_volume"] = int(avg_volume)

                # Helper to calc earn rate
                def calc_rate_safe(df, days_lookback):
                    if len(df) < days_lookback + 1:
                        return None
                    if "Close" not in df.columns:
                        return None

                    # days_lookback 전의 가격과 현재 가격 비교
                    # 예: 1달 수익률 = (오늘종가 - 21거래일전 종가) / 21거래일전 종가
                    # tail을 써서 필요한 만큼만 잘라냄
                    # df는 이미 날짜순 정렬됨
                    subset = df.tail(days_lookback + 1)
                    if len(subset) < 2:
                        return None

                    start_price = subset.iloc[0]["Close"]
                    end_price = subset.iloc[-1]["Close"]

                    if pd.notna(start_price) and pd.notna(end_price) and start_price > 0:
                        return round(((end_price - start_price) / start_price) * 100, 4)
                    return None

                # 2. 수익률 계산 (영업일 기준 근사치)
                # 1달 ~ 21일, 3달 ~ 63일, 6달 ~ 126일, 12달 ~ 252일
                stock["1_month_earn_rate"] = calc_rate_safe(data, 21)
                stock["3_month_earn_rate"] = calc_rate_safe(data, 63)
                stock["6_month_earn_rate"] = calc_rate_safe(data, 126)
                stock["12_month_earn_rate"] = calc_rate_safe(data, 252)

            # 필드 순서 정렬: 1W 거래량 -> 1M, 3M, 6M, 12M 수익률
            ordered_fields = [
                "1_week_avg_volume",
                "1_month_earn_rate",
                "3_month_earn_rate",
                "6_month_earn_rate",
                "12_month_earn_rate",
            ]
            preserved_values = {}
            for key in ordered_fields:
                if key in stock:
                    preserved_values[key] = stock.pop(key)
            for key in ordered_fields:
                if key in preserved_values:
                    stock[key] = preserved_values[key]

            name = stock.get("name") or "-"
            logger.info(f"  -> 메타데이터 획득 중: {idx}/{total_count} - {name}({ticker})")
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

    supported_countries = ["kor", "us"]
    for country in supported_countries:
        stock_file = STOCKS_DIR / f"{country}.json"
        if stock_file.exists():
            logger.info(f"[{country.upper()}] 메타데이터 작업 시작")
            _update_metadata_for_country(country)

    logger.info("모든 종목 메타데이터 업데이트 완료.")


if __name__ == "__main__":
    import argparse

    from utils.account_registry import get_account_settings

    parser = argparse.ArgumentParser(description="Update stock metadata.")
    parser.add_argument("target", nargs="?", help="Account ID or Country Code")
    args = parser.parse_args()

    target = (args.target or "").strip().lower()

    countries_to_update = []

    if not target:
        # No argument -> Update all supported (kor, us)
        countries_to_update = ["kor", "us"]
    else:
        # Check if argument is a known country
        if target in ["kor", "us"]:
            countries_to_update = [target]
        else:
            # Assume it's an account ID
            try:
                settings = get_account_settings(target)
                country = settings.get("country_code", "kor").lower()
                countries_to_update = [country]
            except Exception:
                # If account not found, fallback to treating as country or error?
                # For safety, if not kor/us, we might just try it or warn
                print(f"Warning: Account '{target}' not found or invalid country. Trying as country code.")
                countries_to_update = [target]

    logger = get_app_logger()
    logger.info(f"Updating metadata for: {countries_to_update}")

    for country in countries_to_update:
        stock_file = STOCKS_DIR / f"{country}.json"
        if stock_file.exists():
            logger.info(f"[{country.upper()}] 메타데이터 작업 시작")
            _update_metadata_for_country(country)
        else:
            logger.warning(f"[{country.upper()}] 설정 파일({stock_file})이 없어 건너뜁니다.")

    logger.info("모든 종목 메타데이터 업데이트 완료.")
