"""
종목 파일(zsettings/<account>/stocks.json)에 메타데이터(상장일, 주간 평균 거래량/거래대금)를 업데이트합니다.
"""

import time
import xml.etree.ElementTree as ET
from typing import Any

import pandas as pd
import requests
import yfinance as yf

from utils.logger import get_app_logger
from utils.settings_loader import get_account_settings, list_available_accounts
from utils.stock_list_io import _load_account_stocks_raw, save_etfs


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


def _try_parse_float(value: Any) -> float | None:
    """문자열/숫자 값을 안전하게 float로 변환합니다."""
    if value is None:
        return None
    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _update_metadata_for_account(account_id: str):
    """지정된 계정의 모든 종목에 대한 메타데이터를 업데이트합니다."""
    logger = get_app_logger()
    account_norm = (account_id or "").strip().lower()

    try:
        settings = get_account_settings(account_norm)
        country_code = settings.get("country_code", "kor").lower()
    except Exception as e:
        logger.error(f"계정 설정을 로드할 수 없습니다: {e}")
        return

    stock_data = _load_account_stocks_raw(account_norm)
    if not stock_data:
        logger.warning(f"'{account_norm}' 계정의 종목 데이터가 비어있습니다.")
        return

    updated_count = 0
    ticker_entries: list[dict[str, Any]] = []

    # 구조 Flattening 없이 직접 참조를 가져오기 위해 탐색
    # 하지만 list iteration 중 modification은 주의해야 함
    # stock_data는 list[dict] (Category blocks)
    # 각 block의 'tickers' list의 item들은 dict.
    # 우리는 item dict를 직접 수정해야 함.

    for category in stock_data:
        for stock in category.get("tickers", []):
            ticker_entries.append(stock)

    total_count = len(ticker_entries)
    logger.info(f"[{account_norm.upper()}] 메타데이터 업데이트 시작 (총 {total_count}개 종목)")

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
                    logger.debug(f"[{account_norm.upper()}/{ticker}] 네이버 API에서 상장일 획득: {listing_date_str}")
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
                                logger.debug(f"[{account_norm.upper()}/{ticker}] 종목명 업데이트: {fetched_name}")
                        except Exception as e:
                            logger.warning(f"[{account_norm.upper()}/{ticker}] 종목명 조회 실패: {e}")

                    hist = t.history(period="max")
                    if not hist.empty:
                        first_date = hist.index.min()
                        listing_date_str = first_date.strftime("%Y-%m-%d")
                        logger.debug(f"[{account_norm.upper()}/{ticker}] yfinance에서 상장일 획득: {listing_date_str}")
                    else:
                        logger.warning(f"[{account_norm.upper()}/{ticker}] yfinance 데이터가 비어있습니다.")
                except Exception as e:
                    logger.warning(f"[{account_norm.upper()}/{ticker}] yfinance 메타데이터 조회 조회 실패: {e}")

            if not listing_date_str:
                # 기존 파일의 listing_date 유지
                listing_date_str = stock.get("listing_date")

            if not listing_date_str:
                # logger.warning(f"[{account_norm.upper()}/{ticker}] 상장일을 가져오지 못해 스킵합니다.")
                pass
            else:
                # 상장일 저장
                stock["listing_date"] = listing_date_str

            # 주간 평균 거래량/거래대금 및 3개월 수익률은 항상 업데이트 (yfinance 데이터 필요)
            if data is None:
                # 상장일이 이미 있어서 data가 없는 경우, yfinance로 2년치 데이터 조회 (1년 수익률 계산 확보용)
                data = yf.download(yfinance_ticker, period="2y", progress=False, auto_adjust=True)
                if isinstance(data, pd.DataFrame) and data.empty:
                    pass
                    # logger.warning(f"[{account_norm.upper()}/{ticker}] 거래량/수익률 데이터를 가져올 수 없습니다.")
                elif isinstance(data, pd.DataFrame):
                    if isinstance(data.columns, pd.MultiIndex):
                        data.columns = data.columns.get_level_values(0)
                        data = data.loc[:, ~data.columns.duplicated()]
                    if not data.index.is_unique:
                        data = data[~data.index.duplicated(keep="last")]

            # 구버전 필드 정리
            stock.pop("1_month_avg_volume", None)
            stock.pop("1_week_avg_turnover", None)
            stock.pop("1_month_avg_turnover", None)

            # 신규 필드 초기화 (기존값 유지 or None)
            # stock["1_week_avg_volume"] = None # 유지하거나 업데이트

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

                    subset = df.tail(days_lookback + 1)
                    if len(subset) < 2:
                        return None

                    start_price = subset.iloc[0]["Close"]
                    end_price = subset.iloc[-1]["Close"]

                    if pd.notna(start_price) and pd.notna(end_price) and start_price > 0:
                        return round(((end_price - start_price) / start_price) * 100, 4)
                    return None

                # 2. 수익률 계산 (영업일 기준 근사치)
                stock["1_month_earn_rate"] = calc_rate_safe(data, 21)
                stock["3_month_earn_rate"] = calc_rate_safe(data, 63)
                stock["6_month_earn_rate"] = calc_rate_safe(data, 126)
                stock["12_month_earn_rate"] = calc_rate_safe(data, 252)

            name = stock.get("name") or "-"
            logger.info(f"  -> 메타데이터 획득 중: {idx}/{total_count} - {name}({ticker})")
            updated_count += 1
            time.sleep(0.2)  # API 호출 속도 조절

        except Exception as e:
            logger.error(f"[{account_norm.upper()}/{ticker}] 메타데이터 업데이트 실패: {e}")

    if updated_count > 0:
        try:
            save_etfs(account_norm, stock_data)
        except Exception as e:
            logger.error(f"'{account_id}' 설정 저장 실패: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Update stock metadata.")
    parser.add_argument("target", nargs="?", help="Account ID")
    args = parser.parse_args()

    target = (args.target or "").strip().lower()

    accounts_to_update = []

    logger = get_app_logger()

    if not target:
        # Default: Update basic accounts if nothing specified?
        # Or update all?
        # Let's update valid accounts with country=kor or us
        all_accounts = list_available_accounts()
        for account in all_accounts:
            try:
                settings = get_account_settings(account)
                c_code = settings.get("country_code", "").lower()
                if c_code in ["kor", "us", "usa"]:
                    accounts_to_update.append(account)
            except Exception:
                pass
    else:
        # Check if target is account
        if target in list_available_accounts():
            accounts_to_update = [target]
        else:
            logger.error(f"Target '{target}' is not a valid account ID.")
            exit(1)

    logger.info(f"Updating metadata for accounts: {accounts_to_update}")

    for account in accounts_to_update:
        logger.info(f"[{account.upper()}] 메타데이터 작업 시작")
        _update_metadata_for_account(account)

    logger.info("모든 종목 메타데이터 업데이트 완료.")
