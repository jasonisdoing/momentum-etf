"""
데이터 조회, 파일 입출력 등 공통으로 사용되는 유틸리티 함수 모음.
"""
import functools
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm

# 웹 스크레이핑을 위한 라이브러리
try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    requests = None
    BeautifulSoup = None

# yfinance가 설치되지 않았을 경우를 대비한 예외 처리
try:
    import yfinance as yf
except ImportError:
    yf = None

# pykrx가 설치되지 않았을 경우를 대비한 예외 처리
try:
    from pykrx import stock as _stock
except Exception:
    _stock = None


def is_pykrx_available() -> bool:
    """pykrx 모듈이 성공적으로 임포트되었는지 확인합니다."""
    return _stock is not None


def format_aus_ticker_for_yfinance(ticker: str) -> str:
    """'ASX:BHP' 또는 'BHP' 같은 티커를 yfinance API 형식인 'BHP.AX'로 변환합니다."""
    if ticker.upper().startswith("ASX:"):
        ticker = ticker[4:]
    if not ticker.upper().endswith(".AX"):
        ticker = f"{ticker.upper()}.AX"
    return ticker


def get_today_str() -> str:
    """오늘 날짜를 'YYYYMMDD' 형식의 문자열로 반환합니다."""
    return datetime.now().strftime("%Y%m%d")


@functools.lru_cache(maxsize=10)
def get_trading_days(start_date: str, end_date: str, country: str) -> List[pd.Timestamp]:
    """
    지정된 기간 내의 모든 거래일을 pd.Timestamp 리스트로 반환합니다.
    API에서 아직 제공하지 않는 최근 거래일도 포함하도록 시도합니다.
    """
    trading_days_ts = []
    if country == "kor":
        if not is_pykrx_available():
            trading_days_ts = pd.bdate_range(start=start_date, end=end_date).tolist()
        try:
            # KOSPI 대표 종목으로 거래일 조회
            df = _stock.get_market_ohlcv_by_date(start_date.replace("-", ""), end_date.replace("-", ""), "005930")
            trading_days_ts = df.index.tolist()
        except Exception as e:
            logging.getLogger(__name__).warning(f"pykrx로 거래일 조회 중 오류: {e}. 주말 제외 날짜를 사용합니다.")
            trading_days_ts = pd.bdate_range(start=start_date, end=end_date).tolist()
    elif country == "aus":
        if yf is None:
            trading_days_ts = pd.bdate_range(start=start_date, end=end_date).tolist()
        try:
            # ASX 200 지수로 거래일 조회
            # yfinance의 end는 exclusive이므로 하루를 더해줍니다.
            end_date_plus_one = (pd.to_datetime(end_date) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
            df = yf.download("^AXJO", start=start_date, end=end_date_plus_one, progress=False, auto_adjust=True)
            # yfinance는 timezone-aware index를 반환하므로, naive date-only timestamp로 변환합니다.
            trading_days_ts = [pd.Timestamp(d.date()) for d in df.index]
        except Exception as e:
            logging.getLogger(__name__).warning(f"yfinance로 거래일 조회 중 오류: {e}. 주말 제외 날짜를 사용합니다.")
            trading_days_ts = pd.bdate_range(start=start_date, end=end_date).tolist()

    # API가 아직 오늘 데이터를 제공하지 않는 경우를 대비하여, 오늘이 평일이면 추가합니다.
    today = pd.Timestamp.now().normalize()
    end_date_ts = pd.to_datetime(end_date).normalize()

    if end_date_ts >= today and today.weekday() < 5: # 조회 종료일이 오늘이거나 미래이고, 오늘이 평일인 경우
        # API 결과에 오늘 날짜가 포함되어 있는지 확인
        if not any(d.date() == today.date() for d in trading_days_ts):
            trading_days_ts.append(today)

    # 최종적으로 start_date와 end_date 사이의 날짜만 반환하고, 중복 제거 및 정렬합니다.
    start_date_ts = pd.to_datetime(start_date).normalize()
    final_list = [d for d in trading_days_ts if start_date_ts <= d <= end_date_ts]
    
    return sorted(list(set(final_list)))


def fetch_ohlcv(
    ticker: str,
    country: str = "kor",
    months_back: int = None,
    months_range: Optional[List[int]] = None,
    date_range: Optional[List[str]] = None,
    base_date: Optional[pd.Timestamp] = None,
) -> Optional[pd.DataFrame]:
    """
    pykrx를 통해 OHLCV 데이터를 조회합니다.

    Args:
        ticker (str): 조회할 종목의 티커.
        date_range (Optional[List[str]]): ['YYYY-MM-DD', 'YYYY-MM-DD'] 형식의 조회 기간.
    """
    # 날짜 범위 결정
    if date_range and len(date_range) == 2:
        try:
            start_dt = pd.to_datetime(date_range[0])
            end_dt = pd.to_datetime(date_range[1])
        except (ValueError, TypeError):
            logging.getLogger(__name__).error(
                f"잘못된 date_range 형식: {date_range}. 'YYYY-MM-DD' 형식을 사용해야 합니다."
            )
            return None
    else:
        now = base_date if base_date is not None else pd.to_datetime(get_today_str())
        if months_range is not None and len(months_range) == 2:
            start_off, end_off = months_range
            start_dt = now - pd.DateOffset(months=int(start_off))
            end_dt = now - pd.DateOffset(months=int(end_off))
        else:
            if months_back is None:
                months_back = 12
            start_dt = now - pd.DateOffset(months=int(months_back))
            end_dt = now

    if start_dt > end_dt:
        start_dt, end_dt = end_dt, start_dt

    if country == "kor":
        if not is_pykrx_available():
            logging.getLogger(__name__).error(
                "pykrx 라이브러리가 설치되지 않았습니다. 'pip install pykrx'로 설치해주세요."
            )
            return None
        # pykrx API 안정성을 위해 긴 기간 조회 시 1년 단위로 나누어 요청합니다.
        all_dfs = []
        current_start = start_dt
        while current_start <= end_dt:
            current_end = min(
                current_start + pd.DateOffset(years=1) - pd.DateOffset(days=1), end_dt
            )
            start_str = current_start.strftime("%Y%m%d")
            end_str = current_end.strftime("%Y%m%d")

            try:
                df_part = _stock.get_market_ohlcv_by_date(start_str, end_str, ticker)
                if df_part is not None and not df_part.empty:
                    all_dfs.append(df_part)
            except Exception as e:
                # 특정 기간 조회 실패 시 경고만 하고 계속 진행
                logging.getLogger(__name__).warning(
                    f"{ticker}의 {start_str}~{end_str} 기간 데이터 조회 중 오류: {e}"
                )

            current_start += pd.DateOffset(years=1)

        if not all_dfs:
            return None

        # 조회된 모든 데이터프레임을 하나로 합칩니다.
        full_df = pd.concat(all_dfs)
        # 중복된 인덱스(날짜)가 있을 경우 첫 번째 데이터만 남깁니다.
        full_df = full_df[~full_df.index.duplicated(keep="first")]

        return full_df.rename(
            columns={
                "시가": "Open",
                "고가": "High",
                "저가": "Low",
                "종가": "Close",
                "거래량": "Volume",
            }
        )
    elif country == "aus":
        if yf is None:
            logging.getLogger(__name__).error(
                "yfinance 라이브러리가 설치되지 않았습니다. 'pip install yfinance'로 설치해주세요."
            )
            return None

        ticker_yf = format_aus_ticker_for_yfinance(ticker)

        try:
            # yfinance는 start/end를 사용하며, end 날짜를 포함하려면 하루를 더해야 할 수 있습니다.
            df = yf.download(
                ticker_yf,
                start=start_dt,
                end=end_dt + pd.Timedelta(days=1),
                progress=False,
                auto_adjust=True,
            )
            if df.empty:
                return None
            # yfinance는 이미 컬럼명이 영어로 되어있음
            return df
        except Exception as e:
            logging.getLogger(__name__).warning(f"{ticker}의 데이터 조회 중 오류: {e}")
            return None
    else:
        logging.getLogger(__name__).error(f"지원하지 않는 국가 코드입니다: {country}")
        return None


def fetch_ohlcv_for_tickers(
    tickers: List[str],
    country: str,
    date_range: Optional[List[str]] = None,
    warmup_days: int = 0,
) -> Dict[str, pd.DataFrame]:
    """
    주어진 티커 목록에 대해 OHLCV 데이터를 병렬로 조회합니다.
    """
    prefetched_data = {}
    
    if not date_range or len(date_range) != 2:
        return {}

    core_start = pd.to_datetime(date_range[0])
    warmup_start = core_start - pd.DateOffset(days=warmup_days)
    adjusted_date_range = [warmup_start.strftime("%Y-%m-%d"), date_range[1]]

    def worker(ticker):
        return ticker, fetch_ohlcv(ticker, country=country, date_range=adjusted_date_range)

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(worker, tkr) for tkr in tickers]
        for future in tqdm(as_completed(futures), total=len(tickers), desc="전체 시세 데이터 로딩"):
            tkr, df = future.result()
            if df is not None and not df.empty:
                prefetched_data[tkr] = df
    
    return prefetched_data


def fetch_naver_realtime_price(ticker: str) -> Optional[float]:
    """
    네이버 금융 웹 스크레이핑을 통해 종목의 실시간 현재가를 조회합니다.
    주의: 이 방법은 웹페이지 구조 변경에 취약하며, 비공식적인 방법입니다.
    """
    if not requests or not BeautifulSoup:
        return None

    try:
        url = f"https://finance.naver.com/item/sise.naver?code={ticker}"
        # 네이버의 차단을 피하기 위해 브라우저처럼 보이는 User-Agent를 설정합니다.
        headers = {  # noqa: F841
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()  # HTTP 오류 발생 시 예외 발생

        soup = BeautifulSoup(response.text, "html.parser")
        # 현재가를 담고 있는 HTML 요소를 id를 통해 찾습니다.
        price_element = soup.select_one("#_nowVal")

        if price_element:
            price_str = price_element.get_text().replace(",", "")
            return float(price_str)
    except Exception as e:
        logging.getLogger(__name__).warning(f"{ticker}의 실시간 가격 조회 중 오류 발생: {e}")
    return None


_pykrx_name_cache: Dict[str, str] = {}


def fetch_pykrx_name(ticker: str) -> str:
    """
    pykrx를 통해 종목의 이름을 가져옵니다. ETF와 일반 주식을 모두 시도합니다.
    결과는 단일 실행 내에서 캐시됩니다.
    """
    if ticker in _pykrx_name_cache:
        return _pykrx_name_cache[ticker]

    if not is_pykrx_available():
        return ""

    stock_name = ""
    try:
        # 1. ETF 이름 조회 시도
        name_candidate = _stock.get_etf_ticker_name(ticker)
        if isinstance(name_candidate, str) and name_candidate:
            stock_name = name_candidate
    except Exception:
        pass

    if not stock_name:
        try:
            # 2. 주식 이름 조회 시도
            name_candidate = _stock.get_market_ticker_name(ticker)
            if isinstance(name_candidate, str) and name_candidate:
                stock_name = name_candidate
        except Exception:
            pass

    _pykrx_name_cache[ticker] = stock_name
    return stock_name


_yfinance_name_cache: Dict[str, str] = {}


def fetch_yfinance_name(ticker: str) -> str:
    """
    yfinance를 통해 종목의 이름을 가져옵니다. 결과를 캐시하여 중복 요청을 방지합니다.
    """
    if yf is None:
        return ""

    cache_key = ticker
    if cache_key in _yfinance_name_cache:
        return _yfinance_name_cache[cache_key]

    try:
        ticker_yf = format_aus_ticker_for_yfinance(ticker)
        stock = yf.Ticker(ticker_yf)
        name = stock.info.get("longName") or stock.info.get("shortName") or ""
        _yfinance_name_cache[cache_key] = name
        return name
    except Exception as e:
        logging.getLogger(__name__).warning(f"{cache_key}의 이름 조회 중 오류 발생: {e}")
        _yfinance_name_cache[cache_key] = ""  # 실패도 캐시하여 재시도 방지
        return ""


@functools.lru_cache(maxsize=None)
def fetch_exchange_rate(ticker: str = "AUDKRW=X", as_of_date: Optional[str] = None) -> float | None:
    """
    yfinance를 통해 실시간 환율 정보를 조회합니다.
    결과는 단일 실행 내에서 캐시됩니다.
    """
    if yf is None:
        return None
    try:
        if as_of_date:
            end_date = pd.to_datetime(as_of_date)
            # 해당 날짜의 데이터를 포함하기 위해 end_date에 하루를 더하고, 휴일을 고려해 7일 전부터 조회
            start_date = end_date - pd.Timedelta(days=7)
            data = yf.Ticker(ticker).history(
                start=start_date, end=end_date + pd.Timedelta(days=1), auto_adjust=True, progress=False
            )
        else:
            # as_of_date가 없으면 최신 데이터를 가져옵니다 (기존 동작).
            data = yf.Ticker(ticker).history(period="2d", auto_adjust=True, progress=False)

        if not data.empty:
            return data["Close"].iloc[-1]
    except Exception as e:
        logging.getLogger(__name__).warning(f"{ticker} 환율 조회 중 오류 발생: {e}")
    return None
