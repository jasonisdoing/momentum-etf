"""
데이터 조회, 파일 입출력 등 공통으로 사용되는 유틸리티 함수 모음.
"""

import functools
import json
import logging
import warnings
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

# 웹 스크레이핑을 위한 라이브러리
try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    requests = None
    BeautifulSoup = None

# yfinance가 설치되지 않았을 경우를 대비한 예외 처리
try:
    # Silence noisy yfinance "Failed download" console messages
    import logging as _yf_logging  # noqa: E402

    import yfinance as yf

    _yf_logging.getLogger("yfinance").setLevel(_yf_logging.ERROR)
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
    # 지수 티커(예: ^AXJO)는 변환하지 않습니다.
    if ticker.startswith("^"):
        return ticker
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
    한국(KRX), 호주(ASX)는 pandas_market_calendars만 사용합니다.
    """
    trading_days_ts: List[pd.Timestamp] = []

    def _pmc(country_code: str) -> List[pd.Timestamp]:
        import pandas_market_calendars as mcal  # type: ignore

        cal_code = {"kor": "XKRX", "aus": "ASX"}.get(country_code)
        if not cal_code:
            return []
        try:
            cal = mcal.get_calendar(cal_code)
            # Latest pmc: remove all discontinued market_times using the new API
            try:
                dmt = getattr(cal, "discontinued_market_times", {})
                for tname in getattr(dmt, "keys", lambda: [])():
                    try:
                        cal.remove_time(tname)  # type: ignore[attr-defined]
                    except Exception:
                        pass
            except Exception:
                # Older versions fallback
                for tname in ("break_start", "break_end"):
                    try:
                        cal.remove_time(tname)  # type: ignore[attr-defined]
                    except Exception:
                        pass

            # Prefer valid_days (dates only)
            try:
                days_idx = cal.valid_days(start_date=start_date, end_date=end_date)
                if days_idx is not None and len(days_idx) > 0:
                    return [pd.Timestamp(pd.Timestamp(d).date()) for d in days_idx]
            except Exception:
                pass

            # Fallback to schedule, suppress deprecation warning
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=r"\['break_start', 'break_end'\] are discontinued",
                    category=UserWarning,
                )
                sched = cal.schedule(start_date=start_date, end_date=end_date)
            if sched is not None and not sched.empty:
                return [pd.Timestamp(d.date()) for d in sched.index]
        except Exception as e:
            logging.getLogger(__name__).warning(
                f"pandas_market_calendars({country_code}:{cal_code}) 조회 실패: {e}"
            )
        return []

    if country == "kor":
        trading_days_ts = _pmc("kor")
    elif country == "aus":
        trading_days_ts = _pmc("aus")
    elif country == "coin":
        # 암호화폐는 24/7 거래되므로, 단순히 날짜 범위 내의 모든 날짜를 반환합니다.
        # 실제 거래가 없는 날(예: 거래소 점검)은 고려하지 않습니다.
        trading_days_ts = pd.date_range(start=start_date, end=end_date, freq="D").tolist()
    else:
        logging.getLogger(__name__).error(f"지원하지 않는 국가 코드입니다: {country}")
        return []

    # 최종적으로 start_date와 end_date 사이의 날짜만 반환하고, 중복 제거 및 정렬합니다.
    start_date_ts = pd.to_datetime(start_date).normalize()
    end_date_ts = pd.to_datetime(end_date).normalize()
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
    OHLCV 데이터를 조회합니다.
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

    # 지수 티커는 국가와 상관없이 yfinance로 조회합니다.
    if ticker.startswith("^"):
        if yf is None:
            logging.getLogger(__name__).error(
                "yfinance 라이브러리가 설치되지 않았습니다. 'pip install yfinance'로 설치해주세요."
            )
            return None
        try:
            df = yf.download(
                ticker,
                start=start_dt,
                end=end_dt + pd.Timedelta(days=1),
                progress=False,
                auto_adjust=True,
            )
            if df.empty:
                return None

            # yfinance가 가끔 MultiIndex 컬럼을 반환하는 경우에 대비하여,
            # 컬럼을 단순화하고 중복을 제거합니다.
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                df = df.loc[:, ~df.columns.duplicated()]

            # yfinance는 timezone-aware index를 반환할 수 있습니다.
            # 데이터 일관성을 위해 중복을 제거하고 naive timestamp로 변환합니다.
            if not df.index.is_unique:
                df = df[~df.index.duplicated(keep="first")]
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            return df
        except Exception as e:
            logging.getLogger(__name__).warning(f"{ticker}의 데이터 조회 중 오류: {e}")
            return None

    if country == "kor":
        if not is_pykrx_available():
            logging.getLogger(__name__).error(
                "pykrx 라이브러리가 설치되지 않았습니다. 'pip install pykrx'로 설치해주세요."
            )
            return None

        # pykrx API 안정성을 위해 긴 기간 조회 시 1년 단위로 나누어 요청합니다.
        # JSONDecodeError 발생 시 yfinance로 폴백합니다.
        all_dfs = []
        pykrx_failed_with_json_error = False

        current_start = start_dt
        while current_start <= end_dt:
            current_end = min(
                current_start + pd.DateOffset(years=1) - pd.DateOffset(days=1), end_dt
            )
            start_str = current_start.strftime("%Y%m%d")
            end_str = current_end.strftime("%Y%m%d")

            try:
                df_part = _stock.get_etf_ohlcv_by_date(start_str, end_str, ticker)
                if df_part is not None and not df_part.empty:
                    all_dfs.append(df_part)
            except (json.JSONDecodeError, KeyError) as e:  # Catch KeyError as well
                # pykrx에서 JSONDecodeError 또는 KeyError가 발생하면 (KRX 웹사이트 응답 문제 또는 데이터 구조 문제),
                # 루프를 중단하고 yfinance로 전체 기간 폴백을 시도합니다.
                pykrx_failed_with_json_error = True  # Renamed variable for clarity, but same logic
                logging.getLogger(__name__).warning(
                    f"pykrx 조회 실패({type(e).__name__}), yfinance로 전체 기간 대체 시도: {ticker}"
                )
                break  # while 루프 중단
            except Exception as e:
                # 다른 종류의 예외인 경우, 기존처럼 경고만 로깅합니다。
                logging.getLogger(__name__).warning(
                    f"{ticker}의 {start_str}~{end_str} 기간 데이터 조회 중 오류: {e}"
                )

            current_start += pd.DateOffset(years=1)

        if pykrx_failed_with_json_error:
            if yf is None:
                return None  # yfinance가 없으면 실패 처리
            try:
                y_ticker = ticker
                if ticker.isdigit() and len(ticker) == 6:
                    y_ticker = f"{ticker}.KS"

                df_yf = yf.download(
                    y_ticker,
                    start=start_dt,
                    end=end_dt + pd.Timedelta(days=1),
                    progress=False,
                    auto_adjust=True,
                )

                if df_yf is not None and not df_yf.empty:
                    if isinstance(df_yf.columns, pd.MultiIndex):
                        df_yf.columns = df_yf.columns.get_level_values(0)
                        df_yf = df_yf.loc[:, ~df_yf.columns.duplicated()]
                    if df_yf.index.tz is not None:
                        df_yf.index = df_yf.index.tz_localize(None)
                    # yfinance는 이미 영어 컬럼이므로, 바로 반환합니다.
                    return df_yf
                else:
                    return None  # yfinance 조회도 실패
            except Exception as yf_e:
                logging.getLogger(__name__).warning(f"yfinance 대체 조회도 실패: {ticker} ({yf_e})")
                return None

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

            # yfinance가 가끔 MultiIndex 컬럼을 반환하는 경우에 대비하여,
            # 컬럼을 단순화하고 중복을 제거합니다.
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                df = df.loc[:, ~df.columns.duplicated()]

            # yfinance는 timezone-aware index를 반환할 수 있습니다.
            # 데이터 일관성을 위해 중복을 제거하고 naive timestamp로 변환합니다.
            if not df.index.is_unique:
                df = df[~df.index.duplicated(keep="first")]
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            # yfinance는 이미 컬럼명이 영어로 되어있음
            return df
        except Exception as e:
            logging.getLogger(__name__).warning(f"{ticker}의 데이터 조회 중 오류: {e}")
            return None
    elif country == "coin":
        # 코인: 빗썸 퍼블릭 캔들스틱 API로 일봉(24h) OHLCV를 조회합니다.
        try:
            from datetime import datetime, timezone

            import pandas as _pd

            base = ticker.upper()
            url = f"https://api.bithumb.com/public/candlestick/{base}_KRW/24h"
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            j = r.json() or {}
            data = j.get("data") or []
            if not data:
                return None
            # data: list of arrays [ts(ms), open, close, high, low, volume]
            rows = []
            for arr in data:
                try:
                    ts = int(arr[0])
                    o = float(arr[1])
                    c = float(arr[2])
                    h = float(arr[3])
                    l = float(arr[4])
                    v = float(arr[5])
                except Exception:
                    continue
                # Convert ms epoch to naive timestamp (UTC-based)
                dt = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc).replace(tzinfo=None)
                rows.append((dt, o, h, l, c, v))
            if not rows:
                return None
            rows.sort(key=lambda x: x[0])
            df = _pd.DataFrame(
                rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"]
            ).set_index("Date")
            # Optional date filtering
            if date_range and len(date_range) == 2:
                try:
                    start_dt = _pd.to_datetime(date_range[0])
                    end_dt = _pd.to_datetime(date_range[1])
                    df = df[(df.index >= start_dt) & (df.index <= end_dt)]
                except Exception:
                    pass
            return df if not df.empty else None
        except Exception as e:
            logging.getLogger(__name__).warning(f"{ticker} 코인 OHLCV 조회 중 오류: {e}")
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
    주어진 티커 목록에 대해 OHLCV 데이터를 직렬로 조회합니다.
    """
    prefetched_data = {}

    if not date_range or len(date_range) != 2:
        return {}

    core_start = pd.to_datetime(date_range[0])
    warmup_start = core_start - pd.DateOffset(days=warmup_days)
    adjusted_date_range = [warmup_start.strftime("%Y-%m-%d"), date_range[1]]

    for tkr in tickers:
        df = fetch_ohlcv(ticker=tkr, country=country, date_range=adjusted_date_range)
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

    etf_name = ""
    try:
        # 1. ETF 이름 조회 시도
        name_candidate = _stock.get_etf_ticker_name(ticker)
        if isinstance(name_candidate, str) and name_candidate:
            etf_name = name_candidate
    except Exception:
        pass

    _pykrx_name_cache[ticker] = etf_name
    return etf_name


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
    return None
