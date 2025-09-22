"""
데이터 조회, 파일 입출력 등 공통으로 사용되는 유틸리티 함수 모음.
"""

import functools
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from contextlib import contextmanager

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
    import yfinance as yf
except ImportError:
    yf = None

# pykrx가 설치되지 않았을 경우를 대비한 예외 처리
try:
    from pykrx import stock as _stock
except Exception:
    _stock = None

from utils.cache_utils import load_cached_frame, save_cached_frame
from utils.stock_list_io import get_etfs
from utils.notify import send_verbose_log_to_slack

import warnings

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings(
    "ignore",
    message=r"\['break_start', 'break_end'\] are discontinued",
    category=UserWarning,
    module=r"^pandas_market_calendars\.",
)

CACHE_START_DATE_FALLBACK = "2020-01-01"

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore


class PykrxDataUnavailable(Exception):
    """pykrx 데이터가 제공되지 않을 때 사용되는 예외."""

    def __init__(
        self,
        country: str,
        start_dt: pd.Timestamp,
        end_dt: pd.Timestamp,
        detail: str,
    ) -> None:
        self.country = country
        self.start_dt = start_dt
        self.end_dt = end_dt
        self.detail = detail
        message = (
            f"[{country.upper()}] pykrx data unavailable "
            f"({start_dt.date()}~{end_dt.date()}): {detail}"
        )
        super().__init__(message)


def _get_cache_start_dt() -> Optional[pd.Timestamp]:
    """환경 변수 또는 기본값에서 캐시 시작 날짜를 로드합니다."""
    raw = os.environ.get("CACHE_START_DATE", CACHE_START_DATE_FALLBACK)
    if not raw:
        return None
    try:
        dt = pd.to_datetime(raw)
    except Exception:
        return None
    if isinstance(dt, pd.DatetimeIndex):
        dt = dt[0]
    if isinstance(dt, pd.Timestamp):
        if dt.tzinfo is not None:
            dt = dt.tz_localize(None)
        return dt.normalize()
    return None


def _should_skip_pykrx_fetch(
    country: str,
    cache_end: Optional[pd.Timestamp],
    miss_start: pd.Timestamp,
) -> bool:
    """장 시작 전에는 캐시만 사용하도록 pykrx 호출을 지연합니다."""

    if country != "kor" or cache_end is None:
        return False

    if ZoneInfo is not None:
        now_local = datetime.now(ZoneInfo("Asia/Seoul"))
    else:  # pragma: no cover
        now_local = datetime.now()

    # pykrx 데이터가 당일 분이 아직 나오지 않은 장 시작 전(16시 이전)이라면 생략
    if miss_start.normalize() == pd.Timestamp(now_local.date()) and now_local.hour < 16:
        return True

    return False


@contextmanager
def _silence_yfinance_logs():
    import logging

    targets = [
        logging.getLogger("yfinance"),
        logging.getLogger("yfinance.utils"),
        logging.getLogger("yfinance.data"),
    ]
    prev_levels = [lg.level for lg in targets]
    try:
        for lg in targets:
            lg.setLevel(logging.CRITICAL)
        yield
    finally:
        for lg, lvl in zip(targets, prev_levels):
            lg.setLevel(lvl)


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
            # 최신 pandas_market_calendars에서는 폐지된 장중 시간대를 새 API로 제거합니다.
            try:
                dmt = getattr(cal, "discontinued_market_times", {})
                for tname in getattr(dmt, "keys", lambda: [])():
                    try:
                        cal.remove_time(tname)  # type: ignore[attr-defined]
                    except Exception:
                        pass
            except Exception:
                # 이전 버전에서는 기존 방식으로 폴백합니다.
                for tname in ("break_start", "break_end"):
                    try:
                        cal.remove_time(tname)  # type: ignore[attr-defined]
                    except Exception:
                        pass

            # 가능하다면 날짜만 반환하는 valid_days 결과를 우선 사용합니다.
            try:
                days_idx = cal.valid_days(start_date=start_date, end_date=end_date)
                if days_idx is not None and len(days_idx) > 0:
                    return [pd.Timestamp(pd.Timestamp(d).date()) for d in days_idx]
            except Exception:
                pass

            # 위 단계가 실패하면 schedule 기반으로 폴백하고 폐기 경고를 숨깁니다.
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
            print(f"경고: pandas_market_calendars({country_code}:{cal_code}) 조회 실패: {e}")
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
        print(f"오류: 지원하지 않는 국가 코드입니다: {country}")
        return []

    # 최종적으로 start_date와 end_date 사이의 날짜만 반환하고, 중복 제거 및 정렬합니다.
    start_date_ts = pd.to_datetime(start_date).normalize()
    end_date_ts = pd.to_datetime(end_date).normalize()
    final_list = [d for d in trading_days_ts if start_date_ts <= d <= end_date_ts]

    return sorted(list(set(final_list)))


@functools.lru_cache(maxsize=5)
def get_latest_trading_day(country: str) -> pd.Timestamp:
    """
    오늘 또는 가장 가까운 과거의 '데이터가 있을 것으로 예상되는' 거래일을 pd.Timestamp 형식으로 반환합니다.
    """
    end_dt = pd.Timestamp.now()
    if country == "coin":
        return end_dt.normalize()

    # 한국 시장의 경우, 장 마감 데이터가 집계되기 전(오후 4시 이전)이라면,
    # 조회 기준일을 하루 전으로 설정하여 어제까지의 데이터만 사용하도록 합니다.
    if country == "kor":
        try:
            if ZoneInfo is not None:
                local_now = datetime.now(ZoneInfo("Asia/Seoul"))
            else:  # pragma: no cover
                local_now = datetime.now()
            if local_now.hour < 16:
                end_dt = end_dt - pd.DateOffset(days=1)
        except Exception:
            # 타임존 처리 실패 시 안전하게 폴백
            pass

    # end_dt부터 과거로 하루씩 이동하며 거래일을 찾습니다.
    for i in range(15):
        check_date = end_dt - pd.DateOffset(days=i)
        check_date_str = check_date.strftime("%Y-%m-%d")
        try:
            if get_trading_days(check_date_str, check_date_str, country):
                return check_date.normalize()
        except Exception as e:
            print(f"경고: 거래일 조회 중 오류 발생({check_date_str}): {e}")
            # 오류 발생 시 다음 날짜로 계속 탐색
            continue

    # 15일 동안 거래일을 찾지 못하면 오늘 날짜를 정규화하여 반환합니다.
    print(f"경고: 최근 15일 내에 거래일을 찾지 못했습니다. 오늘 날짜({end_dt.strftime('%Y-%m-%d')})를 사용합니다.")
    return end_dt.normalize()


def fetch_ohlcv(
    ticker: str,
    country: str = "kor",
    months_back: int = None,
    months_range: Optional[List[int]] = None,
    date_range: Optional[List[Optional[str]]] = None,
    base_date: Optional[pd.Timestamp] = None,
) -> Optional[pd.DataFrame]:
    """OHLCV 데이터를 조회합니다. 캐시를 우선 사용하고 부족분만 원천에서 보충합니다."""

    if date_range and len(date_range) == 2:
        try:
            start_dt = pd.to_datetime(date_range[0])
            if date_range[1] is None:
                # date_range의 두 번째 인자가 None이면 오늘까지 조회합니다.
                end_dt = pd.to_datetime(get_today_str())
            else:
                end_dt = pd.to_datetime(date_range[1])
        except (ValueError, TypeError):
            print(f"오류: 잘못된 date_range 형식: {date_range}. 'YYYY-MM-DD' 형식을 사용해야 합니다.")
            return None
    else:
        now = base_date if base_date is not None else pd.Timestamp.now()
        if months_range is not None and len(months_range) == 2:
            start_off, end_off = months_range
            start_dt = now - pd.DateOffset(months=int(start_off))
            end_dt = now - pd.DateOffset(months=int(end_off))
        else:
            months_back = months_back or 12
            start_dt = now - pd.DateOffset(months=int(months_back))
            end_dt = now

    # 조회 종료일(end_dt)이 실제 데이터가 있는 마지막 거래일을 초과하지 않도록 보정합니다.
    # 이는 주말이나 휴일에 다음 거래일을 기준으로 데이터를 조회할 때, 아직 존재하지 않는
    # 미래 데이터를 조회하려는 시도를 방지합니다.
    latest_known_trading_day = get_latest_trading_day(country)
    if end_dt > latest_known_trading_day:
        end_dt = latest_known_trading_day

    if start_dt > end_dt:
        # 보정 후 시작일이 종료일보다 미래가 될 수 있으므로, 이 경우 데이터를 조회하지 않습니다.
        return None

    return _fetch_ohlcv_with_cache(ticker, country, start_dt.normalize(), end_dt.normalize())


def _fetch_ohlcv_with_cache(
    ticker: str,
    country: str,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
) -> Optional[pd.DataFrame]:
    cached_df = load_cached_frame(country, ticker)
    missing_ranges: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    cache_start: Optional[pd.Timestamp] = None
    cache_end: Optional[pd.Timestamp] = None
    if cached_df is None or cached_df.empty:
        cached_df = None
        missing_ranges.append((start_dt, end_dt))
    else:
        cache_start = cached_df.index.min().normalize()
        cache_end = cached_df.index.max().normalize()

        if start_dt < cache_start:
            missing_ranges.append((start_dt, cache_start - pd.Timedelta(days=1)))
        if end_dt > cache_end:
            missing_ranges.append((cache_end + pd.Timedelta(days=1), end_dt))

    new_frames: List[pd.DataFrame] = []
    for miss_start, miss_end in missing_ranges:
        if miss_start > miss_end:
            continue
        if cache_start is not None and miss_end < cache_start:
            continue
        if cache_end is not None and _should_skip_pykrx_fetch(country, cache_end, miss_start):
            continue

        # Check if there are any trading days in the missing range before attempting to fetch.
        # This prevents errors when the gap consists only of non-trading days (weekends, holidays).
        trading_days_in_gap = get_trading_days(
            miss_start.strftime("%Y-%m-%d"), miss_end.strftime("%Y-%m-%d"), country
        )
        if not trading_days_in_gap:
            continue

        fetched = _fetch_ohlcv_core(ticker, country, miss_start, miss_end, cached_df)
        if fetched is not None and not fetched.empty:
            new_frames.append(fetched)

    combined_df = cached_df
    prev_count = 0 if cached_df is None else cached_df.shape[0]
    added_count = 0

    if new_frames:
        frames = []
        if cached_df is not None and not cached_df.empty:
            frames.append(cached_df)
        frames.extend(new_frames)
        combined_df = pd.concat(frames)
        combined_df.sort_index(inplace=True)
        combined_df = combined_df[~combined_df.index.duplicated(keep="last")]
        save_cached_frame(country, ticker, combined_df)

        new_total = combined_df.shape[0]
        added_count = max(0, new_total - prev_count)
        if added_count > 0:
            try:
                display_name = _get_display_name(country, ticker)
                suffix = f"({display_name})" if display_name else ""
                send_verbose_log_to_slack(
                    f"[CACHE] {country.upper()}/{ticker}{suffix} {new_total:,} rows (+{added_count:,} rows)"
                )
            except Exception:
                pass

    if combined_df is None or combined_df.empty:
        return None

    cache_min = combined_df.index.min()
    cache_max = combined_df.index.max()

    effective_start = start_dt
    if start_dt > cache_max:
        effective_start = cache_max
    elif start_dt < cache_min:
        effective_start = cache_min

    effective_end = end_dt if end_dt <= cache_max else cache_max

    mask = (combined_df.index >= effective_start) & (combined_df.index <= effective_end)
    sliced = combined_df.loc[mask].copy()
    return sliced if not sliced.empty else None


def _fetch_ohlcv_core(
    ticker: str,
    country: str,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    existing_df: Optional[pd.DataFrame] = None,
) -> Optional[pd.DataFrame]:
    """실제 원천 API에서 OHLCV를 조회합니다."""

    if ticker.startswith("^"):
        if existing_df is not None and not existing_df.empty:
            fallback = existing_df[(existing_df.index >= start_dt) & (existing_df.index <= end_dt)]
            if not fallback.empty:
                return fallback
        if yf is None:
            print("오류: yfinance 라이브러리가 설치되지 않았습니다. 'pip install yfinance'로 설치해주세요.")
            return None
        try:
            with _silence_yfinance_logs():
                df = yf.download(
                    ticker,
                    start=start_dt,
                    end=end_dt + pd.Timedelta(days=1),
                    progress=False,
                    auto_adjust=True,
                )
            if df.empty:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                df = df.loc[:, ~df.columns.duplicated()]
            if not df.index.is_unique:
                df = df[~df.index.duplicated(keep="first")]
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            return df
        except Exception as e:
            print(f"경고: {ticker}의 데이터 조회 중 오류: {e}")
            if existing_df is not None and not existing_df.empty:
                fallback_df = existing_df[
                    (existing_df.index >= start_dt) & (existing_df.index <= end_dt)
                ]
                if not fallback_df.empty:
                    return fallback_df
            return None

    if country == "kor":
        if _stock is None:
            print("오류: pykrx 라이브러리가 설치되지 않았습니다. 'pip install pykrx'로 설치해주세요.")
            return None

    if country == "kor":
        # pykrx에 데이터를 요청하기 전에, 해당 기간에 거래일이 있는지 먼저 확인합니다.
        # 거래일이 없는 기간(예: 주말, 연휴)에 대해 불필요한 예외 발생을 방지합니다.
        trading_days_in_range = get_trading_days(
            start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"), "kor"
        )
        if not trading_days_in_range:
            return None  # 거래일이 없으므로 데이터를 가져올 수 없는 것이 정상입니다.

        all_dfs = []
        pykrx_failed = False
        pykrx_error_msg = None

        current_start = start_dt
        while current_start <= end_dt:
            current_end = min(
                current_start + pd.DateOffset(years=1) - pd.DateOffset(days=1), end_dt
            )
            start_str = current_start.strftime("%Y%m%d")
            end_str = current_end.strftime("%Y%m%d")

            try:
                df_part = _stock.get_etf_ohlcv_by_date(start_str, end_str, ticker)
                if df_part is None or df_part.empty:
                    df_part = _stock.get_market_ohlcv_by_date(start_str, end_str, ticker)
                if df_part is None or df_part.empty:
                    get_etn_func = getattr(_stock, "get_etn_ohlcv_by_date", None)
                    if callable(get_etn_func):
                        df_part = get_etn_func(start_str, end_str, ticker)
                if df_part is not None and not df_part.empty:
                    all_dfs.append(df_part)
            except (json.JSONDecodeError, KeyError) as err:
                pykrx_failed = True
                pykrx_error_msg = str(err) or "JSON/KeyError"
                print(f"경고: {ticker}의 {start_str}~{end_str} 기간 pykrx 조회 중 오류: {pykrx_error_msg}")
                break
            except Exception as e:
                err_text = str(e)
                print(f"경고: {ticker}의 {start_str}~{end_str} 기간 데이터 조회 중 오류: {err_text}")
                if isinstance(e, KeyError) or "are in the [columns]" in err_text:
                    pykrx_failed = True
                    pykrx_error_msg = err_text
                    break

            current_start += pd.DateOffset(years=1)

        if not all_dfs:
            pykrx_failed = True
            if pykrx_error_msg is None:
                # 요청 기간의 마지막 거래일이 오늘인 경우, 데이터가 아직 집계되지 않았을 가능성을 안내합니다.
                last_expected_day = max(trading_days_in_range)
                if last_expected_day.date() == datetime.now().date():
                    pykrx_error_msg = "데이터 없음 (장 마감 후 데이터가 집계되지 않았을 수 있습니다)"
                else:
                    pykrx_error_msg = "데이터 없음"

        if pykrx_failed:
            raise PykrxDataUnavailable(country, start_dt, end_dt, pykrx_error_msg)

        full_df = pd.concat(all_dfs)
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

    if country == "aus":
        if yf is None:
            print("오류: yfinance 라이브러리가 설치되지 않았습니다. 'pip install yfinance'로 설치해주세요.")
            return None

        ticker_yf = format_aus_ticker_for_yfinance(ticker)
        try:
            df = yf.download(
                ticker_yf,
                start=start_dt,
                end=end_dt + pd.Timedelta(days=1),
                progress=False,
                auto_adjust=False,  # 원본 데이터를 모두 가져옵니다.
            )
            if df.empty:
                return None

            # 실제 마감가를 unadjusted_close 컬럼에 백업합니다.
            if "Close" in df.columns:
                df["unadjusted_close"] = df["Close"]

            # Adj Close가 있는 경우, 이를 계산의 기준으로 삼고 Close 컬럼에 덮어씁니다.
            if "Adj Close" in df.columns:
                df["Close"] = df["Adj Close"]

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                df = df.loc[:, ~df.columns.duplicated()]
            if not df.index.is_unique:
                df = df[~df.index.duplicated(keep="first")]
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            return df
        except Exception as e:
            print(f"경고: {ticker}의 데이터 조회 중 오류: {e}")
            return None

    if country == "coin":
        try:
            from datetime import timezone

            import pandas as _pd

            base = ticker.upper()
            url = f"https://api.bithumb.com/public/candlestick/{base}_KRW/24h"
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            j = r.json() or {}
            data = j.get("data") or []
            if not data:
                return None
            rows = []
            for arr in data:
                try:
                    ts = int(arr[0])
                    o = float(arr[1])
                    c = float(arr[2])
                    h = float(arr[3])
                    low = float(arr[4])
                    v = float(arr[5])
                except Exception:
                    continue
                dt = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc).replace(tzinfo=None)
                rows.append((dt, o, h, low, c, v))
            if not rows:
                return None
            df = _pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
            df.set_index("Date", inplace=True)
            df.sort_index(inplace=True)
            cache_start_dt = _get_cache_start_dt()
            window_start = start_dt
            if cache_start_dt is not None and cache_start_dt > window_start:
                window_start = cache_start_dt
            if window_start > end_dt:
                return None
            df = df[(df.index >= window_start) & (df.index <= end_dt)]
            if df.empty:
                return None
            return df
        except Exception as e:
            print(f"경고: {ticker} 코인 OHLCV 조회 중 오류: {e}")
            return None

    print(f"오류: 지원하지 않는 국가 코드입니다: {country}")
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
        print(f"경고: {ticker}의 실시간 가격 조회 중 오류 발생: {e}")
    return None


_pykrx_name_cache: Dict[str, str] = {}


def fetch_pykrx_name(ticker: str) -> str:
    """
    pykrx를 통해 종목의 이름을 가져옵니다. ETF, 일반 주식, ETN을 모두 시도합니다.
    결과는 단일 실행 내에서 캐시됩니다.
    """
    if ticker in _pykrx_name_cache:
        return _pykrx_name_cache[ticker]

    if _stock is None:
        return ""

    name = ""
    try:
        # 1. ETF 이름 조회 시도
        name_candidate = _stock.get_etf_ticker_name(ticker)
        if isinstance(name_candidate, str) and name_candidate:
            name = name_candidate
    except Exception:
        pass

    # 2. ETF 조회가 실패하면 일반 주식으로 간주하고 다시 시도
    if not name:
        try:
            name_candidate = _stock.get_market_ticker_name(ticker)
            if isinstance(name_candidate, str) and name_candidate:
                name = name_candidate
        except Exception:
            pass

    # 3. 주식 조회도 실패하면 ETN으로 간주하고 다시 시도
    if not name:
        try:
            name_candidate = _stock.get_etn_ticker_name(ticker)
            if isinstance(name_candidate, str) and name_candidate:
                name = name_candidate
        except Exception:
            pass

    _pykrx_name_cache[ticker] = name
    return name


_yfinance_name_cache: Dict[str, str] = {}
_etf_name_cache: Dict[Tuple[str, str], str] = {}


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
        print(f"경고: {cache_key}의 이름 조회 중 오류 발생: {e}")
        _yfinance_name_cache[cache_key] = ""  # 실패도 캐시하여 재시도 방지
    return ""


def _get_display_name(country: str, ticker: str) -> str:
    key = (country.lower(), (ticker or "").upper())
    if key in _etf_name_cache:
        return _etf_name_cache[key]

    name = ""
    try:
        etf_blocks = get_etfs(country) or []
        for block in etf_blocks:
            if isinstance(block, dict):
                if "tickers" in block:
                    for item in block.get("tickers", []):
                        if isinstance(item, dict):
                            tkr = (item.get("ticker") or "").upper()
                            if tkr == key[1]:
                                name = item.get("name") or block.get("name") or ""
                                break
                    if name:
                        break
                else:
                    tkr = (block.get("ticker") or "").upper()
                    if tkr == key[1]:
                        name = block.get("name", "")
                        break
    except Exception:
        pass

    if not name:
        try:
            if country == "kor":
                name = fetch_pykrx_name(ticker)
            elif country == "aus":
                name = fetch_yfinance_name(ticker)
        except Exception:
            pass

    _etf_name_cache[key] = name or ""
    return _etf_name_cache[key]


def fetch_latest_unadjusted_price(ticker: str, country: str) -> Optional[float]:
    """Fetches the latest unadjusted closing price for a ticker."""
    if not yf:
        return None

    yfinance_ticker = ticker
    if country == "aus":
        if not ticker.upper().endswith(".AX"):
            yfinance_ticker = f"{ticker.upper()}.AX"
    elif country == "kor":
        if ticker.isdigit() and len(ticker) == 6:
            yfinance_ticker = f"{ticker.KS}"

    latest_trade_day = get_latest_trading_day(country)
    if not latest_trade_day:
        print(f"[ERROR] Could not determine latest trading day for country {country}.")
        return None

    start_date = latest_trade_day
    end_date = latest_trade_day + pd.Timedelta(days=1)
    date_str_for_log = start_date.strftime("%Y-%m-%d")

    try:
        print(
            f"[INFO] Fetching unadjusted price for {yfinance_ticker} for trading day: {date_str_for_log}, {start_date.strftime('%Y-%m-%d')}, {end_date.strftime('%Y-%m-%d')}"
        )

        df = yf.download(
            yfinance_ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            auto_adjust=False,
            progress=False,
            show_errors=False,  # 에러 로그를 직접 제어하기 위해 False로 설정
        )

        if df is not None and not df.empty:
            return df["Close"].iloc[-1]
        else:
            print(f"[WARN] No data returned for {yfinance_ticker} for date {date_str_for_log}.")
            return None

    except Exception as e:
        print(
            f"[ERROR] yfinance download failed for {yfinance_ticker} (date: {date_str_for_log}): {e}"
        )
        return None
