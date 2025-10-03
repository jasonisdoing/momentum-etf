"""
데이터 조회, 파일 입출력 등 공통으로 사용되는 유틸리티 함수 모음.
"""

import functools
import json
import logging
import os
import warnings
from datetime import datetime, time
from typing import Dict, List, Optional, Tuple, Union
from contextlib import contextmanager

import pandas as pd

# pkg_resources 워닝 억제 (가장 강력한 방법)
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
warnings.filterwarnings("ignore", category=UserWarning, module="pykrx")

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
from settings.common import REALTIME_PRICE_ENABLED

# from utils.notify import send_verbose_log_to_slack

import warnings

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings(
    "ignore",
    message=r"\['break_start', 'break_end'\] are discontinued",
    category=UserWarning,
    module=r"^pandas_market_calendars\.",
)


class _PykrxLogFilter(logging.Filter):
    """Suppress malformed pykrx util logs that break formatting."""  # pragma: no cover - log hygiene

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.msg
        args = record.args
        if (
            isinstance(msg, tuple)
            and len(msg) == 3
            and all(isinstance(m, str) for m in msg)
            and isinstance(args, tuple)
            and len(args) == 1
            and isinstance(args[0], dict)
            and not args[0]
        ):
            return False
        try:
            formatted = record.getMessage()
        except Exception:  # pragma: no cover - defensive
            formatted = ""
        if "None of [Index(['" in formatted:
            return False
        return True


_root_logger = logging.getLogger()
if not any(isinstance(f, _PykrxLogFilter) for f in _root_logger.filters):
    _root_logger.addFilter(_PykrxLogFilter())

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

    country_code = (country or "").strip().lower()

    if country_code != "kor" or cache_end is None:
        return False

    if ZoneInfo is not None:
        now_local = datetime.now(ZoneInfo("Asia/Seoul"))
    else:  # pragma: no cover
        now_local = datetime.now()

    # pykrx 데이터가 당일 분이 아직 나오지 않은 장 시작 전(16시 이전)이라면 생략
    if miss_start.normalize() == pd.Timestamp(now_local.date()) and now_local.hour < 16:
        return True

    return False


def _now_with_zone(tz_name: str) -> datetime:
    try:
        if ZoneInfo is not None:
            return datetime.now(ZoneInfo(tz_name))
    except Exception:
        pass
    return datetime.now()


def _is_kor_realtime_window(now_kst: datetime) -> bool:
    start = time(9, 0)
    end = time(23, 59, 59)
    current = now_kst.time()
    return start <= current <= end


def _should_use_realtime_price(country: str) -> bool:
    if not REALTIME_PRICE_ENABLED:
        return False
    country_code = (country or "").strip().lower()

    if country_code != "kor":
        return False

    now_kst = _now_with_zone("Asia/Seoul")
    if not _is_kor_realtime_window(now_kst):
        return False

    today_str = now_kst.strftime("%Y-%m-%d")
    try:
        return bool(get_trading_days(today_str, today_str, "kor"))
    except Exception:
        return False


@functools.lru_cache(maxsize=1)
def get_aud_to_krw_rate() -> Optional[float]:
    """yfinance를 사용하여 AUD/KRW 환율을 조회합니다."""
    if not yf:
        return None
    try:
        ticker = yf.Ticker("AUDKRW=X")
        # 가장 최근 가격을 가져오기 위해 2일간의 1분 단위 데이터 시도
        data = ticker.history(period="2d", interval="1m")
        if not data.empty:
            return data["Close"].iloc[-1]
        # 1m 데이터가 없으면 일 단위 데이터로 폴백
        data = ticker.history(period="2d")
        if not data.empty:
            return data["Close"].iloc[-1]
    except Exception as e:
        print(f"AUD/KRW 환율 정보를 가져오는 데 실패했습니다: {e}")
        return None
    return None


@functools.lru_cache(maxsize=1)
def get_usd_to_krw_rate() -> Optional[float]:
    """yfinance를 사용하여 USD/KRW 환율을 조회합니다."""
    if not yf:
        return None
    try:
        ticker = yf.Ticker("USDKRW=X")
        # 가장 최근 가격을 가져오기 위해 2일간의 1분 단위 데이터 시도
        data = ticker.history(period="2d", interval="1m")
        if not data.empty:
            return data["Close"].iloc[-1]
        # 1m 데이터가 없으면 일 단위 데이터로 폴백
        data = ticker.history(period="2d")
        if not data.empty:
            return data["Close"].iloc[-1]
    except Exception as e:
        print(f"USD/KRW 환율 정보를 가져오는 데 실패했습니다: {e}")
    return None


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

        cal_code = {"kor": "XKRX", "aus": "ASX", "us": "NYSE"}.get(country_code)
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

    country_code = (country or "").strip().lower()

    if country_code == "kor":
        trading_days_ts = _pmc("kor")
    elif country_code == "aus":
        trading_days_ts = _pmc("aus")
    elif country_code == "us":
        trading_days_ts = _pmc("us")
    else:
        print(f"오류: 지원하지 않는 국가 코드입니다: {country_code}")
        return []

    # 최종적으로 start_date와 end_date 사이의 날짜만 반환하고, 중복 제거 및 정렬합니다.
    start_date_ts = pd.to_datetime(start_date).normalize()
    end_date_ts = pd.to_datetime(end_date).normalize()
    final_list = [d for d in trading_days_ts if start_date_ts <= d <= end_date_ts]

    return sorted(list(set(final_list)))


def count_trading_days(
    country: str,
    start_date: Union[str, datetime, pd.Timestamp],
    end_date: Union[str, datetime, pd.Timestamp],
) -> int:
    """Return number of trading days between two dates (inclusive)."""

    start_ts = pd.to_datetime(start_date).normalize()
    end_ts = pd.to_datetime(end_date).normalize()

    if start_ts > end_ts:
        return 0

    country_code = (country or "").strip().lower()

    days = get_trading_days(
        start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d"), country_code
    )
    return len(days)


@functools.lru_cache(maxsize=5)
def get_latest_trading_day(country: str) -> pd.Timestamp:
    """
    오늘 또는 가장 가까운 과거의 '데이터가 있을 것으로 예상되는' 거래일을 pd.Timestamp 형식으로 반환합니다.
    """
    country_code = (country or "").strip().lower()

    end_dt = pd.Timestamp.now()

    # 한국 시장의 경우, 장 마감 데이터가 집계되기 전(오후 4시 이전)이라면,
    # 조회 기준일을 하루 전으로 설정하여 어제까지의 데이터만 사용하도록 합니다.
    if country_code == "kor":
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
            if get_trading_days(check_date_str, check_date_str, country_code):
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

    country_code = (country or "").strip().lower() or "kor"

    if date_range and len(date_range) == 2:
        try:  # date_range가 있으면, 다른 기간 인자들을 무시하고 이를 기준으로 start_dt, end_dt를 설정합니다.
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
        if months_range is not None and len(months_range) == 2:  # months_range가 있으면 사용
            start_off, end_off = months_range
            start_dt = now - pd.DateOffset(months=int(start_off))
            end_dt = now - pd.DateOffset(months=int(end_off))
        else:
            months_back = months_back or 3  # months_back의 기본값은 3개월
            start_dt = now - pd.DateOffset(months=int(months_back))
            end_dt = now

    # 조회 종료일(end_dt)이 실제 데이터가 있는 마지막 거래일을 초과하지 않도록 보정합니다.
    # 이는 주말이나 휴일에 다음 거래일을 기준으로 데이터를 조회할 때, 아직 존재하지 않는
    # 미래 데이터를 조회하려는 시도를 방지합니다.
    latest_known_trading_day = get_latest_trading_day(country_code)
    if end_dt > latest_known_trading_day:
        end_dt = latest_known_trading_day

    if start_dt > end_dt:
        # 보정 후 시작일이 종료일보다 미래가 될 수 있으므로, 이 경우 데이터를 조회하지 않습니다.
        return None

    df = _fetch_ohlcv_with_cache(ticker, country_code, start_dt.normalize(), end_dt.normalize())

    if df is None or df.empty:
        return df

    if _should_use_realtime_price(country_code):
        df = _overlay_realtime_price(df, ticker, country_code)

    return df


def _fetch_ohlcv_with_cache(
    ticker: str,
    country: str,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
) -> Optional[pd.DataFrame]:
    country_code = (country or "").strip().lower()

    cached_df = load_cached_frame(country_code, ticker)
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
        if cache_end is not None and _should_skip_pykrx_fetch(country_code, cache_end, miss_start):
            continue

        # Check if there are any trading days in the missing range before attempting to fetch.
        # This prevents errors when the gap consists only of non-trading days (weekends, holidays).
        trading_days_in_gap = get_trading_days(
            miss_start.strftime("%Y-%m-%d"), miss_end.strftime("%Y-%m-%d"), country_code
        )
        if not trading_days_in_gap:
            continue

        try:
            fetched = _fetch_ohlcv_core(ticker, country_code, miss_start, miss_end, cached_df)
        except PykrxDataUnavailable:
            # 신규 상장 등으로 과거 데이터가 존재하지 않거나, 최신 데이터만 캐시에 있는 경우
            # 기존 캐시로 충분하면 네트워크 오류를 무시하고 계속 진행합니다.
            if cached_df is not None:
                if cache_start is not None and miss_end < cache_start:
                    continue
                if cache_end is not None and miss_start > cache_end:
                    continue
            raise

        if fetched is not None and not fetched.empty:
            new_frames.append(fetched)

    combined_df = cached_df
    # prev_count = 0 if cached_df is None else cached_df.shape[0]
    # added_count = 0

    if new_frames:
        frames = []
        if cached_df is not None and not cached_df.empty:
            frames.append(cached_df)
        frames.extend(new_frames)
        combined_df = pd.concat(frames)
        combined_df.sort_index(inplace=True)
        combined_df = combined_df[~combined_df.index.duplicated(keep="last")]
        save_cached_frame(country_code, ticker, combined_df)

        # new_total = combined_df.shape[0]
        # added_count = max(0, new_total - prev_count)
        # if added_count > 0:
        #     try:
        #         display_name = _get_display_name(country, ticker)
        #         suffix = f"({display_name})" if display_name else ""
        #         send_verbose_log_to_slack(
        #             f"[CACHE] {country.upper()}/{ticker}{suffix} {new_total:,} rows (+{added_count:,} rows)"
        #         )
        #     except Exception:
        #         pass

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


def _overlay_realtime_price(df: pd.DataFrame, ticker: str, country: str) -> pd.DataFrame:
    """개장 이후 실시간 가격을 캐시 데이터에 덧씌웁니다."""

    country_code = (country or "").strip().lower()

    if country_code != "kor":
        return df
    # 네이버 실시간 가격은 한국 상장 종목(숫자/알파벳 코드)에만 적용
    if ticker.startswith("^"):
        return df

    price = fetch_naver_realtime_price(ticker)
    if price is None or price <= 0:
        return df

    df = df.copy()
    df.sort_index(inplace=True)

    now_kst = _now_with_zone("Asia/Seoul")
    today = pd.Timestamp(now_kst.date())

    if not isinstance(df.index, pd.DatetimeIndex):
        return df

    last_idx = df.index[-1]
    last_date = pd.Timestamp(last_idx).normalize()

    target_idx = None
    if last_date == today:
        target_idx = last_idx
    elif last_date < today:
        # 새로운 거래일 행을 추가합니다.
        new_row = df.iloc[-1].copy()
        for col in ("Open", "High", "Low", "Close", "Adj Close"):
            if col in df.columns:
                new_row[col] = price
        if "Volume" in new_row.index:
            new_row["Volume"] = 0.0
        target_idx = today
        df.loc[target_idx] = new_row
        df.sort_index(inplace=True)
    else:
        # 미래 날짜가 이미 존재하는 경우는 그대로 둡니다.
        return df

    if target_idx is not None:
        for col in ("Close", "Adj Close", "Open", "High", "Low"):
            if col in df.columns:
                df.loc[target_idx, col] = price

    return df


def _fetch_ohlcv_core(
    ticker: str,
    country: str,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    existing_df: Optional[pd.DataFrame] = None,
) -> Optional[pd.DataFrame]:
    """실제 원천 API에서 OHLCV를 조회합니다."""

    country_code = (country or "").strip().lower()

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
                    auto_adjust=True,
                    progress=False,
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

    if country_code == "kor":
        if _stock is None:
            print("오류: pykrx 라이브러리가 설치되지 않았습니다. 'pip install pykrx'로 설치해주세요.")
            return None

    if country_code == "kor":
        # pykrx에 데이터를 요청하기 전에, 해당 기간에 거래일이 있는지 먼저 확인합니다.
        # 거래일이 없는 기간(예: 주말, 연휴)에 대해 불필요한 예외 발생을 방지합니다.
        trading_days_in_range = get_trading_days(
            start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"), country_code
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
            raise PykrxDataUnavailable(country_code, start_dt, end_dt, pykrx_error_msg)

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

    if country_code == "aus":
        if yf is None:
            print("오류: yfinance 라이브러리가 설치되지 않았습니다. 'pip install yfinance'로 설치해주세요.")
            return None

        ticker_yf = format_aus_ticker_for_yfinance(ticker)
        try:
            # yfinance에서 데이터를 가져올 때 auto_adjust=False로 설정하고, 수동으로 조정합니다.
            df = yf.download(
                ticker_yf,
                start=start_dt,
                end=end_dt + pd.Timedelta(days=1),
                auto_adjust=False,  # 수동으로 조정
                progress=False,
            )
            if df.empty:
                return None

            # MultiIndex 컬럼을 정리합니다.
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
                df = df.loc[:, ~df.columns.duplicated()]

            # 조정되지 않은 종가를 백업
            if "Close" in df.columns:
                df["unadjusted_close"] = df["Close"]

            # 조정된 종가가 있으면 사용하고, 없으면 원본 종가를 사용
            if "Adj Close" in df.columns and not df["Adj Close"].isnull().all():
                df["Close"] = df["Adj Close"]

            # 필요한 컬럼만 남기고 나머지는 제거
            required_columns = ["Open", "High", "Low", "Close", "Volume", "unadjusted_close"]
            df = df[[col for col in required_columns if col in df.columns]]
            if not df.index.is_unique:
                df = df[~df.index.duplicated(keep="first")]
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            return df
        except Exception as e:
            print(f"경고: {ticker}의 데이터 조회 중 오류: {e}")
            return None

    if country_code == "us":
        if yf is None:
            print("오류: yfinance 라이브러리가 설치되지 않았습니다. 'pip install yfinance'로 설치해주세요.")
            return None

        try:
            df = yf.download(
                ticker,
                start=start_dt,
                end=end_dt + pd.Timedelta(days=1),
                auto_adjust=True,  # 수정 종가 사용
                progress=False,
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
            return None

    print(f"오류: 지원하지 않는 국가 코드입니다: {country_code}")
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


def fetch_au_realtime_price(ticker: str) -> Optional[float]:
    """
    yfinance 라이브러리를 통해 호주 종목의 실시간 현재가를 조회합니다.
    """
    if not yf:
        return None

    ticker_yf = format_aus_ticker_for_yfinance(ticker)
    try:
        # yfinance 라이브러리를 사용하여 가격 정보를 가져옵니다.
        stock = yf.Ticker(ticker_yf)
        # 'regularMarketPrice'는 장중 실시간 가격, 'previousClose'는 전일 종가입니다.
        # 실시간 데이터가 없을 경우를 대비하여 두 값을 모두 확인합니다.
        price = stock.info.get("regularMarketPrice") or stock.info.get("previousClose")
        if price:
            return float(price)
    except Exception as e_yf:
        print(f"경고: {ticker}의 호주 실시간 가격 조회(yfinance) 실패: {e_yf}")
    return None


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

    if not name:
        name = _get_display_name("kor", ticker)

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


def resolve_security_name(country: str, ticker: str) -> str:
    """지정한 국가/티커의 표시용 이름을 반환합니다."""
    if not ticker:
        return ""

    ticker_upper = ticker.strip().upper()
    if not ticker_upper:
        return ""

    country_lower = (country or "").strip().lower()

    name = ""
    if country_lower == "kor":
        name = fetch_pykrx_name(ticker_upper)
    elif country_lower == "aus":
        name = fetch_yfinance_name(ticker_upper)

    if not name:
        name = _get_display_name(country_lower, ticker_upper)

    return name


def _get_display_name(country: str, ticker: str) -> str:
    country_code = (country or "").strip().lower()

    key = (country_code, (ticker or "").upper())
    if key in _etf_name_cache:
        return _etf_name_cache[key]

    name = ""
    try:
        etf_blocks = get_etfs(country_code) or []
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
            if country_code == "kor":
                name = fetch_pykrx_name(ticker)
            elif country_code == "aus":
                name = fetch_yfinance_name(ticker)
        except Exception:
            pass

    _etf_name_cache[key] = name or ""
    return _etf_name_cache[key]


def fetch_latest_unadjusted_price(ticker: str, country: str) -> Optional[float]:
    """Fetches the latest unadjusted closing price for a ticker."""
    if not yf:
        return None

    country_code = (country or "").strip().lower() or "kor"

    yfinance_ticker = ticker
    if country_code == "aus":
        if not ticker.upper().endswith(".AX"):
            yfinance_ticker = f"{ticker.upper()}.AX"
    elif country_code == "kor":
        if ticker.isdigit() and len(ticker) == 6:
            yfinance_ticker = f"{ticker.KS}"

    latest_trade_day = get_latest_trading_day(country_code)
    if not latest_trade_day:
        print(f"[ERROR] Could not determine latest trading day for country {country_code}.")
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
