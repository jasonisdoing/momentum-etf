"""
데이터 조회, 파일 입출력 등 공통으로 사용되는 유틸리티 함수 모음.
"""

import functools
import json
import logging
import os
import warnings
from datetime import datetime, time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from contextlib import contextmanager

import pandas as pd

from config import KOR_REALTIME_ETF_PRICE_SOURCE

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
from utils.stock_list_io import get_etfs, get_listing_date, set_listing_date
from utils.logger import get_app_logger

# from utils.notification import send_verbose_log_to_slack

import warnings

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings(
    "ignore",
    message=r"\['break_start', 'break_end'\] are discontinued",
    category=UserWarning,
    module=r"^pandas_market_calendars\.",
)


class _PykrxLogFilter(logging.Filter):
    """형식이 무너지는 pykrx util 로그를 억제한다."""  # pragma: no cover - 로그 정리 목적

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

logger = get_app_logger()

_KOR_PRICE_SOURCE_NORMALIZED = (KOR_REALTIME_ETF_PRICE_SOURCE or "").strip().lower()
_KOR_ALLOWED_PRICE_SOURCES = {"price", "nav"}

if _KOR_PRICE_SOURCE_NORMALIZED not in _KOR_ALLOWED_PRICE_SOURCES:
    raise ValueError("KOR_REALTIME_ETF_PRICE_SOURCE must be one of {'Price', 'Nav'}")


def _safe_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
        message = f"[{country.upper()}] pykrx data unavailable " f"({start_dt.date()}~{end_dt.date()}): {detail}"
        super().__init__(message)


class RateLimitException(Exception):
    """API rate limit에 도달했을 때 사용되는 예외."""

    def __init__(self, ticker: str, detail: str) -> None:
        self.ticker = ticker
        self.detail = detail
        message = f"Rate limit exceeded for {ticker}: {detail}"
        super().__init__(message)


def _get_cache_start_dt() -> Optional[pd.Timestamp]:
    """config.py에서 캐시 시작 날짜를 로드합니다."""
    try:
        from utils.settings_loader import load_common_settings

        common_settings = load_common_settings()
        raw = common_settings.get("CACHE_START_DATE")
    except Exception:
        return None

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


MARKET_OPEN_INFO = {
    "kor": ("Asia/Seoul", time(9, 0)),
    "aus": ("Asia/Seoul", time(8, 0)),
    "us": ("America/New_York", time(9, 30)),
}


def _should_skip_today_range(country_code: str, target_end: pd.Timestamp) -> bool:
    if ZoneInfo is None:
        return False

    info = MARKET_OPEN_INFO.get((country_code or "").strip().lower())
    if not info:
        return False

    tz_name, open_time = info
    try:
        now_local = _now_with_zone(tz_name)
    except Exception:
        return False

    if target_end.normalize() != pd.Timestamp(now_local.date()):
        return False

    if now_local.time() >= open_time:
        return False

    return True


def _is_time_in_window(now_dt: datetime, start: time, end: time) -> bool:
    current = now_dt.time()
    return start <= current <= end


def _should_use_realtime_price(country: str) -> bool:
    country_code = (country or "").strip().lower()

    if country_code == "kor":
        now_kst = _now_with_zone("Asia/Seoul")
        if not _is_time_in_window(now_kst, time(9, 0), time(23, 59, 59)):
            return False
        today_str = now_kst.strftime("%Y-%m-%d")
        try:
            return bool(get_trading_days(today_str, today_str, "kor"))
        except Exception:
            return False

    if country_code == "aus":
        now_kst = _now_with_zone("Asia/Seoul")
        if not _is_time_in_window(now_kst, time(10, 0), time(23, 59, 59)):
            return False
        today_str = now_kst.strftime("%Y-%m-%d")
        try:
            return bool(get_trading_days(today_str, today_str, "aus"))
        except Exception:
            return False

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
        error_msg = str(e)
        if "Too Many Requests" in error_msg or "Rate limited" in error_msg or "429" in error_msg:
            logger.error("AUD/KRW 환율 조회 Rate Limit 에러: %s", e)
            raise RateLimitException("AUDKRW=X", error_msg)
        logger.warning("AUD/KRW 환율 정보를 가져오는 데 실패했습니다: %s", e)
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
        error_msg = str(e)
        if "Too Many Requests" in error_msg or "Rate limited" in error_msg or "429" in error_msg:
            logger.error("USD/KRW 환율 조회 Rate Limit 에러: %s", e)
            raise RateLimitException("USDKRW=X", error_msg)
        logger.warning("USD/KRW 환율 정보를 가져오는 데 실패했습니다: %s", e)
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
            logger.warning("pandas_market_calendars(%s:%s) 조회 실패: %s", country_code, cal_code, e)
        return []

    country_code = (country or "").strip().lower()

    if country_code == "kor":
        trading_days_ts = _pmc("kor")
    elif country_code == "aus":
        trading_days_ts = _pmc("aus")
    elif country_code == "us":
        trading_days_ts = _pmc("us")
    else:
        logger.error("지원하지 않는 국가 코드입니다: %s", country_code)
        return []

    # 최종적으로 start_date와 end_date 사이의 날짜만 반환하고, 중복 제거 및 정렬합니다.
    start_date_ts = pd.to_datetime(start_date).normalize()
    end_date_ts = pd.to_datetime(end_date).normalize()
    final_list = [d for d in trading_days_ts if start_date_ts <= d <= end_date_ts]

    return sorted(list(set(final_list)))


def is_trading_day(
    country: str,
    date: Union[str, datetime, pd.Timestamp, None] = None,
) -> bool:
    """주어진 날짜가 해당 국가의 거래일인지 여부를 반환합니다."""

    target = pd.Timestamp(date if date is not None else datetime.now())
    target = target.tz_localize(None) if getattr(target, "tzinfo", None) else target
    target_norm = target.normalize()
    date_str = target_norm.strftime("%Y-%m-%d")

    try:
        return bool(get_trading_days(date_str, date_str, country))
    except Exception:
        return False


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

    days = get_trading_days(start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d"), country_code)
    return len(days)


@functools.lru_cache(maxsize=5)
def get_latest_trading_day(country: str) -> pd.Timestamp:
    """
    오늘 또는 가장 가까운 과거의 '데이터가 있을 것으로 예상되는' 거래일을 pd.Timestamp 형식으로 반환합니다.
    """
    country_code = (country or "").strip().lower()

    end_dt = pd.Timestamp.now()

    tz_info = MARKET_OPEN_INFO.get(country_code)
    if tz_info is not None:
        tz_name, open_time = tz_info
        try:
            local_now = _now_with_zone(tz_name)
            candidate_date = local_now.date()
            if local_now.time() < open_time:
                candidate_date = candidate_date - pd.Timedelta(days=1)
            end_dt = pd.Timestamp(candidate_date)
        except Exception:
            # 타임존 처리 실패 시 안전하게 폴백
            pass
    elif country_code == "kor":
        # 과거 코드와의 호환을 위해 기본 동작 유지 (이 경로는 사실상 사용되지 않음)
        try:
            if ZoneInfo is not None:
                local_now = datetime.now(ZoneInfo("Asia/Seoul"))
            else:  # pragma: no cover
                local_now = datetime.now()
            if local_now.hour < 9:
                end_dt = end_dt - pd.DateOffset(days=1)
            else:
                end_dt = pd.Timestamp(local_now.date())
        except Exception:
            pass
    else:
        # 타임존 정보를 모르면 현재 날짜 기준으로 처리
        end_dt = end_dt.normalize()

    # 최근 10일간의 거래일을 한 번에 조회 (효율성 개선)
    start_date = (end_dt - pd.DateOffset(days=10)).strftime("%Y-%m-%d")
    end_date = end_dt.strftime("%Y-%m-%d")

    try:
        # 최근 10일간의 모든 거래일을 한 번에 가져옴
        trading_days = get_trading_days(start_date, end_date, country_code)

        if trading_days:
            # 가장 최근 거래일을 반환
            latest_trading_day = max(trading_days)
            return latest_trading_day.normalize()
    except Exception as e:
        logger.warning("거래일 일괄 조회 중 오류 발생: %s", e)

    # 폴백: 10일간 거래일을 찾지 못하면 오늘 날짜를 정규화하여 반환합니다.
    logger.warning(
        "최근 10일 내에 거래일을 찾지 못했습니다. 오늘 날짜(%s)를 사용합니다.",
        end_dt.strftime("%Y-%m-%d"),
    )
    return end_dt.normalize()


def get_next_trading_day(
    country: str,
    reference_date: Optional[pd.Timestamp] = None,
    *,
    search_horizon_days: int = 30,
) -> Optional[pd.Timestamp]:
    """reference_date 이후의 다음 거래일을 반환한다."""

    country_code = (country or "").strip().lower()
    ref = (reference_date or pd.Timestamp.now()).normalize()
    search_end = ref + pd.DateOffset(days=search_horizon_days)

    trading_days = get_trading_days(ref.strftime("%Y-%m-%d"), search_end.strftime("%Y-%m-%d"), country_code)
    for day in trading_days:
        day_norm = pd.Timestamp(day).normalize()
        if day_norm > ref:
            return day_norm
    return None


def fetch_ohlcv(
    ticker: str,
    country: str = "kor",
    months_back: int = None,
    months_range: Optional[List[int]] = None,
    date_range: Optional[List[Optional[str]]] = None,
    base_date: Optional[pd.Timestamp] = None,
    *,
    cache_country: Optional[str] = None,
    force_refresh: bool = False,
    skip_realtime: bool = False,
    update_listing_meta: bool = False,
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
            logger.error("잘못된 date_range 형식: %s. 'YYYY-MM-DD' 형식을 사용해야 합니다.", date_range)
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

    df = _fetch_ohlcv_with_cache(
        ticker,
        country_code,
        start_dt.normalize(),
        end_dt.normalize(),
        cache_country_override=cache_country,
        force_refresh=force_refresh,
        skip_realtime=skip_realtime,
        update_listing_meta=update_listing_meta,
    )

    if df is None or df.empty:
        logger.debug("%s (%s) 가격 데이터를 가져오지 못했습니다.", ticker, country_code.upper())
        return None

    return df


def _fetch_ohlcv_with_cache(
    ticker: str,
    country: str,
    start_dt: pd.Timestamp,
    end_dt: pd.Timestamp,
    *,
    cache_country_override: Optional[str] = None,
    force_refresh: bool = False,
    skip_realtime: bool = False,
    update_listing_meta: bool = False,
) -> Optional[pd.DataFrame]:
    country_code = (country or "").strip().lower()
    cache_country_code = (cache_country_override or country_code).strip().lower() or country_code

    listing_date_str = get_listing_date(country_code, ticker)
    listing_ts = None
    if listing_date_str:
        try:
            listing_ts = pd.to_datetime(listing_date_str).normalize()
        except Exception:
            listing_ts = None

    # 캐시 시작일 가져오기
    cache_seed_dt = _get_cache_start_dt()

    # 데이터 다운로드 시작일 결정: max(요청 시작일, 실제 상장일, CACHE_START_DATE)
    request_start_dt = start_dt
    if listing_ts is not None and start_dt < listing_ts:
        request_start_dt = listing_ts

    # CACHE_START_DATE가 있고, 실제 상장일보다 늦으면 CACHE_START_DATE 사용
    if cache_seed_dt is not None:
        if listing_ts is None or cache_seed_dt > listing_ts:
            request_start_dt = max(request_start_dt, cache_seed_dt)

    cache_country_display = cache_country_code.upper()

    missing_ranges: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    cache_start: Optional[pd.Timestamp] = None
    cache_end: Optional[pd.Timestamp] = None

    if force_refresh:
        cached_df = None
        missing_ranges.append((request_start_dt, end_dt))
    else:
        cached_df = load_cached_frame(cache_country_code, ticker)
        # cache_seed_dt는 이미 위에서 가져왔으므로 중복 제거
        if (cached_df is None or cached_df.empty) and cache_seed_dt is not None:
            if request_start_dt > cache_seed_dt:
                request_start_dt = cache_seed_dt

        if cached_df is None or cached_df.empty:
            cached_df = None
            if listing_ts is not None and end_dt < listing_ts:
                missing_ranges = []
                return None
            missing_ranges.append((request_start_dt, end_dt))
        else:
            cache_start = cached_df.index.min().normalize()
            cache_end = cached_df.index.max().normalize()

            if request_start_dt < cache_start:
                lower_bound = request_start_dt
                if listing_ts is not None and cache_start > listing_ts:
                    lower_bound = max(request_start_dt, listing_ts)
                missing_ranges.append((lower_bound, cache_start - pd.Timedelta(days=1)))
            if end_dt > cache_end:
                upper_bound = end_dt
                if listing_ts is not None and listing_ts > cache_end:
                    upper_bound = max(end_dt, listing_ts)
                missing_ranges.append((cache_end + pd.Timedelta(days=1), upper_bound))

    new_frames: List[pd.DataFrame] = []
    unfilled_ranges: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    for miss_start, miss_end in missing_ranges:
        if miss_start > miss_end:
            continue

        effective_end = miss_end
        log_pending = False
        start_str = miss_start.strftime("%Y-%m-%d")
        today_norm = pd.Timestamp.now().normalize()
        if skip_realtime and effective_end >= today_norm:
            effective_end = today_norm - pd.Timedelta(days=1)
            log_pending = False
        end_str = miss_end.strftime("%Y-%m-%d")
        if _should_skip_today_range(country_code, miss_end):
            effective_end = miss_end - pd.Timedelta(days=1)
            if effective_end < miss_start:
                continue
            end_str = effective_end.strftime("%Y-%m-%d")
            logger.debug(
                "[CACHE] %s/%s 오늘 개장 전이므로 조회 범위를 조정합니다: %s ~ %s",
                cache_country_display,
                ticker,
                start_str,
                end_str,
            )
        else:
            log_pending = True

        if effective_end < miss_start:
            continue

        if cache_end is not None and _should_skip_pykrx_fetch(country_code, cache_end, miss_start):
            continue

        trading_days_in_gap = get_trading_days(miss_start.strftime("%Y-%m-%d"), effective_end.strftime("%Y-%m-%d"), country_code)
        if not trading_days_in_gap:
            if log_pending:
                logger.debug(
                    "[CACHE] %s/%s 범위(%s~%s)에 거래일이 없어 캐시 갱신을 건너뜁니다.",
                    cache_country_display,
                    ticker,
                    start_str,
                    end_str,
                )
            continue

        if log_pending:
            logger.info(
                "[CACHE] %s/%s 누락 구간을 조회합니다: %s ~ %s",
                cache_country_display,
                ticker,
                start_str,
                end_str,
            )

        try:
            fetched = _fetch_ohlcv_core(ticker, country_code, miss_start, effective_end, cached_df)
        except PykrxDataUnavailable:
            if cached_df is not None:
                if cache_start is not None and effective_end < cache_start:
                    continue
                if cache_end is not None and miss_start > cache_end:
                    continue
            raise

        if fetched is not None and not fetched.empty:
            new_frames.append(fetched)
        else:
            unfilled_ranges.append((miss_start, effective_end))

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
        save_cached_frame(cache_country_code, ticker, combined_df)

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

    if unfilled_ranges:
        ranges_text = ", ".join(f"{start.strftime('%Y-%m-%d')}~{end.strftime('%Y-%m-%d')}" for start, end in unfilled_ranges)
        raise RuntimeError(f"{ticker}의 가격 데이터 누락 구간을 가져오지 못했습니다: {ranges_text}")

    cache_min = combined_df.index.min()
    cache_max = combined_df.index.max()

    effective_start = request_start_dt
    if request_start_dt > cache_max:
        effective_start = cache_max
    elif request_start_dt < cache_min:
        effective_start = cache_min

    effective_end = end_dt if end_dt <= cache_max else cache_max

    if not skip_realtime and _should_use_realtime_price(cache_country_code):
        updated_df = _overlay_realtime_price(combined_df, ticker, cache_country_code)
        if not updated_df.equals(combined_df):
            combined_df = updated_df
            save_cached_frame(cache_country_code, ticker, combined_df)
        else:
            combined_df = updated_df

    mask = (combined_df.index >= effective_start) & (combined_df.index <= effective_end)
    sliced = combined_df.loc[mask].copy()
    if sliced.empty:
        return None

    first_available = combined_df.index.min().normalize()
    target_listing_ts = first_available
    if cache_seed_dt is not None and target_listing_ts < cache_seed_dt:
        target_listing_ts = cache_seed_dt

    should_update_listing = False
    if listing_ts is None:
        should_update_listing = True
    else:
        if target_listing_ts < listing_ts:
            should_update_listing = True
        elif cache_seed_dt is not None and listing_ts < cache_seed_dt <= target_listing_ts:
            should_update_listing = True

    if should_update_listing and update_listing_meta:
        try:
            set_listing_date(
                country_code,
                ticker,
                target_listing_ts.strftime("%Y-%m-%d"),
            )
        except Exception as exc:
            logger.debug("[CACHE] 상장일 저장 실패 (%s/%s): %s", country_code.upper(), ticker, exc)

    return sliced


def _overlay_realtime_price(df: pd.DataFrame, ticker: str, country: str) -> pd.DataFrame:
    """개장 이후 실시간 가격을 캐시 데이터에 덧씌웁니다."""

    country_code = (country or "").strip().lower()

    price: Optional[float] = None
    nav_value: Optional[float] = None

    if country_code == "kor":
        # 네이버 실시간 가격은 한국 상장 종목(숫자/알파벳 코드)에만 적용
        if ticker.startswith("^"):
            return df
        snapshot_entry = get_cached_naver_etf_snapshot_entry(ticker)
        if snapshot_entry:
            price_candidate = _safe_float(snapshot_entry.get("nowVal"))
            nav_candidate = _safe_float(snapshot_entry.get("nav"))
            price = price_candidate if price_candidate is not None else _safe_float(snapshot_entry.get("price"))
            nav_value = nav_candidate
        if price is None or price <= 0:
            price = fetch_naver_realtime_price(ticker)
    elif country_code == "aus":
        # ASX 지수 티커(예: ^AXJO)는 그대로 둡니다.
        if ticker.startswith("^"):
            return df
        price = fetch_au_realtime_price(ticker)
    else:
        return df

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
        if country_code == "kor" and (nav_value is not None or _KOR_PRICE_SOURCE_NORMALIZED == "nav"):
            effective_nav = nav_value if nav_value is not None else price
            if "NAV" in df.columns:
                new_row["NAV"] = effective_nav
            else:
                new_row = new_row.reindex(list(new_row.index) + ["NAV"])
                new_row["NAV"] = effective_nav
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
        if country_code == "kor" and (nav_value is not None or _KOR_PRICE_SOURCE_NORMALIZED == "nav"):
            effective_nav = nav_value if nav_value is not None else price
            if "NAV" not in df.columns:
                df["NAV"] = pd.NA
            df.loc[target_idx, "NAV"] = effective_nav

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
            logger.error("yfinance 라이브러리가 설치되어 있지 않습니다. 'pip install yfinance'로 설치해주세요.")
            return None
        try:
            with _silence_yfinance_logs():
                # signal 대신 간단하게 yfinance 자체 타임아웃 사용
                df = yf.download(
                    ticker,
                    start=start_dt,
                    end=end_dt + pd.Timedelta(days=1),
                    auto_adjust=True,
                    progress=False,
                    timeout=30,
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
        except TimeoutError as e:
            logger.warning("%s 데이터 조회 타임아웃 (30초): %s", ticker, e)
            if existing_df is not None and not existing_df.empty:
                fallback_df = existing_df[(existing_df.index >= start_dt) & (existing_df.index <= end_dt)]
                if not fallback_df.empty:
                    return fallback_df
            return None
        except Exception as e:
            error_msg = str(e)
            if "Too Many Requests" in error_msg or "Rate limited" in error_msg or "429" in error_msg:
                logger.error("%s 데이터 조회 Rate Limit 에러: %s", ticker, e)
                raise RateLimitException(ticker, error_msg)
            logger.warning("%s의 데이터 조회 중 오류: %s", ticker, e)
            if existing_df is not None and not existing_df.empty:
                fallback_df = existing_df[(existing_df.index >= start_dt) & (existing_df.index <= end_dt)]
                if not fallback_df.empty:
                    return fallback_df
            return None

    if country_code == "kor":
        if _stock is None:
            logger.error("pykrx 라이브러리가 설치되어 있지 않습니다. 'pip install pykrx'로 설치해주세요.")
            return None

    if country_code == "kor":
        # pykrx에 데이터를 요청하기 전에, 해당 기간에 거래일이 있는지 먼저 확인합니다.
        # 거래일이 없는 기간(예: 주말, 연휴)에 대해 불필요한 예외 발생을 방지합니다.
        trading_days_in_range = get_trading_days(start_dt.strftime("%Y-%m-%d"), end_dt.strftime("%Y-%m-%d"), country_code)
        if not trading_days_in_range:
            return None  # 거래일이 없으므로 데이터를 가져올 수 없는 것이 정상입니다.

        all_dfs = []
        pykrx_failed = False
        pykrx_error_msg = None

        current_start = start_dt
        while current_start <= end_dt:
            current_end = min(current_start + pd.DateOffset(years=1) - pd.DateOffset(days=1), end_dt)
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
                logger.warning(
                    "%s의 %s~%s 기간 pykrx 조회 중 오류: %s",
                    ticker,
                    start_str,
                    end_str,
                    pykrx_error_msg,
                )
                break
            except Exception as e:
                err_text = str(e)
                logger.warning(
                    "%s의 %s~%s 기간 데이터 조회 중 오류: %s",
                    ticker,
                    start_str,
                    end_str,
                    err_text,
                )
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
            logger.error("yfinance 라이브러리가 설치되지 않았습니다. 'pip install yfinance'로 설치해주세요.")
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

            # 멀티 인덱스 형태로 저장된 컬럼을 단일 인덱스로 정리합니다.
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
            error_msg = str(e)
            if "Too Many Requests" in error_msg or "Rate limited" in error_msg or "429" in error_msg:
                logger.error("%s 데이터 조회 Rate Limit 에러: %s", ticker, e)
                raise RateLimitException(ticker, error_msg)
            logger.warning("%s의 데이터 조회 중 오류: %s", ticker, e)
            return None

    if country_code == "us":
        if yf is None:
            logger.error("yfinance 라이브러리가 설치되지 않았습니다. 'pip install yfinance'로 설치해주세요.")
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
            error_msg = str(e)
            if "Too Many Requests" in error_msg or "Rate limited" in error_msg or "429" in error_msg:
                logger.error("%s 데이터 조회 Rate Limit 에러: %s", ticker, e)
                raise RateLimitException(ticker, error_msg)
            logger.warning("%s의 데이터 조회 중 오류: %s", ticker, e)
            return None

    logger.error("지원하지 않는 국가 코드입니다: %s", country_code)
    return None


def fetch_ohlcv_for_tickers(
    tickers: List[str],
    country: str,
    date_range: Optional[List[str]] = None,
    warmup_days: int = 0,
    skip_realtime: bool = False,
) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """
    주어진 티커 목록에 대해 OHLCV 데이터를 직렬로 조회합니다.
    """
    prefetched_data: Dict[str, pd.DataFrame] = {}

    if not date_range or len(date_range) != 2:
        return {}, []

    core_start = pd.to_datetime(date_range[0])
    warmup_start = core_start - pd.DateOffset(days=warmup_days)
    adjusted_date_range = [warmup_start.strftime("%Y-%m-%d"), date_range[1]]

    missing: List[str] = []

    is_kor_market = (country or "").strip().lower() in {"kr", "kor"}
    if is_kor_market and not skip_realtime:
        try:
            prime_naver_etf_realtime_snapshot(tickers)
        except Exception as exc:  # pragma: no cover - 방어 목적
            logger.debug("네이버 실시간 스냅샷 초기화 실패(%s): %s", country, exc)

    for tkr in tickers:
        df = fetch_ohlcv(ticker=tkr, country=country, date_range=adjusted_date_range, skip_realtime=skip_realtime)
        if df is None or df.empty:
            missing.append(tkr)
            continue
        prefetched_data[tkr] = df

    return prefetched_data, missing


def prepare_price_data(
    *,
    tickers: Sequence[str],
    country: str,
    start_date: str,
    end_date: str,
    warmup_days: int = 0,
    skip_realtime: bool = False,
) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """Shared helper to populate cache-backed OHLCV data consistently across workflows."""

    tickers_list = [str(t).strip() for t in tickers if str(t or "").strip()]
    if not tickers_list:
        return {}, []

    date_range = [start_date, end_date]
    prefetched, missing = fetch_ohlcv_for_tickers(
        tickers_list,
        country,
        date_range=date_range,
        warmup_days=warmup_days,
        skip_realtime=skip_realtime,
    )
    return prefetched, missing


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
        error_msg = str(e_yf)
        # Rate limit 에러 감지
        if "Too Many Requests" in error_msg or "Rate limited" in error_msg or "429" in error_msg:
            logger.error("%s의 호주 실시간 가격 조회(yfinance) Rate Limit 에러: %s", ticker, e_yf)
            raise RateLimitException(ticker, error_msg)
        logger.warning("%s의 호주 실시간 가격 조회(yfinance) 실패: %s", ticker, e_yf)
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
        logger.warning("%s의 실시간 가격 조회 중 오류 발생: %s", ticker, e)
    return None


def fetch_naver_etf_inav_snapshot(tickers: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """네이버 API에서 한국 ETF의 실시간 NAV 정보를 조회합니다."""

    normalized_codes = {str(t).strip().upper() for t in tickers if str(t or "").strip()}
    if not normalized_codes:
        return {}

    if not _should_use_realtime_price("kor"):
        return {}

    if not requests:
        logger.debug("requests 라이브러리가 없어 네이버 iNAV 조회를 건너뜁니다.")
        return {}

    from config import NAVER_FINANCE_ETF_API_URL, NAVER_FINANCE_HEADERS

    url = NAVER_FINANCE_ETF_API_URL
    headers = NAVER_FINANCE_HEADERS

    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
    except Exception as exc:
        logger.warning("네이버 ETF iNAV 조회 실패: %s", exc)
        return {}

    try:
        payload = response.json()
    except Exception as exc:
        logger.warning("네이버 ETF iNAV 응답 파싱 실패: %s", exc)
        return {}

    items = payload.get("result", {}).get("etfItemList")
    if not isinstance(items, list):
        return {}

    snapshot: Dict[str, Dict[str, float]] = {}

    for item in items:
        if not isinstance(item, dict):
            continue

        code = str(item.get("itemcode") or "").strip().upper()
        if not code or code not in normalized_codes:
            continue

        nav_raw = item.get("nav")
        price_raw = item.get("nowVal")

        try:
            nav_value = float(str(nav_raw).replace(",", ""))
            price_value = float(str(price_raw).replace(",", ""))
        except (TypeError, ValueError):
            continue

        if nav_value <= 0 or price_value <= 0:
            continue

        deviation = ((price_value / nav_value) - 1.0) * 100.0

        selected_price = price_value if _KOR_PRICE_SOURCE_NORMALIZED == "price" else nav_value

        snapshot[code] = {
            "nav": nav_value,
            "nowVal": price_value,
            "price": selected_price,
            "deviation": deviation,
        }

    return snapshot


_NAVER_ETF_SNAPSHOT_CACHE: Dict[str, Dict[str, float]] = {}
_NAVER_ETF_SNAPSHOT_FETCHED_AT: Optional[pd.Timestamp] = None


def prime_naver_etf_realtime_snapshot(tickers: Sequence[str]) -> None:
    """Fetch and cache real-time NAV/price snapshot for given Korean ETF tickers."""

    global _NAVER_ETF_SNAPSHOT_CACHE, _NAVER_ETF_SNAPSHOT_FETCHED_AT

    try:
        snapshot = fetch_naver_etf_inav_snapshot(tickers)
    except Exception as exc:  # pragma: no cover - 외부 요청 방어
        logger.warning("네이버 ETF 실시간 스냅샷 조회 실패: %s", exc)
        return

    if snapshot:
        _NAVER_ETF_SNAPSHOT_CACHE = snapshot
        _NAVER_ETF_SNAPSHOT_FETCHED_AT = pd.Timestamp.now()
    else:
        _NAVER_ETF_SNAPSHOT_CACHE = {}
        _NAVER_ETF_SNAPSHOT_FETCHED_AT = None


def get_cached_naver_etf_snapshot_entry(ticker: str) -> Optional[Dict[str, float]]:
    """Return cached NAV snapshot entry for the given Korean ETF ticker."""

    key = str(ticker or "").strip().upper()
    if not key:
        return None
    return _NAVER_ETF_SNAPSHOT_CACHE.get(key)


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
        error_msg = str(e)
        if "Too Many Requests" in error_msg or "Rate limited" in error_msg or "429" in error_msg:
            logger.error("%s 이름 조회 Rate Limit 에러: %s", cache_key, e)
            raise RateLimitException(ticker, error_msg)
        logger.warning("%s의 이름 조회 중 오류 발생: %s", cache_key, e)
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
        logger.error("%s 국가의 최근 거래일을 확인하지 못했습니다.", country_code)
        return None

    start_date = latest_trade_day
    end_date = latest_trade_day + pd.Timedelta(days=1)
    date_str_for_log = start_date.strftime("%Y-%m-%d")

    try:
        logger.info(
            "%s - 거래일 %s (범위: %s ~ %s) 비조정 가격 조회",
            yfinance_ticker,
            date_str_for_log,
            start_date.strftime("%Y-%m-%d"),
            end_date.strftime("%Y-%m-%d"),
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
            logger.warning(
                "%s에 대한 %s 날짜 데이터가 반환되지 않았습니다.",
                yfinance_ticker,
                date_str_for_log,
            )
            return None

    except Exception as e:
        error_msg = str(e)
        if "Too Many Requests" in error_msg or "Rate limited" in error_msg or "429" in error_msg:
            logger.error("yfinance Rate Limit 에러: %s (날짜: %s) - %s", yfinance_ticker, date_str_for_log, e)
            raise RateLimitException(yfinance_ticker, error_msg)
        logger.error(
            "yfinance 다운로드 실패: %s (날짜: %s) - %s",
            yfinance_ticker,
            date_str_for_log,
            e,
        )
        return None
