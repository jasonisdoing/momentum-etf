"""거래일 캘린더 유틸 — `data/country/<국가>/market_calendars.json` 단일 소스.

`utils/data_loader.py` 에서 분리(이동만, 로직 불변). 기존 임포트 경로는
data_loader 가 re-export 로 유지한다.
"""

import functools
import json
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pandas as pd

from config import MARKET_SCHEDULES
from utils.logger import get_app_logger

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore

logger = get_app_logger()


def _now_with_zone(tz_name: str) -> datetime:
    try:
        if ZoneInfo is not None:
            return datetime.now(ZoneInfo(tz_name))
    except Exception:
        pass
    return datetime.now()


def _today_in_korea() -> pd.Timestamp:
    """한국 기준 오늘 날짜를 반환합니다."""
    try:
        if ZoneInfo is not None:
            return pd.Timestamp(datetime.now(ZoneInfo("Asia/Seoul")).date())
    except Exception:
        pass
    return pd.Timestamp.now().normalize()


def _build_market_open_info() -> dict[str, tuple[str, time]]:
    info: dict[str, tuple[str, time]] = {}
    for code, schedule in (MARKET_SCHEDULES or {}).items():
        if not isinstance(schedule, dict):
            continue
        open_time = schedule.get("open") or time(9, 0)
        tz_name = schedule.get("timezone") or "UTC"
        info[code.lower()] = (tz_name, open_time)
    return info


MARKET_OPEN_INFO = _build_market_open_info()


def _is_market_day_completed(country_code: str, trading_day: pd.Timestamp) -> bool:
    """해당 시장의 현지 마감 시간이 지나야 최신 거래일로 인정한다."""
    country_key = (country_code or "").strip().lower()
    schedule = (MARKET_SCHEDULES or {}).get(country_key)
    if not isinstance(schedule, dict):
        return True

    tz_name = str(schedule.get("timezone") or "").strip() or "UTC"
    close_time = schedule.get("close") or time(23, 59)
    close_offset_minutes = int(schedule.get("close_offset_minutes") or 0)

    try:
        now_local = _now_with_zone(tz_name)
    except Exception:
        return True

    trading_day_norm = pd.Timestamp(trading_day).normalize()
    now_local_day = pd.Timestamp(now_local.date()).normalize()
    if trading_day_norm < now_local_day:
        return True
    if trading_day_norm > now_local_day:
        return False

    cutoff_minutes = (close_time.hour * 60) + close_time.minute + close_offset_minutes
    now_minutes = (now_local.hour * 60) + now_local.minute
    return now_minutes >= cutoff_minutes


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


def get_today_str() -> str:
    """오늘 날짜를 'YYYYMMDD' 형식의 문자열로 반환합니다."""
    return datetime.now().strftime("%Y%m%d")


@functools.lru_cache(maxsize=10)
def get_trading_days(start_date: str, end_date: str, country: str) -> list[pd.Timestamp]:
    """
    지정된 기간 내의 모든 거래일을 pd.Timestamp 리스트로 반환합니다.
    국가별 거래일 파일(data/country/{country}/market_calendars.json)만 사용합니다.
    """
    country_code = (country or "").strip().lower()
    if not country_code:
        raise ValueError("거래일 조회 국가 코드가 필요합니다.")

    calendar_path = Path(__file__).resolve().parents[1] / "data" / "country" / country_code / "market_calendars.json"
    if not calendar_path.exists():
        raise FileNotFoundError(f"거래일 캘린더 파일이 없습니다: {calendar_path}")

    try:
        payload = json.loads(calendar_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"거래일 캘린더 파일을 읽을 수 없습니다: {calendar_path}") from exc

    if not isinstance(payload, dict):
        raise RuntimeError(f"거래일 캘린더 파일 형식이 올바르지 않습니다: {calendar_path}")

    raw_start = str(payload.get("start_date") or "").strip()
    raw_end = str(payload.get("end_date") or "").strip()
    raw_days = payload.get("trading_days")
    if not raw_start or not raw_end or not isinstance(raw_days, list):
        raise RuntimeError(f"거래일 캘린더 필수 필드가 없습니다: {calendar_path}")

    start_date_ts = pd.to_datetime(start_date).normalize()
    end_date_ts = pd.to_datetime(end_date).normalize()
    file_start_ts = pd.to_datetime(raw_start).normalize()
    file_end_ts = pd.to_datetime(raw_end).normalize()
    if start_date_ts < file_start_ts or end_date_ts > file_end_ts:
        raise RuntimeError(
            "거래일 캘린더 범위를 벗어났습니다: "
            f"country={country_code}, requested_start={start_date_ts.strftime('%Y-%m-%d')}, "
            f"requested_end={end_date_ts.strftime('%Y-%m-%d')}, "
            f"file_start={file_start_ts.strftime('%Y-%m-%d')}, file_end={file_end_ts.strftime('%Y-%m-%d')}"
        )

    trading_days_ts: list[pd.Timestamp] = []
    for raw_day in raw_days:
        normalized_day = str(raw_day or "").strip()
        if not normalized_day:
            continue
        trading_days_ts.append(pd.to_datetime(normalized_day).normalize())

    # 최종적으로 start_date와 end_date 사이의 날짜만 반환하고, 중복 제거 및 정렬합니다.
    final_list = [d for d in trading_days_ts if start_date_ts <= d <= end_date_ts]

    return sorted(list(set(final_list)))


def get_trading_days_any(start_date: str, end_date: str, countries: list[str]) -> list[pd.Timestamp]:
    """여러 국가 중 하나라도 개장한 날짜를 거래일로 반환한다."""
    if not countries:
        raise ValueError("거래일 조회 국가 코드 목록이 필요합니다.")

    merged: set[pd.Timestamp] = set()
    for country in countries:
        merged.update(get_trading_days(start_date, end_date, country))
    return sorted(merged)


# 자산 집계의 기준일로 쓰는 국가 조합 — 한국 또는 호주 중 하나라도 열린 날.
ASSET_TRADING_COUNTRIES = ["kor", "au"]

# 미국 장은 한국시간으로 **다음 날 새벽**에 끝나므로 거래일을 하루 밀어서 합친다.
# 이 합집합이 없으면 한국·호주가 둘 다 쉬는 평일(설 연휴 임시공휴일 등)에 행이 안 생겨,
# 그날 미국 몫이 직전 거래일 행에 얹혀 그 행의 손익이 실제보다 커진다.
ASSET_TRADING_SHIFTED_COUNTRY = "us"


def _shift_to_weekday(day: date) -> date:
    """주말이면 다음 평일로 민다. 토·일에는 행을 만들지 않는다."""
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return day


def resolve_active_trading_date() -> str:
    """오늘 이하의 마지막 자산 거래일(KST, YYYY-MM-DD).

    일별 집계(`daily_fund_data`)와 자산 스냅샷(`daily_snapshots`)이 **같은 날짜 기준**을
    쓰도록 하는 단일 소스다. 예전에는 집계만 거래일, 스냅샷은 달력 날짜를 써서
    토요일에 스냅샷만 새 행이 생겼고, 그 탓에 `/assets` 의 금일 손익 합계와
    계좌별 값이 서로 다른 구간을 비교했다.

    기준일 후보는 **한국 ∪ 호주 ∪ (미국 거래일 + 1일, 주말이면 다음 평일)** 이다.
    주말·휴일에도 직전 거래일을 돌려주므로 그 날짜의 행이 계속 갱신된다 —
    미국·호주 장이 한국시간 새벽까지 이어지는 몫을 그 거래일에 담기 위한 것이다.
    """
    today = _now_with_zone("Asia/Seoul").date()
    search_start = today - timedelta(days=370)
    candidates = {
        day.date() for day in get_trading_days_any(str(search_start), str(today), ASSET_TRADING_COUNTRIES)
    }
    # 미국은 오늘 이후로 밀릴 수 있어 조회 구간을 하루 넉넉히 잡고, 밀린 뒤 오늘 이하만 남긴다.
    for day in get_trading_days(str(search_start), str(today), ASSET_TRADING_SHIFTED_COUNTRY):
        shifted = _shift_to_weekday(day.date() + timedelta(days=1))
        if shifted <= today:
            candidates.add(shifted)
    if not candidates:
        raise RuntimeError("오늘 이하의 자산 거래일을 찾지 못했습니다.")
    return max(candidates).isoformat()


def is_trading_day(
    country: str,
    date: str | datetime | pd.Timestamp | None = None,
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
    start_date: str | datetime | pd.Timestamp,
    end_date: str | datetime | pd.Timestamp,
) -> int:
    """Return number of trading days between two dates (inclusive)."""

    start_ts = pd.to_datetime(start_date).normalize()
    end_ts = pd.to_datetime(end_date).normalize()

    if start_ts > end_ts:
        return 0

    country_code = (country or "").strip().lower()

    # 캐시를 위해 문자열로 변환하여 내부 함수 호출
    return _count_trading_days_cached(country_code, start_ts.strftime("%Y-%m-%d"), end_ts.strftime("%Y-%m-%d"))


@functools.lru_cache(maxsize=500)
def _count_trading_days_cached(country_code: str, start_str: str, end_str: str) -> int:
    """캐시된 거래일 수 계산 (내부 함수)"""
    days = get_trading_days(start_str, end_str, country_code)
    return len(days)


@functools.lru_cache(maxsize=50)
def _get_latest_trading_day_cached(country: str, cache_key: str) -> pd.Timestamp:
    """
    내부 캐시 함수: 날짜/시간 기반 캐시 키를 사용하여 최신 거래일을 반환합니다.

    Args:
        country: 국가 코드
        cache_key: 캐시 무효화용 키 (날짜_시간 형식)
    """
    country_code = (country or "").strip().lower()

    # 최신 거래일 판단의 기준 날짜는 모든 시장 공통으로 한국 날짜를 사용합니다.
    # 이렇게 해야 AU 시장처럼 현지 날짜가 먼저 넘어가더라도 "내일 거래일"이 잡히지 않습니다.
    end_dt = _today_in_korea()

    # 최근 10일간의 거래일을 한 번에 조회 (효율성 개선)
    start_date = (end_dt - pd.DateOffset(days=10)).strftime("%Y-%m-%d")
    end_date = end_dt.strftime("%Y-%m-%d")

    try:
        # 최근 10일간의 모든 거래일을 한 번에 가져옴
        trading_days = get_trading_days(start_date, end_date, country_code)

        if trading_days:
            normalized_days = sorted(pd.Timestamp(day).normalize() for day in trading_days)
            latest_trading_day = normalized_days[-1]
            if _is_market_day_completed(country_code, latest_trading_day):
                return latest_trading_day
            if len(normalized_days) >= 2:
                return normalized_days[-2]
            return latest_trading_day
    except Exception as e:
        logger.warning("거래일 일괄 조회 중 오류 발생: %s", e)

    # 폴백: 10일간 거래일을 찾지 못하면 오늘 날짜를 정규화하여 반환합니다.
    logger.warning(
        "최근 10일 내에 거래일을 찾지 못했습니다. 오늘 날짜(%s)를 사용합니다.",
        end_dt.strftime("%Y-%m-%d"),
    )
    return end_dt.normalize()


def get_latest_trading_day(country: str) -> pd.Timestamp:
    """
    오늘 또는 가장 가까운 과거의 '데이터가 있을 것으로 예상되는' 거래일을 pd.Timestamp 형식으로 반환합니다.

    시간 기반 캐시를 사용하여 날짜가 바뀌거나 시간이 지나면 자동으로 캐시가 무효화됩니다.
    """
    # 현재 날짜와 시간(시 단위)을 캐시 키로 사용
    # 이렇게 하면 매 시간마다 캐시가 갱신되어 장 시작 전/후 데이터 차이를 반영
    now = pd.Timestamp.now()
    cache_key = f"{now.strftime('%Y-%m-%d')}_{now.hour}"
    return _get_latest_trading_day_cached(country, cache_key)


def get_next_trading_day(
    country: str,
    reference_date: pd.Timestamp | None = None,
    *,
    search_horizon_days: int,
) -> pd.Timestamp | None:
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
