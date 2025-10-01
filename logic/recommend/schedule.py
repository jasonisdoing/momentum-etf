"""Schedule and trading-day helpers for signals logic (self-contained)."""
from __future__ import annotations

from datetime import datetime

import pandas as pd

try:  # optional dependency
    import pytz  # type: ignore
except Exception:  # pragma: no cover
    pytz = None

from utils.data_loader import get_trading_days
from logic.recommend.benchmarks import _is_trading_day


def is_market_open(country: str = "kor") -> bool:
    """Return whether specified market is currently open.

    - coin: always True
    - kor/aus: check holiday-aware trading days + local market hours
    """
    if not pytz:
        return False

    if country == "coin":
        return True

    timezones = {"kor": "Asia/Seoul", "aus": "Australia/Sydney"}
    market_hours = {
        "kor": (
            datetime.strptime("09:00", "%H:%M").time(),
            datetime.strptime("15:30", "%H:%M").time(),
        ),
        "aus": (
            datetime.strptime("10:00", "%H:%M").time(),
            datetime.strptime("16:00", "%H:%M").time(),
        ),
    }

    tz_str = timezones.get(country)
    if not tz_str:
        return False

    try:
        local_tz = pytz.timezone(tz_str)
        now_local = datetime.now(local_tz)

        today_str_for_util = now_local.strftime("%Y-%m-%d")
        is_trading_day_today = bool(
            get_trading_days(today_str_for_util, today_str_for_util, country)
        )
        if not is_trading_day_today:
            return False

        market_open_time, market_close_time = market_hours[country]
        return market_open_time <= now_local.time() <= market_close_time
    except Exception:
        return False


def get_next_trading_day(country: str, start_date: pd.Timestamp) -> pd.Timestamp:
    """Return the nearest trading day on/after start_date.

    coin: return start_date as-is.
    """
    if country == "coin":
        return start_date
    try:
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = (start_date + pd.Timedelta(days=14)).strftime("%Y-%m-%d")
        days = get_trading_days(start_str, end_str, country)
        for d in days:
            if d.date() >= start_date.date():
                return pd.Timestamp(d).normalize()
    except Exception:
        pass
    # fallback: if weekend, next Monday; else same day
    wd = start_date.weekday()
    delta = 0 if wd < 5 else (7 - wd)
    return (start_date + pd.Timedelta(days=delta)).normalize()


def determine_target_date_for_scheduler(country: str) -> pd.Timestamp:
    """Decide target calculation date depending on current time.

    - coin: today
    - stocks: until midnight use today; after midnight use next trading day (KST)
    """
    if country == "coin":
        if pytz:
            try:
                seoul_tz = pytz.timezone("Asia/Seoul")
                return pd.Timestamp.now(seoul_tz).normalize()
            except Exception:
                return pd.Timestamp.now().normalize()
        else:
            return pd.Timestamp.now().normalize()

    if not pytz:
        return pd.Timestamp.now().normalize()

    try:
        kst_tz = pytz.timezone("Asia/Seoul")
        now_kst = datetime.now(kst_tz)
    except Exception:
        now_kst = datetime.now()

    today_kst = pd.Timestamp(now_kst).normalize()

    if now_kst.date() > today_kst.date():
        target_date = get_next_trading_day(country, today_kst)
    else:
        if _is_trading_day(country, today_kst.to_pydatetime()):
            target_date = today_kst
        else:
            target_date = get_next_trading_day(country, today_kst)

    return target_date


__all__ = [
    "is_market_open",
    "get_next_trading_day",
    "determine_target_date_for_scheduler",
]
