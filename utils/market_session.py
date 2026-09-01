"""시장 세션 상태 — 지금 프리장인지 정규장인지 애프터인지, 다음 전환은 언제인지.

시간표는 `config.MARKET_SCHEDULES` 가 단일 소스다. 세션 경계를 화면·서비스마다 따로
적어 두면(예전에 `live_24h_service` 안에 08:00·20:00 이 박혀 있었다) 시간표를 고칠 때
한쪽만 바뀐다.

정규장 밖 거래가 없는 시장(호주)은 `premarket_open`·`aftermarket_close` 키가 없고,
그런 시장은 정규장 아니면 곧바로 '마감'이다 — 없는 세션을 만들어 보여주지 않는다.
미국만 데이장(오버나이트)이 있어 자정을 넘기며, 그 판정은 정규 세션과 따로 한다.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from config import MARKET_SCHEDULES

# 세션 코드 — 화면 문구는 프론트가 정한다(여기서는 상태만 준다).
PREMARKET = "premarket"
REGULAR = "regular"
AFTERMARKET = "aftermarket"
DAYMARKET = "daymarket"
CLOSED = "closed"


def _schedule(country: str) -> dict[str, Any]:
    schedule = (MARKET_SCHEDULES or {}).get(str(country or "").strip().lower())
    if not isinstance(schedule, dict):
        raise ValueError(f"시장 시간표가 없습니다: {country}")
    return schedule


def _at(zone: ZoneInfo, day: date, at: time) -> datetime:
    return datetime.combine(day, at, tzinfo=zone)


def _next_trading_day(country: str, after: date) -> date:
    """`after` **다음**의 거래일. 달력을 못 읽으면 다음 평일로 둔다."""
    try:
        from utils.trading_calendar import get_trading_days

        days = get_trading_days(
            (after + timedelta(days=1)).strftime("%Y-%m-%d"),
            (after + timedelta(days=14)).strftime("%Y-%m-%d"),
            country,
        )
        for day in days:
            if day.date() > after:
                return day.date()
    except Exception:
        pass
    day = after + timedelta(days=1)
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return day


def _is_trading_day(country: str, day: date) -> bool:
    """그 날짜가 거래일인지. 달력을 못 읽으면 평일 여부로 본다."""
    try:
        from utils.trading_calendar import get_trading_days

        stamp = day.strftime("%Y-%m-%d")
        return any(item.date() == day for item in get_trading_days(stamp, stamp, country))
    except Exception:
        return day.weekday() < 5


def market_session(country: str, now: datetime | None = None) -> dict[str, Any]:
    """그 시장의 지금 세션과 **다음 전환 시각**.

    반환: ``{country, name, session, next_change_at}``
      - ``session`` — premarket / regular / aftermarket / daymarket / closed
      - ``next_change_at`` — 다음 상태로 넘어가는 시각(ISO, 타임존 포함).
        프리장이면 정규장 개장, 정규장이면 마감, 애프터면 애프터 종료, 데이장이면 데이장
        종료, 마감이면 **다음 거래일의 정규장 개장**이다(화면이 남은 시간을 여기서 센다).
    """
    schedule = _schedule(country)
    zone = ZoneInfo(str(schedule["timezone"]))
    now_local = (now or datetime.now(zone)).astimezone(zone)
    today = now_local.date()

    open_at = _at(zone, today, schedule["open"])
    close_at = _at(zone, today, schedule["close"])
    pre_at = _at(zone, today, schedule["premarket_open"]) if schedule.get("premarket_open") else None
    after_at = _at(zone, today, schedule["aftermarket_close"]) if schedule.get("aftermarket_close") else None

    session, next_change = CLOSED, None
    if _is_trading_day(country, today):
        if pre_at is not None and pre_at <= now_local < open_at:
            session, next_change = PREMARKET, open_at
        elif open_at <= now_local < close_at:
            session, next_change = REGULAR, close_at
        elif after_at is not None and close_at <= now_local < after_at:
            session, next_change = AFTERMARKET, after_at

    # 데이장(오버나이트) — 자정을 넘기므로 정규 세션과 따로 본다.
    # 저녁에 시작하는 쪽은 **다음 달력일이 거래일**이어야 한다(금요일 밤은 열지 않는다).
    if session == CLOSED and schedule.get("daymarket_open") and schedule.get("daymarket_close"):
        day_close_at = _at(zone, today, schedule["daymarket_close"])
        day_open_at = _at(zone, today, schedule["daymarket_open"])
        if now_local < day_close_at and _is_trading_day(country, today):
            session, next_change = DAYMARKET, day_close_at
        elif now_local >= day_open_at and _is_trading_day(country, today + timedelta(days=1)):
            session, next_change = DAYMARKET, _at(zone, today + timedelta(days=1), schedule["daymarket_close"])

    if next_change is None:
        # 마감 — 다음 개장까지. 오늘 개장 전이면 오늘, 아니면 다음 거래일이다.
        if _is_trading_day(country, today) and now_local < open_at:
            next_change = open_at
        else:
            next_change = _at(zone, _next_trading_day(country, today), schedule["open"])

    return {
        "country": str(country).strip().lower(),
        "name": str(schedule.get("name") or country),
        "session": session,
        "next_change_at": next_change.isoformat(),
    }


def market_sessions() -> list[dict[str, Any]]:
    """시간표에 있는 모든 시장의 세션 상태 — 화면 상단이 그대로 그린다."""
    return [market_session(country) for country in MARKET_SCHEDULES]
