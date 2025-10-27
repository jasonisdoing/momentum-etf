"""Market schedule utilities."""

from datetime import datetime, time, timedelta, date
from config import MARKET_SCHEDULES


def get_market_open_time(country_code: str) -> time:
    """국가별 시장 시작 시간 반환"""
    schedule = MARKET_SCHEDULES.get(country_code.lower())
    if not schedule:
        raise ValueError(f"Unknown country: {country_code}")
    return schedule["open"]


def get_market_close_time(country_code: str) -> time:
    """국가별 시장 종료 시간 반환"""
    schedule = MARKET_SCHEDULES.get(country_code.lower())
    if not schedule:
        raise ValueError(f"Unknown country: {country_code}")
    return schedule["close"]


def get_market_cron(country_code: str) -> str:
    """국가별 시장 시간 기반 크론 표현식 생성

    Returns:
        크론 표현식 (예: "1,11,21,31,41,51 9-15 * * 1-5")
    """
    expressions = generate_market_cron_expressions(country_code)
    if not expressions:
        raise ValueError(f"Unable to derive cron expressions for country: {country_code}")
    return expressions[0]


def generate_market_cron_expressions(country_code: str) -> tuple[str, ...]:
    schedule = MARKET_SCHEDULES.get(country_code.lower())
    if not schedule:
        raise ValueError(f"Unknown country: {country_code}")

    interval = int(schedule.get("interval_minutes", 10) or 10)
    if interval <= 0:
        raise ValueError("interval_minutes must be positive")

    start_time = schedule["open"]
    end_time = schedule["close"]

    start_dt = datetime.combine(date(2000, 1, 1), start_time)
    end_dt = datetime.combine(date(2000, 1, 1), end_time)
    if end_dt < start_dt:
        raise ValueError("Market close time must be after open time")

    slots: dict[int, list[int]] = {}
    current = start_dt
    while current <= end_dt:
        slots.setdefault(current.hour, []).append(current.minute)
        current += timedelta(minutes=interval)

    expressions: list[str] = []
    for hour in sorted(slots.keys()):
        minutes = ",".join(str(minute) for minute in sorted(slots[hour]))
        expressions.append(f"{minutes} {hour} * * 1-5")

    return tuple(expressions)
    schedule = MARKET_SCHEDULES.get(country_code.lower())
    if not schedule:
        raise ValueError(f"Unknown country: {country_code}")

    open_hour = schedule["open"].hour
    close_hour = schedule["close"].hour

    # 매 10분마다: 1, 11, 21, 31, 41, 51분
    minutes = "1,11,21,31,41,51"
    hours = f"{open_hour}-{close_hour}"

    return f"{minutes} {hours} * * 1-5"
