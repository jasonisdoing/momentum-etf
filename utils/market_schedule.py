"""Market schedule utilities."""

from datetime import date, datetime, time, timedelta

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

    open_time = schedule["open"]
    close_time = schedule["close"]
    open_offset = int(schedule.get("open_offset_minutes", 0))
    close_offset = int(schedule.get("close_offset_minutes", 0))

    start_dt = datetime.combine(date(2000, 1, 1), open_time)
    end_dt = datetime.combine(date(2000, 1, 1), close_time)
    if end_dt < start_dt:
        raise ValueError("Market close time must be after open time")

    # [User Request] 장 개시 N분 후와 장 종료 M분 전 한번씩 실행
    # 1. Open + Offset
    time1 = start_dt + timedelta(minutes=open_offset)
    # 2. Close - Offset
    time2 = end_dt - timedelta(minutes=close_offset)

    # 중복 제거 및 시간 순서 정렬
    target_times = sorted({time1, time2})

    expressions: list[str] = []
    for t in target_times:
        expressions.append(f"{t.minute} {t.hour} * * 1-5")

    return tuple(expressions)
