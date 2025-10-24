"""Market schedule utilities."""

from datetime import time
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
    schedule = MARKET_SCHEDULES.get(country_code.lower())
    if not schedule:
        raise ValueError(f"Unknown country: {country_code}")

    open_hour = schedule["open"].hour
    close_hour = schedule["close"].hour

    # 매 10분마다: 1, 11, 21, 31, 41, 51분
    minutes = "1,11,21,31,41,51"
    hours = f"{open_hour}-{close_hour}"

    return f"{minutes} {hours} * * 1-5"
