"""
이동평균 계산 유틸리티.

시스템의 모든 전략 이동평균(추세선·이격도·순위·시장추세·이동평균선 크로스 등)은
config.MOVING_AVERAGE_TYPE("SMA" 또는 "EMA") 하나로 종류를 전환한다.
"""

import pandas as pd


def get_moving_average_type() -> str:
    """설정된 이동평균 종류("SMA"/"EMA")를 반환한다(silent default 없음 — 잘못된 값은 에러)."""
    from config import MOVING_AVERAGE_TYPE

    value = str(MOVING_AVERAGE_TYPE or "").strip().upper()
    if value not in {"SMA", "EMA"}:
        raise ValueError(f"MOVING_AVERAGE_TYPE 는 'SMA' 또는 'EMA' 여야 합니다: {MOVING_AVERAGE_TYPE!r}")
    return value


def calculate_sma(prices: pd.Series, period: int, min_periods: int = 1) -> pd.Series:
    """단순 이동평균(SMA)."""
    return prices.rolling(window=period, min_periods=min_periods).mean()


def calculate_ema(prices: pd.Series, period: int, min_periods: int = 1) -> pd.Series:
    """지수 이동평균(EMA). span=period(=SMA와 같은 '기간'), 초기 편향 최소화(adjust=False)."""
    return prices.ewm(span=period, adjust=False, min_periods=min_periods).mean()


def calculate_moving_average(prices: pd.Series, period: int, min_periods: int = 1) -> pd.Series:
    """설정(config.MOVING_AVERAGE_TYPE)에 따라 SMA 또는 EMA 이동평균을 계산한다.

    Args:
        prices: 가격 시리즈
        period: 이동평균 기간
        min_periods: 최소 표본 수(기본 1 = 첫 봉부터 부분평균). 기존 rolling 기본과 맞추려면 period 를 넘긴다.
    """
    if get_moving_average_type() == "EMA":
        return calculate_ema(prices, period, min_periods=min_periods)
    return calculate_sma(prices, period, min_periods=min_periods)


__all__ = [
    "calculate_ema",
    "calculate_moving_average",
    "calculate_sma",
    "get_moving_average_type",
]
