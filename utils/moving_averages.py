"""
이동평균 계산 유틸리티 함수
- 전략 추세선은 SMA만 사용
"""

import pandas as pd


def calculate_sma(prices: pd.Series, period: int) -> pd.Series:
    """
    Simple Moving Average (단순 이동평균)

    Args:
        prices: 가격 시리즈
        period: 이동평균 기간

    Returns:
        SMA 시리즈
    """
    return prices.rolling(window=period, min_periods=1).mean()


def calculate_moving_average(
    prices: pd.Series,
    period: int,
) -> pd.Series:
    """
    SMA 이동평균을 계산합니다.

    Args:
        prices: 가격 시리즈
        period: 이동평균 기간
    Returns:
        계산된 이동평균 시리즈

    Examples:
        >>> prices = pd.Series([100, 102, 104, 103, 105])
        >>> calculate_moving_average(prices, 3)
    """
    return calculate_sma(prices, period)


__all__ = [
    "calculate_moving_average",
    "calculate_sma",
]
