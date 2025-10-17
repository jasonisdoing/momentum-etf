"""
기술적 지표 계산 함수 모음
- 이동평균 기반 추천 계산을 위한 공통 함수들
"""

import pandas as pd
import numpy as np


def calculate_moving_average_signals(close_prices: pd.Series, moving_average_period: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    이동평균 기반 추천을 계산합니다.

    Args:
        close_prices: 종가 시리즈
        moving_average_period: 이동평균 기간

    Returns:
        tuple: (moving_average, buy_signal_active, consecutive_buy_days)
            - moving_average: 이동평균선
            - buy_signal_active: 매수 추천 활성화 여부 (close > ma)
            - consecutive_buy_days: 매수 추천이 연속으로 활성화된 일수
    """
    if close_prices is None:
        return pd.Series(dtype=float), pd.Series(dtype=bool), pd.Series(dtype=int)

    if len(close_prices) < moving_average_period:
        return pd.Series(dtype=float), pd.Series(dtype=bool), pd.Series(dtype=int)

    # MultiIndex 컬럼 처리 (성능 최적화)
    if isinstance(close_prices, pd.DataFrame):
        close_prices = close_prices.iloc[:, 0]

    # 이동평균 계산
    moving_average = close_prices.rolling(window=moving_average_period).mean()

    # 매수 추천 활성화 여부 (종가 > 이동평균)
    buy_signal_active = close_prices > moving_average

    # 매수 추천이 연속으로 활성화된 일수 계산
    consecutive_buy_days = buy_signal_active.groupby((buy_signal_active != buy_signal_active.shift()).cumsum()).cumsum().fillna(0).astype(int)

    return moving_average, buy_signal_active, consecutive_buy_days


def calculate_ma_score(
    close_prices: pd.Series,
    moving_average: pd.Series,
    normalize: bool = False,
    normalize_config: dict | None = None,
) -> pd.Series:
    """
    이동평균 대비 수익률 점수를 계산합니다.

    Args:
        close_prices: 종가 시리즈
        moving_average: 이동평균 시리즈
        normalize: 0~100 스케일로 정규화 여부 (기본값: False)
        normalize_config: 정규화 설정 (normalize=True일 때만 사용)
            - eligibility_threshold: 투자 적격 기준점 (기본값: 0.0)
            - max_bound: 최대 점수 경계 (기본값: 30.0)

    Returns:
        pd.Series: 이동평균 대비 수익률 (%) 또는 정규화된 점수 (0~100)
    """
    # 0으로 나누기 방지를 위해 0을 NaN으로 변경
    safe_moving_average = moving_average.replace(0, np.nan)
    ma_score = ((close_prices / safe_moving_average) - 1.0) * 100
    # 무한대 값을 0으로 변경
    ma_score = ma_score.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 정규화 적용
    if normalize:
        from strategies.maps.scoring import normalize_ma_score_with_config

        config = normalize_config or {"enabled": True, "eligibility_threshold": 0.0, "max_bound": 30.0}
        ma_score = normalize_ma_score_with_config(ma_score, config)

    return ma_score
