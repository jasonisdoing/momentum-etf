"""MAPS 전략 점수 계산 및 정규화 함수."""

from __future__ import annotations

import pandas as pd
import numpy as np


def normalize_ma_score(
    scores: pd.Series | float,
    eligibility_threshold: float = 0.0,
    max_bound: float = 30.0,
) -> pd.Series | float:
    """
    MA 점수를 0~100 스케일로 정규화합니다.

    Args:
        scores: 원본 MA 점수 (이동평균 대비 수익률 %)
        eligibility_threshold: 투자 적격 기준점 (기본값: 0.0 = MA와 동일)
        max_bound: 최대 점수 경계 (기본값: 30.0 = MA 대비 +30%)

    Returns:
        0~100 스케일로 정규화된 점수
        - eligibility_threshold 미만: 0점 (투자 부적격)
        - eligibility_threshold ~ max_bound: 0~100점 선형 변환
        - max_bound 이상: 100점

    Examples:
        >>> normalize_ma_score(0.0)   # MA와 동일
        0.0
        >>> normalize_ma_score(15.0)  # MA 대비 +15%
        50.0
        >>> normalize_ma_score(30.0)  # MA 대비 +30%
        100.0
        >>> normalize_ma_score(-5.0)  # MA 아래
        0.0
    """
    is_series = isinstance(scores, pd.Series)

    if is_series:
        # Series 처리
        result = scores.copy()

        # eligibility_threshold 미만은 0점
        result[result < eligibility_threshold] = 0.0

        # max_bound 이상은 100점
        result[result >= max_bound] = 100.0

        # 중간 범위는 선형 변환
        mask = (result >= eligibility_threshold) & (result < max_bound)
        if mask.any():
            result[mask] = ((result[mask] - eligibility_threshold) / (max_bound - eligibility_threshold)) * 100

        return result
    else:
        # 단일 값 처리
        if scores < eligibility_threshold:
            return 0.0
        elif scores >= max_bound:
            return 100.0
        else:
            return ((scores - eligibility_threshold) / (max_bound - eligibility_threshold)) * 100


def normalize_ma_score_with_config(
    scores: pd.Series | float,
    config: dict | None = None,
) -> pd.Series | float:
    """
    설정 딕셔너리를 사용하여 MA 점수를 정규화합니다.

    Args:
        scores: 원본 MA 점수
        config: 정규화 설정 딕셔너리
            - enabled: 정규화 활성화 여부 (기본값: True)
            - eligibility_threshold: 투자 적격 기준점 (기본값: 0.0)
            - max_bound: 최대 점수 경계 (기본값: 30.0)

    Returns:
        정규화된 점수 (enabled=False면 원본 반환)
    """
    if config is None:
        config = {}

    # 정규화 비활성화 시 원본 반환
    if not config.get("enabled", True):
        return scores

    eligibility_threshold = float(config.get("eligibility_threshold", 0.0))
    max_bound = float(config.get("max_bound", 30.0))

    return normalize_ma_score(scores, eligibility_threshold, max_bound)


def calculate_maps_score(
    close_prices: pd.Series,
    moving_average: pd.Series,
    normalize: bool = False,
    normalize_config: dict | None = None,
) -> pd.Series:
    """
    MAPS(Moving Average Position Score) 점수를 계산합니다.

    Args:
        close_prices: 종가 시리즈
        moving_average: 이동평균 시리즈
        normalize: 0~100 스케일로 정규화 여부 (기본값: False)
        normalize_config: 정규화 설정 (normalize=True일 때만 사용)
            - eligibility_threshold: 투자 적격 기준점 (기본값: 0.0)
            - max_bound: 최대 점수 경계 (기본값: 30.0)

    Returns:
        pd.Series: 이동평균 대비 수익률 (%) 또는 정규화된 점수 (0~100)

    Examples:
        >>> close = pd.Series([110, 115, 120])
        >>> ma = pd.Series([100, 100, 100])
        >>> calculate_maps_score(close, ma)
        0    10.0
        1    15.0
        2    20.0
        dtype: float64
    """
    # 0으로 나누기 방지
    safe_moving_average = moving_average.replace(0, np.nan)
    ma_score = ((close_prices / safe_moving_average) - 1.0) * 100
    # 무한대 값 처리
    ma_score = ma_score.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 정규화 적용
    if normalize:
        config = normalize_config or {"enabled": True, "eligibility_threshold": 0.0, "max_bound": 30.0}
        ma_score = normalize_ma_score_with_config(ma_score, config)

    return ma_score


__all__ = [
    "calculate_maps_score",
    "normalize_ma_score",
    "normalize_ma_score_with_config",
]
