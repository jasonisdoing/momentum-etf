"""RANK 전략 점수 계산 및 정규화 함수."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_maps_score(
    close_prices: pd.Series,
    moving_average: pd.Series,
) -> pd.Series:
    """
    RANK(Moving Average Position Score) 점수를 계산합니다.

    Args:
        close_prices: 종가 시리즈
        moving_average: 이동평균 시리즈

    Returns:
        pd.Series: 이동평균 대비 수익률 (%)

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

    return ma_score


def calculate_signed_percentile_score(data: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Signed percentile 점수를 계산합니다.

    - 양수 값들은 양수 그룹 내에서 백분위 랭킹 → ``[0, 100]``
    - 음수 값들은 절댓값 기준 음수 그룹 내 백분위 랭킹 → ``[-100, 0]``
    - 0 은 그대로 ``0.0``, ``NaN`` 은 유지한다.

    Series 가 입력되면 전체 값들을 한 세트로 보고 랭킹한다.
    DataFrame 이 입력되면 **각 행(일자)** 내부에서 열(티커)들끼리 랭킹한다
    (``axis=1``). 즉 cross-section 점수 계산에 바로 사용할 수 있다.

    랭킹/백테스트 점수 식이 갈라지지 않도록 **반드시 이 함수 하나만** 사용한다.

    Examples:
        >>> s = pd.Series([3, -1, 2, 0])
        >>> calculate_signed_percentile_score(s).tolist()
        [100.0, -100.0, 50.0, 0.0]
    """
    if isinstance(data, pd.Series):
        numeric = pd.to_numeric(data, errors="coerce")
        pos_rank = numeric.where(numeric > 0).rank(method="average", pct=True) * 100.0
        neg_rank = numeric.where(numeric < 0).abs().rank(method="average", pct=True) * -100.0
        result = pos_rank.combine_first(neg_rank)
        result = result.mask(numeric == 0, 0.0)
        return result

    if isinstance(data, pd.DataFrame):
        numeric = data.apply(pd.to_numeric, errors="coerce")
        pos_rank = numeric.where(numeric > 0).rank(axis=1, method="average", pct=True) * 100.0
        neg_rank = numeric.where(numeric < 0).abs().rank(axis=1, method="average", pct=True) * -100.0
        result = pos_rank.combine_first(neg_rank)
        result = result.mask(numeric == 0, 0.0)
        return result

    raise TypeError(f"지원하지 않는 타입입니다: {type(data)!r}. Series 또는 DataFrame 을 전달하세요.")


__all__ = [
    "calculate_maps_score",
    "calculate_signed_percentile_score",
]
