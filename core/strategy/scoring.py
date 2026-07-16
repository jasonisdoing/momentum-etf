"""RANK 전략 점수 계산 및 정규화 함수.

랭킹(utils/rankings.py)과 백테스트(backtest/engine.py)는 **반드시** 이 모듈의
공통 엔진 함수를 통해서만 점수를 계산해야 한다. 점수식이 양쪽으로 갈라지면 백테스트 결과는
의미가 없어진다.

설정 상수(MIN_TRADING_DAYS, TRADING_DAYS_PER_MONTH)도 이 모듈에서 단일 진입점으로 참조한다.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from config import MIN_TRADING_DAYS, TRADING_DAYS_PER_MONTH
from utils.moving_averages import calculate_moving_average


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


# --------------------------- 공통 랭킹 엔진 --------------------------- #


def _ma_days_from_months(ma_months: int) -> int:
    return int(ma_months) * int(TRADING_DAYS_PER_MONTH)


def compute_trend_frame(
    close_frame: pd.DataFrame,
    ma_type: str,
    ma_months: int,
) -> pd.DataFrame:
    """[일자 × 티커] 구조의 종가 프레임으로부터 MA 대비 트렌드(%) 프레임을 계산한다.

    각 MA 함수가 ``min_periods=1`` 로 부분 계산을 지원하므로, MA 성숙 기간
    (``ma_months * TRADING_DAYS_PER_MONTH``) 을 만족하지 못하더라도 가용한 종가로 MA 를
    계산해 트렌드 값을 반환한다. 랭킹 포함 여부는 ``compute_eligibility_mask`` 가
    ``MIN_TRADING_DAYS`` 기준으로 따로 판정한다.
    """
    days = _ma_days_from_months(ma_months)
    ma_cols: dict[str, pd.Series] = {}
    for ticker in close_frame.columns:
        series = close_frame[ticker].dropna()
        if series.empty:
            ma_cols[ticker] = pd.Series(np.nan, index=close_frame.index, dtype=float)
            continue
        ma_series = calculate_moving_average(series, days, ma_type)
        ma_cols[ticker] = ma_series.reindex(close_frame.index)
    ma_frame = pd.DataFrame(ma_cols, index=close_frame.index)
    trend = pd.DataFrame(
        {
            ticker: calculate_maps_score(close_frame[ticker], ma_frame[ticker])
            for ticker in close_frame.columns
        },
        index=close_frame.index,
    )
    return trend


def compute_rule_percentile_frame(
    close_frame: pd.DataFrame,
    ma_type: str,
    ma_months: int,
) -> pd.DataFrame:
    """단일 MA 규칙에 대한 signed-percentile 점수 프레임을 계산한다."""
    trend = compute_trend_frame(close_frame, ma_type, ma_months)
    return calculate_signed_percentile_score(trend)


def compute_eligibility_mask(close_frame: pd.DataFrame) -> pd.DataFrame:
    """각 일자·티커가 ``MIN_TRADING_DAYS`` 이상 종가 데이터를 누적했는지 여부."""
    return close_frame.notna().cumsum() >= int(MIN_TRADING_DAYS)


def combine_rule_percentiles(
    per_rule_frames: Iterable[pd.DataFrame],
    eligibility_mask: pd.DataFrame,
) -> pd.DataFrame:
    """여러 규칙의 percentile 프레임을 합산하고 자격 마스크를 적용한다.

    - 한 규칙이라도 NaN 이면 합산 결과도 NaN (pandas 기본 동작).
    - ``eligibility_mask`` 가 False 인 셀은 NaN 으로 만든다 (랭킹 제외).
    """
    frames = list(per_rule_frames)
    if not frames:
        return pd.DataFrame(
            index=eligibility_mask.index,
            columns=eligibility_mask.columns,
            dtype=float,
        )
    composite = frames[0].copy()
    for pf in frames[1:]:
        composite = composite + pf
    return composite.where(eligibility_mask)


def build_composite_rank_scores(
    close_frame: pd.DataFrame,
    ma_rules: list[dict[str, Any]],
) -> tuple[pd.DataFrame, dict[int, pd.DataFrame], dict[int, pd.DataFrame]]:
    """랭킹/백테스트 공통: MA 규칙들을 받아 ``(composite, trend_by_order, percentile_by_order)`` 반환.

    - ``composite`` : [일자 × 티커] 최종 점수 (자격 마스크 적용).
    - ``trend_by_order`` : ``{order: trend_frame}`` — 원천 추세(%) (화면 표시/보관용).
    - ``percentile_by_order`` : ``{order: percentile_frame}`` — 규칙별 signed-percentile.
    """
    trend_by_order: dict[int, pd.DataFrame] = {}
    percentile_by_order: dict[int, pd.DataFrame] = {}
    for rule in ma_rules:
        order = int(rule["order"])
        trend = compute_trend_frame(close_frame, str(rule["ma_type"]), int(rule["ma_months"]))
        trend_by_order[order] = trend
        percentile_by_order[order] = calculate_signed_percentile_score(trend)

    eligibility = compute_eligibility_mask(close_frame)
    composite = combine_rule_percentiles(
        [percentile_by_order[int(r["order"])] for r in ma_rules],
        eligibility,
    )
    return composite, trend_by_order, percentile_by_order


__all__ = [
    "calculate_maps_score",
    "calculate_signed_percentile_score",
    "compute_trend_frame",
    "compute_rule_percentile_frame",
    "compute_eligibility_mask",
    "combine_rule_percentiles",
    "build_composite_rank_scores",
]
