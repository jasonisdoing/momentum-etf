"""추천과 백테스트에서 공통으로 사용하는 매수/매도 시그널 로직."""

import numpy as np
import pandas as pd


def _consecutive_counter(flags: np.ndarray) -> np.ndarray:
    """파이썬 버전의 연속 True 카운터."""

    result = np.zeros(flags.shape[0], dtype=np.int32)
    streak = 0
    for idx in range(flags.shape[0]):
        if flags[idx]:
            streak += 1
        else:
            streak = 0
        result[idx] = streak
    return result


def has_buy_signal(score: float, min_score: float = 0.0) -> bool:
    """점수를 기반으로 매수 시그널 여부를 판단합니다.

    Args:
        score: MAPS 점수 (이동평균 대비 수익률 %)

    Returns:
        True if 매수 시그널 있음 (점수 > 0), False otherwise
    """
    return score > 0


def calculate_consecutive_days(
    scores: pd.Series,
) -> pd.Series:
    """점수 시리즈를 기반으로 매수 시그널 지속일을 계산합니다.

    Args:
        scores: MAPS 점수 시리즈

    Returns:
        매수 시그널이 연속으로 활성화된 일수 시리즈
    """
    if scores is None or scores.empty:
        return pd.Series([], dtype=int)

    # 점수가 양수인 경우 매수 시그널 활성화
    buy_signal_active = (scores > 0).to_numpy(dtype=bool, copy=False)

    streak_values = _consecutive_counter(buy_signal_active)
    return pd.Series(streak_values, index=scores.index, dtype=int)


def get_buy_signal_streak(score: float, score_series: pd.Series | None = None) -> int:
    """현재 점수와 점수 시리즈를 기반으로 매수 시그널 지속일을 반환합니다.

    Args:
        score: 현재 MAPS 점수
        score_series: 전체 점수 시리즈 (있으면 정확한 지속일 계산)

    Returns:
        매수 시그널 지속일 (점수 <= 0이면 0)
    """
    if score <= 0:
        return 0

    if score_series is not None and not score_series.empty:
        consecutive_days = calculate_consecutive_days(score_series)
        return int(consecutive_days.iloc[-1])

    # score_series가 없으면 최소 1일로 반환 (점수가 양수이므로)
    return 1


__all__ = [
    "has_buy_signal",
    "calculate_consecutive_days",
    "get_buy_signal_streak",
]
