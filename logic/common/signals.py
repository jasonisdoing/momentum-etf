"""추천과 백테스트에서 공통으로 사용하는 매수/매도 시그널 로직."""

from typing import Optional
import pandas as pd


def has_buy_signal(score: float, min_score: float = 0.0) -> bool:
    """점수를 기반으로 매수 시그널 여부를 판단합니다.

    Args:
        score: MAPS 점수 (이동평균 대비 수익률 %)

    Returns:
        True if 매수 시그널 있음 (점수 > 0), False otherwise
    """
    return score > min_score


def calculate_consecutive_days(
    scores: pd.Series,
    min_score: float = 0.0,
) -> pd.Series:
    """점수 시리즈를 기반으로 매수 시그널 지속일을 계산합니다.

    Args:
        scores: MAPS 점수 시리즈

    Returns:
        매수 시그널이 연속으로 활성화된 일수 시리즈
    """
    # 점수가 양수인 경우 매수 시그널 활성화
    buy_signal_active = scores > min_score

    # 매수 시그널이 연속으로 활성화된 일수 계산
    consecutive_days = buy_signal_active.groupby((buy_signal_active != buy_signal_active.shift()).cumsum()).cumsum().fillna(0).astype(int)

    return consecutive_days


def get_buy_signal_streak(score: float, score_series: Optional[pd.Series] = None, min_score: float = 0.0) -> int:
    """현재 점수와 점수 시리즈를 기반으로 매수 시그널 지속일을 반환합니다.

    Args:
        score: 현재 MAPS 점수
        score_series: 전체 점수 시리즈 (있으면 정확한 지속일 계산)

    Returns:
        매수 시그널 지속일 (점수 <= 0이면 0)
    """
    if score <= min_score:
        return 0

    if score_series is not None and not score_series.empty:
        consecutive_days = calculate_consecutive_days(score_series, min_score)
        return int(consecutive_days.iloc[-1])

    # score_series가 없으면 최소 1일로 반환 (점수가 양수이므로)
    return 1


__all__ = [
    "has_buy_signal",
    "calculate_consecutive_days",
    "get_buy_signal_streak",
]
