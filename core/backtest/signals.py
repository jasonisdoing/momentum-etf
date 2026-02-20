"""Common signal logic for recommendations and backtests."""

import numpy as np
import pandas as pd


def _consecutive_counter(flags: np.ndarray) -> np.ndarray:
    """Python version of consecutive True counter."""
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
    """Determines buy signal based on score."""
    return score > 0


def calculate_consecutive_days(scores: pd.Series) -> pd.Series:
    """Calculates consecutive buy signal days based on score series."""
    if scores is None or scores.empty:
        return pd.Series([], dtype=int)

    buy_signal_active = (scores > 0).to_numpy(dtype=bool, copy=False)
    streak_values = _consecutive_counter(buy_signal_active)
    return pd.Series(streak_values, index=scores.index, dtype=int)


def get_buy_signal_streak(score: float, score_series: pd.Series | None = None) -> int:
    """Returns current buy signal streak."""
    if score <= 0:
        return 0
    if score_series is not None and not score_series.empty:
        consecutive_days = calculate_consecutive_days(score_series)
        return int(consecutive_days.iloc[-1])
    return 1


__all__ = [
    "has_buy_signal",
    "calculate_consecutive_days",
    "get_buy_signal_streak",
]
