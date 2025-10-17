"""RSI 전략 점수 계산 및 정규화 함수."""

from __future__ import annotations

import pandas as pd
import numpy as np


def calculate_rsi_ema(
    close_prices: pd.Series,
    period: int = 14,
    ema_smoothing: float = 2.0,
) -> pd.Series:
    """
    지수 이동평균(EMA) 기반 RSI를 계산합니다.

    Args:
        close_prices: 종가 시리즈
        period: RSI 계산 기간 (기본값: 14)
        ema_smoothing: EMA 평활화 계수 (기본값: 2.0)

    Returns:
        pd.Series: RSI 값 (0~100)

    Notes:
        - RSI = 100 - (100 / (1 + RS))
        - RS = EMA(gains) / EMA(losses)
        - EMA 방식을 사용하여 Wilder's RSI보다 최근 가격 변화에 더 민감하게 반응
    """
    if close_prices is None or len(close_prices) < period + 1:
        return pd.Series(dtype=float, index=close_prices.index if close_prices is not None else None)

    # 가격 변화 계산
    delta = close_prices.diff()

    # 상승/하락 분리
    gains = delta.where(delta > 0, 0.0)
    losses = -delta.where(delta < 0, 0.0)

    # EMA 계산
    alpha = ema_smoothing / (period + 1)

    # 초기 평균 (SMA)
    avg_gain = gains.rolling(window=period, min_periods=period).mean()
    avg_loss = losses.rolling(window=period, min_periods=period).mean()

    # EMA 방식으로 평활화
    for i in range(period, len(gains)):
        avg_gain.iloc[i] = (gains.iloc[i] * alpha) + (avg_gain.iloc[i - 1] * (1 - alpha))
        avg_loss.iloc[i] = (losses.iloc[i] * alpha) + (avg_loss.iloc[i - 1] * (1 - alpha))

    # RS 계산 (0으로 나누기 방지)
    rs = avg_gain / avg_loss.replace(0, np.nan)

    # RSI 계산
    rsi = 100 - (100 / (1 + rs))

    # NaN 처리 (초기 period 구간)
    rsi = rsi.fillna(50.0)  # 중립값으로 초기화

    return rsi


def normalize_rsi_score(
    rsi_values: pd.Series | float,
    oversold_threshold: float = 30.0,
    overbought_threshold: float = 70.0,
) -> pd.Series | float:
    """
    RSI 값을 0~100 스케일로 정규화합니다 (구간별 점수).

    Args:
        rsi_values: RSI 값 (0~100)
        oversold_threshold: 과매도 기준 (기본값: 30.0)
        overbought_threshold: 과매수 기준 (기본값: 70.0)

    Returns:
        0~100 스케일로 정규화된 점수

    Examples:
        >>> normalize_rsi_score(20)  # 과매도
        80.0
        >>> normalize_rsi_score(50)  # 중립
        50.0
        >>> normalize_rsi_score(80)  # 과매수
        20.0

    Notes:
        구간별 점수 부여 (비선형):
        - RSI 0~30 (과매도) → 70~100점 (선형)
        - RSI 30~70 (중립) → 30~70점 (선형)
        - RSI 70~100 (과매수) → 0~30점 (선형)
    """
    is_series = isinstance(rsi_values, pd.Series)

    def _normalize_single(rsi_val: float) -> float:
        if pd.isna(rsi_val):
            return 50.0  # 중립값

        if rsi_val <= oversold_threshold:
            # 과매도 구간: 70~100점
            return 70 + (oversold_threshold - rsi_val) / oversold_threshold * 30
        elif rsi_val <= overbought_threshold:
            # 중립 구간: 30~70점
            neutral_range = overbought_threshold - oversold_threshold
            return 70 - (rsi_val - oversold_threshold) / neutral_range * 40
        else:
            # 과매수 구간: 0~30점
            overbought_range = 100 - overbought_threshold
            return max(0, 30 - (rsi_val - overbought_threshold) / overbought_range * 30)

    if is_series:
        return rsi_values.apply(_normalize_single)
    else:
        return _normalize_single(rsi_values)


def normalize_rsi_score_with_config(
    rsi_values: pd.Series | float,
    config: dict | None = None,
) -> pd.Series | float:
    """
    설정 딕셔너리를 사용하여 RSI 점수를 정규화합니다.

    Args:
        rsi_values: RSI 값
        config: 정규화 설정 딕셔너리
            - enabled: 정규화 활성화 여부 (기본값: False)
            - oversold_threshold: 과매도 기준 (기본값: 30.0)
            - overbought_threshold: 과매수 기준 (기본값: 70.0)

    Returns:
        정규화된 점수 (enabled=False면 원본 반환)
    """
    if config is None:
        config = {}

    # 정규화 비활성화 시 원본 반환
    if not config.get("enabled", False):
        return rsi_values

    oversold_threshold = float(config.get("oversold_threshold", 30.0))
    overbought_threshold = float(config.get("overbought_threshold", 70.0))

    return normalize_rsi_score(rsi_values, oversold_threshold, overbought_threshold)


def calculate_rsi_score(
    close_prices: pd.Series,
    period: int = 14,
    ema_smoothing: float = 2.0,
    normalize: bool = False,
    normalize_config: dict | None = None,
) -> pd.Series:
    """
    RSI 점수를 계산합니다 (EMA 기반).

    Args:
        close_prices: 종가 시리즈
        period: RSI 계산 기간 (기본값: 14)
        ema_smoothing: EMA 평활화 계수 (기본값: 2.0)
        normalize: 0~100 스케일로 정규화 여부 (기본값: False)
        normalize_config: 정규화 설정 (normalize=True일 때만 사용)
            - oversold_threshold: 과매도 기준 (기본값: 30.0)
            - overbought_threshold: 과매수 기준 (기본값: 70.0)

    Returns:
        pd.Series: RSI 값 (0~100) 또는 정규화된 점수 (0~100)
    """
    rsi_values = calculate_rsi_ema(close_prices, period, ema_smoothing)

    # 정규화 적용
    if normalize:
        config = normalize_config or {"enabled": True, "oversold_threshold": 30.0, "overbought_threshold": 70.0}
        rsi_values = normalize_rsi_score_with_config(rsi_values, config)

    return rsi_values


__all__ = [
    "calculate_rsi_ema",
    "calculate_rsi_score",
    "normalize_rsi_score",
    "normalize_rsi_score_with_config",
]
