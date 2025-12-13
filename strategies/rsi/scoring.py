"""RSI 전략 점수 계산 및 정규화 함수."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_rsi_ema(
    close_prices: pd.Series,
    period: int,
    ema_smoothing: float,
) -> pd.Series:
    """
    EMA 기반 RSI(민감형)를 계산합니다.
    """
    # 입력 검증
    if close_prices is None:
        raise ValueError("close_prices 시리즈가 필요합니다.")
    if period <= 0:
        raise ValueError("period 값은 1 이상이어야 합니다.")
    if ema_smoothing <= 0:
        raise ValueError("ema_smoothing 값은 0보다 커야 합니다.")
    if len(close_prices) < period + 1:
        raise ValueError("RSI 계산을 위해 최소 period+1개의 종가 데이터가 필요합니다.")

    # 변화량 계산
    delta = close_prices.diff()
    gains = delta.where(delta > 0, 0.0)
    losses = -delta.where(delta < 0, 0.0)

    # EMA 계산
    alpha = ema_smoothing / (period + 1.0)
    alpha = min(alpha, 1.0)
    avg_gain = gains.ewm(alpha=alpha, adjust=False).mean()
    avg_loss = losses.ewm(alpha=alpha, adjust=False).mean()

    # RSI 계산
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.clip(lower=0, upper=100)

    rsi = rsi.mask(avg_loss == 0, 100.0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss != 0), 0.0)
    rsi.iloc[:period] = np.nan

    return rsi.astype(float)


def normalize_rsi_score(
    rsi_values: pd.Series | float,
    oversold_threshold: float,
    overbought_threshold: float,
) -> pd.Series | float:
    """
    RSI 값을 0~100 스케일로 정규화합니다 (구간별 점수).

    Args:
        rsi_values: RSI 값 (0~100)
        oversold_threshold: 과매도 기준
        overbought_threshold: 과매수 기준

    Returns:
        0~100 스케일로 정규화된 점수

    Notes:
        구간별 점수 부여 (비선형):
        - RSI 0~30 (과매도) → 70~100점 (선형)
        - RSI 30~70 (중립) → 30~70점 (선형)
        - RSI 70~100 (과매수) → 0~30점 (선형)
    """
    is_series = isinstance(rsi_values, pd.Series)

    def _normalize_single(rsi_val: float) -> float:
        if pd.isna(rsi_val):
            return float("nan")  # NaN 유지

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
    config: dict,
) -> pd.Series | float:
    """
    설정 딕셔너리를 사용하여 RSI 점수를 정규화합니다.
    """
    required_keys = {"enabled", "oversold_threshold", "overbought_threshold"}
    missing = required_keys - set(config.keys())
    if missing:
        raise KeyError(f"normalize_rsi_score_with_config에 필요한 설정 키가 없습니다: {', '.join(sorted(missing))}")

    if not config["enabled"]:
        return rsi_values

    oversold_threshold = float(config["oversold_threshold"])
    overbought_threshold = float(config["overbought_threshold"])

    return normalize_rsi_score(rsi_values, oversold_threshold, overbought_threshold)


def calculate_rsi_score(
    close_prices: pd.Series,
    period: int,
    ema_smoothing: float,
    normalize: bool,
    normalize_config: dict | None,
) -> pd.Series:
    """
    RSI 점수를 계산합니다 (EMA 기반).
    """
    rsi_values = calculate_rsi_ema(close_prices, period, ema_smoothing)

    if normalize:
        if normalize_config is None:
            raise ValueError("normalize=True일 때 normalize_config가 필요합니다.")
        rsi_values = normalize_rsi_score_with_config(rsi_values, normalize_config)

    return rsi_values


__all__ = [
    "calculate_rsi_ema",
    "calculate_rsi_score",
    "normalize_rsi_score",
    "normalize_rsi_score_with_config",
]
