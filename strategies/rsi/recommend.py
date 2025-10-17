"""RSI 전략 추천 생성 로직."""

from __future__ import annotations

import pandas as pd
from .scoring import calculate_rsi_score
from data.settings.common import RSI_NORMALIZATION_CONFIG, RSI_CALCULATION_CONFIG
from utils.logger import get_app_logger

logger = get_app_logger()


def calculate_rsi_for_ticker(close_prices: pd.Series) -> float:
    """
    단일 종목에 대한 RSI 점수를 계산합니다.

    Args:
        close_prices: 종가 시리즈

    Returns:
        float: 정규화된 RSI 점수 (0~100)
    """
    if close_prices is None or close_prices.empty:
        return 0.0

    # 데이터 길이 확인
    if len(close_prices) < 15:
        return 0.0

    try:
        # RSI 계산 설정
        rsi_period = RSI_CALCULATION_CONFIG.get("period", 14)
        rsi_ema_smoothing = RSI_CALCULATION_CONFIG.get("ema_smoothing", 2.0)

        # RSI 점수 계산 (정규화 포함)
        rsi_score_series = calculate_rsi_score(
            close_prices, period=rsi_period, ema_smoothing=rsi_ema_smoothing, normalize=True, normalize_config=RSI_NORMALIZATION_CONFIG
        )

        # 최신 값 반환
        if rsi_score_series.empty:
            return 0.0

        rsi_score = rsi_score_series.iloc[-1]

        # NaN 처리
        if pd.isna(rsi_score):
            return 0.0

        return float(rsi_score)
    except Exception as e:
        logger.warning(f"[RSI] 계산 실패: {e}")
        return 0.0


__all__ = [
    "calculate_rsi_for_ticker",
]
