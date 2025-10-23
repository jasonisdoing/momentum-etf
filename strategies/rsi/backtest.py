"""RSI 전략 백테스트 로직."""

from __future__ import annotations

from typing import Dict, Optional
import pandas as pd

from .scoring import calculate_rsi_score
from config import RSI_NORMALIZATION_CONFIG, RSI_CALCULATION_CONFIG


def process_ticker_data_rsi(close_prices: pd.Series) -> Optional[Dict]:
    """
    개별 종목의 RSI 지표를 계산합니다.

    Args:
        close_prices: 종가 시리즈

    Returns:
        Dict: RSI 지표 또는 None (처리 실패 시)
    """
    if close_prices is None or close_prices.empty:
        return None

    # RSI 계산 설정
    rsi_period = RSI_CALCULATION_CONFIG.get("period", 14)
    rsi_ema_smoothing = RSI_CALCULATION_CONFIG.get("ema_smoothing", 2.0)

    # 데이터 길이 확인
    if len(close_prices) < rsi_period + 1:
        return None

    # RSI 점수 계산 (정규화 포함)
    rsi_score = calculate_rsi_score(
        close_prices, period=rsi_period, ema_smoothing=rsi_ema_smoothing, normalize=True, normalize_config=RSI_NORMALIZATION_CONFIG
    )

    return {
        "rsi_score": rsi_score,
    }


__all__ = [
    "process_ticker_data_rsi",
]
