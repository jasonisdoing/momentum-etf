"""RSI 전략 추천 생성 로직."""

from __future__ import annotations

import pandas as pd
from .scoring import calculate_rsi_score
from config import RSI_CALCULATION_CONFIG
from utils.logger import get_app_logger

logger = get_app_logger()


def calculate_rsi_for_ticker(close_prices: pd.Series) -> float:
    """
    단일 종목에 대한 RSI 점수를 계산합니다.

    Args:
        close_prices: 종가 시리즈

    Returns:
        float: RSI 값 (0~100, 70 이상 과매수)
    """
    if close_prices is None or close_prices.empty:
        logger.warning("[RSI] 가격 데이터 없음 (None 또는 empty)")
        return 0.0

    # 데이터 타입을 float64로 변환 (uint32 오버플로우 방지)
    close_prices = close_prices.astype("float64")

    # 데이터 길이 확인 (RSI period + 추가 여유분)
    min_required = RSI_CALCULATION_CONFIG.get("period", 14) + 5  # 14 + 5 = 19개
    if len(close_prices) < min_required:
        logger.warning(f"[RSI] 데이터 길이 부족: {len(close_prices)}개 (최소 {min_required}개 필요)")
        return 0.0

    try:
        # RSI 계산 설정
        rsi_period = RSI_CALCULATION_CONFIG.get("period", 14)
        rsi_ema_smoothing = RSI_CALCULATION_CONFIG.get("ema_smoothing", 2.0)

        # RSI 값 계산 (원본 RSI)
        rsi_score_series = calculate_rsi_score(
            close_prices, period=rsi_period, ema_smoothing=rsi_ema_smoothing, normalize=False, normalize_config=None
        )

        # 최신 값 반환
        if rsi_score_series.empty:
            logger.warning("[RSI] 계산 결과가 비어있음")
            return 0.0

        rsi_score = rsi_score_series.iloc[-1]

        # NaN 처리
        if pd.isna(rsi_score):
            logger.warning("[RSI] 계산 결과가 NaN")
            return 0.0

        logger.debug(f"[RSI] 계산 성공: {rsi_score:.2f}")
        return float(rsi_score)
    except Exception as e:
        logger.warning(f"[RSI] 계산 실패: {e}")
        return 0.0


__all__ = [
    "calculate_rsi_for_ticker",
]
