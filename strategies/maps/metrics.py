"""MAPS 전략 지표 계산 모듈"""

from collections.abc import Mapping
from typing import Any

import pandas as pd

import config
from logic.backtest.signals import calculate_consecutive_days
from strategies.rsi.backtest import process_ticker_data_rsi
from utils.indicators import calculate_ma_score
from utils.moving_averages import calculate_moving_average


def process_ticker_data(
    ticker: str,
    df: pd.DataFrame,
    ma_period: int,
    precomputed_entry: Mapping[str, Any] | None = None,
    ma_type: str = "SMA",
) -> dict | None:
    """
    개별 종목의 데이터를 처리하고 지표를 계산합니다.

    Args:
        ticker: 종목 티커
        df: 가격 데이터프레임
        ma_period: 이동평균 기간
        precomputed_entry: 미리 계산된 캐시 데이터 (옵션)
        ma_type: 이동평균 타입 (SMA, EMA, WMA, DEMA, TEMA, HMA)

    Returns:
        Dict: 계산된 지표들 또는 None (처리 실패 시)
    """
    if df is None and precomputed_entry is None:
        return None
    if df is not None and df.empty and not precomputed_entry:
        return None

    working_df = df
    if working_df is None and precomputed_entry:
        # Dummy frame to keep downstream logic consistent
        working_df = pd.DataFrame()

    if working_df is not None and isinstance(working_df.columns, pd.MultiIndex):
        working_df = working_df.copy()
        working_df.columns = working_df.columns.get_level_values(0)
        working_df = working_df.loc[:, ~working_df.columns.duplicated()]

    # 티커 유형에 따른 이동평균 기간 결정 (단일 기간 사용)
    current_ma_period = ma_period

    close_prices = None
    open_prices = None
    if isinstance(precomputed_entry, Mapping):
        close_prices = precomputed_entry.get("close")
        open_prices = precomputed_entry.get("open")

    if close_prices is None:
        if working_df is None:
            return None

        price_series = None
        if isinstance(working_df.columns, pd.MultiIndex):
            cols = working_df.columns.get_level_values(0)
            working_df = working_df.copy()
            working_df.columns = cols
            working_df = working_df.loc[:, ~working_df.columns.duplicated()]

        if "unadjusted_close" in working_df.columns:
            price_series = working_df["unadjusted_close"]
        else:
            price_series = working_df["Close"]

        if isinstance(price_series, pd.DataFrame):
            price_series = price_series.iloc[:, 0]
        close_prices = price_series.astype(float)

    if open_prices is None:
        if working_df is not None and "Open" in working_df.columns:
            open_series = working_df["Open"]
            if isinstance(open_series, pd.DataFrame):
                open_series = open_series.iloc[:, 0]
            open_prices = open_series.astype(float)
        else:
            open_prices = close_prices.copy()

    # 데이터 충분성 검증: MA 타입별 이상적인 데이터 요구량
    if config.ENABLE_DATA_SUFFICIENCY_CHECK:
        ma_type_upper = (ma_type or "SMA").upper()
        if ma_type_upper in {"EMA", "DEMA", "TEMA"}:
            ideal_multiplier = 2.0
        elif ma_type_upper == "HMA":
            ideal_multiplier = 1.5
        else:  # SMA, WMA 등
            ideal_multiplier = 1.0

        ideal_data_required = int(current_ma_period * ideal_multiplier)

        # 데이터가 이상적인 양보다 적으면 완화된 기준 적용
        if len(close_prices) < ideal_data_required:
            # 완화된 기준: multiplier의 절반 (최소 1배)
            relaxed_multiplier = max(ideal_multiplier / 2.0, 1.0)
            min_required_data = int(current_ma_period * relaxed_multiplier)
        else:
            # 충분한 데이터가 있으면 이상적인 기준 적용
            min_required_data = ideal_data_required

        if len(close_prices) < min_required_data:
            return None

    # MAPS 전략 지표 계산
    ma_type_key = (ma_type or "SMA").upper()
    ma_key = f"{ma_type_key}_{int(current_ma_period)}"
    moving_average = None
    ma_score = None
    if isinstance(precomputed_entry, Mapping):
        ma_cache = precomputed_entry.get("ma") or {}
        ma_score_cache = precomputed_entry.get("ma_score") or {}
        moving_average = ma_cache.get(ma_key)
        ma_score = ma_score_cache.get(ma_key)

    if moving_average is None:
        moving_average = calculate_moving_average(close_prices, current_ma_period, ma_type)
    if ma_score is None:
        ma_score = calculate_ma_score(close_prices, moving_average)

    # 점수 기반 매수 시그널 지속일 계산
    consecutive_buy_days = calculate_consecutive_days(ma_score)

    # RSI 전략 지표 계산
    rsi_score = None
    if isinstance(precomputed_entry, Mapping):
        rsi_score = precomputed_entry.get("rsi_score")
    if rsi_score is None or isinstance(rsi_score, float):
        rsi_data = process_ticker_data_rsi(close_prices)
        rsi_score = rsi_data.get("rsi_score") if rsi_data else pd.Series(dtype=float)

    return {
        "df": working_df if working_df is not None else df,
        "close": close_prices,
        "open": open_prices,
        "ma": moving_average,
        "ma_score": ma_score,
        "rsi_score": rsi_score,
        "buy_signal_days": consecutive_buy_days,
        "ma_period": current_ma_period,
    }
