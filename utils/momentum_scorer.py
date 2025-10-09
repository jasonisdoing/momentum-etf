#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
모멘텀 스코어를 계산하는 유틸리티 함수.
"""
import warnings
import numpy as np
import pandas as pd

from utils.logger import get_app_logger

# pkg_resources 워닝 억제
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

try:
    import yfinance as yf
except ImportError:
    yf = None

try:
    from pykrx import stock
except ImportError:
    stock = None

logger = get_app_logger()


def momentum_score_yf(ticker: str, end_date: str = None) -> float | None:
    """
    yfinance를 이용해 SPMO 스타일 모멘텀 스코어를 계산합니다.

    모멘텀 스코어 = ((12-1개월 수익률) - (최근 1개월 수익률)) / (12개월 변동성)
    """
    if yf is None:
        logger.warning("yfinance가 설치되어 있지 않습니다.")
        return None

    if end_date is None:
        end_date = pd.Timestamp.today()
    else:
        end_date = pd.Timestamp(end_date)

    start_date = end_date - pd.DateOffset(months=14)

    try:
        df = yf.download(
            ticker,
            start=start_date.strftime("%Y-%m-%d"),
            end=end_date.strftime("%Y-%m-%d"),
            auto_adjust=True,
            progress=False,
        )

        # yfinance가 가끔 MultiIndex 컬럼을 반환하는 경우에 대비합니다.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            df = df.loc[:, ~df.columns.duplicated()]

        if df.empty:
            return None

        df_close = df["Close"].dropna()

        if len(df_close) < 250:
            return None

        monthly = df_close.resample("ME").last()
        if len(monthly) < 13:
            return None

        total_return = (monthly.iloc[-2] / monthly.iloc[-13]) - 1
        recent_return = (monthly.iloc[-1] / monthly.iloc[-2]) - 1
        momentum_return = total_return - recent_return

        # .last() is deprecated. Use .loc with a date offset instead.
        daily_ret_df = df_close.loc[df_close.index >= (df_close.index[-1] - pd.DateOffset(days=365))]
        daily_ret = np.log(daily_ret_df / daily_ret_df.shift(1)).dropna()
        if len(daily_ret) < 20:
            return None
        volatility = daily_ret.std() * np.sqrt(252)
        if volatility == 0:
            return 0.0

        score = momentum_return / volatility
        return score
    except Exception:
        return None


def momentum_score_krx(ticker: str, end_date: str = None) -> float | None:
    """
    pykrx를 이용해 SPMO 스타일 모멘텀 스코어 계산
    ticker 예시: '069500' (KODEX 200)
    """
    if stock is None:
        logger.warning("pykrx가 설치되어 있지 않습니다.")
        return None

    if end_date is None:
        end_date = pd.Timestamp.today()
    else:
        end_date = pd.Timestamp(end_date)

    start_date = end_date - pd.DateOffset(months=13)

    try:
        df = stock.get_etf_ohlcv_by_date(start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d"), ticker)
        if df.empty:
            return None
        df_close = df["종가"]

        if len(df_close) < 250:
            return None

        monthly = df_close.resample("ME").last()
        if len(monthly) < 13:
            return None

        total_return = (monthly.iloc[-2] / monthly.iloc[-13]) - 1
        recent_return = (monthly.iloc[-1] / monthly.iloc[-2]) - 1
        momentum_return = total_return - recent_return

        # .last() is deprecated. Use .loc with a date offset instead.
        daily_ret_df = df_close.loc[df_close.index >= (df_close.index[-1] - pd.DateOffset(days=365))]
        daily_ret = np.log(daily_ret_df / daily_ret_df.shift(1)).dropna()
        if len(daily_ret) < 20:
            return None
        volatility = daily_ret.std() * np.sqrt(252)

        if volatility == 0:
            return 0.0

        score = momentum_return / volatility
        return score
    except Exception:
        return None
