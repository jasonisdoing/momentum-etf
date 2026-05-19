"""시장지수 추세 데이터 서비스.

코스피/코스피200/S&P500/나스닥/나스닥100 의 현재가, 변동률, MA 대비 추세% 를 계산한다.
가격은 yfinance 에서 직접 가져오며 (인덱스는 stock_cache 미사용), MA 계산은 utils.moving_averages 사용.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import yfinance as yf

from config import TRADING_DAYS_PER_MONTH
from utils.moving_averages import calculate_moving_average

logger = logging.getLogger(__name__)

# 표시 순서대로 정의. yf_ticker 는 Yahoo Finance 인덱스 심볼.
INDICES: list[dict[str, str]] = [
    {"name": "코스피", "yf_ticker": "^KS11"},
    {"name": "코스피 200", "yf_ticker": "^KS200"},
    {"name": "S&P 500", "yf_ticker": "^GSPC"},
    {"name": "나스닥", "yf_ticker": "^IXIC"},
    {"name": "나스닥 100", "yf_ticker": "^NDX"},
]

# 과거 시점 추세 계산용 trading-day offset (오늘 대비 N 거래일 전).
TRADING_DAYS_PER_WEEK = 5
TREND_OFFSETS_DAYS = {
    "trend_pct_w1": TRADING_DAYS_PER_WEEK * 1,            # 1주일 전
    "trend_pct_m1": TRADING_DAYS_PER_MONTH * 1,           # 1달 전
    "trend_pct_m3": TRADING_DAYS_PER_MONTH * 3,           # 3달 전
}


def _to_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(result):
        return None
    return result


def compute_market_trend(ma_type: str, ma_months: int) -> dict[str, Any]:
    """5개 시장지수의 현재가/변동률/MA 추세%(현재 + 과거 3시점) 를 계산해 반환한다.

    Args:
        ma_type: SMA/EMA/WMA/DEMA/TEMA/HMA/ALMA
        ma_months: 1~12 (정수). 내부에서 ``ma_months * TRADING_DAYS_PER_MONTH`` 일로 환산.

    Returns:
        ``{"ma_type", "ma_months", "items": [{
            name, ticker, price, change_pct,
            trend_pct, trend_pct_w1, trend_pct_m1, trend_pct_m3,
        }, ...]}``
    """

    ma_days = int(ma_months) * int(TRADING_DAYS_PER_MONTH)
    if ma_days < 2:
        ma_days = 2

    tickers = [idx["yf_ticker"] for idx in INDICES]
    # 단일 호출로 일괄 다운로드 (네트워크 효율). MA 가 가장 긴 12개월 = 240일 이라
    # 2년치 데이터면 충분히 안전. 3개월 전 추세 계산용 과거 시점도 같은 윈도우에서 커버.
    try:
        df = yf.download(
            tickers=tickers,
            period="2y",
            interval="1d",
            group_by="ticker",
            progress=False,
            auto_adjust=True,
            threads=True,
        )
    except Exception:
        logger.exception("yfinance 시장지수 다운로드 실패")
        df = None

    items: list[dict[str, Any]] = []
    for idx in INDICES:
        item = _build_item(df, idx["yf_ticker"], idx["name"], ma_days, ma_type)
        items.append(item)

    return {
        "ma_type": ma_type,
        "ma_months": int(ma_months),
        "items": items,
    }


def _build_item(
    df: pd.DataFrame | None,
    yf_ticker: str,
    name: str,
    ma_days: int,
    ma_type: str,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "name": name,
        "ticker": yf_ticker,
        "price": None,
        "change_pct": None,
        "trend_pct": None,
        "trend_pct_w1": None,
        "trend_pct_m1": None,
        "trend_pct_m3": None,
    }
    if df is None or df.empty:
        return base

    # multi-ticker 결과는 컬럼 멀티인덱스(ticker, ohlc). 단일 ticker 결과는 평탄.
    try:
        if (yf_ticker, "Close") in df.columns:
            close_series = df[(yf_ticker, "Close")].dropna()
        elif "Close" in df.columns:
            close_series = df["Close"].dropna()
        else:
            return base
    except Exception:
        return base

    if close_series is None or close_series.empty or len(close_series) < 2:
        return base

    latest_price = _to_float(close_series.iloc[-1])
    prev_price = _to_float(close_series.iloc[-2])
    if latest_price is None or prev_price is None:
        return base

    change_pct: float | None = None
    if prev_price != 0:
        change_pct = (latest_price / prev_price - 1.0) * 100.0

    base["price"] = latest_price
    base["change_pct"] = change_pct

    # MA 시리즈는 전체 가격 시리즈에 대해 한 번만 계산하고, 시점별로 인덱싱한다.
    try:
        ma_series = calculate_moving_average(close_series, ma_days, ma_type)
    except Exception:
        logger.exception("MA 계산 실패: %s (type=%s, days=%d)", yf_ticker, ma_type, ma_days)
        return base

    base["trend_pct"] = _trend_pct_at(close_series, ma_series, offset=0)
    for key, offset in TREND_OFFSETS_DAYS.items():
        base[key] = _trend_pct_at(close_series, ma_series, offset=offset)

    return base


def _trend_pct_at(
    close_series: pd.Series,
    ma_series: pd.Series,
    *,
    offset: int,
) -> float | None:
    """``offset`` 거래일 전의 (종가 / MA - 1) * 100 을 반환한다. offset=0 이면 최신."""
    if close_series is None or ma_series is None:
        return None
    length = min(len(close_series), len(ma_series))
    if length <= offset:
        return None
    idx = -1 - offset
    price = _to_float(close_series.iloc[idx])
    ma_value = _to_float(ma_series.iloc[idx])
    if price is None or ma_value is None or ma_value == 0:
        return None
    return (price / ma_value - 1.0) * 100.0


__all__ = ["compute_market_trend", "INDICES"]
