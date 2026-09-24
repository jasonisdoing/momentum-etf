"""시장 캘린더의 날짜별 거래일·지수·환율·ADR 조회."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd

from config import MARKET_SCHEDULES
from services.price_service import get_exchange_rate_series
from utils.market_breadth_service import load_adr_series, pool_market_key
from utils.market_trend_service import _apply_intraday_boost, load_index_ohlc
from utils.settings_loader import list_available_ticker_types
from utils.trading_calendar import get_trading_days, is_market_day_completed, is_market_day_started

INDEX_COUNTRIES = {"^KS11": "kor", "^KQ11": "kor", "^GSPC": "us", "^NDX": "us"}
CALENDAR_ADR_POOLS = ("kor_stock", "us_stock")
CALENDAR_FX_SYMBOL = "KRW=X"


def _daily_changes(series: pd.Series, start: date, end: date) -> dict[str, dict[str, float | bool | None]]:
    """실제 봉이 있는 날짜에만 전 거래일 대비 등락률을 둔다."""
    close = series.dropna().sort_index()
    close = close[~close.index.duplicated(keep="last")]
    changes = close.pct_change(fill_method=None) * 100.0
    result: dict[str, dict[str, float | bool]] = {}
    for stamp, value in close.items():
        day = pd.Timestamp(stamp).date()
        if start <= day <= end:
            change = changes.loc[stamp]
            result[day.isoformat()] = {
                "close": float(value),
                "change_pct": float(change) if pd.notna(change) else None,
            }
    return result


def _index_changes(start: date, end: date) -> tuple[dict[str, dict[str, Any]], list[str]]:
    result: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    for ticker, country in INDEX_COUNTRIES.items():
        frame = load_index_ohlc(ticker)
        if frame is None or frame.empty:
            warnings.append(f"{ticker} 지수 가격을 조회하지 못했습니다.")
            result[ticker] = {}
            continue
        close = frame["Close"].dropna()
        if country == "us" and end >= pd.Timestamp.now(tz="America/New_York").date():
            close = _apply_intraday_boost(close, ticker)
        if close is None or close.empty:
            warnings.append(f"{ticker} 지수 종가가 없습니다.")
            result[ticker] = {}
            continue
        values = _daily_changes(close, start, end)
        for day, point in values.items():
            point["provisional"] = not is_market_day_completed(country, pd.Timestamp(day))
        result[ticker] = values
    return result, warnings


def _fx_changes(start: date, end: date) -> dict[str, dict[str, float | bool | None]]:
    series = get_exchange_rate_series(
        start - timedelta(days=14), end, symbol=CALENDAR_FX_SYMBOL, allow_partial=True
    )
    if series is None or series.empty:
        return {}
    result = _daily_changes(series, start, end)
    today = pd.Timestamp.now(tz="Asia/Seoul").date().isoformat()
    for day, point in result.items():
        point["provisional"] = day == today
    return result


def _market_sessions(start: date, end: date) -> dict[str, dict[str, str]]:
    start_text, end_text = start.isoformat(), end.isoformat()
    result: dict[str, dict[str, str]] = {}
    for country in ("kor", "us"):
        trading_days = {stamp.date().isoformat() for stamp in get_trading_days(start_text, end_text, country)}
        local_today = pd.Timestamp.now(tz=MARKET_SCHEDULES[country]["timezone"]).date()
        day = start
        while day <= end:
            key = day.isoformat()
            if key not in trading_days:
                status = "closed_future" if day > local_today else "closed"
            elif is_market_day_completed(country, pd.Timestamp(day)):
                status = "finished"
            elif is_market_day_started(country, pd.Timestamp(day)):
                status = "open"
            else:
                status = "scheduled"
            result.setdefault(key, {})[country] = status
            day += timedelta(days=1)
    return result


def _pool_adrs(start: date, end: date) -> dict[str, dict[str, Any]]:
    available = set(list_available_ticker_types())
    missing = [pool for pool in CALENDAR_ADR_POOLS if pool not in available]
    if missing:
        raise ValueError(f"시장 캘린더 종목풀이 없습니다: {', '.join(missing)}")
    return {
        pool: {
            point["date"]: point
            for point in load_adr_series(pool_market_key(pool))
            if start.isoformat() <= point["date"] <= end.isoformat()
        }
        for pool in CALENDAR_ADR_POOLS
    }


def get_market_calendar(start: date, end: date) -> dict[str, Any]:
    """달력에 필요한 날짜 범위를 한 번에 조회한다."""
    if end < start or (end - start).days > 42:
        raise ValueError("시장 캘린더 조회 범위는 최대 43일입니다.")
    sessions = _market_sessions(start, end)
    indices, warnings = _index_changes(start, end)
    fx_values = _fx_changes(start, end)
    if not fx_values:
        warnings.append("USD/KRW 환율 가격을 조회하지 못했습니다.")
    adr_values = _pool_adrs(start, end)
    days = {
        day: {
            "sessions": status,
            "indices": {ticker: points.get(day) for ticker, points in indices.items()},
            "fx": fx_values.get(day),
            "adr": {pool: points.get(day) for pool, points in adr_values.items()},
        }
        for day, status in sessions.items()
    }
    return {"days": days, "warnings": warnings}
