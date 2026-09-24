"""시장 캘린더의 날짜별 거래일·지수·환율·ADR 조회."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd

from config import MARKET_SCHEDULES
from services.price_service import get_exchange_rate_series
from utils.market_breadth_service import load_adr_series, pool_market_key
from utils.market_trend_service import _apply_intraday_boost, load_index_ohlc
from utils.momentum_service import adr_market_of_pool
from utils.settings_loader import get_ticker_type_settings, list_available_ticker_types
from utils.trading_calendar import get_trading_days, is_market_day_completed, is_market_day_started

INDEX_COUNTRIES = {"^KS11": "kor", "^KQ11": "kor", "^GSPC": "us", "^NDX": "us"}
FX_SYMBOLS = {"USD/KRW": "KRW=X", "AUD/KRW": "AUDKRW=X"}


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


def _fx_changes(start: date, end: date, fx: str) -> dict[str, dict[str, float | bool | None]]:
    series = get_exchange_rate_series(
        start - timedelta(days=14), end, symbol=FX_SYMBOLS[fx], allow_partial=True
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


def _pool_adr(pool: str | None, start: date, end: date) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if not pool:
        return {}, None
    if pool not in list_available_ticker_types():
        raise ValueError(f"존재하지 않는 종목풀입니다: {pool}")
    gate_market = adr_market_of_pool(pool)
    market = gate_market or pool_market_key(pool)
    settings = get_ticker_type_settings(pool) or {}
    floor = settings.get("ADR_FLOOR") if gate_market else None
    values = {
        point["date"]: point
        for point in load_adr_series(market)
        if start.isoformat() <= point["date"] <= end.isoformat()
    }
    return values, {"market": market, "floor": float(floor) if isinstance(floor, (int, float)) else None}


def get_market_calendar(start: date, end: date, pool: str | None, fx: str) -> dict[str, Any]:
    """달력에 필요한 날짜 범위를 한 번에 조회한다."""
    if end < start or (end - start).days > 42:
        raise ValueError("시장 캘린더 조회 범위는 최대 43일입니다.")
    if fx not in FX_SYMBOLS:
        raise ValueError(f"지원하지 않는 환율입니다: {fx}")

    sessions = _market_sessions(start, end)
    indices, warnings = _index_changes(start, end)
    fx_values = _fx_changes(start, end, fx)
    if not fx_values:
        warnings.append(f"{fx} 환율 가격을 조회하지 못했습니다.")
    adr_values, adr_meta = _pool_adr(pool, start, end)
    days = {
        day: {
            "sessions": status,
            "indices": {ticker: points.get(day) for ticker, points in indices.items()},
            "fx": fx_values.get(day),
            "adr": adr_values.get(day),
        }
        for day, status in sessions.items()
    }
    return {"days": days, "adr_meta": adr_meta, "warnings": warnings}
