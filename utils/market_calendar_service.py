"""시장 캘린더의 날짜별 거래일·지수·환율·ADR 조회."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from math import isfinite
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from config import MARKET_SCHEDULES
from services.price_service import get_exchange_rate_series, get_yahoo_symbol_snapshot
from utils.market_breadth_service import load_adr_series, pool_market_key
from utils.market_trend_service import _apply_intraday_boost, load_index_ohlc
from utils.momentum_service import adr_market_of_pool
from utils.settings_loader import get_ticker_type_settings, list_available_ticker_types
from utils.slot_positions import adr_entry_gate
from utils.trading_calendar import get_trading_days, is_market_day_completed, is_market_day_started

INDEX_COUNTRIES = {"^KS11": "kor", "^KQ11": "kor", "^GSPC": "us", "^NDX": "us"}
CALENDAR_ADR_POOLS = ("kor_stock", "us_stock")
CALENDAR_FX_SYMBOL = "KRW=X"
FUTURES_BY_INDEX = {"^GSPC": "ES=F", "^NDX": "NQ=F"}
MAX_FUTURE_QUOTE_AGE_SECONDS = 3600


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


def _index_changes(
    start: date, end: date, sessions: dict[str, dict[str, str]]
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, dict[str, str]]], list[str]]:
    result: dict[str, dict[str, Any]] = {}
    issues: dict[str, dict[str, dict[str, str]]] = {}
    warnings: list[str] = []
    for ticker, country in INDEX_COUNTRIES.items():
        frame = load_index_ohlc(ticker)
        if frame is None or frame.empty:
            warnings.append(f"{ticker} 지수 가격을 조회하지 못했습니다.")
            close = pd.Series(dtype=float)
        else:
            close = frame["Close"].dropna()
            if country == "us" and end >= pd.Timestamp.now(tz="America/New_York").date():
                close = _apply_intraday_boost(close, ticker)
            if close is None or close.empty:
                warnings.append(f"{ticker} 지수 종가가 없습니다.")
                close = pd.Series(dtype=float)
        values = _daily_changes(close, start, end)
        for day, point in values.items():
            point["provisional"] = not is_market_day_completed(country, pd.Timestamp(day))
        trading_days = [
            stamp.date().isoformat()
            for stamp in get_trading_days((start - timedelta(days=10)).isoformat(), end.isoformat(), country)
        ]
        close_days = {pd.Timestamp(stamp).date().isoformat() for stamp in close.index}
        for previous_day, day in zip(trading_days, trading_days[1:]):
            if day < start.isoformat() or sessions.get(day, {}).get(country) != "finished":
                continue
            if day not in values:
                issues.setdefault(day, {})[ticker] = {
                    "label": "종가 누락",
                    "reason": f"{day} 거래일의 확정 종가 데이터가 조회되지 않았습니다.",
                }
            elif previous_day not in close_days:
                values[day]["change_pct"] = None
                issues.setdefault(day, {})[ticker] = {
                    "label": "전일 종가 누락",
                    "reason": f"전 거래일({previous_day}) 종가가 없어 당일 변동률을 계산할 수 없습니다.",
                }
        result[ticker] = values
    return result, issues, warnings


def _fx_changes(start: date, end: date) -> dict[str, dict[str, float | bool | None]]:
    series = get_exchange_rate_series(start - timedelta(days=14), end, symbol=CALENDAR_FX_SYMBOL, allow_partial=True)
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


def _pool_adrs(start: date, end: date) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    available = set(list_available_ticker_types())
    missing = [pool for pool in CALENDAR_ADR_POOLS if pool not in available]
    if missing:
        raise ValueError(f"시장 캘린더 종목풀이 없습니다: {', '.join(missing)}")
    values: dict[str, dict[str, Any]] = {}
    meta: dict[str, dict[str, Any]] = {}
    for pool in CALENDAR_ADR_POOLS:
        settings = get_ticker_type_settings(pool) or {}
        floor = settings.get("ADR_FLOOR")
        entry_blocked, gate_adr_at = adr_entry_gate(pool, floor)
        meta[pool] = {"floor": floor, "gate_market": adr_market_of_pool(pool)}
        values[pool] = {
            point["date"]: point
            for point in load_adr_series(pool_market_key(pool))
            if start.isoformat() <= point["date"] <= end.isoformat()
        }
        for day, point in values[pool].items():
            stamp = pd.Timestamp(day)
            point["entry_allowed"] = not entry_blocked(stamp)
            point["gate_adr"] = gate_adr_at(stamp)
    return values, meta


def _today_us_futures(
    start: date, end: date, sessions: dict[str, dict[str, str]], indices: dict[str, dict[str, Any]]
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """오늘 미국 정규장 개장 전 빈 지수 칸에만 신선한 선물 시세를 제공한다."""
    now = datetime.now(timezone.utc)
    today = now.astimezone(ZoneInfo("Asia/Seoul")).date()
    day = today.isoformat()
    if not (start <= today <= end) or sessions.get(day, {}).get("us") != "scheduled":
        return {}, []

    result: dict[str, dict[str, Any]] = {}
    warnings: list[str] = []
    for index_ticker, future_symbol in FUTURES_BY_INDEX.items():
        if indices.get(index_ticker, {}).get(day, {}).get("change_pct") is not None:
            continue
        try:
            quote = get_yahoo_symbol_snapshot([future_symbol]).get(future_symbol) or {}
            price = float(quote["nowVal"])
            change = float(quote["changeRate"])
            quoted_at = datetime.fromtimestamp(float(quote["quoteTime"]), tz=timezone.utc)
            age = (now - quoted_at).total_seconds()
            if not (isfinite(price) and isfinite(change) and price > 0):
                raise ValueError("가격 또는 변동률이 유효하지 않습니다")
            if (
                not (0 <= age <= MAX_FUTURE_QUOTE_AGE_SECONDS)
                or quoted_at.astimezone(ZoneInfo("Asia/Seoul")).date() != today
            ):
                raise ValueError("오늘의 최신 선물 시세가 아닙니다")
            result[index_ticker] = {
                "close": price,
                "change_pct": change,
                "provisional": True,
                "quote_at": quoted_at.isoformat(),
            }
        except Exception as exc:
            warnings.append(f"{future_symbol} 선물 시세를 표시하지 못했습니다: {exc}")
    return {day: result} if result else {}, warnings


def get_market_calendar(start: date, end: date) -> dict[str, Any]:
    """달력에 필요한 날짜 범위를 한 번에 조회한다."""
    if end < start or (end - start).days > 42:
        raise ValueError("시장 캘린더 조회 범위는 최대 43일입니다.")
    sessions = _market_sessions(start, end)
    indices, index_issues, warnings = _index_changes(start, end, sessions)
    futures, future_warnings = _today_us_futures(start, end, sessions, indices)
    warnings.extend(future_warnings)
    fx_values = _fx_changes(start, end)
    if not fx_values:
        warnings.append("USD/KRW 환율 가격을 조회하지 못했습니다.")
    adr_values, adr_meta = _pool_adrs(start, end)
    days = {
        day: {
            "sessions": status,
            "indices": {ticker: points.get(day) for ticker, points in indices.items()},
            "index_issues": index_issues.get(day, {}),
            "futures": futures.get(day),
            "fx": fx_values.get(day),
            "adr": {pool: points.get(day) for pool, points in adr_values.items()},
        }
        for day, status in sessions.items()
    }
    return {"days": days, "adr_meta": adr_meta, "warnings": warnings}
