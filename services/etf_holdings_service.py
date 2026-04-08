from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import requests
import yfinance as yf
from pykrx import stock

from utils.data_loader import get_trading_days
from utils.logger import get_app_logger

logger = get_app_logger()

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
NAVER_ETF_COMPONENT_URL = "https://stock.naver.com/api/domestic/detail/{ticker}/ETFComponent"
_NAVER_ETF_COMPONENT_CACHE: dict[str, dict[str, Any]] = {}
_NAVER_ETF_COMPONENT_TTL_SECONDS = 300
_FOREIGN_PRICE_CACHE: dict[str, dict[str, Any]] = {}
_FOREIGN_PRICE_TTL_SECONDS = 300


def _normalize_ticker(ticker: str) -> str:
    return str(ticker or "").strip().upper().replace("ASX:", "")


def _normalize_date(value: str | None) -> str | None:
    normalized = str(value or "").strip()
    return normalized or None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.replace(",", "").strip()
        if not normalized or normalized == "-":
            return None
        value = normalized
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        return None
    return float(parsed)


def _to_int(value: Any) -> int | None:
    parsed = _to_float(value)
    if parsed is None:
        return None
    return int(parsed)


def _is_cache_alive(cache_entry: dict[str, Any] | None, now: datetime) -> bool:
    if not cache_entry:
        return False
    expires_at = cache_entry.get("expires_at")
    if not isinstance(expires_at, datetime):
        return False
    return now < expires_at


def _create_naver_session() -> requests.Session:
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": DEFAULT_USER_AGENT,
            "Referer": "https://stock.naver.com/",
            "Accept": "application/json, text/plain, */*",
        }
    )
    return session


def _extract_symbol_from_reuters_code(value: str | None) -> str | None:
    normalized = str(value or "").strip().upper()
    if not normalized:
        return None
    return normalized.split(".", 1)[0].strip() or None


def _normalize_reference_date(value: str | None) -> str | None:
    normalized = str(value or "").strip()
    if not normalized:
        return None
    return normalized.replace("-", "")


def _normalize_contracts(value: Any) -> int | float | None:
    parsed = _to_float(value)
    if parsed is None:
        return None
    if float(parsed).is_integer():
        return int(parsed)
    return round(parsed, 2)


def fetch_korean_etf_holdings_from_naver(ticker: str) -> dict[str, Any]:
    normalized_ticker = _normalize_ticker(ticker)
    if not normalized_ticker:
        raise ValueError("ETF 티커가 필요합니다.")

    cache_key = f"naver-holdings:{normalized_ticker}"
    now = datetime.now()
    cached_entry = _NAVER_ETF_COMPONENT_CACHE.get(cache_key)
    if _is_cache_alive(cached_entry, now):
        return dict(cached_entry["data"])

    session = _create_naver_session()
    response = session.get(NAVER_ETF_COMPONENT_URL.format(ticker=normalized_ticker), timeout=15)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise RuntimeError(f"네이버 ETFComponent 응답 형식이 올바르지 않습니다: {normalized_ticker}")
    if not payload:
        raise RuntimeError(f"네이버 ETFComponent 응답이 비어 있습니다: {normalized_ticker}")

    holdings: list[dict[str, Any]] = []
    as_of_date: str | None = None

    for item in payload:
        raw_code = str(item.get("componentIsinCode") or item.get("componentItemCode") or "").strip().upper()
        raw_name = str(item.get("componentName") or "").strip()

        component_item_code = str(item.get("componentItemCode") or "").strip().upper() or None
        component_reuters_code = str(item.get("componentReutersCode") or "").strip().upper() or None
        display_ticker = component_item_code or _extract_symbol_from_reuters_code(component_reuters_code) or raw_code
        reference_date = _normalize_reference_date(item.get("referenceDate"))
        if reference_date:
            as_of_date = reference_date

        holdings.append(
            {
                "ticker": display_ticker,
                "name": raw_name,
                "raw_code": raw_code,
                "raw_name": raw_name,
                "reuters_code": component_reuters_code,
                "yahoo_symbol": _extract_symbol_from_reuters_code(component_reuters_code),
                "contracts": _normalize_contracts(item.get("cuUnitQuantity")),
                "amount": _to_int(item.get("evalAmount")),
                "weight": _to_float(item.get("weight")),
                "market_type": str(item.get("componentMarketType") or "").strip() or None,
            }
        )

    if not holdings:
        raise RuntimeError(f"네이버 ETFComponent에서 저장 가능한 구성종목이 없습니다: {normalized_ticker}")
    if not as_of_date:
        raise RuntimeError(f"네이버 ETFComponent 기준일(referenceDate)을 찾지 못했습니다: {normalized_ticker}")

    holdings.sort(key=lambda row: (row.get("weight") is None, -(row.get("weight") or 0)))
    document = {
        "ticker": normalized_ticker,
        "country_code": "kor",
        "source": "naver_etf_component",
        "as_of_date": as_of_date,
        "holdings_count": len(holdings),
        "holdings": holdings,
        "fetched_at": now.isoformat(),
    }
    _NAVER_ETF_COMPONENT_CACHE[cache_key] = {
        "data": dict(document),
        "expires_at": now + timedelta(seconds=_NAVER_ETF_COMPONENT_TTL_SECONDS),
    }
    return document


def fetch_korean_stock_price_snapshot(tickers: list[str], as_of_date: str) -> dict[str, dict[str, Any]]:
    normalized_date = _normalize_date(as_of_date)
    if not normalized_date:
        raise ValueError("현재가 조회 기준일(as_of_date)이 필요합니다.")

    normalized_tickers = [_normalize_ticker(ticker) for ticker in tickers if _normalize_ticker(ticker)]
    if not normalized_tickers:
        return {}

    target_date = pd.Timestamp(normalized_date)
    trading_days = get_trading_days(
        (target_date - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
        target_date.strftime("%Y-%m-%d"),
        "kor",
    )
    normalized_trading_days = [pd.Timestamp(day).strftime("%Y%m%d") for day in trading_days]
    if normalized_date not in normalized_trading_days:
        raise RuntimeError(f"한국 거래일이 아닌 날짜입니다: {normalized_date}")

    target_index = normalized_trading_days.index(normalized_date)
    if target_index == 0:
        raise RuntimeError(f"전일 거래일을 계산할 수 없습니다: {normalized_date}")
    previous_date = normalized_trading_days[target_index - 1]

    result: dict[str, dict[str, Any]] = {}
    for ticker in normalized_tickers:
        if not ticker.isdigit() or len(ticker) != 6:
            continue
        df = stock.get_market_ohlcv_by_date(previous_date, normalized_date, ticker)
        if df is None or df.empty:
            continue
        working_df = df.copy().sort_index()
        if len(working_df) < 2:
            continue

        previous_close = _to_int(working_df.iloc[-2].get("종가"))
        current_price = _to_int(working_df.iloc[-1].get("종가"))
        if previous_close is None or current_price is None or previous_close == 0:
            continue

        change_pct = round(((current_price / previous_close) - 1.0) * 100.0, 2)
        result[ticker] = {
            "current_price": current_price,
            "previous_close": previous_close,
            "change_pct": change_pct,
            "price_currency": "KRW",
        }

    return result


def _fetch_single_foreign_stock_price_snapshot(symbol: str) -> dict[str, Any] | None:
    ticker = yf.Ticker(symbol)
    history = ticker.history(period="5d", auto_adjust=False)
    if history is None or history.empty:
        return None

    working_df = history.copy().sort_index()
    working_df = working_df[pd.to_numeric(working_df.get("Close"), errors="coerce").notna()]
    if len(working_df) < 2:
        return None

    previous_close = _to_float(working_df.iloc[-2].get("Close"))
    current_price = _to_float(working_df.iloc[-1].get("Close"))
    if previous_close is None or current_price is None or previous_close == 0:
        return None

    as_of_date = pd.Timestamp(working_df.index[-1]).strftime("%Y%m%d")
    metadata = getattr(ticker, "history_metadata", None)
    price_currency = None
    if isinstance(metadata, dict):
        price_currency = str(metadata.get("currency") or "").strip().upper() or None

    return {
        "current_price": round(current_price, 2),
        "previous_close": round(previous_close, 2),
        "change_pct": round(((current_price / previous_close) - 1.0) * 100.0, 2),
        "price_currency": price_currency,
        "as_of_date": as_of_date,
    }


def fetch_foreign_stock_price_snapshot(symbols: list[str]) -> tuple[dict[str, dict[str, Any]], str | None]:
    normalized_symbols = sorted({str(symbol or "").strip().upper() for symbol in symbols if str(symbol or "").strip()})
    if not normalized_symbols:
        return {}, None

    now = datetime.now()
    result: dict[str, dict[str, Any]] = {}
    as_of_dates: set[str] = set()
    for symbol in normalized_symbols:
        cache_key = f"foreign:{symbol}"
        cached_entry = _FOREIGN_PRICE_CACHE.get(cache_key)
        snapshot: dict[str, Any] | None
        if _is_cache_alive(cached_entry, now):
            snapshot = dict(cached_entry["data"])
        else:
            snapshot = _fetch_single_foreign_stock_price_snapshot(symbol)
            if snapshot is None:
                continue
            _FOREIGN_PRICE_CACHE[cache_key] = {
                "data": dict(snapshot),
                "expires_at": now + timedelta(seconds=_FOREIGN_PRICE_TTL_SECONDS),
            }

        result[symbol] = snapshot
        as_of_date = str(snapshot.get("as_of_date") or "").strip()
        if as_of_date:
            as_of_dates.add(as_of_date)

    if not as_of_dates:
        return result, None
    return result, max(as_of_dates)
