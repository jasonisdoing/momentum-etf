from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd

from services.price_service import get_realtime_snapshot, get_worldstock_snapshot, get_yahoo_symbol_snapshot
from utils.cache_utils import load_cached_close_series_bulk_with_fallback
from utils.logger import get_app_logger

logger = get_app_logger()


def infer_yahoo_symbol_currency(symbol: str) -> str:
    normalized = str(symbol or "").strip().upper()
    if normalized.endswith(".TW"):
        return "TWD"
    if normalized.endswith(".HK"):
        return "HKD"
    if normalized.endswith(".T"):
        return "JPY"
    if normalized.endswith(".L"):
        return "GBP"
    if normalized.endswith((".KS", ".KQ")):
        return "KRW"
    if normalized.endswith(".AX"):
        return "AUD"
    return "USD"


def enrich_component_prices(
    holdings: Iterable[dict[str, Any]],
    *,
    price_fetch_limit: int,
    preserve_existing: bool = False,
    cumulative_base_date: str | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    """ETF 구성종목에 현재가/등락률/통화 정보를 붙인다."""
    holdings_list = [dict(item) for item in holdings]
    if not holdings_list:
        return [], None

    def _weight_value(item: dict[str, Any]) -> float:
        try:
            return float(item.get("weight") or item.get("total_weight") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    holdings_for_pricing = sorted(holdings_list, key=_weight_value, reverse=True)[:price_fetch_limit]
    pricing_ids = {id(item) for item in holdings_for_pricing}

    korean_tickers: list[str] = []
    us_tickers: list[str] = []
    au_tickers: list[str] = []
    worldstock_codes: list[str] = []
    yahoo_exchange_symbols: list[str] = []
    baseline_yahoo_symbols: list[str] = []
    # 한국 ETF 마감(15:30 KST) 시점에 미국장은 아직 열리지 않았으므로(US 22:30 KST 개장),
    # 한국 base_date 마감 시점에 반영된 미국 가격은 전 미국 거래일 종가다.
    # 따라서 미국 종목 baseline 은 base_date 미만(< base_ts)으로 한 번 더 소급해야 한다.
    previous_trading_day_baseline_symbols: set[str] = set()

    for item in holdings_for_pricing:
        if _is_cash_like_holding(item):
            continue

        component_ticker = _normalize_upper(item.get("ticker"))
        if _is_korean_six_digit_holding(item):
            korean_tickers.append(component_ticker)
            continue

        yahoo_symbol = _normalize_upper(item.get("yahoo_symbol")) or component_ticker
        if not yahoo_symbol:
            continue
        if cumulative_base_date:
            baseline_yahoo_symbols.append(yahoo_symbol)
            if _is_us_yahoo_symbol(yahoo_symbol):
                previous_trading_day_baseline_symbols.add(yahoo_symbol)
        if yahoo_symbol.endswith(".AX"):
            au_tickers.append(yahoo_symbol[:-3])
        elif _is_worldstock_symbol(yahoo_symbol):
            worldstock_codes.append(_normalize_upper(item.get("reuters_code")) or yahoo_symbol)
        elif _is_yahoo_exchange_symbol(yahoo_symbol):
            yahoo_exchange_symbols.append(yahoo_symbol)
        elif "." not in yahoo_symbol:
            us_tickers.append(yahoo_symbol)

    kor_price_map = _safe_fetch_snapshot("kor", korean_tickers)
    us_price_map = _safe_fetch_snapshot("us", us_tickers)
    au_price_map = _safe_fetch_snapshot("au", au_tickers)
    worldstock_price_map = _safe_fetch_worldstock(worldstock_codes)
    yahoo_exchange_price_map = _safe_fetch_yahoo(yahoo_exchange_symbols)
    korean_baseline_price_map = _safe_fetch_cached_baseline_prices("kor", korean_tickers, cumulative_base_date)
    baseline_price_map = _safe_fetch_yahoo_baseline_prices(
        baseline_yahoo_symbols,
        cumulative_base_date,
        previous_trading_day_baseline_symbols,
    )

    enriched: list[dict[str, Any]] = []
    price_as_of_dates: set[str] = set()
    for item in holdings_list:
        enriched_item = dict(item)
        component_ticker = _normalize_upper(item.get("ticker"))
        yahoo_symbol = _normalize_upper(item.get("yahoo_symbol")) or None
        enriched_item["yahoo_symbol"] = yahoo_symbol

        if _is_cash_like_holding(item):
            _clear_price_fields(enriched_item, preserve_existing=preserve_existing)
            enriched_item["price_currency"] = "KRW"
        elif id(item) not in pricing_ids:
            _clear_price_fields(enriched_item, preserve_existing=preserve_existing)
        elif _is_korean_six_digit_holding(item):
            _apply_price_snapshot(
                enriched_item,
                kor_price_map.get(component_ticker, {}),
                "KRW",
                preserve_existing=preserve_existing,
            )
            if cumulative_base_date:
                _apply_cumulative_change(enriched_item, korean_baseline_price_map.get(component_ticker))
        elif yahoo_symbol and yahoo_symbol.endswith(".AX"):
            _apply_price_snapshot(
                enriched_item,
                au_price_map.get(yahoo_symbol[:-3], {}),
                "AUD",
                preserve_existing=preserve_existing,
            )
        elif yahoo_symbol and _is_worldstock_symbol(yahoo_symbol):
            lookup_code = _normalize_upper(item.get("reuters_code")) or yahoo_symbol
            snapshot = worldstock_price_map.get(lookup_code, {})
            _apply_price_snapshot(
                enriched_item,
                snapshot,
                str(snapshot.get("currency") or infer_yahoo_symbol_currency(yahoo_symbol)),
                preserve_existing=preserve_existing,
            )
        elif yahoo_symbol and _is_yahoo_exchange_symbol(yahoo_symbol):
            _apply_price_snapshot(
                enriched_item,
                yahoo_exchange_price_map.get(yahoo_symbol, {}),
                infer_yahoo_symbol_currency(yahoo_symbol),
                preserve_existing=preserve_existing,
            )
        elif yahoo_symbol and "." in yahoo_symbol:
            _clear_price_fields(enriched_item, preserve_existing=preserve_existing)
        else:
            lookup_symbol = yahoo_symbol or component_ticker
            _apply_price_snapshot(
                enriched_item,
                us_price_map.get(lookup_symbol, {}),
                "USD",
                preserve_existing=preserve_existing,
            )

        if cumulative_base_date and yahoo_symbol and not _is_korean_six_digit_holding(item):
            baseline_item = baseline_price_map.get(yahoo_symbol)
            _apply_cumulative_change(enriched_item, baseline_item)

        as_of_date = str(enriched_item.get("price_as_of_date") or "").strip()
        if as_of_date:
            price_as_of_dates.add(as_of_date)
        enriched.append(enriched_item)

    price_as_of = max(price_as_of_dates) if price_as_of_dates else None
    return enriched, price_as_of


def _normalize_upper(value: Any) -> str:
    return str(value or "").strip().upper()


def _is_cash_like_holding(item: dict[str, Any]) -> bool:
    ticker = _normalize_upper(item.get("ticker"))
    raw_code = _normalize_upper(item.get("raw_code"))
    name = str(item.get("name") or item.get("raw_name") or "").strip()
    return ticker.startswith("KRD") or raw_code.startswith("KRD") or "현금" in name


def _is_korean_six_digit_holding(item: dict[str, Any]) -> bool:
    component_ticker = _normalize_upper(item.get("ticker"))
    raw_code = _normalize_upper(item.get("raw_code"))
    yahoo_symbol = _normalize_upper(item.get("yahoo_symbol"))
    if not component_ticker.isdigit() or len(component_ticker) != 6:
        return False
    if yahoo_symbol and not yahoo_symbol.endswith((".KS", ".KQ")):
        return False
    if raw_code.startswith("CNE"):
        return False
    return True


def _is_worldstock_symbol(symbol: str) -> bool:
    return _normalize_upper(symbol).endswith((".T", ".HK"))


def _is_yahoo_exchange_symbol(symbol: str) -> bool:
    return _normalize_upper(symbol).endswith((".TW", ".L"))


def _is_us_yahoo_symbol(symbol: str) -> bool:
    normalized = _normalize_upper(symbol)
    return bool(normalized) and "." not in normalized


def _safe_fetch_snapshot(country_code: str, tickers: list[str]) -> dict[str, dict[str, Any]]:
    if not tickers:
        return {}
    try:
        return get_realtime_snapshot(country_code, tickers)
    except Exception as exc:
        logger.warning("구성종목 가격 조회 실패 (%s): %s", country_code, exc)
        return {}


def _safe_fetch_worldstock(reuters_codes: list[str]) -> dict[str, dict[str, Any]]:
    if not reuters_codes:
        return {}
    try:
        return get_worldstock_snapshot(reuters_codes)
    except Exception as exc:
        logger.warning("구성종목 worldstock 가격 조회 실패: %s", exc)
        return {}


def _safe_fetch_yahoo(symbols: list[str]) -> dict[str, dict[str, Any]]:
    if not symbols:
        return {}
    try:
        return get_yahoo_symbol_snapshot(symbols)
    except Exception as exc:
        logger.warning("구성종목 Yahoo 가격 조회 실패: %s", exc)
        return {}


def _safe_fetch_yahoo_baseline_prices(
    symbols: list[str],
    base_date: str | None,
    previous_trading_day_symbols: set[str],
) -> dict[str, dict[str, Any]]:
    if not symbols or not base_date:
        return {}
    try:
        return _fetch_yahoo_baseline_prices(symbols, base_date, previous_trading_day_symbols)
    except Exception as exc:
        logger.warning("구성종목 기준일 가격 조회 실패(base_date=%s): %s", base_date, exc)
        return {}


def _safe_fetch_cached_baseline_prices(
    ticker_type: str,
    tickers: list[str],
    base_date: str | None,
) -> dict[str, dict[str, Any]]:
    if not tickers or not base_date:
        return {}
    try:
        return _fetch_cached_baseline_prices(ticker_type, tickers, base_date)
    except Exception as exc:
        logger.warning("구성종목 국내 기준일 가격 조회 실패(base_date=%s): %s", base_date, exc)
        return {}


def _fetch_cached_baseline_prices(
    ticker_type: str,
    tickers: list[str],
    base_date: str,
) -> dict[str, dict[str, Any]]:
    normalized_tickers = sorted({str(ticker or "").strip().upper() for ticker in tickers if str(ticker or "").strip()})
    if not normalized_tickers:
        return {}

    base_ts = pd.Timestamp(base_date).normalize()
    close_series_map = load_cached_close_series_bulk_with_fallback(ticker_type, normalized_tickers)
    result: dict[str, dict[str, Any]] = {}
    for ticker, series in close_series_map.items():
        if series is None or series.empty:
            continue
        close_series = pd.to_numeric(series, errors="coerce").dropna()
        close_series = close_series[close_series > 0]
        if close_series.empty:
            continue

        normalized_index = pd.to_datetime(close_series.index).tz_localize(None).normalize()
        base_series = close_series[normalized_index <= base_ts]
        if base_series.empty:
            continue

        baseline_date = base_series.index[-1]
        result[str(ticker).strip().upper()] = {
            "price": float(base_series.iloc[-1]),
            "date": pd.Timestamp(baseline_date).strftime("%Y-%m-%d"),
        }
    return result


import threading

_YAHOO_BASELINE_CACHE: dict[tuple[str, str, bool], dict[str, Any]] = {}
_YAHOO_BASELINE_LOCK = threading.Lock()

def _fetch_yahoo_baseline_prices(
    symbols: list[str],
    base_date: str,
    previous_trading_day_symbols: set[str],
) -> dict[str, dict[str, Any]]:
    import yfinance as yf

    normalized_symbols = sorted({str(symbol or "").strip().upper() for symbol in symbols if str(symbol or "").strip()})
    if not normalized_symbols:
        return {}

    with _YAHOO_BASELINE_LOCK:
        result: dict[str, dict[str, Any]] = {}
        symbols_to_fetch: list[str] = []
        previous_day_symbols = {str(symbol or "").strip().upper() for symbol in previous_trading_day_symbols}

        for symbol in normalized_symbols:
            is_prev = symbol in previous_day_symbols
            cache_key = (symbol, base_date, is_prev)
            if cache_key in _YAHOO_BASELINE_CACHE:
                result[symbol] = _YAHOO_BASELINE_CACHE[cache_key]
            else:
                symbols_to_fetch.append(symbol)

        if not symbols_to_fetch:
            return result

        base_ts = pd.Timestamp(base_date).normalize()
        start_ts = base_ts - pd.Timedelta(days=10)
        end_ts = base_ts + pd.Timedelta(days=1)
        downloaded = yf.download(
            symbols_to_fetch,
            start=start_ts.strftime("%Y-%m-%d"),
            end=end_ts.strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=True,
        )
        if downloaded is None or downloaded.empty:
            return result

        def _extract_symbol_frame(symbol: str) -> pd.DataFrame | None:
            if isinstance(downloaded.columns, pd.MultiIndex):
                if symbol not in downloaded.columns.get_level_values(0):
                    return None
                return downloaded[symbol].copy()
            if len(symbols_to_fetch) != 1:
                return None
            return downloaded.copy()

        for symbol in symbols_to_fetch:
            frame = _extract_symbol_frame(symbol)
            if frame is None or frame.empty or "Close" not in frame.columns:
                continue
            close_series = pd.to_numeric(frame["Close"], errors="coerce").dropna()
            normalized_index = pd.to_datetime(close_series.index).tz_localize(None).normalize()

            is_prev = symbol in previous_day_symbols
            if is_prev:
                # 한국 base_date 마감 시점에는 미국장이 열리기 전이므로 base_date 미만 종가 사용
                close_series = close_series[normalized_index < base_ts]
            else:
                close_series = close_series[normalized_index <= base_ts]

            if close_series.empty:
                continue
            data = {
                "price": float(close_series.iloc[-1]),
                "date": pd.Timestamp(close_series.index[-1]).strftime("%Y-%m-%d"),
            }
            result[symbol] = data
            _YAHOO_BASELINE_CACHE[(symbol, base_date, is_prev)] = data

        return result


def _apply_price_snapshot(
    enriched_item: dict[str, Any],
    snapshot: dict[str, Any],
    currency: str,
    *,
    preserve_existing: bool,
) -> None:
    current_price = snapshot.get("nowVal") if snapshot.get("nowVal") is not None else snapshot.get("price")
    previous_close = snapshot.get("prevClose")
    change_pct = snapshot.get("changeRate")
    if change_pct is None:
        change_pct = snapshot.get("change_pct")
    if current_price is None and change_pct is None and preserve_existing:
        return

    enriched_item["current_price"] = float(current_price) if current_price is not None else None
    enriched_item["previous_close"] = float(previous_close) if previous_close is not None else None
    enriched_item["change_pct"] = float(change_pct) if change_pct is not None else None
    enriched_item["price_currency"] = currency
    enriched_item["price_as_of_date"] = snapshot.get("as_of_date")


def _apply_cumulative_change(enriched_item: dict[str, Any], baseline_item: dict[str, Any] | None) -> None:
    current_price = enriched_item.get("current_price")
    baseline_price = baseline_item.get("price") if isinstance(baseline_item, dict) else None
    if baseline_price is None or current_price is None:
        enriched_item["baseline_price"] = None
        enriched_item["baseline_price_date"] = None
        enriched_item["cumulative_change_pct"] = None
        return
    try:
        base_val = float(baseline_price)
        current_val = float(current_price)
    except (TypeError, ValueError):
        enriched_item["baseline_price"] = None
        enriched_item["baseline_price_date"] = None
        enriched_item["cumulative_change_pct"] = None
        return
    if base_val <= 0:
        enriched_item["baseline_price"] = None
        enriched_item["baseline_price_date"] = None
        enriched_item["cumulative_change_pct"] = None
        return
    enriched_item["baseline_price"] = base_val
    enriched_item["baseline_price_date"] = baseline_item.get("date") if isinstance(baseline_item, dict) else None
    enriched_item["cumulative_change_pct"] = ((current_val / base_val) - 1.0) * 100.0


def _clear_price_fields(enriched_item: dict[str, Any], *, preserve_existing: bool) -> None:
    if preserve_existing:
        return
    enriched_item["current_price"] = None
    enriched_item["previous_close"] = None
    enriched_item["change_pct"] = None
    enriched_item["price_currency"] = None
    enriched_item["price_as_of_date"] = None
