from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from services.price_service import get_realtime_snapshot, get_worldstock_snapshot, get_yahoo_symbol_snapshot
from utils.cache_utils import load_cached_close_series_bulk_with_fallback
from utils.formatters import clean_holding_display_name
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
    if normalized.endswith((".SZ", ".SS")):
        return "CNY"
    if normalized.endswith(".AX"):
        return "AUD"
    return "USD"


def enrich_component_prices(
    holdings: Iterable[dict[str, Any]],
    *,
    price_fetch_limit: int | None,
    preserve_existing: bool = False,
    cumulative_base_date: str | None = None,
    component_price_snapshot: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], str | None]:
    """ETF 구성종목에 현재가/등락률/통화 정보를 붙인다."""
    holdings_list = [dict(item) for item in holdings]
    if not holdings_list:
        return [], None

    component_price_snapshot = component_price_snapshot or {}
    holdings_for_pricing = select_component_holdings_for_pricing(holdings_list, price_fetch_limit)
    pricing_ids = {id(item) for item in holdings_for_pricing}

    korean_tickers: list[str] = []
    korean_baseline_tickers: list[str] = []
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
        # 호주: ASX: 접두사를 가장 먼저 인식해 호주 시장으로 라우팅 (docs/developer_guide 참조).
        # ASX:TECH 형태가 시스템 표준이고 .AX 는 yahoo 심볼 등 외부에서 들어오는 변형 형태이다.
        if component_ticker.startswith("ASX:"):
            if _component_price_key(item) not in component_price_snapshot:
                au_tickers.append(component_ticker[4:])
            continue

        if _is_korean_six_digit_holding(item):
            if cumulative_base_date:
                korean_baseline_tickers.append(component_ticker)
            if _component_price_key(item) not in component_price_snapshot:
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
            if _component_price_key(item) not in component_price_snapshot:
                au_tickers.append(yahoo_symbol[:-3])
        elif _is_worldstock_symbol(yahoo_symbol):
            if _component_price_key(item) not in component_price_snapshot:
                worldstock_codes.append(_normalize_upper(item.get("reuters_code")) or yahoo_symbol)
        elif _is_yahoo_exchange_symbol(yahoo_symbol):
            if _component_price_key(item) not in component_price_snapshot:
                yahoo_exchange_symbols.append(yahoo_symbol)
        elif "." not in yahoo_symbol:
            if _component_price_key(item) not in component_price_snapshot:
                us_tickers.append(yahoo_symbol)

    kor_price_map = _safe_fetch_snapshot("kor", sorted(set(korean_tickers)))
    us_price_map = _safe_fetch_snapshot("us", sorted(set(us_tickers)))
    au_price_map = _safe_fetch_snapshot("au", sorted(set(au_tickers)))
    worldstock_price_map = _safe_fetch_worldstock(sorted(set(worldstock_codes)))
    yahoo_exchange_price_map = _safe_fetch_yahoo(sorted(set(yahoo_exchange_symbols)))
    korean_baseline_price_map = _safe_fetch_cached_baseline_prices(
        "kor", korean_baseline_tickers, cumulative_base_date
    )
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
        # 표시 이름 정제 (예: "AMD(어드밴스드 마이크로 디바이시스)" → "AMD").
        # 원본은 raw_name 에 보존되어 있다.
        enriched_item["name"] = clean_holding_display_name(item.get("name"))

        if _is_cash_like_holding(item):
            _clear_price_fields(enriched_item, preserve_existing=preserve_existing)
            enriched_item["price_currency"] = "KRW"
        elif id(item) not in pricing_ids:
            _clear_price_fields(enriched_item, preserve_existing=preserve_existing)
        elif component_ticker and component_ticker.startswith("ASX:"):
            # 호주 ASX: 접두사 — 시스템 표준 호주 시장 식별자.
            _apply_price_snapshot(
                enriched_item,
                au_price_map.get(component_ticker[4:], {}),
                "AUD",
                preserve_existing=preserve_existing,
            )
        elif _component_price_key(item) in component_price_snapshot:
            snapshot = component_price_snapshot[_component_price_key(item)]
            currency = str(snapshot.get("currency") or _infer_component_currency(item)).strip().upper()
            _apply_price_snapshot(
                enriched_item,
                snapshot,
                currency,
                preserve_existing=preserve_existing,
            )
            if cumulative_base_date and _is_korean_six_digit_holding(item):
                _apply_cumulative_change(enriched_item, korean_baseline_price_map.get(component_ticker))
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
            if _is_baseline_suspicious(enriched_item, baseline_item):
                # Yahoo 가 간헐적으로 가격을 10배 등 잘못된 스케일로 반환하는 glitch.
                # (관측: 2026-06-12 KLAC baseline $2,411 vs 실제 $241 — 미국 그룹 -90% 오염)
                # baseline 폐기 — 누적 변동은 None 으로 표시 (fallback 없음).
                logger.warning(
                    "구성종목 baseline 이상치 폐기: %s baseline=%s previous_close=%s",
                    yahoo_symbol,
                    baseline_item.get("price") if isinstance(baseline_item, dict) else None,
                    enriched_item.get("previous_close"),
                )
                baseline_item = None
            _apply_cumulative_change(enriched_item, baseline_item)

        as_of_date = str(enriched_item.get("price_as_of_date") or "").strip()
        if as_of_date:
            price_as_of_dates.add(as_of_date)
        enriched.append(enriched_item)

    price_as_of = max(price_as_of_dates) if price_as_of_dates else None
    return enriched, price_as_of


def select_component_holdings_for_pricing(
    holdings: Iterable[dict[str, Any]],
    price_fetch_limit: int | None,
) -> list[dict[str, Any]]:
    """비중 상위 구성종목만 가격 조회 대상으로 선택한다.

    price_fetch_limit=None 이면 전체 holdings 를 가격 조회 대상으로 사용한다
    (보유 상세 화면처럼 전 종목 가격이 필요한 케이스).
    """
    holdings_list = list(holdings)
    if price_fetch_limit is None:
        return holdings_list
    if price_fetch_limit <= 0:
        return []
    return sorted(holdings_list, key=_weight_value, reverse=True)[:price_fetch_limit]


def build_component_price_snapshot(
    holdings: Iterable[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """중복 구성종목을 한 번만 조회해 공통 가격 스냅샷을 만든다."""
    holdings_list = [dict(item) for item in holdings]
    if not holdings_list:
        return {}

    korean_tickers: set[str] = set()
    us_tickers: set[str] = set()
    au_tickers: set[str] = set()
    worldstock_codes: set[str] = set()
    yahoo_exchange_symbols: set[str] = set()

    for item in holdings_list:
        if _is_cash_like_holding(item):
            continue
        component_ticker = _normalize_upper(item.get("ticker"))
        yahoo_symbol = _normalize_upper(item.get("yahoo_symbol")) or component_ticker
        if _is_korean_six_digit_holding(item):
            korean_tickers.add(component_ticker)
        elif yahoo_symbol.endswith(".AX"):
            au_tickers.add(yahoo_symbol[:-3])
        elif _is_worldstock_symbol(yahoo_symbol):
            worldstock_codes.add(_normalize_upper(item.get("reuters_code")) or yahoo_symbol)
        elif _is_yahoo_exchange_symbol(yahoo_symbol):
            yahoo_exchange_symbols.add(yahoo_symbol)
        elif yahoo_symbol and "." not in yahoo_symbol:
            us_tickers.add(yahoo_symbol)

    kor_price_map = _safe_fetch_snapshot("kor", sorted(korean_tickers))
    us_price_map = _safe_fetch_snapshot("us", sorted(us_tickers))
    au_price_map = _safe_fetch_snapshot("au", sorted(au_tickers))
    worldstock_price_map = _safe_fetch_worldstock(sorted(worldstock_codes))
    yahoo_exchange_price_map = _safe_fetch_yahoo(sorted(yahoo_exchange_symbols))

    snapshot: dict[str, dict[str, Any]] = {}
    for item in holdings_list:
        key = _component_price_key(item)
        if not key or key in snapshot:
            continue
        component_ticker = _normalize_upper(item.get("ticker"))
        yahoo_symbol = _normalize_upper(item.get("yahoo_symbol")) or component_ticker
        price_item: dict[str, Any] | None = None
        currency = _infer_component_currency(item)

        if _is_korean_six_digit_holding(item):
            price_item = kor_price_map.get(component_ticker)
            currency = "KRW"
        elif yahoo_symbol.endswith(".AX"):
            price_item = au_price_map.get(yahoo_symbol[:-3])
            currency = "AUD"
        elif _is_worldstock_symbol(yahoo_symbol):
            lookup_code = _normalize_upper(item.get("reuters_code")) or yahoo_symbol
            price_item = worldstock_price_map.get(lookup_code)
            currency = str((price_item or {}).get("currency") or infer_yahoo_symbol_currency(yahoo_symbol))
        elif _is_yahoo_exchange_symbol(yahoo_symbol):
            price_item = yahoo_exchange_price_map.get(yahoo_symbol)
            currency = infer_yahoo_symbol_currency(yahoo_symbol)
        elif yahoo_symbol and "." not in yahoo_symbol:
            price_item = us_price_map.get(yahoo_symbol)
            currency = "USD"

        if not price_item:
            continue
        snapshot[key] = {**price_item, "currency": currency}

    return snapshot


def _weight_value(item: dict[str, Any]) -> float:
    try:
        return float(item.get("weight") or item.get("total_weight") or 0.0)
    except (TypeError, ValueError):
        return 0.0


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
    return _normalize_upper(symbol).endswith((".TW", ".L", ".SZ", ".SS"))


def _is_us_yahoo_symbol(symbol: str) -> bool:
    normalized = _normalize_upper(symbol)
    return bool(normalized) and "." not in normalized


def _component_price_key(item: dict[str, Any]) -> str | None:
    if _is_cash_like_holding(item):
        return None
    component_ticker = _normalize_upper(item.get("ticker"))
    if _is_korean_six_digit_holding(item):
        return f"kor:{component_ticker}"

    yahoo_symbol = _normalize_upper(item.get("yahoo_symbol")) or component_ticker
    if not yahoo_symbol:
        return None
    if yahoo_symbol.endswith(".AX"):
        return f"au:{yahoo_symbol[:-3]}"
    if _is_worldstock_symbol(yahoo_symbol):
        return f"world:{_normalize_upper(item.get('reuters_code')) or yahoo_symbol}"
    if _is_yahoo_exchange_symbol(yahoo_symbol):
        return f"yahoo:{yahoo_symbol}"
    if "." in yahoo_symbol:
        return None
    return f"us:{yahoo_symbol}"


def _infer_component_currency(item: dict[str, Any]) -> str:
    component_ticker = _normalize_upper(item.get("ticker"))
    yahoo_symbol = _normalize_upper(item.get("yahoo_symbol")) or component_ticker
    if _is_korean_six_digit_holding(item):
        return "KRW"
    if yahoo_symbol.endswith(".AX"):
        return "AUD"
    return infer_yahoo_symbol_currency(yahoo_symbol)


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

    missing = [t for t in normalized_tickers if t not in result]
    if missing:
        pykrx_result = _fetch_pykrx_baseline_prices(missing, base_date)
        result.update(pykrx_result)

    return result


def _fetch_pykrx_baseline_prices(
    tickers: list[str],
    base_date: str,
) -> dict[str, dict[str, Any]]:
    try:
        from pykrx import stock as _stock
    except ImportError:
        return {}

    base_ts = pd.Timestamp(base_date).normalize()
    start_ts = base_ts - pd.Timedelta(days=10)
    start_str = start_ts.strftime("%Y%m%d")
    end_str = base_ts.strftime("%Y%m%d")

    result: dict[str, dict[str, Any]] = {}
    for ticker in tickers:
        try:
            df = _stock.get_market_ohlcv_by_date(start_str, end_str, ticker)
            if df is None or df.empty:
                continue
            close_col = "종가" if "종가" in df.columns else "Close" if "Close" in df.columns else None
            if close_col is None:
                continue
            close_series = pd.to_numeric(df[close_col], errors="coerce").dropna()
            close_series = close_series[close_series > 0]
            if close_series.empty:
                continue
            result[ticker] = {
                "price": float(close_series.iloc[-1]),
                "date": pd.Timestamp(close_series.index[-1]).strftime("%Y-%m-%d"),
            }
        except Exception as exc:
            logger.warning("pykrx 기준일 가격 조회 실패(ticker=%s): %s", ticker, exc)
    return result


import threading

_YAHOO_BASELINE_CACHE: dict[tuple[str, str, bool], dict[str, Any]] = {}
_YAHOO_BASELINE_LOCK = threading.Lock()

# mongodb 영속 캐시 컬렉션명. baseline 은 "특정 base_date 의 종가" 라 한 번 구하면 불변 —
# 프로세스 재시작/배포 후에도 yfinance 재호출 없이 재사용한다.
_YAHOO_BASELINE_COLLECTION = "yahoo_baseline_prices"


def _baseline_doc_id(symbol: str, base_date: str, is_prev: bool) -> str:
    return f"{symbol}|{base_date}|{int(is_prev)}"


def _load_persisted_yahoo_baselines(
    keys: list[tuple[str, str, bool]],
) -> dict[tuple[str, str, bool], dict[str, Any]]:
    """mongodb 에서 (symbol, base_date, is_prev) 키들의 baseline 을 일괄 조회한다."""
    if not keys:
        return {}
    try:
        from utils.db_manager import get_db_connection

        db = get_db_connection()
        if db is None:
            return {}
        ids = [_baseline_doc_id(*key) for key in keys]
        cursor = db[_YAHOO_BASELINE_COLLECTION].find(
            {"_id": {"$in": ids}},
            {"symbol": 1, "base_date": 1, "is_prev": 1, "price": 1, "date": 1},
        )
        result: dict[tuple[str, str, bool], dict[str, Any]] = {}
        for doc in cursor:
            key = (str(doc.get("symbol")), str(doc.get("base_date")), bool(doc.get("is_prev")))
            price = doc.get("price")
            if price is None:
                continue
            result[key] = {"price": float(price), "date": doc.get("date")}
        return result
    except Exception as exc:
        logger.warning("yahoo baseline mongodb 조회 실패: %s", exc)
        return {}


def _persist_yahoo_baselines(entries: dict[tuple[str, str, bool], dict[str, Any]]) -> None:
    """yfinance 로 새로 받은 baseline 을 mongodb 에 upsert 한다 (실패해도 동작에는 영향 없음)."""
    if not entries:
        return
    try:
        from pymongo import UpdateOne

        from utils.db_manager import get_db_connection

        db = get_db_connection()
        if db is None:
            return
        now = datetime.now()
        ops = [
            UpdateOne(
                {"_id": _baseline_doc_id(symbol, base_date, is_prev)},
                {
                    "$set": {
                        "symbol": symbol,
                        "base_date": base_date,
                        "is_prev": is_prev,
                        "price": data.get("price"),
                        "date": data.get("date"),
                        "updated_at": now,
                    }
                },
                upsert=True,
            )
            for (symbol, base_date, is_prev), data in entries.items()
        ]
        db[_YAHOO_BASELINE_COLLECTION].bulk_write(ops, ordered=False)
    except Exception as exc:
        logger.warning("yahoo baseline mongodb 저장 실패: %s", exc)


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

        from utils.symbol_resolution_blacklist import get_active_blacklist, mark_failed
        blacklist = get_active_blacklist()

        for symbol in normalized_symbols:
            is_prev = symbol in previous_day_symbols
            cache_key = (symbol, base_date, is_prev)
            if cache_key in _YAHOO_BASELINE_CACHE:
                result[symbol] = _YAHOO_BASELINE_CACHE[cache_key]
            elif symbol in blacklist:
                # 24시간 내 yfinance/토스에서 실패한 심볼은 재시도하지 않음
                continue
            else:
                symbols_to_fetch.append(symbol)

        # 메모리 캐시 miss 분은 mongodb 영속 캐시에서 일괄 조회 — 프로세스 재시작 후에도
        # yfinance 재호출 없이 복원된다 (cold start 시 30초 timeout 의 주범 제거).
        if symbols_to_fetch:
            persisted = _load_persisted_yahoo_baselines(
                [(symbol, base_date, symbol in previous_day_symbols) for symbol in symbols_to_fetch]
            )
            if persisted:
                still_missing: list[str] = []
                for symbol in symbols_to_fetch:
                    key = (symbol, base_date, symbol in previous_day_symbols)
                    data = persisted.get(key)
                    if data is not None:
                        result[symbol] = data
                        _YAHOO_BASELINE_CACHE[key] = data
                    else:
                        still_missing.append(symbol)
                symbols_to_fetch = still_missing

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

        newly_fetched: dict[tuple[str, str, bool], dict[str, Any]] = {}
        for symbol in symbols_to_fetch:
            frame = _extract_symbol_frame(symbol)
            if frame is None or frame.empty or "Close" not in frame.columns:
                # yfinance 가 반환하지 않은 심볼은 상폐/미상장 가능성 → 블랙리스트
                mark_failed(symbol, source="야후", reason="yfinance: 가격 데이터 없음")
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
                mark_failed(symbol, source="야후", reason="yfinance: base_date 이전 종가 없음")
                continue
            data = {
                "price": float(close_series.iloc[-1]),
                "date": pd.Timestamp(close_series.index[-1]).strftime("%Y-%m-%d"),
            }
            result[symbol] = data
            _YAHOO_BASELINE_CACHE[(symbol, base_date, is_prev)] = data
            # 당일 종가는 장중/마감 직후 잠정치일 수 있어 영속화하지 않는다 (메모리 캐시만).
            # (관측: 6981.T 당일 잠정 종가 9,454 가 영속화돼 확정 종가 8,556 과 10% 괴리)
            today_kst = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d")
            if str(data.get("date") or "") < today_kst:
                newly_fetched[(symbol, base_date, is_prev)] = data

        _persist_yahoo_baselines(newly_fetched)
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


def _is_baseline_suspicious(enriched_item: dict[str, Any], baseline_item: dict[str, Any] | None) -> bool:
    """Yahoo baseline 의 스케일 오류(10배 등) 를 토스 previous_close 와 교차 검증으로 감지한다.

    baseline 과 전일 종가가 40% 이상 괴리하면서 당일 변동률은 정상 범위(<15%)면
    데이터 오류로 판단한다. 일간 변동이 ±15% 이상인 급변동 장에서는 진짜 괴리일 수
    있으므로 판정을 보류한다 (베이스라인 유지).
    """
    if not isinstance(baseline_item, dict):
        return False
    try:
        baseline_price = float(baseline_item.get("price"))
        previous_close = float(enriched_item.get("previous_close"))
    except (TypeError, ValueError):
        return False
    if baseline_price <= 0 or previous_close <= 0:
        return False
    change_pct = enriched_item.get("change_pct")
    try:
        if change_pct is not None and abs(float(change_pct)) >= 15.0:
            return False
    except (TypeError, ValueError):
        pass
    ratio = baseline_price / previous_close
    return not (0.6 <= ratio <= 1.67)


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
