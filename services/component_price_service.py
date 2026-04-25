from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from services.price_service import get_realtime_snapshot, get_worldstock_snapshot, get_yahoo_symbol_snapshot
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

    for item in holdings_for_pricing:
        component_ticker = _normalize_upper(item.get("ticker"))
        if _is_korean_six_digit_holding(item):
            korean_tickers.append(component_ticker)
            continue

        yahoo_symbol = _normalize_upper(item.get("yahoo_symbol")) or component_ticker
        if not yahoo_symbol:
            continue
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

    enriched: list[dict[str, Any]] = []
    price_as_of_dates: set[str] = set()
    for item in holdings_list:
        enriched_item = dict(item)
        component_ticker = _normalize_upper(item.get("ticker"))
        yahoo_symbol = _normalize_upper(item.get("yahoo_symbol")) or None
        enriched_item["yahoo_symbol"] = yahoo_symbol

        if id(item) not in pricing_ids:
            _clear_price_fields(enriched_item, preserve_existing=preserve_existing)
        elif _is_korean_six_digit_holding(item):
            _apply_price_snapshot(
                enriched_item,
                kor_price_map.get(component_ticker, {}),
                "KRW",
                preserve_existing=preserve_existing,
            )
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

        as_of_date = str(enriched_item.get("price_as_of_date") or "").strip()
        if as_of_date:
            price_as_of_dates.add(as_of_date)
        enriched.append(enriched_item)

    price_as_of = max(price_as_of_dates) if price_as_of_dates else None
    return enriched, price_as_of


def _normalize_upper(value: Any) -> str:
    return str(value or "").strip().upper()


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


def _clear_price_fields(enriched_item: dict[str, Any], *, preserve_existing: bool) -> None:
    if preserve_existing:
        return
    enriched_item["current_price"] = None
    enriched_item["previous_close"] = None
    enriched_item["change_pct"] = None
    enriched_item["price_currency"] = None
    enriched_item["price_as_of_date"] = None
