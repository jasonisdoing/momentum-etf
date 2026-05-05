"""ETF 포트폴리오 변동 계산 공통 서비스.

/ticker, /compare, /holdings 화면이 동일한 base_date 와 캐시 결과를 공유한다.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from services.component_price_service import enrich_component_prices
from services.price_service import get_exchange_rates, get_exchange_rate_series
from services.stock_cache_service import get_stock_cache_meta
from utils.stock_cache_meta_io import get_previous_stock_cache_meta_history

logger = logging.getLogger(__name__)

_HOLDINGS_PRICE_FETCH_LIMIT = 100
_TTL_SECONDS = 300

_PORTFOLIO_CHANGE_CACHE: dict[str, dict[str, Any]] = {}
_PORTFOLIO_CHANGE_LOCK = threading.Lock()

_FX_SYMBOL_BY_CURRENCY = {
    "USD": "KRW=X",
    "AUD": "AUDKRW=X",
    "JPY": "JPYKRW=X",
    "CNY": "CNYKRW=X",
    "TWD": "TWDKRW=X",
    "HKD": "HKDKRW=X",
    "GBP": "GBPKRW=X",
    "EUR": "EURKRW=X",
}


def _cache_key(ticker_type: str, ticker: str) -> str:
    return f"{(ticker_type or '').strip().lower()}:{(ticker or '').strip().upper()}"


def _is_cache_alive(entry: dict[str, Any], now: datetime) -> bool:
    expires = entry.get("expires_at")
    return isinstance(expires, datetime) and now < expires


def determine_portfolio_change_base_date(ticker_type: str, ticker: str) -> str | None:
    """오늘 자정+1일을 넘겨 오늘 히스토리도 포함한 가장 최근 기준일을 반환한다."""
    today_dt = datetime.now(ZoneInfo("Asia/Seoul"))
    tomorrow_str = (today_dt + timedelta(days=1)).strftime("%Y-%m-%d")
    hist = get_previous_stock_cache_meta_history(ticker_type, ticker, tomorrow_str)
    if not hist:
        return None
    return str(hist.get("date") or "").strip() or None


def _build_fx_rates_for_currencies(currencies: set[str], rates: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for currency in currencies:
        info = rates.get(currency)
        if not isinstance(info, dict):
            continue
        rate = info.get("rate")
        if rate is None:
            continue
        result.append({"currency": currency, "rate": rate})
    return result


def build_cumulative_fx_rates(
    holdings: list[dict[str, Any]],
    rates: dict[str, Any],
    base_date: str | None,
) -> list[dict[str, Any]]:
    """구성종목 통화별 기준일 이후 환율 변동률을 구성한다."""
    if not base_date:
        return []

    currencies: set[str] = set()
    for h in holdings:
        currency = str(h.get("price_currency") or "").strip().upper()
        if currency and currency != "KRW":
            currencies.add(currency)

    fx_rates = _build_fx_rates_for_currencies(currencies, rates)
    if not fx_rates:
        return []

    base_ts = pd.Timestamp(base_date).normalize()
    end_ts = datetime.now(ZoneInfo("Asia/Seoul")).replace(tzinfo=None)
    result: list[dict[str, Any]] = []
    for item in fx_rates:
        currency = str(item.get("currency") or "").strip().upper()
        current_rate = item.get("rate")
        symbol = _FX_SYMBOL_BY_CURRENCY.get(currency)
        if not symbol or current_rate is None:
            continue
        try:
            series = get_exchange_rate_series(
                (base_ts - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
                end_ts,
                symbol=symbol,
                allow_partial=True,
            )
        except Exception as exc:
            logger.warning("환율 시계열 조회 실패(symbol=%s): %s", symbol, exc)
            continue
        if series is None or series.empty:
            continue
        base_series = series[pd.to_datetime(series.index).normalize() <= base_ts]
        if base_series.empty:
            continue
        base_rate = float(base_series.iloc[-1])
        if base_rate <= 0:
            continue
        change_pct = ((float(current_rate) / base_rate) - 1.0) * 100.0
        result.append(
            {
                "currency": currency,
                "rate": current_rate,
                "change_pct": change_pct,
                "base_rate": base_rate,
            }
        )
    return result


def _calc_breakdown_and_total(
    holdings: list[dict[str, Any]],
    fx_rates: list[dict[str, Any]],
) -> tuple[float | None, list[dict[str, Any]], float]:
    """portfolio-change.ts 의 calcPortfolioChange 와 동일 로직 (Python 포팅)."""
    fx_change_by_currency: dict[str, float] = {}
    for fx in fx_rates:
        cur = str(fx.get("currency") or "").strip().upper()
        chg = fx.get("change_pct")
        if cur and chg is not None:
            try:
                fx_change_by_currency[cur] = float(chg)
            except (TypeError, ValueError):
                continue

    groups: dict[str, dict[str, float]] = {}
    for h in holdings:
        try:
            weight = float(h.get("weight") or 0)
        except (TypeError, ValueError):
            continue
        if weight <= 0:
            continue
        comp = h.get("cumulative_change_pct")
        if comp is None:
            continue
        try:
            comp_val = float(comp)
        except (TypeError, ValueError):
            continue
        currency = str(h.get("price_currency") or "").strip().upper() or "KRW"
        is_foreign = currency != "KRW"
        change_krw = comp_val
        if is_foreign:
            fx_chg = fx_change_by_currency.get(currency)
            if fx_chg is None:
                continue
            change_krw = ((1 + comp_val / 100) * (1 + fx_chg / 100) - 1) * 100

        g = groups.setdefault(currency, {"weight": 0.0, "weighted_sum": 0.0})
        g["weight"] += weight
        g["weighted_sum"] += weight * change_krw

    coverage = 0.0
    total_weighted_sum = 0.0
    breakdown: list[dict[str, Any]] = []
    for currency, g in groups.items():
        if g["weight"] <= 0:
            continue
        chg = g["weighted_sum"] / g["weight"]
        breakdown.append({"currency": currency, "change_pct": chg, "weight": g["weight"]})
        coverage += g["weight"]
        total_weighted_sum += g["weight"] * chg

    breakdown.sort(key=lambda x: -x["weight"])

    if coverage <= 0:
        return None, breakdown, 0.0
    return total_weighted_sum / 100, breakdown, coverage


def compute_portfolio_change_bundle(
    ticker: str,
    ticker_type: str,
    *,
    use_cache: bool = True,
) -> dict[str, Any] | None:
    """ETF 1개의 포트폴리오 변동 계산 결과(캐시 포함).

    반환: {
        base_date, priced_holdings, fx_rates, total_pct, breakdown, coverage_weight
    }
    """
    norm_ticker = (ticker or "").strip().upper()
    norm_type = (ticker_type or "").strip().lower()
    if not norm_ticker or not norm_type:
        return None

    key = _cache_key(norm_type, norm_ticker)
    now = datetime.now()

    if use_cache:
        with _PORTFOLIO_CHANGE_LOCK:
            cached = _PORTFOLIO_CHANGE_CACHE.get(key)
            if cached and _is_cache_alive(cached, now):
                return cached["data"]

    cache_doc = get_stock_cache_meta(norm_type, norm_ticker)
    if not isinstance(cache_doc, dict):
        return None
    holdings_cache = dict(cache_doc.get("holdings_cache") or {})
    holdings = list(holdings_cache.get("items") or [])
    if not holdings:
        return None

    base_date = determine_portfolio_change_base_date(norm_type, norm_ticker)
    if not base_date:
        return None

    priced_holdings, _ = enrich_component_prices(
        holdings,
        price_fetch_limit=_HOLDINGS_PRICE_FETCH_LIMIT,
        cumulative_base_date=base_date,
    )
    fx_rates = build_cumulative_fx_rates(priced_holdings, get_exchange_rates(), base_date)
    total_pct, breakdown, coverage = _calc_breakdown_and_total(priced_holdings, fx_rates)

    result = {
        "base_date": base_date,
        "priced_holdings": priced_holdings,
        "fx_rates": fx_rates,
        "total_pct": total_pct,
        "breakdown": breakdown,
        "coverage_weight": coverage,
    }

    with _PORTFOLIO_CHANGE_LOCK:
        _PORTFOLIO_CHANGE_CACHE[key] = {
            "data": result,
            "expires_at": now + timedelta(seconds=_TTL_SECONDS),
        }

    return result


def clear_portfolio_change_cache() -> None:
    with _PORTFOLIO_CHANGE_LOCK:
        _PORTFOLIO_CHANGE_CACHE.clear()
