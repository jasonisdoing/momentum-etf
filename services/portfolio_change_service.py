"""ETF 포트폴리오 변동 계산 공통 서비스.

/ticker, /compare, /holdings 화면이 동일한 base_date 와 캐시 결과를 공유한다.
"""

from __future__ import annotations

import logging
import math
import threading
from datetime import date, datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from services.component_price_service import enrich_component_prices
from services.price_service import get_exchange_rate_series, get_exchange_rates
from services.stock_cache_service import get_stock_cache_meta, refresh_stock_portfolio_change_cache
from utils.data_loader import fetch_naver_etf_inav_snapshot
from utils.stock_cache_meta_io import get_previous_stock_cache_meta_history

logger = logging.getLogger(__name__)

_HOLDINGS_PRICE_FETCH_LIMIT = 100
_TTL_SECONDS = 300
_PORTFOLIO_CHANGE_CALC_VERSION = 4  # 합계 계산을 base_date 누적(cumulative_change_pct)로 전환

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


def _build_snapshot_from_cached_holdings(holdings: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """holdings_cache.items 에 component_prices_updater 가 미리 박아둔 가격을 snapshot 으로 변환.

    enrich_component_prices 의 component_price_snapshot 인자에 그대로 전달하면 외부 API
    호출이 우회된다. current_price 가 없는 항목은 snapshot 에 들어가지 않으므로 그런
    항목만 enrich 가 외부에서 조회한다 (혼합 동작 안전).
    """
    from services.component_price_service import _component_price_key, _is_cash_like_holding

    snapshot: dict[str, dict[str, Any]] = {}
    for item in holdings:
        if not isinstance(item, dict):
            continue
        if _is_cash_like_holding(item):
            continue
        if item.get("current_price") is None:
            continue
        key = _component_price_key(item)
        if not key:
            continue
        entry: dict[str, Any] = {"nowVal": item.get("current_price")}
        if item.get("previous_close") is not None:
            entry["prevClose"] = item.get("previous_close")
        if item.get("change_pct") is not None:
            entry["changeRate"] = item.get("change_pct")
        currency = item.get("price_currency")
        if currency:
            entry["currency"] = currency
        if item.get("price_as_of_date"):
            entry["as_of_date"] = item.get("price_as_of_date")
        snapshot[key] = entry
    return snapshot


def _cache_key(ticker_type: str, ticker: str) -> str:
    return f"{(ticker_type or '').strip().lower()}:{(ticker or '').strip().upper()}"


def _to_jsonable(value: Any) -> Any:
    """MongoDB 저장이 가능한 기본 타입으로 변환한다."""
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if value is not None and not isinstance(value, (str, bytes)):
        try:
            if bool(pd.isna(value)):
                return None
        except (TypeError, ValueError):
            pass
    if hasattr(value, "item"):
        try:
            return _to_jsonable(value.item())
        except Exception:
            pass
    return value


def _is_cache_alive(entry: dict[str, Any], now: datetime) -> bool:
    expires = entry.get("expires_at")
    return isinstance(expires, datetime) and now < expires


def _to_utc_naive(value: Any) -> datetime | None:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _is_persisted_cache_alive(cache_doc: dict[str, Any], now: datetime) -> bool:
    updated_at = _to_utc_naive(cache_doc.get("portfolio_change_cache_updated_at"))
    if updated_at is None:
        return False
    return now - updated_at < timedelta(seconds=_TTL_SECONDS)


def _is_portfolio_change_cache_usable(
    cache_data: Any,
    holdings_reference_date: str | None,
    expected_base_date: str | None = None,
) -> bool:
    """계산 실패로 저장된 포트폴리오 변동 캐시는 재계산 대상이다.

    expected_base_date 가 주어지면 캐시의 base_date 와 일치할 때만 재사용한다.
    (휴장일 캘린더 수정 등으로 base_date 가 변경됐을 때 stale 캐시 회피)
    """
    if not isinstance(cache_data, dict):
        return False
    if cache_data.get("calc_version") != _PORTFOLIO_CHANGE_CALC_VERSION:
        return False
    cached_base_date = cache_data.get("base_date")
    if not cached_base_date:
        return False
    if expected_base_date is not None and cached_base_date != expected_base_date:
        return False
    if cache_data.get("holdings_reference_date") != holdings_reference_date:
        return False
    if cache_data.get("total_pct") is None:
        return False
    return not _has_missing_korean_component_cumulative_change(cache_data)


def _build_storable_result(result: dict[str, Any], source: str) -> dict[str, Any]:
    return _to_jsonable(
        {
            **result,
            "updated_at": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
            "source": source,
        }
    )


def _store_portfolio_change_result(
    ticker_type: str,
    ticker: str,
    result: dict[str, Any],
    *,
    source: str,
) -> dict[str, Any]:
    stored = _build_storable_result(result, source)
    refresh_stock_portfolio_change_cache(ticker_type, ticker, stored)

    key = _cache_key(ticker_type, ticker)
    with _PORTFOLIO_CHANGE_LOCK:
        _PORTFOLIO_CHANGE_CACHE[key] = {
            "data": stored,
            "expires_at": datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(seconds=_TTL_SECONDS),
        }
    return stored


def _has_missing_korean_component_cumulative_change(cache_data: dict[str, Any]) -> bool:
    """국내 6자리 구성종목의 기준가 계산이 누락된 기존 캐시는 폐기한다."""
    priced_holdings = cache_data.get("priced_holdings")
    if not isinstance(priced_holdings, list):
        return False

    for item in priced_holdings:
        if not isinstance(item, dict):
            continue
        ticker = str(item.get("ticker") or "").strip()
        if not ticker.isdigit() or len(ticker) != 6:
            continue
        try:
            weight = float(item.get("weight") or 0)
        except (TypeError, ValueError):
            continue
        if weight <= 0:
            continue
        if str(item.get("price_currency") or "").strip().upper() != "KRW":
            continue
        if item.get("current_price") is None:
            continue
        if item.get("cumulative_change_pct") is None:
            return True
    return False


def _is_trading_day_kor(date_str: str) -> bool:
    """date_str(YYYY-MM-DD)가 한국 거래일인지 캘린더로 확인한다. 실패 시 True 로 간주(보수적)."""
    try:
        from utils.data_loader import get_trading_days

        days = get_trading_days(date_str, date_str, "kor")
        return bool(days)
    except Exception:
        # 캘린더 조회 실패 시 보정 로직을 건너뛰어 기존 동작 유지
        return True


def _resolve_base_date_to_trading_day(ticker_type: str, ticker: str, raw_date: str) -> str | None:
    """히스토리에 기록된 raw_date 가 휴장일이면 그 이전 거래일로 보정한다.

    캘린더가 갱신되어 과거에 거래일로 잘못 기록된 스냅샷이 남아 있는 경우,
    화면 표시·계산이 잘못된 날짜로 진행되는 것을 방지한다.
    """
    candidate = raw_date
    for _ in range(7):  # 최대 7번까지 거슬러 올라가며 거래일 탐색
        if not candidate:
            return None
        if _is_trading_day_kor(candidate):
            return candidate
        prev = get_previous_stock_cache_meta_history(ticker_type, ticker, candidate)
        if not prev:
            return None
        candidate = str(prev.get("date") or "").strip()
    return candidate or None


def determine_portfolio_change_base_date(ticker_type: str, ticker: str) -> str | None:
    """오늘 자정+1일을 넘겨 오늘 히스토리도 포함한 가장 최근 기준일을 반환한다.

    한국 종목인 경우 반환 날짜가 휴장일이면 그 이전 거래일로 보정한다.
    """
    today_dt = datetime.now(ZoneInfo("Asia/Seoul"))
    tomorrow_str = (today_dt + timedelta(days=1)).strftime("%Y-%m-%d")
    hist = get_previous_stock_cache_meta_history(ticker_type, ticker, tomorrow_str)
    if not hist:
        return None
    raw_date = str(hist.get("date") or "").strip()
    if not raw_date:
        return None
    # 한국 종목 풀만 캘린더 보정 (다른 국가는 별도 캘린더라 영향 없음)
    if str(ticker_type or "").strip().lower().startswith("kor"):
        return _resolve_base_date_to_trading_day(ticker_type, ticker, raw_date)
    return raw_date


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


def build_daily_fx_rates(
    holdings: list[dict[str, Any]],
    rates: dict[str, Any],
) -> list[dict[str, Any]]:
    """구성종목 통화별 최신 일간 환율 변동률을 구성한다."""
    currencies: set[str] = set()
    for h in holdings:
        currency = str(h.get("price_currency") or "").strip().upper()
        if currency and currency != "KRW":
            currencies.add(currency)

    result: list[dict[str, Any]] = []
    for item in _build_fx_rates_for_currencies(currencies, rates):
        currency = str(item.get("currency") or "").strip().upper()
        info = rates.get(currency)
        if not isinstance(info, dict):
            continue
        result.append(
            {
                **item,
                "change_pct": info.get("change_pct"),
            }
        )
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
    """기준일 이후 누적 포트폴리오 변동률을 계산한다."""
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


def _calc_realtime_portfolio_change(
    holdings: list[dict[str, Any]],
    fx_rates: list[dict[str, Any]],
) -> tuple[float | None, list[dict[str, Any]], float, float | None]:
    """구성종목 일간 변동과 환율 변동을 합산한 포트폴리오 변동률을 계산한다.

    ETF의 공식 iNAV 변동률은 별도 지표로 보관하지만 여기서는 차감하지 않는다.
    이 값은 전일 기준 구성종목과 환율만으로 추정한 포트폴리오 변동이다.
    """
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
        # base_date 이후 누적 변동을 사용해야 한국 휴장일에도 "직전 거래일 종가 vs 현재가" 가
        # 정확히 반영된다. cumulative_change_pct 가 있으면 우선 사용하고, 없으면 일간(change_pct)
        # 으로 폴백한다. (cumulative_change_pct 는 yahoo baseline 과 토스 현재가의 비율로
        # 계산되어 base_date 종가 → 현재 시점 누적률을 의미한다.)
        comp = h.get("cumulative_change_pct")
        if comp is None:
            comp = h.get("change_pct")
        if comp is None:
            continue
        try:
            comp_val = float(comp)
        except (TypeError, ValueError):
            continue

        currency = str(h.get("price_currency") or "").strip().upper() or "KRW"
        if currency != "KRW" and currency not in fx_change_by_currency:
            continue

        g = groups.setdefault(currency, {"weight": 0.0, "component_weighted_sum": 0.0})
        g["weight"] += weight
        g["component_weighted_sum"] += weight * comp_val

    coverage = 0.0
    gross_weighted_sum = 0.0
    breakdown: list[dict[str, Any]] = []
    for currency, g in groups.items():
        if g["weight"] <= 0:
            continue
        component_change = g["component_weighted_sum"] / g["weight"]
        adjusted_change = component_change
        if currency != "KRW":
            fx_change = fx_change_by_currency[currency]
            adjusted_change = ((1 + component_change / 100) * (1 + fx_change / 100) - 1) * 100

        breakdown.append({"currency": currency, "change_pct": component_change, "weight": g["weight"]})
        coverage += g["weight"]
        gross_weighted_sum += g["weight"] * adjusted_change

    breakdown.sort(key=lambda x: -x["weight"])

    if coverage <= 0:
        return None, breakdown, 0.0, None

    gross_pct = gross_weighted_sum / 100
    return gross_pct, breakdown, coverage, gross_pct


def _resolve_nav_change_pct(ticker_type: str, ticker: str, current_nav: Any) -> float | None:
    """네이버 현재 iNAV와 메타 히스토리로 공식 iNAV 장중 변동률을 계산한다."""
    try:
        nav_value = float(current_nav)
    except (TypeError, ValueError):
        return None
    if nav_value <= 0:
        return None

    today_dt = datetime.now(ZoneInfo("Asia/Seoul"))
    tomorrow_str = (today_dt + timedelta(days=1)).strftime("%Y-%m-%d")
    latest_history = get_previous_stock_cache_meta_history(ticker_type, ticker, tomorrow_str)
    if not latest_history or "meta_cache" not in latest_history:
        return None

    latest_history_nav = latest_history["meta_cache"].get("nav")
    raw_base_date = str(latest_history.get("date") or "").strip() or None
    # 한국 종목 풀: base_date 가 휴장일이면 직전 거래일로 보정
    if raw_base_date and str(ticker_type or "").strip().lower().startswith("kor"):
        portfolio_change_base_date = _resolve_base_date_to_trading_day(ticker_type, ticker, raw_base_date)
    else:
        portfolio_change_base_date = raw_base_date
    prev_nav = None
    try:
        if (
            latest_history_nav is not None
            and float(nav_value) == float(latest_history_nav)
            and portfolio_change_base_date
        ):
            prev_history = get_previous_stock_cache_meta_history(ticker_type, ticker, portfolio_change_base_date)
            if prev_history and "meta_cache" in prev_history:
                prev_nav = prev_history["meta_cache"].get("nav")
        else:
            prev_nav = latest_history_nav
    except (TypeError, ValueError):
        return None

    try:
        prev_nav_value = float(prev_nav)
    except (TypeError, ValueError):
        return None
    if prev_nav_value <= 0:
        return None

    return ((nav_value / prev_nav_value) - 1.0) * 100.0


def compute_portfolio_change_bundle(
    ticker: str,
    ticker_type: str,
    *,
    use_cache: bool = True,
    component_price_snapshot: dict[str, dict[str, Any]] | None = None,
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
    now = datetime.now(timezone.utc).replace(tzinfo=None)

    cache_doc = get_stock_cache_meta(norm_type, norm_ticker)
    if not isinstance(cache_doc, dict):
        return None
    holdings_cache = dict(cache_doc.get("holdings_cache") or {})
    holdings = list(holdings_cache.get("items") or [])
    if not holdings:
        return None
    holdings_reference_date = str(holdings_cache.get("reference_date") or "").strip() or None

    # 캐시 검증 단계에서 base_date 변경 여부를 확인하기 위해 먼저 결정한다.
    # 휴장일 캘린더 수정 등으로 base_date 가 바뀐 경우 stale 캐시를 회피한다.
    base_date = determine_portfolio_change_base_date(norm_type, norm_ticker)
    if not base_date:
        return None

    if use_cache:
        with _PORTFOLIO_CHANGE_LOCK:
            cached = _PORTFOLIO_CHANGE_CACHE.get(key)
            if cached and _is_cache_alive(cached, now):
                cached_data = cached.get("data")
                if _is_portfolio_change_cache_usable(cached_data, holdings_reference_date, base_date):
                    return cached_data
                _PORTFOLIO_CHANGE_CACHE.pop(key, None)

        persisted = cache_doc.get("portfolio_change_cache")
        if _is_portfolio_change_cache_usable(
            persisted, holdings_reference_date, base_date
        ) and _is_persisted_cache_alive(cache_doc, now):
            with _PORTFOLIO_CHANGE_LOCK:
                _PORTFOLIO_CHANGE_CACHE[key] = {
                    "data": persisted,
                    "expires_at": now + timedelta(seconds=_TTL_SECONDS),
                }
            return persisted

    # holdings_cache.items 에 component_prices_updater 배치가 미리 채워둔 가격이 있으면
    # 그것을 snapshot 으로 전달해 enrich 가 외부 API 호출(토스/네이버/야후 등)을 우회한다.
    # 호출자가 명시적으로 snapshot 을 전달했으면 그것을 우선한다 (예: 같은 화면 안에서
    # 가격을 한 번만 갱신하고 여러 ETF 에 분배하는 케이스).
    effective_snapshot = (
        component_price_snapshot
        if component_price_snapshot is not None
        else _build_snapshot_from_cached_holdings(holdings)
    )
    priced_holdings, _ = enrich_component_prices(
        holdings,
        price_fetch_limit=_HOLDINGS_PRICE_FETCH_LIMIT,
        cumulative_base_date=base_date,
        component_price_snapshot=effective_snapshot,
    )
    rates = get_exchange_rates()
    # 합계 계산은 base_date 이후 누적 변동을 사용하므로 환율도 누적률을 적용한다.
    # 누적 환율 조회 실패 시(휴장/네트워크 문제) 일간 환율로 폴백.
    cumulative_fx = build_cumulative_fx_rates(priced_holdings, rates, base_date)
    daily_fx = build_daily_fx_rates(priced_holdings, rates)
    fx_rates_for_calc = cumulative_fx if cumulative_fx else daily_fx
    inav_snapshot = fetch_naver_etf_inav_snapshot([norm_ticker]).get(norm_ticker, {})
    nav_change_pct = _resolve_nav_change_pct(norm_type, norm_ticker, inav_snapshot.get("nav"))
    total_pct, breakdown, coverage, gross_portfolio_pct = _calc_realtime_portfolio_change(
        priced_holdings,
        fx_rates_for_calc,
    )
    # 화면 표시용 fx_rates 는 일간 변동률 유지 (기존 UI 호환)
    fx_rates = daily_fx

    result = {
        "calc_version": _PORTFOLIO_CHANGE_CALC_VERSION,
        "base_date": base_date,
        "priced_holdings": priced_holdings,
        "fx_rates": fx_rates,
        "total_pct": total_pct,
        "gross_portfolio_pct": gross_portfolio_pct,
        "inav_change_pct": nav_change_pct,
        "breakdown": breakdown,
        "coverage_weight": coverage,
        "holdings_reference_date": holdings_reference_date,
    }

    if use_cache:
        return _store_portfolio_change_result(norm_type, norm_ticker, result, source="realtime_ttl")

    with _PORTFOLIO_CHANGE_LOCK:
        _PORTFOLIO_CHANGE_CACHE[key] = {
            "data": result,
            "expires_at": now + timedelta(seconds=_TTL_SECONDS),
        }

    return result


def compute_and_store_portfolio_change_bundle(
    ticker: str,
    ticker_type: str,
    *,
    component_price_snapshot: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """포트폴리오 변동을 새로 계산해 stock_cache_meta 에 저장한다."""
    result = compute_portfolio_change_bundle(
        ticker,
        ticker_type,
        use_cache=False,
        component_price_snapshot=component_price_snapshot,
    )
    if not result:
        return None

    return _store_portfolio_change_result(ticker_type, ticker, result, source="manual_refresh")


def clear_portfolio_change_cache() -> None:
    with _PORTFOLIO_CHANGE_LOCK:
        _PORTFOLIO_CHANGE_CACHE.clear()
