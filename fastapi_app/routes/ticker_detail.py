from __future__ import annotations

import threading
import time as _time
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from fastapi import APIRouter, Body, Depends, Query

from config import CACHE_TTL_LIVE, HOLDING_CHART_SHOW_AVG_BUY_PRICE, MARKET_SCHEDULES
from fastapi_app.dependencies import require_internal_token
from fastapi_app.streaming import sse_stream
from services.component_price_service import build_component_price_snapshot, enrich_component_prices
from services.portfolio_change_service import (
    build_daily_fx_rates as _build_daily_fx_rates_for_holdings,
)
from services.portfolio_change_service import (
    compute_portfolio_change_bundle,
)
from services.price_service import (
    get_exchange_rates,
    get_realtime_snapshot,
    get_realtime_snapshot_meta,
)
from services.stock_cache_service import get_stock_cache_meta
from utils.cache_utils import (
    get_cache_refresh_completed_at,
    load_cached_close_series_bulk_before_or_at_with_fallback,
    load_cached_close_series_bulk_with_fallback,
    load_cached_updated_at_bulk_before_or_at_with_fallback,
    load_cached_updated_at_bulk_with_fallback,
)
from utils.cash_model import currency_for_country
from utils.data_loader import (
    fetch_naver_etf_inav_snapshot,
    fetch_ohlcv,
    fetch_overseas_etf_nav_snapshot,
    get_latest_trading_day,
    get_trading_days,
)
from utils.kis_market import load_cached_kis_domestic_etf_master
from utils.portfolio_io import load_portfolio_master
from utils.settings_loader import list_available_accounts, load_common_settings
from utils.stock_cache_meta_io import get_previous_stock_cache_meta
from utils.stock_list_io import get_etfs
from utils.ticker_registry import load_ticker_type_configs
from utils.ticker_resolver import resolve_ticker_meta

router = APIRouter(prefix="/internal/ticker-detail", tags=["ticker-detail"])

# 비교(/compare) 결과 캐시 — **종목 단위** 짧은 TTL. 세트 단위로 캐시하면 8종목 중 7개가
# 끝나고 하나에서 실패했을 때 아무것도 남지 않아 새로고침이 전부 재계산이 됐다. 종목마다
# 끝나는 즉시 저장하므로, 실패 후 재시도는 성공분을 캐시에서 읽고 실패분만 다시 계산한다.
_COMPARE_CACHE: dict[tuple[str, str, str, bool], tuple[dict[str, object], float]] = {}
_COMPARE_CACHE_LOCK = threading.Lock()
_COMPARE_CACHE_TTL = CACHE_TTL_LIVE


def _load_us_pool_ticker_set() -> set[str]:
    return {
        str(item.get("ticker") or "").strip().upper()
        for item in get_etfs("us")
        if str(item.get("ticker") or "").strip()
    }


def _load_kor_pool_ticker_set() -> set[str]:
    return {
        str(item.get("ticker") or "").strip().upper()
        for item in get_etfs("kor")
        if str(item.get("ticker") or "").strip()
    }


def _load_domestic_etf_ticker_set() -> set[str]:
    df, _ = load_cached_kis_domestic_etf_master()
    if "티커" not in df.columns:
        raise RuntimeError("KIS ETF 마스터 캐시에 티커 컬럼이 없습니다.")
    return {str(value or "").strip().upper() for value in df["티커"].tolist() if str(value or "").strip()}


def _is_us_pool_candidate(item: dict[str, object]) -> bool:
    component_ticker = str(item.get("ticker") or "").strip().upper()
    raw_code = str(item.get("raw_code") or "").strip().upper()
    yahoo_symbol = str(item.get("yahoo_symbol") or "").strip().upper()
    price_currency = str(item.get("price_currency") or "").strip().upper()
    if not component_ticker:
        return False
    if ":" in component_ticker:
        return False
    if raw_code.startswith("KRD"):
        return False
    if yahoo_symbol and "." in yahoo_symbol:
        return False
    if price_currency and price_currency != "USD":
        return False
    return component_ticker.isalpha()


def _is_kor_pool_candidate(item: dict[str, object], domestic_etf_tickers: set[str]) -> bool:
    component_ticker = str(item.get("ticker") or "").strip().upper()
    raw_code = str(item.get("raw_code") or "").strip().upper()
    yahoo_symbol = str(item.get("yahoo_symbol") or "").strip().upper()
    if not component_ticker.isdigit() or len(component_ticker) != 6:
        return False
    if raw_code.startswith("KRD"):
        return False
    if yahoo_symbol and not yahoo_symbol.endswith((".KS", ".KQ")):
        return False
    if raw_code.startswith("CNE"):
        return False
    return component_ticker not in domestic_etf_tickers


def _serialize_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


def _infer_yahoo_symbol_currency(symbol: str) -> str | None:
    normalized = str(symbol or "").strip().upper()
    if normalized.endswith(".TW"):
        return "TWD"
    if normalized.endswith(".HK"):
        return "HKD"
    if normalized.endswith((".SS", ".SZ", ".BJ")):
        return "CNY"
    if normalized.endswith(".T"):
        return "JPY"
    if normalized.endswith(".L"):
        return "GBP"
    if normalized.endswith((".KS", ".KQ")):
        return "KRW"
    if normalized.endswith(".AX"):
        return "AUD"
    return None


def _build_fx_rates_for_holdings(
    holdings: list[dict[str, object]], rates: dict[str, object]
) -> list[dict[str, object]]:
    currencies: set[str] = set()
    for item in holdings:
        ticker = str(item.get("ticker") or "").strip().upper()
        raw_code = str(item.get("raw_code") or "").strip().upper()
        name = str(item.get("name") or item.get("raw_name") or "").strip()
        if ticker.startswith("KRD") or raw_code.startswith("KRD") or "현금" in name:
            continue

        currency = str(item.get("price_currency") or "").strip().upper()
        if not currency:
            inferred_currency = _infer_yahoo_symbol_currency(str(item.get("yahoo_symbol") or ""))
            currency = str(inferred_currency or "").strip().upper()
        if currency:
            currencies.add(currency)

    result: list[dict[str, object]] = []
    for currency in sorted(currencies):
        if currency == "KRW":
            continue
        rate_info = rates.get(currency)
        if not isinstance(rate_info, dict):
            continue
        result.append(
            {
                "currency": currency,
                "rate": rate_info.get("rate"),
                "change_pct": rate_info.get("change_pct"),
            }
        )
    return result


def _calculate_consolidated_average_buy_price(ticker: str, currency: str | None = None) -> float | None:
    """모든 계좌의 동일 티커 보유분을 합산해 통합 평균매입가를 계산한다.

    같은 티커가 여러 시장에 상장된 경우(예: IOO — 미국 USD / 호주 AUD)가 있어,
    ``currency`` 가 주어지면 그 통화 보유분만 합산한다(통화가 섞여 계산 불가한 상황 자체를 없앤다).

    `config.HOLDING_CHART_SHOW_AVG_BUY_PRICE` 가 꺼져 있으면 계산하지 않는다 —
    전략·순위 차트와 같은 스위치다(내 단가를 화면에서 빼는 목적이라 화면마다 다르면 뜻이 없다).
    """
    if not HOLDING_CHART_SHOW_AVG_BUY_PRICE:
        return None
    ticker_key = str(ticker or "").strip().upper()
    if not ticker_key:
        raise ValueError("ticker 값이 필요합니다.")
    target_currency = str(currency or "").strip().upper()

    total_quantity = 0.0
    total_buy_amount = 0.0
    currencies: set[str] = set()

    for account_id in list_available_accounts():
        master = load_portfolio_master(account_id)
        if not master:
            continue
        for holding in master.get("holdings") or []:
            holding_ticker = str(holding.get("ticker") or "").strip().upper()
            if holding_ticker != ticker_key:
                continue

            quantity = float(holding.get("quantity") or 0.0)
            average_buy_price = float(holding.get("average_buy_price") or 0.0)
            if quantity <= 0 or average_buy_price <= 0:
                continue

            holding_currency = str(holding.get("currency") or "").strip().upper()
            # 조회 대상 통화가 정해져 있으면 다른 시장 상장분(동일 티커)은 제외한다.
            if target_currency and holding_currency and holding_currency != target_currency:
                continue
            if holding_currency:
                currencies.add(holding_currency)
            total_quantity += quantity
            total_buy_amount += quantity * average_buy_price

    if len(currencies) > 1:
        raise RuntimeError(
            f"{ticker_key} 보유 통화가 여러 개라 통합 평균단가를 계산할 수 없습니다: {sorted(currencies)}"
        )
    if total_quantity <= 0:
        return None
    return total_buy_amount / total_quantity


def _build_overseas_etf_info_payload(
    *,
    ticker: str,
    cache_document: dict[str, object] | None,
    holdings_cache: dict[str, object],
    latest_row: dict[str, object] | None,
    country_code: str,
) -> dict[str, object]:
    """해외(미국·호주) ETF 기본 정보 페이로드.

    한국은 iNAV/괴리율을 계산하는 별도 페이로드를 쓰고, 해외는 메타 캐시에 저장된 값
    (운용보수·순자산·배당수익률 등)을 그대로 노출한다. 값이 없으면 None(화면은 '-').
    시가총액은 화면이 국가별로 포맷하므로 현지 통화 원시값을 그대로 넘긴다.
    """
    meta_cache = dict((cache_document or {}).get("meta_cache") or {}) if isinstance(cache_document, dict) else {}

    def _as_float(value: object) -> float | None:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return None

    payload: dict[str, object] = {
        "source": holdings_cache.get("source"),
        "reference_date": holdings_cache.get("reference_date"),
        "expense_ratio": _as_float(meta_cache.get("expense_ratio")),
        "dividend_yield_ttm": _as_float(meta_cache.get("dividend_yield_ttm")),
        "market_cap_krw": _as_float(meta_cache.get("total_net_assets")),
        "listed_date": meta_cache.get("listed_date"),
        "volume": int(latest_row["volume"]) if latest_row and latest_row.get("volume") is not None else None,
    }

    # NAV·괴리율 — yfinance navPrice 기준(한국 네이버 iNAV 와 계산식 동일, 다만 실시간 추정이 아닌 공시 NAV).
    nav_snapshot = fetch_overseas_etf_nav_snapshot(ticker, country_code)
    payload["nav"] = _as_float(nav_snapshot.get("nav"))
    payload["deviation"] = _as_float(nav_snapshot.get("deviation"))

    # 환율(해외 종목 표기용) — 한국 페이로드와 같은 소스.
    rates = get_exchange_rates()
    currency = "USD" if country_code == "us" else "AUD" if country_code == "au" else None
    if currency:
        rate_info = rates.get(currency)
        if isinstance(rate_info, dict):
            payload["fx_rate"] = rate_info.get("rate")
            payload["fx_change_pct"] = rate_info.get("change_pct")
    return payload


def _build_korean_etf_info_payload(
    *,
    ticker: str,
    ticker_type: str,
    cache_document: dict[str, object] | None,
    latest_row: dict[str, object] | None,
    holdings: list[dict[str, object]],
) -> dict[str, object] | None:
    if not isinstance(cache_document, dict):
        return None

    meta_cache = dict(cache_document.get("meta_cache") or {})
    if not meta_cache:
        return None

    inav_snapshot = fetch_naver_etf_inav_snapshot([ticker]).get(str(ticker or "").strip().upper(), {})
    nav_value = inav_snapshot.get("nav")
    deviation_value = inav_snapshot.get("deviation")

    market_cap_krw = None
    total_net_assets = meta_cache.get("total_net_assets")
    if total_net_assets is not None:
        try:
            # 네이버 ETFBase API의 totalNetAssets는 이미 '원' 단위임
            market_cap_krw = float(total_net_assets)
        except (TypeError, ValueError):
            market_cap_krw = None

    # 비교는 (현재 값 vs 직전 영업일 값) 한 쌍 — 조회 1회.
    # 직전값의 date 는 저장 시점에 이미 거래일이라 휴장일 보정이 필요 없다.
    previous_meta = get_previous_stock_cache_meta(ticker_type, ticker)
    prev_nav = None
    portfolio_change_base_date = None
    if previous_meta and "meta_cache" in previous_meta:
        prev_nav = previous_meta["meta_cache"].get("nav")
        portfolio_change_base_date = str(previous_meta.get("date") or "").strip() or None

    nav_change = None
    nav_change_pct = None
    if nav_value is not None and prev_nav is not None and prev_nav > 0:
        nav_change = float(nav_value) - float(prev_nav)
        nav_change_pct = round((nav_change / float(prev_nav)) * 100, 2)

    # 환율 정보 (무조건 제공)
    rates = get_exchange_rates()
    usd_info = rates.get("USD", {})
    fx_rate = usd_info.get("rate")
    fx_change_pct = usd_info.get("change_pct")
    fx_rates = _build_fx_rates_for_holdings(holdings, rates)

    return {
        "nav": float(nav_value) if nav_value is not None else None,
        "nav_change": nav_change,
        "nav_change_pct": nav_change_pct,
        "fx_rate": fx_rate,
        "fx_change_pct": fx_change_pct,
        "fx_rates": fx_rates,
        "portfolio_change_base_date": portfolio_change_base_date,
        "deviation": float(deviation_value) if deviation_value is not None else None,
        "expense_ratio": float(meta_cache["expense_ratio"]) if meta_cache.get("expense_ratio") is not None else None,
        "dividend_yield_ttm": float(meta_cache["dividend_yield_ttm"])
        if meta_cache.get("dividend_yield_ttm") is not None
        else None,
        "dividend_history": list(meta_cache.get("dividend_history") or []),
        "total_net_assets_eok": float(total_net_assets) if total_net_assets is not None else None,
        "market_cap_krw": market_cap_krw,
        "volume": int(latest_row["volume"]) if latest_row and latest_row.get("volume") is not None else None,
    }


def _is_worldstock_symbol(symbol: str) -> bool:
    normalized = str(symbol or "").strip().upper()
    return normalized.endswith((".T", ".HK"))


def _is_yahoo_tw_symbol(symbol: str) -> bool:
    normalized = str(symbol or "").strip().upper()
    return normalized.endswith(".TW")


def _is_pre_open_top_movers_window() -> bool:
    schedule = MARKET_SCHEDULES.get("kor") or {}
    timezone_name = str(schedule.get("timezone") or "Asia/Seoul").strip() or "Asia/Seoul"
    market_open = schedule.get("open")
    if market_open is None:
        return False

    now_local = datetime.now(ZoneInfo(timezone_name))
    return now_local.time() < market_open


def _is_pre_open_cache_timestamp(value: datetime | None) -> bool:
    if value is None:
        return False

    schedule = MARKET_SCHEDULES.get("kor") or {}
    timezone_name = str(schedule.get("timezone") or "Asia/Seoul").strip() or "Asia/Seoul"
    market_open = schedule.get("open")
    if market_open is None:
        return False

    local_value = (
        value.astimezone(ZoneInfo(timezone_name))
        if value.tzinfo
        else value.replace(
            tzinfo=timezone.utc,
        ).astimezone(ZoneInfo(timezone_name))
    )
    now_local = datetime.now(ZoneInfo(timezone_name))
    return local_value.date() == now_local.date() and local_value.time() < market_open


def _build_price_snapshot(close_series: pd.Series | None) -> tuple[float | None, float | None]:
    if close_series is None or close_series.empty:
        return None, None

    numeric_series = pd.to_numeric(close_series, errors="coerce").dropna()
    if numeric_series.empty:
        return None, None

    current_price = float(numeric_series.iloc[-1])
    if len(numeric_series) < 2:
        return current_price, None

    previous_close = float(numeric_series.iloc[-2])
    if previous_close == 0:
        return current_price, None

    change_pct = round(((current_price / previous_close) - 1.0) * 100.0, 2)
    return current_price, change_pct


def _apply_realtime_snapshot_to_dataframe(
    df: pd.DataFrame,
    *,
    ticker: str,
    country_code: str,
) -> tuple[pd.DataFrame, float | None]:
    country = str(country_code or "").strip().lower()
    if country not in {"kor", "au", "us"}:
        return df, None

    try:
        realtime_map = get_realtime_snapshot(country, [ticker])
    except Exception:
        return df, None

    realtime_entry = realtime_map.get(str(ticker or "").strip().upper()) or {}
    now_val = realtime_entry.get("nowVal")
    if now_val is None:
        return df, None

    try:
        realtime_price = float(now_val)
    except (TypeError, ValueError):
        return df, None

    if realtime_price <= 0:
        return df, None

    realtime_change_pct: float | None = None
    if country == "us" and realtime_entry.get("changeRate") is not None:
        try:
            realtime_change_pct = round(float(realtime_entry["changeRate"]), 2)
        except (TypeError, ValueError):
            realtime_change_pct = None

    target_trading_day = _resolve_realtime_target_trading_day(country)
    if country == "kor" and realtime_entry.get("is_pre_market") is True:
        target_trading_day = pd.Timestamp(datetime.now(ZoneInfo("Asia/Seoul")).date()).normalize()
    latest_trading_day = (target_trading_day or get_latest_trading_day(country)).normalize()
    adjusted = df.copy()

    if adjusted.empty:
        return adjusted, realtime_change_pct

    close_col = "Close" if "Close" in adjusted.columns else "close"
    open_col = "Open" if "Open" in adjusted.columns else "open"
    high_col = "High" if "High" in adjusted.columns else "high"
    low_col = "Low" if "Low" in adjusted.columns else "low"
    volume_col = "Volume" if "Volume" in adjusted.columns else "volume"

    if adjusted.index.max().normalize() == latest_trading_day:
        target_index = adjusted.index.max()
        existing_open = adjusted.at[target_index, open_col] if open_col in adjusted.columns else None
        existing_high = adjusted.at[target_index, high_col] if high_col in adjusted.columns else None
        existing_low = adjusted.at[target_index, low_col] if low_col in adjusted.columns else None
        adjusted.at[target_index, close_col] = realtime_price
        if open_col in adjusted.columns and pd.isna(existing_open):
            adjusted.at[target_index, open_col] = realtime_price
        if high_col in adjusted.columns:
            try:
                adjusted.at[target_index, high_col] = max(float(existing_high), realtime_price)
            except (TypeError, ValueError):
                adjusted.at[target_index, high_col] = realtime_price
        if low_col in adjusted.columns:
            try:
                adjusted.at[target_index, low_col] = min(float(existing_low), realtime_price)
            except (TypeError, ValueError):
                adjusted.at[target_index, low_col] = realtime_price
    else:
        new_row: dict[str, object] = {
            close_col: realtime_price,
            open_col: realtime_entry.get("open", realtime_price),
            high_col: realtime_entry.get("high", realtime_price),
            low_col: realtime_entry.get("low", realtime_price),
        }
        if volume_col in adjusted.columns:
            new_row[volume_col] = realtime_entry.get("volume", 0)
        adjusted.loc[latest_trading_day] = new_row

    adjusted.sort_index(inplace=True)
    return adjusted, realtime_change_pct


def _resolve_realtime_target_trading_day(country_code: str) -> pd.Timestamp | None:
    country = str(country_code or "").strip().lower()
    schedule = MARKET_SCHEDULES.get(country)
    if not isinstance(schedule, dict):
        return None

    timezone_name = str(schedule.get("timezone") or "").strip() or "UTC"
    market_open = schedule.get("open")
    if market_open is None:
        return None

    now_local = datetime.now(ZoneInfo(timezone_name))
    # 미국은 프리마켓(4:00 ET)부터 토스 API로 가격 제공, 한국/호주는 장 시작 기준
    from datetime import time as dt_time

    earliest_time = dt_time(4, 0) if country == "us" else market_open
    if now_local.time() < earliest_time:
        return None

    today_local = pd.Timestamp(now_local.date()).normalize()
    trading_days = get_trading_days(
        today_local.strftime("%Y-%m-%d"),
        today_local.strftime("%Y-%m-%d"),
        country,
    )
    if not trading_days:
        return None

    return pd.Timestamp(trading_days[-1]).normalize()


@router.get("/tickers")
def get_all_tickers(
    _: None = Depends(require_internal_token),
) -> list[dict[str, object]]:
    """전체 종목타입의 활성 종목 목록을 반환합니다."""
    configs = load_ticker_type_configs()
    result: list[dict[str, object]] = []
    for config in configs:
        ticker_type = config["ticker_type"]
        country_code = config.get("country_code", "")
        etfs = get_etfs(ticker_type)
        for etf in etfs:
            tkr = etf.get("ticker", "")
            name = etf.get("name", "")
            if tkr:
                result.append(
                    {
                        "ticker": tkr,
                        "name": name,
                        "ticker_type": ticker_type,
                        "country_code": country_code,
                        "is_etf": bool(etf.get("is_etf", False)),
                    }
                )
    return result


@router.get("/resolve")
def resolve_ticker(
    ticker: str = Query(...),
    ticker_types: str | None = Query(None),
    account_id: str | None = Query(None),
    _: None = Depends(require_internal_token),
) -> dict[str, object]:
    """직접 진입용 티커 메타데이터를 반환합니다.

    ``ticker_types`` (쉼표 구분)를 주면 해당 종목풀로 검색 범위를 제한한다.
    ``account_id`` 를 주면 같은 티커가 여러 시장에 있을 때 계좌 현금 통화로 시장을 판별한다.
    """

    allowed: set[str] | None = None
    if ticker_types is not None:
        allowed = {part.strip() for part in ticker_types.split(",") if part.strip()}
    return resolve_ticker_meta(ticker, allowed_ticker_types=allowed, account_id=account_id)


@router.get("/search-data")
def get_ticker_search_data(
    _: None = Depends(require_internal_token),
) -> dict[str, object]:
    """전역 티커 검색용 메타데이터와 급상승 목록을 반환합니다."""

    configs = load_ticker_type_configs()
    ticker_items: list[dict[str, object]] = []
    top_movers_by_type: list[dict[str, object]] = []
    top_movers_updated_at: datetime | None = None
    top_movers_pre_open = False

    for config in configs:
        ticker_type = config["ticker_type"]
        country_code = config.get("country_code", "")
        ticker_type_name = str(config.get("name") or ticker_type).strip()
        etfs = get_etfs(ticker_type)
        tickers = [str(item.get("ticker") or "").strip().upper() for item in etfs if item.get("ticker")]
        realtime_snapshot_map: dict[str, dict[str, float]] = {}
        type_updated_at: datetime | None = None

        if country_code in {"kor", "au"}:
            realtime_snapshot_map = get_realtime_snapshot(country_code, tickers)
            realtime_meta = get_realtime_snapshot_meta(country_code, tickers) or {}
            fetched_at = realtime_meta.get("fetched_at")
            type_updated_at = fetched_at if isinstance(fetched_at, datetime) else None
            close_series_map = {}
        else:
            completed_at = get_cache_refresh_completed_at(ticker_type)
            if completed_at is not None:
                close_series_map = load_cached_close_series_bulk_before_or_at_with_fallback(
                    ticker_type,
                    tickers,
                    completed_at,
                )
                updated_at_map = load_cached_updated_at_bulk_before_or_at_with_fallback(
                    ticker_type,
                    tickers,
                    completed_at,
                )
                type_updated_at = completed_at
            else:
                close_series_map = load_cached_close_series_bulk_with_fallback(ticker_type, tickers)
                updated_at_map = load_cached_updated_at_bulk_with_fallback(ticker_type, tickers)
                type_updated_at = max(updated_at_map.values()) if updated_at_map else None
        ticker_type_items: list[dict[str, object]] = []

        if type_updated_at is not None:
            if top_movers_updated_at is None or type_updated_at > top_movers_updated_at:
                top_movers_updated_at = type_updated_at

        for etf in etfs:
            ticker = str(etf.get("ticker") or "").strip().upper()
            if not ticker:
                continue

            realtime_entry = realtime_snapshot_map.get(ticker) or {}
            if realtime_entry:
                now_val = realtime_entry.get("nowVal")
                change_rate = realtime_entry.get("changeRate")
                current_price = float(now_val) if now_val is not None else None
                change_pct = float(change_rate) if change_rate is not None else None
            else:
                current_price, change_pct = _build_price_snapshot(close_series_map.get(ticker))
            item = {
                "ticker": ticker,
                "name": str(etf.get("name") or "").strip(),
                "ticker_type": ticker_type,
                "country_code": country_code,
                "is_etf": bool(etf.get("is_etf", False)),
                "current_price": current_price,
                "change_pct": change_pct,
            }
            ticker_items.append(item)
            ticker_type_items.append(item)

        top_movers = sorted(
            [item for item in ticker_type_items if item.get("change_pct") is not None],
            key=lambda item: float(item["change_pct"]),
            reverse=True,
        )[:5]
        top_movers_by_type.append(
            {
                "ticker_type": ticker_type,
                "label": ticker_type_name,
                "items": top_movers,
            }
        )

    top_movers_pre_open = _is_pre_open_cache_timestamp(top_movers_updated_at) or (
        top_movers_updated_at is None and _is_pre_open_top_movers_window()
    )
    if top_movers_pre_open:
        top_movers_by_type = [
            {
                **item,
                "items": [],
            }
            for item in top_movers_by_type
        ]

    return {
        "tickers": ticker_items,
        "top_movers_by_type": top_movers_by_type,
        "top_movers_updated_at": _serialize_datetime(top_movers_updated_at),
        "top_movers_pre_open": top_movers_pre_open,
    }


def build_ticker_detail_payload(
    ticker: str,
    ticker_type: str,
    country_code: str = "kor",
    *,
    component_price_snapshot: dict[str, dict[str, Any]] | None = None,
    use_bundle_cache: bool = True,
    include_holdings: bool = True,
) -> dict[str, object]:
    """단일 ETF/종목의 상세(가격 행 + 구성종목) 페이로드를 만든다.

    ``component_price_snapshot`` 을 주면 구성종목 가격을 그 공유 스냅샷에서 가져온다
    (비교 화면에서 여러 ETF가 같은 종목을 동일 값으로 보도록 합집합 1회 조회 결과 주입).
    그 경우 ETF별 캐시를 우회하려면 ``use_bundle_cache=False`` 로 강제 재계산한다.

    ``include_holdings=False`` 면 구성종목 시세 평가·포트폴리오 변동 계산을 건너뛴다
    (비교 화면 성과분석 탭처럼 가격 시계열만 필요한 경우 — 초기 로딩을 크게 줄인다).
    """
    settings = load_common_settings()
    cache_start_date = str(settings.get("CACHE_START_DATE") or "").strip()
    if not cache_start_date:
        raise RuntimeError("CACHE_START_DATE 설정이 필요합니다.")

    db_ticker = ticker.split(":")[-1] if ":" in ticker else ticker
    fetch_error: str | None = None
    try:
        df = fetch_ohlcv(
            db_ticker,
            country=country_code,
            months_back=None,
            date_range=[cache_start_date, None],
            ticker_type=ticker_type,
        )
    except Exception as exc:
        # pykrx 가 지원하지 않는 신형 알파벳 포함 ETF 코드(예: 0060H0)나
        # 원천 API 일시 장애로 예외가 올라올 수 있으므로 500 대신 에러 메시지로 돌려준다.
        df = None
        fetch_error = f"가격 데이터를 가져오지 못했습니다: {exc}"

    if df is None or df.empty:
        return {
            "ticker": ticker,
            "rows": [],
            "holdings": [],
            "holdings_as_of_date": None,
            "holdings_price_as_of_date": None,
            "holdings_error": None,
            "my_average_buy_price": _calculate_consolidated_average_buy_price(
                db_ticker, currency_for_country(country_code)
            ),
            "error": fetch_error or "가격 데이터를 가져오지 못했습니다.",
        }

    df = df.sort_index()
    df, realtime_change_pct = _apply_realtime_snapshot_to_dataframe(
        df,
        ticker=db_ticker,
        country_code=country_code,
    )

    close_col = "Close" if "Close" in df.columns else "close"
    open_col = "Open" if "Open" in df.columns else "open"
    high_col = "High" if "High" in df.columns else "high"
    low_col = "Low" if "Low" in df.columns else "low"
    volume_col = "Volume" if "Volume" in df.columns else "volume"

    rows: list[dict[str, object]] = []
    prev_close = None
    for date_idx, row in df.iterrows():
        date_str = pd.Timestamp(date_idx).strftime("%Y-%m-%d")
        close = float(row[close_col]) if pd.notna(row.get(close_col)) else None
        open_val = float(row[open_col]) if pd.notna(row.get(open_col)) else None
        high_val = float(row[high_col]) if pd.notna(row.get(high_col)) else None
        low_val = float(row[low_col]) if pd.notna(row.get(low_col)) else None
        volume_val = int(row[volume_col]) if pd.notna(row.get(volume_col)) else None

        change_pct = None
        if close is not None and prev_close is not None and prev_close != 0:
            change_pct = round((close - prev_close) / prev_close * 100, 2)

        rows.append(
            {
                "date": date_str,
                "open": open_val,
                "high": high_val,
                "low": low_val,
                "close": close,
                "volume": volume_val,
                "change_pct": change_pct,
            }
        )
        if close is not None:
            prev_close = close

    # 미국 프리·애프터마켓은 토스가 세션에 맞는 기준가로 계산한 등락률을 사용한다.
    if rows and realtime_change_pct is not None:
        rows[-1]["change_pct"] = realtime_change_pct

    holdings: list[dict[str, object]] = []
    holdings_as_of_date: str | None = None
    holdings_price_as_of_date: str | None = None
    holdings_error: str | None = None
    etf_info: dict[str, object] | None = None
    us_pool_tickers: set[str] = set()
    kor_pool_tickers: set[str] = set()
    domestic_etf_tickers: set[str] = set()
    country_clean = str(country_code or "").strip().lower()
    if country_clean in ("kor", "au", "us"):
        db_ticker = ticker.split(":")[-1] if ":" in ticker else ticker
        cache_document = get_stock_cache_meta(ticker_type, db_ticker)
        holdings_cache = dict(cache_document.get("holdings_cache") or {}) if isinstance(cache_document, dict) else {}
        holdings = list(holdings_cache.get("items") or [])
        if country_clean == "kor":
            etf_info = _build_korean_etf_info_payload(
                ticker=db_ticker,
                ticker_type=ticker_type,
                cache_document=cache_document if isinstance(cache_document, dict) else None,
                latest_row=rows[-1] if rows else None,
                holdings=holdings,
            )
        else:
            # 해외(미국·호주) ETF — 메타 캐시의 기본 정보(보수·순자산·배당 등)를 함께 노출한다.
            # (한국은 iNAV/괴리율까지 계산하는 별도 페이로드를 쓴다.)
            etf_info = _build_overseas_etf_info_payload(
                ticker=db_ticker,
                cache_document=cache_document if isinstance(cache_document, dict) else None,
                holdings_cache=holdings_cache,
                latest_row=rows[-1] if rows else None,
                country_code=country_clean,
            )
        holdings_as_of_date = str(holdings_cache.get("reference_date") or "").strip() or None
        if not include_holdings:
            # 가격 시계열만 필요한 호출(성과분석 탭) — 구성종목 시세 평가·포트폴리오 변동 계산을 건너뛴다.
            holdings = []
        elif not holdings:
            holdings_error = (
                "구성종목 캐시가 없습니다. python scripts/stock_reference_meta_updater.py 실행이 필요합니다."
            )
        elif not holdings_as_of_date:
            holdings_error = "구성종목 캐시 기준일(reference_date)이 없습니다."
        else:
            us_pool_tickers = _load_us_pool_ticker_set()
            kor_pool_tickers = _load_kor_pool_ticker_set()
            domestic_etf_tickers = _load_domestic_etf_ticker_set()

            # /holdings 엔드포인트와 동일한 캐시 결과를 공유한다.
            # 공유 스냅샷이 주어지면(비교 화면) 캐시를 우회해 동일 시세로 재계산한다.
            bundle = compute_portfolio_change_bundle(
                db_ticker,
                ticker_type,
                use_cache=use_bundle_cache,
                component_price_snapshot=component_price_snapshot,
            )
            if bundle:
                priced_holdings = bundle.get("priced_holdings") or []
                holdings_price_as_of_date = None
                bundle_fx_rates = bundle.get("fx_rates") or []
                if etf_info is not None:
                    etf_info["portfolio_change_base_date"] = bundle.get("base_date")
                    etf_info["portfolio_change_base_is_open"] = bool(bundle.get("base_is_open"))
            else:
                priced_holdings, holdings_price_as_of_date = enrich_component_prices(
                    holdings,
                    price_fetch_limit=100,
                    cumulative_base_date=str(etf_info.get("portfolio_change_base_date") or "") if etf_info else None,
                    component_price_snapshot=component_price_snapshot,
                )
                bundle_fx_rates = None

            enriched_holdings: list[dict[str, object]] = []
            for source_item in priced_holdings:
                # 캐시 공유를 위해 dict 복사 후 풀 플래그를 추가한다.
                enriched_item = dict(source_item)
                component_ticker = str(enriched_item.get("ticker") or "").strip().upper()

                enriched_item["is_us_pool_candidate"] = _is_us_pool_candidate(enriched_item)
                enriched_item["in_us_pool"] = component_ticker in us_pool_tickers
                enriched_item["is_kor_pool_candidate"] = _is_kor_pool_candidate(
                    enriched_item,
                    domestic_etf_tickers,
                )
                enriched_item["in_kor_pool"] = component_ticker in kor_pool_tickers

                enriched_holdings.append(enriched_item)
            holdings = enriched_holdings
            if etf_info is not None:
                if bundle_fx_rates is not None:
                    etf_info["fx_rates"] = bundle_fx_rates
                else:
                    etf_info["fx_rates"] = _build_daily_fx_rates_for_holdings(
                        holdings,
                        get_exchange_rates(),
                    )

    return {
        "ticker": ticker,
        "rows": rows,
        "etf_info": etf_info,
        "holdings": holdings,
        "holdings_as_of_date": holdings_as_of_date,
        "holdings_price_as_of_date": holdings_price_as_of_date,
        "holdings_error": holdings_error,
        "my_average_buy_price": _calculate_consolidated_average_buy_price(
            db_ticker, currency_for_country(country_code)
        ),
    }


@router.get("")
def get_ticker_detail(
    ticker: str = Query(...),
    ticker_type: str = Query(...),
    country_code: str = Query(default="kor"),
    _: None = Depends(require_internal_token),
) -> dict[str, object]:
    return build_ticker_detail_payload(ticker, ticker_type, country_code)


@router.post("/compare")
def get_ticker_detail_compare(
    payload: dict[str, object] = Body(...),
    _: None = Depends(require_internal_token),
):
    """여러 ETF 상세를 한 번에 계산한다 — 구성종목 합집합을 1회만 조회해 공유한다.

    같은 구성종목(예: SK스퀘어)이 여러 ETF에 등장해도 **동일한 시세/변동률**로 보이고,
    중복 조회가 사라진다. body: ``{"items": [{"ticker","ticker_type","country_code"}, ...],
    "include_holdings": bool}``.

    ``include_holdings=False`` (성과분석 탭 초기 로딩)면 구성종목 시세 평가·포트폴리오 변동
    계산을 생략해 응답이 크게 빨라진다. 구성종목/기본정보 탭을 열 때 true 로 다시 요청한다.

    응답은 **SSE**(튜닝과 같은 형식)다 — 8종목 × 구성종목 시세 조회가 한 요청에 몰리면
    일반 JSON 응답은 프록시 타임아웃(90초)에 걸렸고, 화면 진행 바도 추정으로만 움직였다.
    단계마다 진행 이벤트를 흘리고, 실패하면 어느 종목·단계인지 에러 이벤트로 알린다.
    """
    return sse_stream(lambda: _compare_events(payload))


def _compare_events(payload: dict[str, object]):
    """compare 계산 본체 — 진행(progress)·결과(result) 이벤트를 흘리는 제너레이터."""
    raw_items = payload.get("items") if isinstance(payload, dict) else None
    items = [it for it in raw_items if isinstance(it, dict)] if isinstance(raw_items, list) else []
    include_holdings = bool(payload.get("include_holdings", True)) if isinstance(payload, dict) else True

    def item_cache_key(item: dict[str, object]) -> tuple[str, str, str, bool]:
        return (
            str(item.get("ticker") or ""),
            str(item.get("ticker_type") or ""),
            str(item.get("country_code") or "kor"),
            include_holdings,
        )

    # 0) 종목 단위 캐시 — TTL 안이면 재계산하지 않는다. 부분 실패 후 재시도가
    #    성공분을 여기서 읽고 실패분만 다시 계산한다(진행 바도 그만큼 건너뛴다).
    now_ts = _time.time()
    detail_by_key: dict[tuple[str, str, str, bool], dict[str, object]] = {}
    with _COMPARE_CACHE_LOCK:
        for item in items:
            cached = _COMPARE_CACHE.get(item_cache_key(item))
            if cached and now_ts - cached[1] < _COMPARE_CACHE_TTL:
                detail_by_key[item_cache_key(item)] = cached[0]
    pending = [item for item in items if item_cache_key(item) not in detail_by_key]

    # 1) 한국 ETF 구성종목 합집합 → 공유 가격 스냅샷 1회 구성 (build_component_price_snapshot 가 중복 제거)
    #    캐시된 종목은 뺀다 — 새로 계산할 종목의 구성종목만 필요하다.
    #    구성종목이 필요 없는 호출이면 이 조회 자체를 건너뛴다(성과분석 탭 초기 로딩 단축).
    shared_snapshot: dict[str, dict[str, Any]] = {}
    if include_holdings and pending:
        union_holdings: list[dict[str, object]] = []
        for item in pending:
            if str(item.get("country_code") or "kor").strip().lower() != "kor":
                continue
            cache_doc = get_stock_cache_meta(str(item.get("ticker_type") or ""), str(item.get("ticker") or ""))
            if not isinstance(cache_doc, dict):
                continue
            holdings_cache = dict(cache_doc.get("holdings_cache") or {})
            union_holdings.extend(list(holdings_cache.get("items") or []))

        if union_holdings:
            yield {
                "type": "progress",
                "percent": 5,
                "message": f"구성종목 합집합 {len(union_holdings):,}건 가격 스냅샷 조회 중"
                + (f" (캐시 재사용 {len(items) - len(pending)}종목 제외)" if len(pending) < len(items) else ""),
            }
            try:
                shared_snapshot = build_component_price_snapshot(union_holdings)
            except Exception as exc:
                raise RuntimeError(f"구성종목 가격 스냅샷 조회 실패: {exc}") from exc

    # 진행·에러 메시지용 종목명 — 풀 목록에서 읽는다(티커만으로는 어떤 종목인지 알기 어렵다).
    name_by_ticker: dict[tuple[str, str], str] = {}
    for pool in {str(item.get("ticker_type") or "") for item in pending if item.get("ticker_type")}:
        try:
            for row in get_etfs(pool):
                name_by_ticker[(pool, str(row.get("ticker") or "").strip().upper())] = str(row.get("name") or "")
        except Exception:  # noqa: BLE001 - 이름은 표시용이라 못 읽으면 티커만 쓴다
            continue

    # 2) ETF 별 detail 을 공유 스냅샷으로 계산 (번들 캐시 우회 → 종목당 동일 값 보장).
    #    **끝나는 즉시 종목 캐시에 저장한다** — 뒤 종목에서 실패해도 앞의 성공분은 남는다.
    total = max(len(pending), 1)
    for index, item in enumerate(pending):
        ticker = str(item.get("ticker") or "")
        name = name_by_ticker.get((str(item.get("ticker_type") or ""), ticker.strip().upper()), "")
        label = f"{name}({ticker})" if name else ticker
        yield {
            "type": "progress",
            "percent": round(30 + 65 * index / total),
            "message": f"{index + 1}/{total} {label} 상세 계산 중",
        }
        try:
            detail = build_ticker_detail_payload(
                ticker,
                str(item.get("ticker_type") or ""),
                str(item.get("country_code") or "kor"),
                component_price_snapshot=shared_snapshot,
                use_bundle_cache=False,
                include_holdings=include_holdings,
            )
        except Exception as exc:
            raise RuntimeError(
                f"「{label}」 상세 계산 실패 ({index + 1}/{total}) — 계산이 끝난 종목은 캐시에 남아 "
                f"재시도 시 이 종목부터 이어집니다: {exc}"
            ) from exc
        # 어느 요청 항목의 결과인지 응답에 실어 둔다 — 화면이 순서(인덱스)가 아니라
        # 이 값으로 짝을 지어야 순서가 바뀌어도 이름과 시세가 어긋나지 않는다.
        detail["ticker_type"] = str(item.get("ticker_type") or "")
        detail["country_code"] = str(item.get("country_code") or "kor")
        detail_by_key[item_cache_key(item)] = detail
        with _COMPARE_CACHE_LOCK:
            _COMPARE_CACHE[item_cache_key(item)] = (detail, now_ts)

    # 요청 순서대로 조립 — 캐시·신규 계산이 섞여도 화면 카드 순서와 일치한다.
    results = [detail_by_key[item_cache_key(item)] for item in items]
    yield {"type": "progress", "percent": 100, "message": "비교 데이터 반영 중"}
    yield {"type": "result", "results": results}
