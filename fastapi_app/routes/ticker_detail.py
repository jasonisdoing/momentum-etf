from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
from fastapi import APIRouter, Depends, Query

from config import MARKET_SCHEDULES
from fastapi_app.dependencies import require_internal_token
from services.component_price_service import enrich_component_prices
from services.price_service import (
    get_exchange_rates,
    get_exchange_rate_series,
    get_realtime_snapshot,
    get_realtime_snapshot_meta,
)
from services.stock_cache_service import get_stock_cache_meta
from utils.stock_cache_meta_io import get_previous_stock_cache_meta_history
from utils.cache_utils import (
    get_cache_refresh_completed_at,
    load_cached_close_series_bulk_before_or_at_with_fallback,
    load_cached_close_series_bulk_with_fallback,
    load_cached_updated_at_bulk_before_or_at_with_fallback,
    load_cached_updated_at_bulk_with_fallback,
)
from utils.data_loader import fetch_ohlcv, get_latest_trading_day, get_trading_days, fetch_naver_etf_inav_snapshot
from utils.kis_market import load_cached_kis_domestic_etf_master
from utils.settings_loader import load_common_settings
from utils.settings_loader import list_available_accounts
from utils.stock_list_io import get_active_holding_tickers, get_etfs
from utils.ticker_registry import load_ticker_type_configs
from utils.portfolio_io import load_portfolio_master

router = APIRouter(prefix="/internal/ticker-detail", tags=["ticker-detail"])


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
    return {
        str(value or "").strip().upper()
        for value in df["티커"].tolist()
        if str(value or "").strip()
    }


def _lookup_domestic_etf_name(ticker: str) -> str | None:
    df, _ = load_cached_kis_domestic_etf_master()
    if "티커" not in df.columns:
        raise RuntimeError("KIS ETF 마스터 캐시에 티커 컬럼이 없습니다.")
    name_column = "종목명" if "종목명" in df.columns else "한글종목명" if "한글종목명" in df.columns else None
    if name_column is None:
        raise RuntimeError("KIS ETF 마스터 캐시에 종목명 컬럼이 없습니다.")

    ticker_norm = str(ticker or "").strip().upper()
    matched = df[df["티커"].astype(str).str.strip().str.upper() == ticker_norm]
    if matched.empty:
        return None

    name = str(matched.iloc[0].get(name_column) or "").strip()
    return name or None


def _resolve_ticker_meta_item(ticker: str) -> dict[str, object]:
    ticker_key = str(ticker or "").strip().upper()
    if not ticker_key:
        raise ValueError("ticker 파라미터가 필요합니다.")

    # 시장 명시 접두사가 있으면 해당 시장으로 강제 지정. 동일 심볼이 여러 풀에 있을 때 구분.
    forced_ticker_type: str | None = None
    if ticker_key.startswith("ASX:"):
        ticker_key = ticker_key[len("ASX:"):]
        forced_ticker_type = "aus"
        if not ticker_key:
            raise ValueError("ASX: 뒤에 티커가 필요합니다.")
    elif ticker_key.startswith("US:"):
        ticker_key = ticker_key[len("US:"):]
        forced_ticker_type = "us"
        if not ticker_key:
            raise ValueError("US: 뒤에 티커가 필요합니다.")

    configs = load_ticker_type_configs()
    matches: list[dict[str, object]] = []
    for config in configs:
        ticker_type = config["ticker_type"]
        if forced_ticker_type is not None and ticker_type != forced_ticker_type:
            continue
        country_code = config.get("country_code", "")
        for item in get_etfs(ticker_type):
            item_ticker = str(item.get("ticker") or "").strip().upper()
            if item_ticker != ticker_key:
                continue
            matches.append(
                {
                    "ticker": ticker_key,
                    "name": str(item.get("name") or "").strip() or ticker_key,
                    "ticker_type": ticker_type,
                    "country_code": country_code,
                    "is_etf": bool(item.get("is_etf", False)),
                    "has_holdings": bool(item.get("has_holdings", False)),
                }
            )

    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        # 접두사 없이 호출된 경우 미국(us) 풀을 우선 선택
        if forced_ticker_type is None:
            us_matches = [m for m in matches if m["ticker_type"] == "us"]
            if len(us_matches) == 1:
                return us_matches[0]
        raise RuntimeError(f"동일한 티커 {ticker_key}가 여러 종목풀에 등록되어 있습니다.")

    # ASX: 접두사로 호주를 명시했는데 풀에 없으면 즉시 호주로 결정 (미국 폴백 차단)
    if forced_ticker_type == "aus":
        return {
            "ticker": ticker_key,
            "name": ticker_key,
            "ticker_type": "aus",
            "country_code": "au",
            "is_etf": True,
            "has_holdings": False,
        }

    holding_matches = [
        ticker_type
        for ticker_type, tickers in get_active_holding_tickers().items()
        if ticker_key in tickers
    ]
    if len(holding_matches) == 1:
        holding_type = holding_matches[0]
        holding_config = next(
            (config for config in configs if str(config.get("ticker_type") or "").strip().lower() == holding_type),
            {},
        )
        cache_doc = get_stock_cache_meta(holding_type, ticker_key)
        holdings_cache = dict(cache_doc.get("holdings_cache") or {}) if isinstance(cache_doc, dict) else {}
        cache_name = cache_doc.get("name") if isinstance(cache_doc, dict) else None
        if not cache_name and ticker_key.isdigit() and len(ticker_key) == 6:
            cache_name = _lookup_domestic_etf_name(ticker_key)

        return {
            "ticker": ticker_key,
            "name": cache_name or ticker_key,
            "ticker_type": holding_type,
            "country_code": str(holding_config.get("country_code") or "").strip().lower(),
            "is_etf": holding_type != "kor",
            "has_holdings": bool(holdings_cache.get("items")),
        }
    if len(holding_matches) > 1:
        joined = ", ".join(sorted(holding_matches))
        raise RuntimeError(f"보유 중인 동일 티커 {ticker_key}가 여러 종목풀에 있습니다: {joined}")

    if ticker_key.isdigit() and len(ticker_key) == 6:
        domestic_etf_tickers = _load_domestic_etf_ticker_set()
        if ticker_key in domestic_etf_tickers:
            return {
                "ticker": ticker_key,
                "name": _lookup_domestic_etf_name(ticker_key) or ticker_key,
                "ticker_type": "kor_kr",
                "country_code": "kor",
                "is_etf": True,
                "has_holdings": True,
            }
        return {
            "ticker": ticker_key,
            "name": ticker_key,
            "ticker_type": "kor",
            "country_code": "kor",
            "is_etf": False,
            "has_holdings": False,
        }

    if ticker_key.endswith(".AX"):
        return {
            "ticker": ticker_key,
            "name": ticker_key,
            "ticker_type": "aus",
            "country_code": "au",
            "is_etf": True,
            "has_holdings": False,
        }

    if ticker_key.isalpha() or "." in ticker_key:
        return {
            "ticker": ticker_key,
            "name": ticker_key,
            "ticker_type": "us",
            "country_code": "us",
            "is_etf": False,
            "has_holdings": False,
        }

    raise RuntimeError(f"{ticker_key} 티커를 찾지 못했습니다.")


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


def _build_fx_rates_for_holdings(holdings: list[dict[str, object]], rates: dict[str, object]) -> list[dict[str, object]]:
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


def _build_cumulative_fx_rates_for_holdings(
    holdings: list[dict[str, object]],
    rates: dict[str, object],
    base_date: str | None,
) -> list[dict[str, object]]:
    """구성종목 통화별 기준일 이후 환율 변동률을 구성한다."""
    if not base_date:
        return []

    fx_rates = _build_fx_rates_for_holdings(holdings, rates)
    if not fx_rates:
        return []

    base_ts = pd.Timestamp(base_date).normalize()
    end_ts = datetime.now(ZoneInfo("Asia/Seoul")).replace(tzinfo=None)
    result: list[dict[str, object]] = []
    for item in fx_rates:
        currency = str(item.get("currency") or "").strip().upper()
        current_rate = item.get("rate")
        symbol = _FX_SYMBOL_BY_CURRENCY.get(currency)
        if not symbol or current_rate is None:
            continue
        series = get_exchange_rate_series(
            (base_ts - pd.Timedelta(days=10)).strftime("%Y-%m-%d"),
            end_ts,
            symbol=symbol,
            allow_partial=True,
        )
        if series.empty:
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


def _calculate_consolidated_average_buy_price(ticker: str) -> float | None:
    """모든 계좌의 동일 티커 보유분을 합산해 통합 평균매입가를 계산한다."""
    ticker_key = str(ticker or "").strip().upper()
    if not ticker_key:
        raise ValueError("ticker 값이 필요합니다.")

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

            currency = str(holding.get("currency") or "").strip().upper()
            if currency:
                currencies.add(currency)
            total_quantity += quantity
            total_buy_amount += quantity * average_buy_price

    if len(currencies) > 1:
        raise RuntimeError(f"{ticker_key} 보유 통화가 여러 개라 통합 평균단가를 계산할 수 없습니다: {sorted(currencies)}")
    if total_quantity <= 0:
        return None
    return total_buy_amount / total_quantity


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

    # 최근 공식 iNAV 기준일과 비교 기준 iNAV 히스토리 조회
    today_str = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d")
    latest_history = get_previous_stock_cache_meta_history(ticker_type, ticker, today_str)
    prev_nav = None
    portfolio_change_base_date = None
    if latest_history and "meta_cache" in latest_history:
        latest_history_nav = latest_history["meta_cache"].get("nav")
        portfolio_change_base_date = str(latest_history.get("date") or "").strip() or None
        if (
            nav_value is not None
            and latest_history_nav is not None
            and float(nav_value) == float(latest_history_nav)
            and portfolio_change_base_date
        ):
            prev_history = get_previous_stock_cache_meta_history(ticker_type, ticker, portfolio_change_base_date)
            if prev_history and "meta_cache" in prev_history:
                prev_nav = prev_history["meta_cache"].get("nav")
        else:
            prev_nav = latest_history_nav

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
        "dividend_yield_ttm": float(meta_cache["dividend_yield_ttm"]) if meta_cache.get("dividend_yield_ttm") is not None else None,
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

    local_value = value.astimezone(ZoneInfo(timezone_name)) if value.tzinfo else value.replace(
        tzinfo=timezone.utc,
    ).astimezone(ZoneInfo(timezone_name))
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
) -> pd.DataFrame:
    country = str(country_code or "").strip().lower()
    if country not in {"kor", "au", "us"}:
        return df

    try:
        realtime_map = get_realtime_snapshot(country, [ticker])
    except Exception:
        return df

    realtime_entry = realtime_map.get(str(ticker or "").strip().upper()) or {}
    now_val = realtime_entry.get("nowVal")
    if now_val is None:
        return df

    try:
        realtime_price = float(now_val)
    except (TypeError, ValueError):
        return df

    if realtime_price <= 0:
        return df

    target_trading_day = _resolve_realtime_target_trading_day(country)
    latest_trading_day = (target_trading_day or get_latest_trading_day(country)).normalize()
    adjusted = df.copy()

    if adjusted.empty:
        return adjusted

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
    return adjusted


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
                result.append({
                    "ticker": tkr,
                    "name": name,
                    "ticker_type": ticker_type,
                    "country_code": country_code,
                    "is_etf": bool(etf.get("is_etf", False)),
                    "has_holdings": bool(etf.get("has_holdings", False)),
                })
    return result


@router.get("/resolve")
def resolve_ticker(
    ticker: str = Query(...),
    _: None = Depends(require_internal_token),
) -> dict[str, object]:
    """직접 진입용 티커 메타데이터를 반환합니다."""

    return _resolve_ticker_meta_item(ticker)


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
                "has_holdings": bool(etf.get("has_holdings", False)),
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


@router.get("")
def get_ticker_detail(
    ticker: str = Query(...),
    ticker_type: str = Query(...),
    country_code: str = Query(default="kor"),
    _: None = Depends(require_internal_token),
) -> dict[str, object]:
    settings = load_common_settings()
    cache_start_date = str(settings.get("CACHE_START_DATE") or "").strip()
    if not cache_start_date:
        raise RuntimeError("CACHE_START_DATE 설정이 필요합니다.")

    fetch_error: str | None = None
    try:
        df = fetch_ohlcv(
            ticker,
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
            "my_average_buy_price": _calculate_consolidated_average_buy_price(ticker),
            "error": fetch_error or "가격 데이터를 가져오지 못했습니다.",
        }

    df = df.sort_index()
    df = _apply_realtime_snapshot_to_dataframe(df, ticker=ticker, country_code=country_code)

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

    holdings: list[dict[str, object]] = []
    holdings_as_of_date: str | None = None
    holdings_price_as_of_date: str | None = None
    holdings_error: str | None = None
    etf_info: dict[str, object] | None = None
    us_pool_tickers: set[str] = set()
    kor_pool_tickers: set[str] = set()
    domestic_etf_tickers: set[str] = set()
    if str(country_code or "").strip().lower() == "kor":
        cache_document = get_stock_cache_meta(ticker_type, ticker)
        holdings_cache = dict(cache_document.get("holdings_cache") or {}) if isinstance(cache_document, dict) else {}
        holdings = list(holdings_cache.get("items") or [])
        etf_info = _build_korean_etf_info_payload(
            ticker=ticker,
            ticker_type=ticker_type,
            cache_document=cache_document if isinstance(cache_document, dict) else None,
            latest_row=rows[-1] if rows else None,
            holdings=holdings,
        )
        holdings_as_of_date = str(holdings_cache.get("reference_date") or "").strip() or None
        if not holdings:
            holdings_error = (
                "구성종목 캐시가 없습니다. "
                "python scripts/stock_meta_cache_updater.py 실행이 필요합니다."
            )
        elif not holdings_as_of_date:
            holdings_error = "구성종목 캐시 기준일(reference_date)이 없습니다."
        else:
            us_pool_tickers = _load_us_pool_ticker_set()
            kor_pool_tickers = _load_kor_pool_ticker_set()
            domestic_etf_tickers = _load_domestic_etf_ticker_set()

            # 구성종목이 수천 개인 글로벌 ETF(예: 0060H0) 는 yfinance 호출이 폭주해
            # 응답이 30초 이상 걸리므로, 비중 상위 종목으로 가격 조회를 제한한다.
            _HOLDINGS_PRICE_FETCH_LIMIT = 100
            priced_holdings, holdings_price_as_of_date = enrich_component_prices(
                holdings,
                price_fetch_limit=_HOLDINGS_PRICE_FETCH_LIMIT,
                cumulative_base_date=str(etf_info.get("portfolio_change_base_date") or "") if etf_info else None,
            )
            enriched_holdings: list[dict[str, object]] = []
            for enriched_item in priced_holdings:
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
                base_date = str(etf_info.get("portfolio_change_base_date") or "").strip() or None
                etf_info["fx_rates"] = _build_cumulative_fx_rates_for_holdings(
                    holdings,
                    get_exchange_rates(),
                    base_date,
                )

    return {
        "ticker": ticker,
        "rows": rows,
        "etf_info": etf_info,
        "holdings": holdings,
        "holdings_as_of_date": holdings_as_of_date,
        "holdings_price_as_of_date": holdings_price_as_of_date,
        "holdings_error": holdings_error,
        "my_average_buy_price": _calculate_consolidated_average_buy_price(ticker),
    }
