from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
from fastapi import APIRouter, Depends, Query

from config import MARKET_SCHEDULES
from fastapi_app.dependencies import require_internal_token
from services.price_service import get_realtime_snapshot
from services.price_service import get_realtime_snapshot_meta
from services.price_service import get_worldstock_snapshot
from services.price_service import get_yahoo_symbol_snapshot
from services.stock_cache_service import get_stock_cache_meta
from utils.cache_utils import (
    get_cache_refresh_completed_at,
    load_cached_close_series_bulk_before_or_at_with_fallback,
    load_cached_close_series_bulk_with_fallback,
    load_cached_updated_at_bulk_before_or_at_with_fallback,
    load_cached_updated_at_bulk_with_fallback,
)
from utils.data_loader import fetch_ohlcv, get_latest_trading_day, get_trading_days
from utils.kis_market import load_cached_kis_domestic_etf_master
from utils.settings_loader import load_common_settings
from utils.stock_list_io import get_etfs
from utils.ticker_registry import load_ticker_type_configs

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


def _infer_yahoo_symbol_currency(symbol: str) -> str:
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
    us_pool_tickers: set[str] = set()
    kor_pool_tickers: set[str] = set()
    domestic_etf_tickers: set[str] = set()
    if str(country_code or "").strip().lower() == "kor":
        cache_document = get_stock_cache_meta(ticker_type, ticker)
        holdings_cache = dict(cache_document.get("holdings_cache") or {}) if isinstance(cache_document, dict) else {}
        holdings = list(holdings_cache.get("items") or [])
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

            def is_korean_six_digit_holding(item: dict[str, object]) -> bool:
                component_ticker = str(item.get("ticker") or "").strip().upper()
                raw_code = str(item.get("raw_code") or "").strip().upper()
                yahoo_symbol = str(item.get("yahoo_symbol") or "").strip().upper()
                if not component_ticker.isdigit() or len(component_ticker) != 6:
                    return False
                # .KS/.KQ 접미사는 한국 종목 (yfinance 표기)
                if yahoo_symbol and not yahoo_symbol.endswith((".KS", ".KQ")):
                    return False
                if raw_code.startswith("CNE"):
                    return False
                return True

            # 구성종목이 수천 개인 글로벌 ETF(예: 0060H0) 는 yfinance 호출이 폭주해
            # 응답이 30초 이상 걸리므로, 비중 상위 종목으로 가격 조회를 제한한다.
            _HOLDINGS_PRICE_FETCH_LIMIT = 100
            def _weight_value(item: dict[str, object]) -> float:
                try:
                    return float(item.get("weight") or 0.0)
                except (TypeError, ValueError):
                    return 0.0

            holdings_for_pricing = sorted(holdings, key=_weight_value, reverse=True)[
                :_HOLDINGS_PRICE_FETCH_LIMIT
            ]
            pricing_ids = {id(item) for item in holdings_for_pricing}

            korean_tickers: list[str] = []
            us_tickers: list[str] = []
            au_tickers: list[str] = []
            worldstock_codes: list[str] = []
            yahoo_tw_symbols: list[str] = []

            for item in holdings_for_pricing:
                if is_korean_six_digit_holding(item):
                    korean_tickers.append(str(item.get("ticker") or "").strip().upper())
                else:
                    yahoo_sym = str(item.get("yahoo_symbol") or "").strip().upper()
                    if not yahoo_sym:
                        continue
                    if yahoo_sym.endswith(".AX"):
                        bare = yahoo_sym[:-3]
                        au_tickers.append(bare)
                    elif _is_worldstock_symbol(yahoo_sym):
                        worldstock_codes.append(str(item.get("reuters_code") or yahoo_sym).strip().upper())
                    elif _is_yahoo_tw_symbol(yahoo_sym):
                        yahoo_tw_symbols.append(yahoo_sym)
                    else:
                        us_tickers.append(yahoo_sym)

            # 통합 가격 조회: get_realtime_snapshot(country, tickers)
            kor_price_map: dict[str, dict[str, float]] = {}
            us_price_map: dict[str, dict[str, float]] = {}
            au_price_map: dict[str, dict[str, float]] = {}
            worldstock_price_map: dict[str, dict[str, float | str]] = {}
            yahoo_tw_price_map: dict[str, dict[str, float]] = {}

            if korean_tickers:
                try:
                    kor_price_map = get_realtime_snapshot("kor", korean_tickers)
                except Exception:
                    pass
            if us_tickers:
                try:
                    us_price_map = get_realtime_snapshot("us", us_tickers)
                except Exception:
                    pass
            if au_tickers:
                try:
                    au_price_map = get_realtime_snapshot("au", au_tickers)
                except Exception:
                    pass
            if worldstock_codes:
                try:
                    worldstock_price_map = get_worldstock_snapshot(worldstock_codes)
                except Exception:
                    pass
            if yahoo_tw_symbols:
                try:
                    yahoo_tw_price_map = get_yahoo_symbol_snapshot(yahoo_tw_symbols)
                except Exception:
                    pass

            enriched_holdings: list[dict[str, object]] = []
            for item in holdings:
                component_ticker = str(item.get("ticker") or "").strip().upper()
                yahoo_symbol = str(item.get("yahoo_symbol") or "").strip().upper()
                enriched_item = dict(item)
                enriched_item["yahoo_symbol"] = yahoo_symbol or None

                if id(item) not in pricing_ids:
                    enriched_item["current_price"] = None
                    enriched_item["previous_close"] = None
                    enriched_item["change_pct"] = None
                    enriched_item["price_currency"] = None
                elif is_korean_six_digit_holding(item):
                    rt = kor_price_map.get(component_ticker, {})
                    enriched_item["current_price"] = float(rt["nowVal"]) if rt.get("nowVal") is not None else None
                    enriched_item["previous_close"] = float(rt["prevClose"]) if rt.get("prevClose") is not None else None
                    enriched_item["change_pct"] = float(rt["changeRate"]) if rt.get("changeRate") is not None else None
                    enriched_item["price_currency"] = "KRW"
                elif yahoo_symbol.endswith(".AX"):
                    bare = yahoo_symbol[:-3]
                    rt = au_price_map.get(bare, {})
                    enriched_item["current_price"] = float(rt["nowVal"]) if rt.get("nowVal") is not None else None
                    enriched_item["previous_close"] = float(rt["prevClose"]) if rt.get("prevClose") is not None else None
                    enriched_item["change_pct"] = float(rt["changeRate"]) if rt.get("changeRate") is not None else None
                    enriched_item["price_currency"] = "AUD"
                elif _is_worldstock_symbol(yahoo_symbol):
                    lookup_code = str(item.get("reuters_code") or yahoo_symbol).strip().upper()
                    rt = worldstock_price_map.get(lookup_code, {})
                    enriched_item["current_price"] = float(rt["nowVal"]) if rt.get("nowVal") is not None else None
                    enriched_item["previous_close"] = float(rt["prevClose"]) if rt.get("prevClose") is not None else None
                    enriched_item["change_pct"] = float(rt["changeRate"]) if rt.get("changeRate") is not None else None
                    enriched_item["price_currency"] = str(rt.get("currency") or _infer_yahoo_symbol_currency(yahoo_symbol))
                elif _is_yahoo_tw_symbol(yahoo_symbol):
                    rt = yahoo_tw_price_map.get(yahoo_symbol, {})
                    enriched_item["current_price"] = float(rt["nowVal"]) if rt.get("nowVal") is not None else None
                    enriched_item["previous_close"] = float(rt["prevClose"]) if rt.get("prevClose") is not None else None
                    enriched_item["change_pct"] = float(rt["changeRate"]) if rt.get("changeRate") is not None else None
                    enriched_item["price_currency"] = _infer_yahoo_symbol_currency(yahoo_symbol)
                elif "." in yahoo_symbol:
                    enriched_item["current_price"] = None
                    enriched_item["previous_close"] = None
                    enriched_item["change_pct"] = None
                    enriched_item["price_currency"] = None
                else:
                    rt = us_price_map.get(yahoo_symbol or component_ticker, {})
                    enriched_item["current_price"] = float(rt["nowVal"]) if rt.get("nowVal") is not None else None
                    enriched_item["previous_close"] = float(rt["prevClose"]) if rt.get("prevClose") is not None else None
                    enriched_item["change_pct"] = float(rt["changeRate"]) if rt.get("changeRate") is not None else None
                    enriched_item["price_currency"] = "USD"

                enriched_item["is_us_pool_candidate"] = _is_us_pool_candidate(enriched_item)
                enriched_item["in_us_pool"] = component_ticker in us_pool_tickers
                enriched_item["is_kor_pool_candidate"] = _is_kor_pool_candidate(
                    enriched_item,
                    domestic_etf_tickers,
                )
                enriched_item["in_kor_pool"] = component_ticker in kor_pool_tickers

                enriched_holdings.append(enriched_item)
            holdings = enriched_holdings

    return {
        "ticker": ticker,
        "rows": rows,
        "holdings": holdings,
        "holdings_as_of_date": holdings_as_of_date,
        "holdings_price_as_of_date": holdings_price_as_of_date,
        "holdings_error": holdings_error,
    }
