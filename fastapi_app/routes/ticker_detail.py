from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, Depends, Query

from fastapi_app.dependencies import require_internal_token
from services.etf_holdings_service import (
    fetch_foreign_stock_price_snapshot,
    fetch_korean_etf_holdings_from_naver,
    fetch_korean_stock_price_snapshot,
)
from utils.cache_utils import load_cached_close_series_bulk_with_fallback
from utils.data_loader import fetch_ohlcv
from utils.settings_loader import load_common_settings
from utils.stock_list_io import get_etfs
from utils.ticker_registry import load_ticker_type_configs

router = APIRouter(prefix="/internal/ticker-detail", tags=["ticker-detail"])


def _is_cash_holding(item: dict[str, object]) -> bool:
    ticker = str(item.get("ticker") or "").strip().upper()
    raw_code = str(item.get("raw_code") or "").strip().upper()
    name = str(item.get("name") or item.get("raw_name") or "").strip()
    if ticker.startswith("CASH") or ticker.startswith("KRD"):
        return True
    if raw_code.startswith("CASH") or raw_code.startswith("KRD"):
        return True
    return "현금" in name


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


@router.get("/tickers")
def get_all_tickers(
    _: None = Depends(require_internal_token),
) -> list[dict[str, str]]:
    """전체 종목타입의 활성 종목 목록을 반환합니다."""
    configs = load_ticker_type_configs()
    result: list[dict[str, str]] = []
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

    for config in configs:
        ticker_type = config["ticker_type"]
        country_code = config.get("country_code", "")
        ticker_type_name = str(config.get("name") or ticker_type).strip()
        etfs = get_etfs(ticker_type)
        tickers = [str(item.get("ticker") or "").strip().upper() for item in etfs if item.get("ticker")]
        close_series_map = load_cached_close_series_bulk_with_fallback(ticker_type, tickers)
        ticker_type_items: list[dict[str, object]] = []

        for etf in etfs:
            ticker = str(etf.get("ticker") or "").strip().upper()
            if not ticker:
                continue

            current_price, change_pct = _build_price_snapshot(close_series_map.get(ticker))
            item = {
                "ticker": ticker,
                "name": str(etf.get("name") or "").strip(),
                "ticker_type": ticker_type,
                "country_code": country_code,
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

    return {
        "tickers": ticker_items,
        "top_movers_by_type": top_movers_by_type,
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

    df = fetch_ohlcv(
        ticker,
        country=country_code,
        date_range=[cache_start_date, None],
        ticker_type=ticker_type,
    )

    if df is None or df.empty:
        return {
            "ticker": ticker,
            "rows": [],
            "holdings": [],
            "holdings_as_of_date": None,
            "holdings_price_as_of_date": None,
            "holdings_error": None,
            "error": "가격 데이터를 가져오지 못했습니다.",
        }

    df = df.sort_index()

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
    if str(country_code or "").strip().lower() == "kor":
        try:
            holdings_document = fetch_korean_etf_holdings_from_naver(ticker)
        except Exception as exc:
            holdings_error = str(exc).strip() or "구성종목 데이터를 확인할 수 없습니다."
        else:
            holdings = [item for item in list(holdings_document.get("holdings") or []) if not _is_cash_holding(item)]
            holdings_as_of_date = str(holdings_document.get("as_of_date") or "").strip() or None
            if holdings and holdings_as_of_date:
                korean_tickers = [
                    str(item.get("ticker") or "").strip().upper()
                    for item in holdings
                    if str(item.get("ticker") or "").strip().upper().isdigit()
                    and len(str(item.get("ticker") or "").strip().upper()) == 6
                ]
                foreign_symbols = [
                    str(item.get("yahoo_symbol") or "").strip().upper()
                    for item in holdings
                    if not (
                        str(item.get("ticker") or "").strip().upper().isdigit()
                        and len(str(item.get("ticker") or "").strip().upper()) == 6
                    )
                    and str(item.get("yahoo_symbol") or "").strip()
                ]
                price_snapshot_map = fetch_korean_stock_price_snapshot(korean_tickers, holdings_as_of_date)
                foreign_price_snapshot_map, holdings_price_as_of_date = fetch_foreign_stock_price_snapshot(foreign_symbols)
                enriched_holdings: list[dict[str, object]] = []
                for item in holdings:
                    component_ticker = str(item.get("ticker") or "").strip().upper()
                    yahoo_symbol = str(item.get("yahoo_symbol") or "").strip().upper()
                    if component_ticker.isdigit() and len(component_ticker) == 6:
                        snapshot = price_snapshot_map.get(component_ticker, {})
                    else:
                        snapshot = foreign_price_snapshot_map.get(yahoo_symbol or component_ticker, {})
                    enriched_item = dict(item)
                    enriched_item["current_price"] = snapshot.get("current_price")
                    enriched_item["previous_close"] = snapshot.get("previous_close")
                    enriched_item["change_pct"] = snapshot.get("change_pct")
                    enriched_item["price_currency"] = snapshot.get("price_currency")
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
