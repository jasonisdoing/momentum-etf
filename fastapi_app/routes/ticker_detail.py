from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, Depends, Query

from fastapi_app.dependencies import require_internal_token
from utils.data_loader import fetch_ohlcv
from utils.stock_list_io import get_etfs
from utils.ticker_registry import load_ticker_type_configs

router = APIRouter(prefix="/internal/ticker-detail", tags=["ticker-detail"])


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


@router.get("")
def get_ticker_detail(
    ticker: str = Query(...),
    ticker_type: str = Query(...),
    country_code: str = Query(default="kor"),
    months: int = Query(default=12),
    _: None = Depends(require_internal_token),
) -> dict[str, object]:
    df = fetch_ohlcv(
        ticker,
        country=country_code,
        months_back=months,
        ticker_type=ticker_type,
    )

    if df is None or df.empty:
        return {"ticker": ticker, "rows": [], "error": "가격 데이터를 가져오지 못했습니다."}

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

    return {"ticker": ticker, "rows": rows}
