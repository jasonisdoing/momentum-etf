from __future__ import annotations

import threading
import time
from typing import Any

import pandas as pd
from pykrx import stock

_CACHE_TTL_SECONDS = 60 * 30
_cache_lock = threading.Lock()
_holdings_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}


def _normalize_ticker(ticker: str) -> str:
    return str(ticker or "").strip().upper().replace("ASX:", "")


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        return None
    return float(parsed)


def load_korean_etf_holdings(ticker: str, date: str | None = None) -> list[dict[str, Any]]:
    normalized_ticker = _normalize_ticker(ticker)
    if not normalized_ticker:
        raise ValueError("ETF 티커가 필요합니다.")
    normalized_date = str(date or "").strip()

    now = time.time()
    cache_key = f"{normalized_ticker}:{normalized_date or 'latest'}"
    with _cache_lock:
        cached = _holdings_cache.get(cache_key)
        if cached and now - cached[0] < _CACHE_TTL_SECONDS:
            return cached[1]

    try:
        df = stock.get_etf_portfolio_deposit_file(normalized_ticker, normalized_date or None)
    except Exception:
        return []
    if df is None or df.empty:
        return []

    working_df = df.copy()
    working_df.index = working_df.index.map(lambda value: str(value).strip().upper())
    working_df["종목명"] = [stock.get_market_ticker_name(component_ticker) for component_ticker in working_df.index]

    records: list[dict[str, Any]] = []
    for component_ticker, row in working_df.sort_values("비중", ascending=False).iterrows():
        contracts = _to_float(row.get("계약수"))
        amount = _to_float(row.get("금액"))
        weight = _to_float(row.get("비중"))
        records.append(
            {
                "ticker": component_ticker,
                "name": str(row.get("종목명") or "").strip(),
                "contracts": int(contracts) if contracts is not None else None,
                "amount": int(amount) if amount is not None else None,
                "weight": round(weight, 2) if weight is not None else None,
            }
        )

    with _cache_lock:
        _holdings_cache[cache_key] = (now, records)

    return records
