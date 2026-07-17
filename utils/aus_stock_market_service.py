"""호주 개별주 인덱스 구성종목 서비스."""

from __future__ import annotations

from typing import Any

from utils.index_constituents_loader import load_index_constituents, load_index_meta
from utils.market_service import load_ticker_pool_map
from utils.portfolio_io import load_all_holding_tickers


def _normalize_asx_ticker(value: object) -> str:
    ticker = str(value or "").strip().upper()
    if not ticker:
        return ""
    if ticker.startswith("ASX:"):
        return ticker
    if ticker.endswith(".AX"):
        return f"ASX:{ticker[:-3]}"
    return f"ASX:{ticker}"


def load_aus_index_stock_market(index: str = "ASX200", min_market_cap_ukm: int = 0) -> dict[str, Any]:
    """S&P/ASX 200 구성종목을 JSON에서 읽어 반환한다."""
    if str(index or "").strip().upper() != "ASX200":
        raise ValueError(f"지원하지 않는 호주 인덱스입니다: {index}")

    constituents = load_index_constituents("ASX200")
    meta = load_index_meta("ASX200")

    ticker_pool_map = load_ticker_pool_map(country_code="au")
    held_tickers = load_all_holding_tickers(country_code="au")
    held_keys = {_normalize_asx_ticker(ticker) for ticker in held_tickers}
    min_market_cap_aud = min_market_cap_ukm * 100_000_000

    rows: list[dict[str, Any]] = []
    for item in constituents:
        ticker = _normalize_asx_ticker(item.get("ticker"))
        if not ticker:
            continue
        market_cap = item.get("market_cap")
        if min_market_cap_aud > 0 and (market_cap is None or market_cap < min_market_cap_aud):
            continue
        rows.append(
            {
                "rank": 0,
                "ticker": ticker,
                "name": item.get("name") or ticker,
                "english_name": item.get("name") or "",
                "industry": item.get("industry") or item.get("sector") or "",
                "sector": item.get("sector") or "",
                "market": "ASX",
                "ticker_pools": ", ".join(ticker_pool_map.get(ticker.replace("ASX:", ""), []) or ticker_pool_map.get(ticker, [])),
                "is_held": ticker in held_keys,
                "current_price": item.get("current_price") or item.get("return_3m_latest_price"),
                "change_pct": item.get("change_pct"),
                "volume": item.get("volume"),
                "market_cap": market_cap,
                "return_1m_base_date": item.get("return_1m_base_date"),
                "return_1m_base_price": item.get("return_1m_base_price"),
                "return_1m_pct": item.get("return_1m_pct"),
                "return_3m_base_date": item.get("return_3m_base_date"),
                "return_3m_base_price": item.get("return_3m_base_price"),
                "return_3m_pct": item.get("return_3m_pct"),
                "return_12m_base_date": item.get("return_12m_base_date"),
                "return_12m_base_price": item.get("return_12m_base_price"),
                "return_12m_pct": item.get("return_12m_pct"),
                "mdd_12m_pct": item.get("mdd_12m_pct"),
            }
        )

    rows.sort(key=lambda row: (-(row["market_cap"] or 0), row["ticker"]))
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx

    return {
        "index": "ASX200",
        "updated_at": meta.get("updated_at", ""),
        "total_count": len(rows),
        "count": len(rows),
        "rows": rows,
    }
