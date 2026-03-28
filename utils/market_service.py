from __future__ import annotations

from datetime import date, datetime
from typing import Any

import requests

from utils.db_manager import get_db_connection


def _normalize_number(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _normalize_nullable_number(value: Any) -> float | None:
    if value in (None, "", "-"):
        return None

    try:
        return float(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _to_updated_at_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _load_kor_etf_realtime_snapshot(tickers: list[str]) -> dict[str, dict[str, float | None]]:
    if not tickers:
        return {}

    response = requests.get(
        "https://finance.naver.com/api/sise/etfItemList.nhn",
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0.0.0 Safari/537.36"
            ),
            "Referer": "https://finance.naver.com/sise/etfList.nhn",
            "Accept": "application/json, text/plain, */*",
        },
        timeout=20,
    )
    response.raise_for_status()

    payload = response.json()
    items = payload.get("result", {}).get("etfItemList")
    if not isinstance(items, list):
        raise ValueError("네이버 ETF 실시간 스냅샷 응답 형식이 올바르지 않습니다.")

    ticker_set = {ticker.upper() for ticker in tickers if ticker}
    snapshot: dict[str, dict[str, float | None]] = {}

    for item in items:
        code = str(item.get("itemcode") or "").strip().upper()
        if not code or code not in ticker_set:
            continue

        now_val = _normalize_nullable_number(item.get("nowVal"))
        nav = _normalize_nullable_number(item.get("nav"))
        change_rate = _normalize_nullable_number(item.get("changeRate"))
        three_month_earn_rate = _normalize_nullable_number(item.get("threeMonthEarnRate"))
        deviation = ((now_val / nav) - 1) * 100 if now_val is not None and nav not in (None, 0) else None

        snapshot[code] = {
            "changeRate": change_rate,
            "nowVal": now_val,
            "nav": nav,
            "deviation": deviation,
            "threeMonthEarnRate": three_month_earn_rate,
        }

    return snapshot


def load_market_data() -> dict[str, Any]:
    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결에 실패했습니다.")

    doc = db.etf_market_master.find_one({"master_id": "kor_etf_market"}) or {}
    rows = doc.get("rows") or []
    if not rows:
        raise RuntimeError("ETF 마켓 캐시가 없습니다. stock_meta_updater를 먼저 실행하세요.")

    normalized_rows = [
        {
            "ticker": _normalize_text(row.get("티커")),
            "name": _normalize_text(row.get("종목명")),
            "listed_at": _normalize_text(row.get("상장일")),
            "prev_volume": int(_normalize_number(row.get("전일거래량"))),
            "market_cap": int(_normalize_number(row.get("시가총액"))),
        }
        for row in rows
    ]

    snapshot = _load_kor_etf_realtime_snapshot([row["ticker"] for row in normalized_rows if row["ticker"]])

    return {
        "updated_at": _to_updated_at_text(doc.get("updated_at")),
        "rows": [
            {
                **row,
                "daily_change_pct": snapshot.get(row["ticker"], {}).get("changeRate"),
                "current_price": snapshot.get(row["ticker"], {}).get("nowVal"),
                "nav": snapshot.get(row["ticker"], {}).get("nav"),
                "deviation": snapshot.get(row["ticker"], {}).get("deviation"),
                "return_3m_pct": snapshot.get(row["ticker"], {}).get("threeMonthEarnRate"),
            }
            for row in normalized_rows
        ],
    }
