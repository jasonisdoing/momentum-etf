"""지수 구성종목 배치의 종목풀 미등록 알림."""

from __future__ import annotations

from typing import Any

from utils.market_service import load_ticker_pool_type_map
from utils.notification import app_link, send_slack_message_v2


def notify_unregistered_index_stocks(country: str, index_items: dict[str, list[dict[str, Any]]]) -> int:
    """해당 국가의 어느 종목풀에도 없는 지수 종목을 한 메시지로 알린다."""
    registered = set(load_ticker_pool_type_map(country))
    missing_by_index: dict[str, list[str]] = {}
    for index, items in index_items.items():
        tickers = {
            str(item.get("ticker") or "").strip().upper()
            for item in items
            if country != "us" or not item.get("duplicate_class")
        }
        missing = sorted(ticker for ticker in tickers - registered if ticker)
        if missing:
            missing_by_index[index] = missing

    if not missing_by_index:
        return 0

    country_label = {"us": "미국", "kor": "한국"}[country]
    unique_missing = set().union(*(set(tickers) for tickers in missing_by_index.values()))
    lines = [
        "<!channel>",
        f"⚠️ *{country_label} 지수 구성종목 중 종목풀 미등록 {len(unique_missing)}개*",
    ]
    for index, tickers in missing_by_index.items():
        label = "S&P500 시총 상위 300" if index == "SP500" else index
        lines.append(f"*{label} ({len(tickers)}개)*: {', '.join(tickers)}")

    if not send_slack_message_v2("\n".join(lines)):
        raise RuntimeError(f"{country_label} 지수 구성종목 미등록 알림을 슬랙으로 보내지 못했습니다.")
    return len(unique_missing)
