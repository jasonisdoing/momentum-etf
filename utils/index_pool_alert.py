"""시장 화면 「통합」 종목의 종목풀 미등록 알림 — 미국·한국 공용."""

from __future__ import annotations

from typing import Any

from utils.index_constituents_loader import US_SP500_MARKET_CAP_LIMIT
from utils.kor_stock_market_service import KOR_MARKET_TOP_COUNTS
from utils.market_service import load_ticker_pool_type_map
from utils.notification import send_slack_message_v2

# 명단별 표시 이름 — 개수는 명단을 정하는 상수에서 가져온다(문구와 실제 명단이 갈리지 않게).
_LIST_LABELS: dict[str, str] = {
    "SP500": f"S&P500 시총 상위 {US_SP500_MARKET_CAP_LIMIT}",
    "NDX100": "NDX100",
    "KOSPI": f"코스피 시총 상위 {KOR_MARKET_TOP_COUNTS['KOSPI']}",
    "KOSDAQ": f"코스닥 시총 상위 {KOR_MARKET_TOP_COUNTS['KOSDAQ']}",
}


def notify_unregistered_market_stocks(country: str, list_items: dict[str, list[dict[str, Any]]]) -> int:
    """시장 화면 「통합」 명단 중 그 국가의 어느 종목풀에도 없는 종목을 한 메시지로 알린다.

    명단은 호출부가 화면 「통합」 보기와 **같은 함수**로 만들어 넘긴다 — 미국은 S&P500 시총
    상위·나스닥100, 한국은 코스피·코스닥 시총 상위다. 알림 대상과 화면이 갈리면 화면에 보이는
    종목이 빠지거나 안 보이는 종목이 알림에 섞인다.
    """
    registered = set(load_ticker_pool_type_map(country))
    missing_by_list: dict[str, list[str]] = {}
    for key, items in list_items.items():
        # 같은 회사의 다른 주식 종류(미국 GOOG 등)는 화면도 기본으로 숨기므로 뺀다.
        tickers = {str(item.get("ticker") or "").strip().upper() for item in items if not item.get("duplicate_class")}
        missing = sorted(ticker for ticker in tickers - registered if ticker)
        if missing:
            missing_by_list[key] = missing

    if not missing_by_list:
        return 0

    country_label = {"us": "미국", "kor": "한국"}[country]
    unique_missing = set().union(*(set(tickers) for tickers in missing_by_list.values()))
    lines = [
        "<!channel>",
        f"⚠️ *{country_label} 시장 통합 종목 중 종목풀 미등록 {len(unique_missing)}개*",
    ]
    for key, tickers in missing_by_list.items():
        lines.append(f"*{_LIST_LABELS[key]} ({len(tickers)}개)*: {', '.join(tickers)}")

    if not send_slack_message_v2("\n".join(lines)):
        raise RuntimeError(f"{country_label} 시장 통합 종목 미등록 알림을 슬랙으로 보내지 못했습니다.")
    return len(unique_missing)
