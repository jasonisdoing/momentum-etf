"""실시간 시세 조회 — 화면이 표시용 현재가·등락률만 자주 갱신할 때 쓴다.

전략 화면의 선정·판정은 무거워서 5분 캐시(`CACHE_TTL_COMPUTE`)를 쓴다. 그 안에서
바뀌는 건 표시용 가격뿐이므로, 계산 전체를 다시 돌리는 대신 이 API 로 가격만 덮어쓴다.
시세 자체는 `price_service` 의 60초 캐시(`CACHE_TTL_LIVE`)를 그대로 쓴다.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from fastapi_app.dependencies import require_internal_token

router = APIRouter(prefix="/internal/quotes", tags=["quotes"])


@router.get("")
def get_quotes(
    country: str = Query(...),
    tickers: str = Query(..., description="쉼표로 구분한 티커 목록"),
    _: None = Depends(require_internal_token),
) -> dict:
    """티커별 현재가와 등락률을 돌려준다 — `{ticker: {price, change_pct}}`."""
    from services.price_service import get_realtime_snapshot

    codes = [code.strip() for code in tickers.split(",") if code.strip()]
    if not codes:
        return {"quotes": {}}

    snapshot = get_realtime_snapshot(country, codes)
    quotes: dict[str, dict[str, float | None]] = {}
    for ticker, entry in (snapshot or {}).items():
        price = (entry or {}).get("nowVal")
        if price is None:
            continue
        change = (entry or {}).get("changeRate")
        quotes[ticker] = {
            "price": float(price),
            "change_pct": float(change) if change is not None else None,
        }
    return {"quotes": quotes}
