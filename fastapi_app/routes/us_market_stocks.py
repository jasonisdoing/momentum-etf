from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from fastapi_app.dependencies import require_internal_token
from utils.index_pool_alert import US_ALERT_BASIS
from utils.us_stock_market_service import load_index_stock_market, load_us_stock_market

router = APIRouter(prefix="/internal/us-market-stocks", tags=["us-market-stocks"])


@router.get("")
def get_us_market_stocks(
    market: Annotated[str, Query(pattern="^(NYS|NSQ)$")],
    limit: Annotated[int, Query(ge=50, le=200)],
    min_market_cap_ukm: Annotated[int, Query(ge=0)] = 0,
    _: None = Depends(require_internal_token),
) -> dict[str, object]:
    return load_us_stock_market(market=market, limit=limit, min_market_cap_ukm=min_market_cap_ukm)


@router.get("/index")
def get_us_index_stocks(
    index: Annotated[str, Query(pattern="^(USTOP1000|SP500|NDX100)$")],
    min_market_cap_ukm: Annotated[int, Query(ge=0)] = 0,
    _: None = Depends(require_internal_token),
) -> dict[str, object]:
    return load_index_stock_market(index=index, min_market_cap_ukm=min_market_cap_ukm)


@router.get("/alert-basis")
def get_us_alert_basis(_: None = Depends(require_internal_token)) -> dict[str, str]:
    """슬랙 미등록 알림의 대상 명단 설명 — 고정 문구라 종목 데이터 로딩과 따로 준다."""
    return {"alert_basis": US_ALERT_BASIS}
