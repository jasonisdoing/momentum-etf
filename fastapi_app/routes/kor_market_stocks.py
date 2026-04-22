from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from fastapi_app.dependencies import require_internal_token
from utils.kor_stock_market_service import load_kor_stock_market

router = APIRouter(prefix="/internal/kor-market-stocks", tags=["kor-market-stocks"])


@router.get("")
def get_kor_market_stocks(
    market: str = Query("KOSPI", pattern="^(KOSPI|KOSDAQ)$"),
    limit: int = Query(50, ge=1, le=100),
    min_market_cap: int = Query(1000, ge=0),
    _: None = Depends(require_internal_token),
) -> dict[str, object]:
    return load_kor_stock_market(market=market, limit=limit, min_market_cap=min_market_cap)
