from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from fastapi_app.dependencies import require_internal_token
from utils.us_stock_market_service import load_us_stock_market

router = APIRouter(prefix="/internal/us-market-stocks", tags=["us-market-stocks"])


@router.get("")
def get_us_market_stocks(
    market: Annotated[str, Query(pattern="^(NYS|NSQ)$")],
    limit: Annotated[int, Query(ge=50, le=200)],
    min_market_cap_ukm: Annotated[int, Query(ge=0)] = 400,
    _: None = Depends(require_internal_token),
) -> dict[str, object]:
    return load_us_stock_market(market=market, limit=limit, min_market_cap_ukm=min_market_cap_ukm)
