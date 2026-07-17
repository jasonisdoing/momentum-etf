from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from fastapi_app.dependencies import require_internal_token
from utils.aus_stock_market_service import load_aus_index_stock_market

router = APIRouter(prefix="/internal/aus-market-stocks", tags=["aus-market-stocks"])


@router.get("/index")
def get_aus_index_stocks(
    index: Annotated[str, Query(pattern="^ASX200$")] = "ASX200",
    min_market_cap_ukm: Annotated[int, Query(ge=0)] = 0,
    _: None = Depends(require_internal_token),
) -> dict[str, object]:
    return load_aus_index_stock_market(index=index, min_market_cap_ukm=min_market_cap_ukm)
