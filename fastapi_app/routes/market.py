from __future__ import annotations

from fastapi import APIRouter, Depends

from fastapi_app.dependencies import require_internal_token
from utils.market_service import load_market_data

router = APIRouter(prefix="/internal/market", tags=["market"])


@router.get("")
def get_market_data(_: None = Depends(require_internal_token)) -> dict[str, object]:
    return load_market_data()
