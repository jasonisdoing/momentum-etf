from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query

from fastapi_app.dependencies import require_internal_token
from utils.holdings_detail_service import load_all_holdings_detail

router = APIRouter(prefix="/internal/holdings", tags=["holdings"])


@router.get("")
def get_all_holdings(
    account: str | None = Query(default=None),
    _: None = Depends(require_internal_token),
) -> dict[str, Any]:
    return load_all_holdings_detail(account_id=account)
