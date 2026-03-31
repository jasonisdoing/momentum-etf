from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends, Query

from fastapi_app.dependencies import require_internal_token
from utils.holdings_detail_service import delete_holding, load_all_holdings_detail, update_holding

router = APIRouter(prefix="/internal/holdings", tags=["holdings"])


@router.get("")
def get_all_holdings(
    account: str | None = Query(default=None),
    _: None = Depends(require_internal_token),
) -> dict[str, Any]:
    return load_all_holdings_detail(account_id=account)


@router.delete("")
def delete_one_holding(
    account: str = Query(...),
    ticker: str = Query(...),
    _: None = Depends(require_internal_token),
) -> dict[str, Any]:
    return delete_holding(account_id=account, ticker=ticker)


@router.patch("")
def patch_one_holding(
    body: dict[str, Any] = Body(...),
    _: None = Depends(require_internal_token),
) -> dict[str, Any]:
    return update_holding(
        account_id=body.get("account_id", ""),
        ticker=body.get("ticker", ""),
        quantity=body.get("quantity"),
        average_buy_price=body.get("average_buy_price"),
    )
