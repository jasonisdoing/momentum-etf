from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from fastapi_app.dependencies import require_internal_token
from utils.stocks_service import (
    add_active_stock,
    hard_delete_stocks,
    load_active_stocks_table,
    load_deleted_stocks_table,
    restore_deleted_stocks,
    soft_delete_stock,
    update_stock_bucket,
    validate_stock_candidate,
)

router = APIRouter(prefix="/internal/stocks", tags=["stocks"])


class BucketUpdatePayload(BaseModel):
    account_id: str
    ticker: str
    bucket_id: int


class StockDeletePayload(BaseModel):
    account_id: str
    ticker: str
    reason: str | None = None


class DeletedStocksPayload(BaseModel):
    account_id: str
    tickers: list[str]


class StockValidationPayload(BaseModel):
    account_id: str
    ticker: str


class StockCreatePayload(BaseModel):
    account_id: str
    ticker: str
    bucket_id: int


@router.get("")
def get_active_stocks(
    account_id: str | None = Query(default=None),
    _: None = Depends(require_internal_token),
) -> dict[str, object]:
    return load_active_stocks_table(account_id)


@router.patch("")
def patch_active_stock(payload: BucketUpdatePayload, _: None = Depends(require_internal_token)) -> dict[str, bool]:
    update_stock_bucket(payload.account_id, payload.ticker, payload.bucket_id)
    return {"ok": True}


@router.delete("")
def delete_active_stock(payload: StockDeletePayload, _: None = Depends(require_internal_token)) -> dict[str, bool]:
    soft_delete_stock(payload.account_id, payload.ticker, payload.reason)
    return {"ok": True}


@router.post("/validate")
def post_validate_stock(
    payload: StockValidationPayload, _: None = Depends(require_internal_token)
) -> dict[str, object]:
    return validate_stock_candidate(payload.account_id, payload.ticker)


@router.post("")
def post_active_stock(payload: StockCreatePayload, _: None = Depends(require_internal_token)) -> dict[str, object]:
    return add_active_stock(payload.account_id, payload.ticker, payload.bucket_id)


@router.get("/deleted")
def get_deleted_stocks(
    account_id: str | None = Query(default=None),
    _: None = Depends(require_internal_token),
) -> dict[str, object]:
    return load_deleted_stocks_table(account_id)


@router.patch("/deleted")
def patch_deleted_stocks(
    payload: DeletedStocksPayload, _: None = Depends(require_internal_token)
) -> dict[str, int | bool]:
    restored_count = restore_deleted_stocks(payload.account_id, payload.tickers)
    return {"ok": True, "restored_count": restored_count}


@router.delete("/deleted")
def delete_deleted_stocks(
    payload: DeletedStocksPayload, _: None = Depends(require_internal_token)
) -> dict[str, int | bool]:
    deleted_count = hard_delete_stocks(payload.account_id, payload.tickers)
    return {"ok": True, "deleted_count": deleted_count}
