from __future__ import annotations

import os

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from pydantic import BaseModel

from utils.stocks_service import (
    hard_delete_stocks,
    load_active_stocks_table,
    load_deleted_stocks_table,
    restore_deleted_stocks,
    soft_delete_stock,
    update_stock_bucket,
)

router = APIRouter(prefix="/internal/stocks", tags=["stocks"])


def _require_internal_token(x_internal_token: str | None = Header(default=None, alias="X-Internal-Token")) -> None:
    expected_token = os.getenv("FASTAPI_INTERNAL_TOKEN", "").strip()
    if not expected_token:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="FASTAPI_INTERNAL_TOKEN 환경변수가 필요합니다.",
        )

    if x_internal_token != expected_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="내부 API 토큰이 올바르지 않습니다.")


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


@router.get("")
def get_active_stocks(
    account_id: str | None = Query(default=None),
    _: None = Depends(_require_internal_token),
) -> dict[str, object]:
    return load_active_stocks_table(account_id)


@router.patch("")
def patch_active_stock(payload: BucketUpdatePayload, _: None = Depends(_require_internal_token)) -> dict[str, bool]:
    update_stock_bucket(payload.account_id, payload.ticker, payload.bucket_id)
    return {"ok": True}


@router.delete("")
def delete_active_stock(payload: StockDeletePayload, _: None = Depends(_require_internal_token)) -> dict[str, bool]:
    soft_delete_stock(payload.account_id, payload.ticker, payload.reason)
    return {"ok": True}


@router.get("/deleted")
def get_deleted_stocks(
    account_id: str | None = Query(default=None),
    _: None = Depends(_require_internal_token),
) -> dict[str, object]:
    return load_deleted_stocks_table(account_id)


@router.patch("/deleted")
def patch_deleted_stocks(
    payload: DeletedStocksPayload, _: None = Depends(_require_internal_token)
) -> dict[str, int | bool]:
    restored_count = restore_deleted_stocks(payload.account_id, payload.tickers)
    return {"ok": True, "restored_count": restored_count}


@router.delete("/deleted")
def delete_deleted_stocks(
    payload: DeletedStocksPayload, _: None = Depends(_require_internal_token)
) -> dict[str, int | bool]:
    deleted_count = hard_delete_stocks(payload.account_id, payload.tickers)
    return {"ok": True, "deleted_count": deleted_count}
