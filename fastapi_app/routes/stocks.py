from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from fastapi_app.dependencies import require_internal_token
from utils.stocks_service import (
    add_active_stock,
    hard_delete_stocks,
    load_active_stocks_table,
    load_deleted_stocks_table,
    movable_pools,
    move_active_stock,
    refresh_single_stock,
    restore_deleted_stocks,
    toggle_exclude_from_ranking,
    update_stock_bucket,
    validate_stock_candidate,
)
from utils.stocks_service import (
    delete_active_stock as delete_active_stock_entry,
)

router = APIRouter(prefix="/internal/stocks", tags=["stocks"])


class BucketUpdatePayload(BaseModel):
    ticker_type: str
    ticker: str
    bucket_id: int


class StockExcludePayload(BaseModel):
    ticker_type: str
    ticker: str
    exclude: bool


class StockDeletePayload(BaseModel):
    ticker_type: str
    ticker: str


class DeletedStocksPayload(BaseModel):
    ticker_type: str
    tickers: list[str]


class StockRefreshPayload(BaseModel):
    ticker_type: str
    ticker: str


class StockValidationPayload(BaseModel):
    ticker_type: str
    ticker: str


class StockMemoPayload(BaseModel):
    ticker: str
    memo: str = ""


class StockCreatePayload(BaseModel):
    ticker_type: str
    ticker: str
    bucket_id: int


class StockMovePayload(BaseModel):
    from_pool: str
    to_pool: str
    ticker: str


@router.get("")
def get_active_stocks(
    ticker_type: str | None = Query(default=None),
    _: None = Depends(require_internal_token),
) -> dict[str, object]:
    return load_active_stocks_table(ticker_type)


@router.patch("")
def patch_active_stock(payload: BucketUpdatePayload, _: None = Depends(require_internal_token)) -> dict[str, bool]:
    update_stock_bucket(payload.ticker_type, payload.ticker, payload.bucket_id)
    return {"ok": True}


@router.patch("/memo")
def patch_stock_memo(payload: StockMemoPayload, _: None = Depends(require_internal_token)) -> dict[str, bool]:
    """종목 메모 — 계좌가 아니라 종목에 붙는다(자산 관리·순위 화면 공용)."""
    from utils.stock_memo_store import set_stock_memo

    if not set_stock_memo(payload.ticker, payload.memo):
        raise HTTPException(status_code=404, detail=f"종목을 찾을 수 없습니다: {payload.ticker}")
    return {"ok": True}


@router.patch("/exclude")
def patch_exclude_stock(payload: StockExcludePayload, _: None = Depends(require_internal_token)) -> dict[str, bool]:
    toggle_exclude_from_ranking(payload.ticker_type, payload.ticker, payload.exclude)
    return {"ok": True}


@router.delete("")
def delete_active_stock(payload: StockDeletePayload, _: None = Depends(require_internal_token)) -> dict[str, bool]:
    delete_active_stock_entry(payload.ticker_type, payload.ticker)
    return {"ok": True}


@router.post("/refresh")
def post_refresh_stock(payload: StockRefreshPayload, _: None = Depends(require_internal_token)) -> dict[str, str]:
    return refresh_single_stock(payload.ticker_type, payload.ticker)


@router.post("/validate")
def post_validate_stock(
    payload: StockValidationPayload, _: None = Depends(require_internal_token)
) -> dict[str, object]:
    return validate_stock_candidate(payload.ticker_type, payload.ticker)


@router.post("")
def post_active_stock(payload: StockCreatePayload, _: None = Depends(require_internal_token)) -> dict[str, object]:
    return add_active_stock(payload.ticker_type, payload.ticker, payload.bucket_id)


@router.get("/movable-pools")
def get_movable_pools(ticker_type: str = Query(...), _: None = Depends(require_internal_token)) -> dict[str, object]:
    """그 종목풀에서 옮길 수 있는 대상 풀 목록 (같은 국가·구분만)."""
    return {"pools": movable_pools(ticker_type)}


@router.post("/move")
def post_move_stock(payload: StockMovePayload, _: None = Depends(require_internal_token)) -> dict[str, object]:
    """종목 하나를 다른 종목풀로 옮긴다. 종목마다 한 번씩 호출한다(화면이 진행도를 보여준다)."""
    return move_active_stock(payload.from_pool, payload.to_pool, payload.ticker)


@router.get("/deleted")
def get_deleted_stocks(
    ticker_type: str | None = Query(default=None),
    _: None = Depends(require_internal_token),
) -> dict[str, object]:
    return load_deleted_stocks_table(ticker_type)


@router.patch("/deleted")
def patch_deleted_stocks(
    payload: DeletedStocksPayload, _: None = Depends(require_internal_token)
) -> dict[str, int | bool]:
    restored_count = restore_deleted_stocks(payload.ticker_type, payload.tickers)
    return {"ok": True, "restored_count": restored_count}


@router.delete("/deleted")
def delete_deleted_stocks(
    payload: DeletedStocksPayload, _: None = Depends(require_internal_token)
) -> dict[str, int | bool]:
    deleted_count = hard_delete_stocks(payload.ticker_type, payload.tickers)
    return {"ok": True, "deleted_count": deleted_count}
