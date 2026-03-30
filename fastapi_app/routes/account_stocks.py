from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from fastapi_app.dependencies import require_internal_token
from utils.account_stocks_service import load_account_stocks_data
from utils.account_stocks_io import get_account_targets, save_account_targets

router = APIRouter(prefix="/internal/account-stocks", tags=["account-stocks"])

class TargetItemPayload(BaseModel):
    ticker: str
    ratio: float = Field(..., gt=0, le=100, description="비중 (0.1 ~ 100.0%)")
    name: str | None = None

class AccountTargetsPayload(BaseModel):
    account_id: str
    targets: list[TargetItemPayload]

class SingleTargetPayload(BaseModel):
    account_id: str
    ticker: str
    ratio: float = Field(..., gt=0, le=100, description="비중 (0.1 ~ 100.0%)")
    name: str | None = None

class SingleDeletePayload(BaseModel):
    account_id: str
    ticker: str

@router.get("")
def get_account_stocks(
    account_id: str | None = Query(default=None),
    _: None = Depends(require_internal_token),
) -> dict[str, object]:
    return load_account_stocks_data(account_id)

@router.post("/batch")
def post_account_stocks_batch(
    payload: AccountTargetsPayload,
    _: None = Depends(require_internal_token),
) -> dict[str, object]:
    items = []
    for item in payload.targets:
        items.append({
            "ticker": item.ticker,
            "ratio": item.ratio,
            "name": item.name or "",
        })
    save_account_targets(payload.account_id, items)
    return {"ok": True}

@router.post("")
def post_account_stocks_single(
    payload: SingleTargetPayload,
    _: None = Depends(require_internal_token),
) -> dict[str, object]:
    # 기존 타겟 리스트를 가져와서 추가/수정합니다.
    targets = get_account_targets(payload.account_id)
    items = []
    found = False
    for t in targets:
        if str(t.get("ticker")).upper() == str(payload.ticker).upper():
            t["ratio"] = payload.ratio
            if payload.name:
                t["name"] = payload.name
            found = True
        items.append(t)
    
    if not found:
        items.append({
            "ticker": payload.ticker,
            "ratio": payload.ratio,
            "name": payload.name or "",
        })

    save_account_targets(payload.account_id, items)
    return {"ok": True, "action": "added/updated", "ticker": payload.ticker}

@router.patch("")
def patch_account_stocks_single(
    payload: SingleTargetPayload,
    _: None = Depends(require_internal_token),
) -> dict[str, object]:
    # POST와 동일하게 취급 가능합니다. (비중 수정)
    return post_account_stocks_single(payload)

@router.delete("")
def delete_account_stocks_single(
    payload: SingleDeletePayload,
    _: None = Depends(require_internal_token),
) -> dict[str, object]:
    targets = get_account_targets(payload.account_id)
    items = [t for t in targets if str(t.get("ticker")).upper() != str(payload.ticker).upper()]
    save_account_targets(payload.account_id, items)
    return {"ok": True, "action": "deleted", "ticker": payload.ticker}
