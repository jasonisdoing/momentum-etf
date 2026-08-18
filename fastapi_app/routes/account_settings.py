"""계좌 설정(account_settings) 조회/저장 API.

DB(account_settings)가 단일 소스. 값 수정만 지원 — 계좌 추가/삭제는 화면 미지원.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from fastapi_app.dependencies import require_internal_token
from utils.account_settings_store import (
    EDITABLE_KEYS,
    AccountSettingsStoreError,
    create_account,
    delete_account,
    get_account_settings_updated_at,
    load_account_docs,
    save_account_settings,
)
from utils.market_trend_service import INDICES

router = APIRouter(prefix="/internal/account-settings", tags=["account-settings"])


class AccountSettingsUpdatePayload(BaseModel):
    account_id: str
    values: dict[str, Any]


class AccountCreatePayload(BaseModel):
    account_id: str
    name: str
    icon: str = ""
    order: int = 0
    country_code: str = "kor"
    currency: str = "KRW"


class AccountDeletePayload(BaseModel):
    account_id: str


@router.get("")
def get_account_settings_list(_: None = Depends(require_internal_token)) -> dict[str, object]:
    try:
        accounts = load_account_docs()
    except AccountSettingsStoreError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    from utils.momentum_service import pool_options

    return {
        "accounts": accounts,
        "editable_keys": list(EDITABLE_KEYS),
        "market_indices": [{"ticker": item["yf_ticker"], "name": item["name"]} for item in INDICES],
        # 합성 전략 종목풀 선택지 — 다른 화면의 풀 셀렉터와 같은 표기 필드를 쓴다.
        "pool_options": pool_options(),
    }


@router.put("")
def put_account_settings(
    payload: AccountSettingsUpdatePayload, _: None = Depends(require_internal_token)
) -> dict[str, object]:
    try:
        saved = save_account_settings(payload.account_id, payload.values)
    except AccountSettingsStoreError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "ok": True,
        "account_id": payload.account_id,
        "saved": saved,
        "updated_at": get_account_settings_updated_at(payload.account_id),
    }


@router.post("")
def post_account(payload: AccountCreatePayload, _: None = Depends(require_internal_token)) -> dict[str, object]:
    try:
        created = create_account(
            payload.account_id,
            payload.name,
            icon=payload.icon,
            order=payload.order,
            country_code=payload.country_code,
            currency=payload.currency,
        )
    except AccountSettingsStoreError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, **created}


@router.delete("")
def delete_account_route(payload: AccountDeletePayload, _: None = Depends(require_internal_token)) -> dict[str, object]:
    try:
        result = delete_account(payload.account_id)
    except AccountSettingsStoreError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, **result}
