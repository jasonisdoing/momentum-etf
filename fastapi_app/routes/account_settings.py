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
    get_account_settings_updated_at,
    load_account_docs,
    save_account_settings,
)

router = APIRouter(prefix="/internal/account-settings", tags=["account-settings"])


class AccountSettingsUpdatePayload(BaseModel):
    account_id: str
    values: dict[str, Any]


@router.get("")
def get_account_settings_list(_: None = Depends(require_internal_token)) -> dict[str, object]:
    try:
        accounts = load_account_docs()
    except AccountSettingsStoreError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"accounts": accounts, "editable_keys": list(EDITABLE_KEYS)}


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
