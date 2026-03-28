from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from pydantic import BaseModel

from utils.note_service import load_note_page_data, save_note_page_data

router = APIRouter(prefix="/internal/note", tags=["note"])


def _require_internal_token(x_internal_token: str | None = Header(default=None, alias="X-Internal-Token")) -> None:
    expected_token = os.getenv("FASTAPI_INTERNAL_TOKEN", "").strip()
    if not expected_token:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="FASTAPI_INTERNAL_TOKEN 환경변수가 필요합니다.",
        )

    if x_internal_token != expected_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="내부 API 토큰이 올바르지 않습니다.")


class NoteSavePayload(BaseModel):
    account_id: str
    content: str


@router.get("")
def get_note_data(
    account_id: str | None = Query(default=None),
    _: None = Depends(_require_internal_token),
) -> dict[str, Any]:
    return load_note_page_data(account_id)


@router.patch("")
def patch_note_data(payload: NoteSavePayload, _: None = Depends(_require_internal_token)) -> dict[str, str]:
    return save_note_page_data(payload.account_id, payload.content)
