from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel

from utils.cash_service import load_cash_accounts, save_cash_accounts

router = APIRouter(prefix="/internal/cash", tags=["cash"])


def _require_internal_token(x_internal_token: str | None = Header(default=None, alias="X-Internal-Token")) -> None:
    expected_token = os.getenv("FASTAPI_INTERNAL_TOKEN", "").strip()
    if not expected_token:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="FASTAPI_INTERNAL_TOKEN 환경변수가 필요합니다.",
        )

    if x_internal_token != expected_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="내부 API 토큰이 올바르지 않습니다.")


class CashSavePayload(BaseModel):
    accounts: list[dict[str, Any]]


@router.get("")
def get_cash_accounts(_: None = Depends(_require_internal_token)) -> dict[str, Any]:
    return load_cash_accounts()


@router.post("")
def post_cash_accounts(payload: CashSavePayload, _: None = Depends(_require_internal_token)) -> dict[str, str]:
    return save_cash_accounts(payload.accounts)
