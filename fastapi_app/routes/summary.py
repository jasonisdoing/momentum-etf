from __future__ import annotations

import os

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from pydantic import BaseModel

from utils.summary_service import generate_summary_data, load_summary_page_data

router = APIRouter(prefix="/internal/summary", tags=["summary"])


def _require_internal_token(x_internal_token: str | None = Header(default=None, alias="X-Internal-Token")) -> None:
    expected_token = os.getenv("FASTAPI_INTERNAL_TOKEN", "").strip()
    if not expected_token:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="FASTAPI_INTERNAL_TOKEN 환경변수가 필요합니다.",
        )

    if x_internal_token != expected_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="내부 API 토큰이 올바르지 않습니다.")


class SummaryGenerateRequest(BaseModel):
    account_id: str


@router.get("")
def get_summary_data(
    account_id: str | None = Query(default=None),
    _: None = Depends(_require_internal_token),
) -> dict[str, object]:
    return load_summary_page_data(account_id=account_id)


@router.post("")
def post_summary_data(
    payload: SummaryGenerateRequest,
    _: None = Depends(_require_internal_token),
) -> dict[str, object]:
    return generate_summary_data(payload.account_id)
