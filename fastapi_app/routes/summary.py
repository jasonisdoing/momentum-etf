from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from fastapi_app.dependencies import require_internal_token
from utils.summary_service import generate_summary_data, load_summary_page_data

router = APIRouter(prefix="/internal/summary", tags=["summary"])


class SummaryGenerateRequest(BaseModel):
    account_id: str


@router.get("")
def get_summary_data(
    account_id: str | None = Query(default=None),
    _: None = Depends(require_internal_token),
) -> dict[str, object]:
    return load_summary_page_data(account_id=account_id)


@router.post("")
def post_summary_data(
    payload: SummaryGenerateRequest,
    _: None = Depends(require_internal_token),
) -> dict[str, object]:
    return generate_summary_data(payload.account_id)
