from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel

from utils.import_service import parse_bulk_import_text, save_bulk_import_rows

router = APIRouter(prefix="/internal/import", tags=["import"])


def _require_internal_token(x_internal_token: str | None = Header(default=None, alias="X-Internal-Token")) -> None:
    expected_token = os.getenv("FASTAPI_INTERNAL_TOKEN", "").strip()
    if not expected_token:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="FASTAPI_INTERNAL_TOKEN 환경변수가 필요합니다.",
        )

    if x_internal_token != expected_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="내부 API 토큰이 올바르지 않습니다.")


class ImportPreviewPayload(BaseModel):
    text: str


class ImportSavePayload(BaseModel):
    rows: list[dict[str, Any]]


@router.post("/preview")
def post_import_preview(payload: ImportPreviewPayload, _: None = Depends(_require_internal_token)) -> dict[str, Any]:
    return parse_bulk_import_text(payload.text)


@router.post("/save")
def post_import_save(payload: ImportSavePayload, _: None = Depends(_require_internal_token)) -> dict[str, int]:
    return save_bulk_import_rows(payload.rows)
