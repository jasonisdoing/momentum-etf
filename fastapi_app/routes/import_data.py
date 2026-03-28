from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from fastapi_app.dependencies import require_internal_token
from utils.import_service import parse_bulk_import_text, save_bulk_import_rows

router = APIRouter(prefix="/internal/import", tags=["import"])


class ImportPreviewPayload(BaseModel):
    text: str


class ImportSavePayload(BaseModel):
    rows: list[dict[str, Any]]


@router.post("/preview")
def post_import_preview(payload: ImportPreviewPayload, _: None = Depends(require_internal_token)) -> dict[str, Any]:
    return parse_bulk_import_text(payload.text)


@router.post("/save")
def post_import_save(payload: ImportSavePayload, _: None = Depends(require_internal_token)) -> dict[str, int]:
    return save_bulk_import_rows(payload.rows)
