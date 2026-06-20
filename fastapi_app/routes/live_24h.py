"""24H 실시간 주식 및 선물 시세 API."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from fastapi_app.dependencies import require_internal_token
from utils.live_24h_service import load_live_24h_quotes

router = APIRouter(prefix="/internal/live-24h", tags=["live-24h"])


@router.get("")
def get_live_24h(_: None = Depends(require_internal_token)) -> dict[str, object]:
    try:
        return load_live_24h_quotes()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
