"""Hyperliquid 24시간 토큰화 주식 시세 API."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from fastapi_app.dependencies import require_internal_token
from utils.hyperliquid_service import load_hyperliquid_quotes

router = APIRouter(prefix="/internal/hyperliquid", tags=["hyperliquid"])


@router.get("")
def get_hyperliquid(_: None = Depends(require_internal_token)) -> dict[str, object]:
    try:
        return load_hyperliquid_quotes()
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
