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


@router.get("/nq-future")
def get_nq_future(_: None = Depends(require_internal_token)) -> dict[str, object]:
    """헤더 표시용 나스닥 100 선물 현재가 + 전일 기준 변동률 (토스, 5초 TTL)."""
    try:
        from services.toss_market_service import fetch_toss_indicator_prices

        info = fetch_toss_indicator_prices().get("RFU.NQc1") or {}
        latest = info.get("latest")
        base = info.get("base")
        change_pct = ((float(latest) / float(base) - 1.0) * 100.0) if (latest and base) else None
        return {"price": latest, "change_pct": change_pct}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
