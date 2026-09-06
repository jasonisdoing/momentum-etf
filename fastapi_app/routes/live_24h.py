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
    """헤더 표시용 나스닥 100 선물 현재가 + 전일 종가 대비 변동률.

    소스는 야후(NQ=F, CME 시세라 약 10~15분 지연) — /live-24h 지표 카드와 같은 소스다.
    """
    try:
        from services.price_service import get_yahoo_symbol_snapshot

        info = get_yahoo_symbol_snapshot(["NQ=F"]).get("NQ=F") or {}
        return {"price": info.get("nowVal"), "change_pct": info.get("changeRate")}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
