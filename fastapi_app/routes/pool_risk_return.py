"""종목풀 위험·수익 산점도 API — 읽기 전용."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from fastapi_app.dependencies import require_internal_token
from utils.pool_risk_return_service import compute_pool_risk_return

router = APIRouter(prefix="/internal/pool-risk-return", tags=["pool-risk-return"])


@router.get("")
def get_pool_risk_return(
    pool_id: str = Query(...),
    months: int = Query(default=60, ge=1, le=120),
    _: None = Depends(require_internal_token),
) -> dict[str, object]:
    """선택 종목풀 종목별 CAGR·MDD. 기간(개월)보다 상장이 짧은 종목은 뺀다."""
    try:
        return compute_pool_risk_return(pool_id, months)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
