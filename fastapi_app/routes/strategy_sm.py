"""Steady Momentum(전략-ST) 설정·선정 API."""

from fastapi import APIRouter, Body, Depends

from fastapi_app.dependencies import require_internal_token
from utils.steady_momentum_service import compute_picks, load_settings, pool_labels, save_settings

router = APIRouter(prefix="/internal/strategy-sm", tags=["strategy-sm"])


@router.get("")
def get_strategy_sm(
    _: None = Depends(require_internal_token),
) -> dict:
    """저장된 설정을 반환한다. 저장된 값이 없거나 깨졌으면 에러다(기본값 대체 없음)."""
    return {"settings": load_settings(), "pool_labels": pool_labels(), "picks": None}


@router.put("/settings")
def put_strategy_sm_settings(
    payload: dict = Body(...),
    _: None = Depends(require_internal_token),
) -> dict:
    """설정을 검증 후 저장한다. body: ``{"settings": {...}}``."""
    settings = payload.get("settings") if isinstance(payload, dict) else None
    if not isinstance(settings, dict):
        raise ValueError("저장할 'settings' 가 필요합니다.")
    return {"settings": save_settings(settings), "pool_labels": pool_labels(), "picks": None}


@router.post("/picks")
def post_strategy_sm_picks(
    _: None = Depends(require_internal_token),
) -> dict:
    """현재 월 확정 포트폴리오 선정을 실행한다 (가격 캐시 기반 — 수 초)."""
    return compute_picks()


@router.post("/backtest")
def post_strategy_sm_backtest(
    payload: dict = Body(...),
    _: None = Depends(require_internal_token),
) -> dict:
    """월간 리밸런싱 백테스트. body: ``{"months": 12}`` (1~24)."""
    from utils.steady_momentum_backtest import run_backtest

    months = payload.get("months") if isinstance(payload, dict) else None
    if not isinstance(months, int) or isinstance(months, bool):
        raise ValueError("'months' 는 정수여야 합니다.")
    return run_backtest(months)
