"""Steady Momentum(전략-ST) 설정·선정 API."""

from fastapi import APIRouter, Body, Depends

from fastapi_app.dependencies import require_internal_token
from utils.steady_momentum_service import compute_picks, load_settings, pool_labels, save_settings

router = APIRouter(prefix="/internal/strategy-st", tags=["strategy-st"])


@router.get("")
def get_strategy_st(
    _: None = Depends(require_internal_token),
) -> dict:
    """저장된 설정(없으면 기본값 + is_saved=false)을 반환한다. 선정은 별도 실행."""
    settings, is_saved = load_settings()
    return {"settings": settings, "is_saved": is_saved, "pool_labels": pool_labels(), "picks": None}


@router.put("/settings")
def put_strategy_st_settings(
    payload: dict = Body(...),
    _: None = Depends(require_internal_token),
) -> dict:
    """설정을 검증 후 저장한다. body: ``{"settings": {...}}``."""
    settings = payload.get("settings") if isinstance(payload, dict) else None
    if not isinstance(settings, dict):
        raise ValueError("저장할 'settings' 가 필요합니다.")
    saved = save_settings(settings)
    return {"settings": saved, "is_saved": True, "pool_labels": pool_labels(), "picks": None}


@router.post("/picks")
def post_strategy_st_picks(
    _: None = Depends(require_internal_token),
) -> dict:
    """현재 월 확정 포트폴리오 선정을 실행한다 (가격 캐시 기반 — 수 초)."""
    return compute_picks()


@router.post("/backtest")
def post_strategy_st_backtest(
    payload: dict = Body(...),
    _: None = Depends(require_internal_token),
) -> dict:
    """월간 리밸런싱 백테스트. body: ``{"months": 12}`` (1~24)."""
    from utils.steady_momentum_backtest import run_backtest

    months = payload.get("months") if isinstance(payload, dict) else None
    if not isinstance(months, int) or isinstance(months, bool):
        raise ValueError("'months' 는 정수여야 합니다.")
    return run_backtest(months)
