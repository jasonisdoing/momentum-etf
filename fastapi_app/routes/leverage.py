"""레버리지 전략 설정·상태 API."""

from fastapi import APIRouter, Body, Depends, Query

from fastapi_app.dependencies import require_internal_token
from utils.leverage_service import (
    leverage_tune_status,
    load_leverage_settings,
    save_leverage_settings,
    trigger_leverage_tune,
)

router = APIRouter(prefix="/internal/leverage", tags=["leverage"])


@router.get("/config")
def get_leverage_config(
    profile: str = Query(default="switch"),
    _: None = Depends(require_internal_token),
) -> dict:
    """레버리지 설정 + 직전 추천 상태를 반환한다."""
    return load_leverage_settings(profile)


@router.post("/config")
def post_leverage_config(
    payload: dict = Body(...),
    profile: str = Query(default="switch"),
    _: None = Depends(require_internal_token),
) -> dict:
    """편집된 설정을 검증 후 저장한다. body: ``{"config": {...}}``. 검증 실패 → 400."""
    config = payload.get("config") if isinstance(payload, dict) else None
    if not isinstance(config, dict):
        raise ValueError("저장할 'config' 가 필요합니다.")
    return save_leverage_settings(profile, config)


@router.post("/tune")
def post_leverage_tune(
    profile: str = Query(default="switch"),
    _: None = Depends(require_internal_token),
) -> dict:
    """튜닝 작업을 배치 큐에 추가한다(워커가 실행). 이미 대기/실행 중이면 무시."""
    return trigger_leverage_tune(profile)


@router.get("/tune/status")
def get_leverage_tune_status(
    profile: str = Query(default="switch"),
    date: str | None = Query(default=None),
    _: None = Depends(require_internal_token),
) -> dict:
    """튜닝 실행 상태 + 선택 날짜(없으면 최신) 로그 + 날짜 목록을 반환한다."""
    return leverage_tune_status(profile, date)


@router.get("/resolve-ticker")
def resolve_leverage_ticker(
    ticker: str = Query(...),
    _: None = Depends(require_internal_token),
) -> dict:
    """종목풀(db.stock_meta) 내에서 티커를 조회하여 종목명을 반환합니다."""
    from utils.leverage_service import resolve_pool_ticker
    return resolve_pool_ticker(ticker)
