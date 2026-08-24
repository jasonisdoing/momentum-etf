"""종목풀 편집 가능 설정(pool_settings) 조회/저장 API."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from fastapi_app.dependencies import require_internal_token
from utils.ma_options import ma_options_by_country
from utils.market_trend_service import INDICES
from utils.pool_settings_store import (
    POOL_EDITABLE_KEYS,
    SLIPPAGE_PCT_OPTIONS,
    PoolSettingsError,
    create_pool,
    delete_pool,
    get_pool_delete_impact,
    load_pool_definitions,
    save_pool_settings,
    update_pool,
)

router = APIRouter(prefix="/internal/pool-settings", tags=["pool-settings"])


class PoolSettingsUpdatePayload(BaseModel):
    pool_id: str
    values: dict[str, Any]
    save_method: str = "수동"


class PoolDefinitionPayload(BaseModel):
    values: dict[str, Any]
    save_method: str = "사용자"


def _editable(settings: dict[str, Any]) -> dict[str, Any]:
    """편집 가능한 키의 현재(DB) 값을 반환한다."""
    return {key: {"value": settings.get(key)} for key in POOL_EDITABLE_KEYS}


def _pool_payload(settings: dict[str, Any]) -> dict[str, Any]:
    return {
        "ticker_type": settings["ticker_type"],
        "name": settings["name"],
        "icon": settings["icon"],
        "order": settings["order"],
        "country_code": settings["country_code"],
        "currency": settings["currency"],
        # 풀 성격(stock/etf) — 미설정이면 None (화면이 '미설정' 으로 보여준다).
        "pool_kind": settings.get("pool_kind"),
        "type_source": settings.get("type_source"),
        "settings": _editable(settings),
        "updated_at": settings.get("updated_at"),
        "save_method": settings.get("save_method"),
    }


@router.get("")
def get_pool_settings(_: None = Depends(require_internal_token)) -> dict[str, object]:
    """풀별 편집 가능 설정과 입력 제약을 반환한다."""
    try:
        pools = [_pool_payload(settings) for settings in load_pool_definitions()]
    except PoolSettingsError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "pools": pools,
        "constraints": {
            "ma_options_by_country": ma_options_by_country(),
            "slippage_pct_options": list(SLIPPAGE_PCT_OPTIONS),
            "editable_keys": list(POOL_EDITABLE_KEYS),
            "market_indices": [{"ticker": item["yf_ticker"], "name": item["name"]} for item in INDICES],
        },
    }


@router.put("")
def put_pool_settings(
    payload: PoolSettingsUpdatePayload, _: None = Depends(require_internal_token)
) -> dict[str, object]:
    """편집한 값을 저장한다 (pool_id = ticker_type)."""
    try:
        saved = save_pool_settings(payload.pool_id, payload.values, save_method=payload.save_method)
    except PoolSettingsError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "pool_id": payload.pool_id, "saved": saved}


@router.post("/pools")
def post_pool_definition(
    payload: PoolDefinitionPayload, _: None = Depends(require_internal_token)
) -> dict[str, object]:
    """신규 종목풀을 생성한다."""
    try:
        saved = create_pool(payload.values, save_method=payload.save_method)
    except PoolSettingsError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "pool": saved}


@router.patch("/pools/{pool_id}")
def patch_pool_definition(
    pool_id: str, payload: PoolDefinitionPayload, _: None = Depends(require_internal_token)
) -> dict[str, object]:
    """기존 종목풀의 메타/설정을 수정한다. ticker_type 은 변경 불가."""
    try:
        saved = update_pool(pool_id, payload.values, save_method=payload.save_method)
    except PoolSettingsError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "pool_id": pool_id, "saved": saved}


@router.get("/pools/{pool_id}/delete-impact")
def get_pool_delete_impact_route(pool_id: str, _: None = Depends(require_internal_token)) -> dict[str, object]:
    """종목풀 삭제 영향도를 반환한다."""
    try:
        return get_pool_delete_impact(pool_id)
    except PoolSettingsError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.delete("/pools/{pool_id}")
def delete_pool_definition(pool_id: str, _: None = Depends(require_internal_token)) -> dict[str, object]:
    """계좌 연결이 없는 종목풀을 하드 삭제한다."""
    try:
        deleted = delete_pool(pool_id)
    except PoolSettingsError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "deleted": deleted}
