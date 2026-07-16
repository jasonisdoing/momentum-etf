"""종목풀 편집 가능 설정(pool_settings) 조회/저장 API."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from fastapi_app.dependencies import require_internal_token
from utils.pool_settings_store import (
    MA_DAY_OPTIONS,
    OVERRIDABLE_KEYS,
    PoolSettingsError,
    save_pool_settings,
)
from utils.settings_loader import get_ticker_type_settings
from utils.ticker_registry import load_ticker_type_configs

router = APIRouter(prefix="/internal/pool-settings", tags=["pool-settings"])


class PoolSettingsUpdatePayload(BaseModel):
    pool_id: str
    values: dict[str, Any]
    save_method: str = "수동"


def _editable(settings: dict[str, Any]) -> dict[str, Any]:
    """편집 가능한 키의 현재(DB) 값을 반환한다."""
    return {key: {"value": settings.get(key)} for key in OVERRIDABLE_KEYS}


@router.get("")
def get_pool_settings(_: None = Depends(require_internal_token)) -> dict[str, object]:
    """풀별 편집 가능 설정과 입력 제약을 반환한다."""
    pools: list[dict[str, Any]] = []
    for config in load_ticker_type_configs():
        t_id = str(config["ticker_type"])
        settings = get_ticker_type_settings(t_id)
        pools.append(
            {
                "ticker_type": t_id,
                "name": config["name"],
                "icon": config["icon"],
                "order": config["order"],
                "settings": _editable(settings),
                "updated_at": settings.get("updated_at"),
                "save_method": settings.get("save_method"),
            }
        )

    return {
        "pools": pools,
        "constraints": {
            "ma_day_options": list(MA_DAY_OPTIONS),
            "editable_keys": list(OVERRIDABLE_KEYS),
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
