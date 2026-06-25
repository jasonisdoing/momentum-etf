"""종목풀 편집 가능 설정(pool_settings) 조회/저장 API.

pools.json 의 구조는 유지하고, 자주 바뀌는 5개 값(TOP_N_HOLD/HOLDING_BONUS_SCORE/
MA_TYPE/MA_MONTHS/RSI_LIMIT)만 DB 오버라이드로 수정한다 (utils.pool_settings_store).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from fastapi_app.dependencies import require_internal_token
from utils.pool_settings_store import (
    ALL_POOL_ID,
    OVERRIDABLE_KEYS,
    PoolSettingsError,
    save_pool_settings,
)
from utils.rankings import ALLOWED_MA_TYPES, get_rank_months_max
from utils.settings_loader import get_all_pool_settings, get_ticker_type_settings
from utils.ticker_registry import load_ticker_type_configs

router = APIRouter(prefix="/internal/pool-settings", tags=["pool-settings"])


class PoolSettingsUpdatePayload(BaseModel):
    pool_id: str
    values: dict[str, Any]


def _editable(settings: dict[str, Any]) -> dict[str, Any]:
    """편집 가능한 5개 키의 현재(DB) 값을 반환한다."""
    return {key: {"value": settings.get(key)} for key in OVERRIDABLE_KEYS}


@router.get("")
def get_pool_settings(_: None = Depends(require_internal_token)) -> dict[str, object]:
    """전체(all) + 풀별 편집 가능 설정과 입력 제약(MA 타입/개월 범위)을 반환한다."""
    all_settings = get_all_pool_settings()
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
            }
        )

    return {
        "all": {
            "pool_id": ALL_POOL_ID,
            "name": "전체 (가상 종목풀)",
            "settings": _editable(all_settings),
        },
        "pools": pools,
        "constraints": {
            "ma_types": ALLOWED_MA_TYPES,
            "ma_months_max": get_rank_months_max(),
            "editable_keys": list(OVERRIDABLE_KEYS),
        },
    }


@router.put("")
def put_pool_settings(
    payload: PoolSettingsUpdatePayload, _: None = Depends(require_internal_token)
) -> dict[str, object]:
    """편집한 값을 저장한다 (pool_id = '__all__' 또는 ticker_type)."""
    try:
        saved = save_pool_settings(payload.pool_id, payload.values)
    except PoolSettingsError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "pool_id": payload.pool_id, "saved": saved}
