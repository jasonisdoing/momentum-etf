from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from fastapi_app.dependencies import require_internal_token
from utils.weekly_service import aggregate_active_week_data, load_weekly_table_data, update_weekly_row

router = APIRouter(prefix="/internal/weekly", tags=["weekly"])


@router.get("")
def get_weekly_data(_: None = Depends(require_internal_token)) -> dict[str, Any]:
    return load_weekly_table_data()


@router.post("")
def post_weekly_aggregate(_: None = Depends(require_internal_token)) -> dict[str, str]:
    return aggregate_active_week_data()


@router.patch("")
def patch_weekly_row(payload: dict[str, Any], _: None = Depends(require_internal_token)) -> dict[str, str]:
    return update_weekly_row(str(payload.get("week_date") or "").strip(), payload)
