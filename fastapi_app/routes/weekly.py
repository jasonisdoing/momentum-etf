from __future__ import annotations

import os
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, status

from utils.weekly_service import aggregate_active_week_data, load_weekly_table_data, update_weekly_row

router = APIRouter(prefix="/internal/weekly", tags=["weekly"])


def _require_internal_token(x_internal_token: str | None = Header(default=None, alias="X-Internal-Token")) -> None:
    expected_token = os.getenv("FASTAPI_INTERNAL_TOKEN", "").strip()
    if not expected_token:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="FASTAPI_INTERNAL_TOKEN 환경변수가 필요합니다.",
        )

    if x_internal_token != expected_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="내부 API 토큰이 올바르지 않습니다.")


@router.get("")
def get_weekly_data(_: None = Depends(_require_internal_token)) -> dict[str, Any]:
    return load_weekly_table_data()


@router.post("")
def post_weekly_aggregate(_: None = Depends(_require_internal_token)) -> dict[str, str]:
    return aggregate_active_week_data()


@router.patch("")
def patch_weekly_row(payload: dict[str, Any], _: None = Depends(_require_internal_token)) -> dict[str, str]:
    return update_weekly_row(str(payload.get("week_date") or "").strip(), payload)
