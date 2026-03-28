from __future__ import annotations

import os

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel

from utils.system_service import SystemAction, load_system_data, trigger_system_action

router = APIRouter(prefix="/internal/system", tags=["system"])


class SystemActionRequest(BaseModel):
    action: SystemAction


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
def get_system_data(_: None = Depends(_require_internal_token)) -> dict[str, object]:
    return load_system_data()


@router.post("")
def post_system_action(payload: SystemActionRequest, _: None = Depends(_require_internal_token)) -> dict[str, str]:
    return {"message": trigger_system_action(payload.action)}
