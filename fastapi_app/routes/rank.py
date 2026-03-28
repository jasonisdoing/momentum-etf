from __future__ import annotations

import os

from fastapi import APIRouter, Depends, Header, HTTPException, Query, status

from utils.rank_service import load_rank_data

router = APIRouter(prefix="/internal/rank", tags=["rank"])


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
def get_rank_data(
    account_id: str | None = Query(default=None),
    ma_type: str | None = Query(default=None),
    ma_months: int | None = Query(default=None),
    _: None = Depends(_require_internal_token),
) -> dict[str, object]:
    return load_rank_data(account_id=account_id, ma_type=ma_type, ma_months=ma_months)
