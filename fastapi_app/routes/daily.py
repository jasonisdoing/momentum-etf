from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, model_validator

from fastapi_app.dependencies import require_internal_token
from utils.daily_fund_service import load_daily_table_data, update_daily_row

router = APIRouter(prefix="/internal/daily", tags=["daily"])


class DailyRowUpdatePayload(BaseModel):
    """일별 데이터 수정 페이로드. date 필수, 나머지는 FIELD_DEFS 기반 동적 필드."""

    model_config = {"extra": "allow"}

    date: str

    @model_validator(mode="after")
    def strip_date(self) -> "DailyRowUpdatePayload":
        self.date = self.date.strip()
        if not self.date:
            raise ValueError("date는 필수입니다.")
        return self


@router.get("")
def get_daily_data(_: None = Depends(require_internal_token)) -> dict[str, Any]:
    return load_daily_table_data()


@router.patch("")
def patch_daily_row(payload: DailyRowUpdatePayload, _: None = Depends(require_internal_token)) -> dict[str, str]:
    return update_daily_row(payload.date, payload.model_dump())
