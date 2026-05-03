from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, model_validator

from fastapi_app.dependencies import require_internal_token
from utils.monthly_service import aggregate_active_month_data, load_monthly_table_data, update_monthly_row

router = APIRouter(prefix="/internal/monthly", tags=["monthly"])


class MonthlyRowUpdatePayload(BaseModel):
    """월별 데이터 수정 페이로드. month_date 필수, 나머지는 FIELD_DEFS 기반 동적 필드."""

    model_config = {"extra": "allow"}

    month_date: str

    @model_validator(mode="after")
    def strip_month_date(self) -> MonthlyRowUpdatePayload:
        self.month_date = self.month_date.strip()
        if not self.month_date:
            raise ValueError("month_date는 필수입니다.")
        return self


@router.get("")
def get_monthly_data(_: None = Depends(require_internal_token)) -> dict[str, Any]:
    return load_monthly_table_data()


@router.post("")
def post_monthly_aggregate(_: None = Depends(require_internal_token)) -> dict[str, str]:
    return aggregate_active_month_data()


@router.patch("")
def patch_monthly_row(payload: MonthlyRowUpdatePayload, _: None = Depends(require_internal_token)) -> dict[str, str]:
    return update_monthly_row(payload.month_date, payload.model_dump())
