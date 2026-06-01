from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, model_validator

from fastapi_app.dependencies import require_internal_token
from utils.yearly_service import aggregate_active_year_data, load_yearly_table_data, update_yearly_row

router = APIRouter(prefix="/internal/yearly", tags=["yearly"])


class YearlyRowUpdatePayload(BaseModel):
    """년별 데이터 수정 페이로드. year_date 필수, 나머지는 FIELD_DEFS 기반 동적 필드."""

    model_config = {"extra": "allow"}

    year_date: str

    @model_validator(mode="after")
    def strip_year_date(self) -> "YearlyRowUpdatePayload":
        self.year_date = self.year_date.strip()
        if not self.year_date:
            raise ValueError("year_date는 필수입니다.")
        return self


@router.get("")
def get_yearly_data(_: None = Depends(require_internal_token)) -> dict[str, Any]:
    return load_yearly_table_data()


@router.post("")
def post_yearly_aggregate(_: None = Depends(require_internal_token)) -> dict[str, str]:
    return aggregate_active_year_data()


@router.patch("")
def patch_yearly_row(payload: YearlyRowUpdatePayload, _: None = Depends(require_internal_token)) -> dict[str, str]:
    return update_yearly_row(payload.year_date, payload.model_dump())
