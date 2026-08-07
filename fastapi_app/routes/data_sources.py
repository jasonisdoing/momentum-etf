"""데이터 소스 카탈로그 API 라우트."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from fastapi_app.dependencies import require_internal_token
from utils.data_source_catalog import build_data_source_payload

router = APIRouter(prefix="/internal/data-sources", tags=["data-sources"])


@router.get("")
def get_data_sources(_: None = Depends(require_internal_token)) -> dict[str, object]:
    """외부 데이터 소스 목록과 호주 ETF 구성종목 수집 현황을 반환한다."""
    return build_data_source_payload()
