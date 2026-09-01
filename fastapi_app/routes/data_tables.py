"""DB 테이블 카탈로그 API 라우트."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from fastapi_app.dependencies import require_internal_token
from utils.data_table_catalog import build_data_table_payload

router = APIRouter(prefix="/internal/data-tables", tags=["data-tables"])


@router.get("")
def get_data_tables(_: None = Depends(require_internal_token)) -> dict[str, object]:
    """분류별 컬렉션 목록과 실측 통계(문서 수·데이터/인덱스 크기)를 반환한다."""
    return build_data_table_payload()
