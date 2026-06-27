"""모멘텀 백테스트 실행/상태 API."""

from fastapi import APIRouter, Depends, Query

from fastapi_app.dependencies import require_internal_token
from utils.momentum_backtest_service import (
    momentum_backtest_status,
    trigger_momentum_backtest,
)

router = APIRouter(prefix="/internal/momentum/backtest", tags=["momentum-backtest"])


@router.post("")
def post_momentum_backtest(_: None = Depends(require_internal_token)) -> dict:
    """백테스트 작업을 배치 큐에 추가한다(워커가 실행). 이미 대기/실행 중이면 무시."""
    return trigger_momentum_backtest()


@router.get("/status")
def get_momentum_backtest_status(
    file: str | None = Query(default=None),
    _: None = Depends(require_internal_token),
) -> dict:
    """백테스트 실행 상태 + 선택 파일(없으면 최신) 결과 + 파일 목록을 반환한다."""
    return momentum_backtest_status(file)
