"""종목풀 신호(이격/배열) 실증 백테스트 API — 읽기 전용."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from fastapi_app.dependencies import require_internal_token
from utils.pool_signal_backtest_service import (
    FORWARD_DAY_OPTIONS,
    compute_pool_signal_backtest,
    get_max_backtest_months,
    get_month_options,
)

router = APIRouter(prefix="/internal/pool-backtest", tags=["pool-backtest"])


@router.get("/options")
def get_pool_backtest_options(
    _: None = Depends(require_internal_token),
) -> dict[str, object]:
    """화면 셀렉트용 선택지."""
    return {
        "forward_day_options": list(FORWARD_DAY_OPTIONS),
        "month_options": get_month_options(),
        "max_months": get_max_backtest_months(),
    }


@router.get("")
def get_pool_backtest(
    pool_id: str = Query(...),
    forward_days: int = Query(default=20),
    months: int = Query(default=12, ge=1, le=120),
    # 파라미터 오버라이드(실험용). 미지정이면 종목풀 설정값을 쓴다.
    top_n: int | None = Query(default=None),
    short_ma_days: int | None = Query(default=None),
    long_ma_days: int | None = Query(default=None),
    hold_threshold_k: float | None = Query(default=None),
    _: None = Depends(require_internal_token),
) -> dict[str, object]:
    """선택 종목풀의 이격/배열 → 향후 N일 상승확률 실증 결과."""
    try:
        return compute_pool_signal_backtest(
            pool_id,
            forward_days=forward_days,
            months=months,
            top_n=top_n,
            short_ma_days=short_ma_days,
            long_ma_days=long_ma_days,
            hold_threshold_k=hold_threshold_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
