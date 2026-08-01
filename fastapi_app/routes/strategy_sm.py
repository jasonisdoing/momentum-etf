"""Steady Momentum(전략-ST) 설정·선정 API."""

from fastapi import APIRouter, Body, Depends

from fastapi_app.dependencies import require_internal_token
from utils.steady_momentum_service import compute_picks, load_settings, pool_labels, save_settings

router = APIRouter(prefix="/internal/strategy-sm", tags=["strategy-sm"])


def _month_options(settings: dict) -> list[int]:
    """기간 선택지 — 종목풀 백테스트와 같은 목록을 쓰되, 이 전략이 실제로 돌릴 수
    있는 개월 수까지만 남긴다. 상한은 종목풀 데이터와 룩백이 함께 정한다."""
    from utils.pool_signal_backtest_service import get_month_options
    from utils.steady_momentum_service import available_backtest_months, load_benchmark_close

    limit = available_backtest_months(
        load_benchmark_close(settings["pool"]), settings["lookback_months"]
    )
    options = [month for month in get_month_options() if month <= limit]
    if limit not in options:
        options.append(limit)
    # 상한이 줄기 전에 저장된 값은 선택지에 남겨 둔다 — 빼면 셀렉트가 빈칸이 되어
    # 무엇이 저장돼 있는지 알 수 없다. 실행·저장할 때 명시적 에러로 안내된다.
    saved = settings.get("backtest_months")
    if isinstance(saved, int) and saved not in options:
        options.append(saved)
    return sorted(options)


@router.get("")
def get_strategy_sm(
    _: None = Depends(require_internal_token),
) -> dict:
    """저장된 설정을 반환한다. 저장된 값이 없거나 깨졌으면 에러다(기본값 대체 없음)."""
    settings = load_settings()
    return {
        "settings": settings,
        "pool_labels": pool_labels(),
        "month_options": _month_options(settings),
        "picks": None,
    }


@router.put("/settings")
def put_strategy_sm_settings(
    payload: dict = Body(...),
    _: None = Depends(require_internal_token),
) -> dict:
    """설정을 검증 후 저장한다. body: ``{"settings": {...}}``."""
    settings = payload.get("settings") if isinstance(payload, dict) else None
    if not isinstance(settings, dict):
        raise ValueError("저장할 'settings' 가 필요합니다.")
    saved = save_settings(settings)
    return {
        "settings": saved,
        "pool_labels": pool_labels(),
        "month_options": _month_options(saved),
        "picks": None,
    }


@router.post("/picks")
def post_strategy_sm_picks(
    _: None = Depends(require_internal_token),
) -> dict:
    """현재 월 확정 포트폴리오 선정을 실행한다 (가격 캐시 기반 — 수 초)."""
    return compute_picks()


@router.post("/backtest")
def post_strategy_sm_backtest(
    payload: dict = Body(...),
    _: None = Depends(require_internal_token),
) -> dict:
    """월간 리밸런싱 백테스트. body: ``{"months": 12}`` (1~24)."""
    from utils.steady_momentum_backtest import run_backtest

    months = payload.get("months") if isinstance(payload, dict) else None
    if not isinstance(months, int) or isinstance(months, bool):
        raise ValueError("'months' 는 정수여야 합니다.")
    return run_backtest(months)
