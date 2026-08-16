"""Steady Momentum(전략-ST) 설정·선정 API."""

from fastapi import APIRouter, Body, Depends

from fastapi_app.dependencies import require_internal_token
from utils.steady_momentum_service import (
    compute_picks,
    load_settings,
    load_settings_map,
    pool_options,
    save_settings,
)

router = APIRouter(prefix="/internal/strategy-sm", tags=["strategy-sm"])


def _month_options(settings: dict) -> list[int]:
    """기간 선택지 — 종목풀 백테스트와 같은 목록을 쓰되, 이 전략이 실제로 돌릴 수
    있는 개월 수까지만 남긴다. 상한은 벤치마크 데이터와 전략 장기 이평선이 정한다."""
    from utils.pool_signal_backtest_service import get_month_options
    from utils.steady_momentum_service import available_backtest_months, load_benchmark_close

    limit = available_backtest_months(load_benchmark_close(settings["pool"]), int(settings["long_ma_days"]))
    options = [month for month in get_month_options() if month <= limit]
    if limit not in options:
        options.append(limit)
    # 상한이 줄기 전에 저장된 값은 선택지에 남겨 둔다 — 빼면 셀렉트가 빈칸이 되어
    # 무엇이 저장돼 있는지 알 수 없다. 실행·저장할 때 명시적 에러로 안내된다.
    saved = settings.get("backtest_months")
    if isinstance(saved, int) and saved not in options:
        options.append(saved)
    return sorted(options)


def _ma_rule_payload(settings: dict) -> dict:
    """전략 전용 이평선 + 선택지 — steady_momentum_settings 가 단일 소스다."""
    from utils.pool_settings_store import MA_DAY_OPTIONS

    return {
        "short_ma_days": int(settings["short_ma_days"]),
        "long_ma_days": int(settings["long_ma_days"]),
        "ma_day_options": list(MA_DAY_OPTIONS),
    }


def _constraints_payload() -> dict:
    """화면 셀렉트 선택지 — 백엔드 상수가 단일 소스(프론트 복사본 제거)."""
    from utils.steady_momentum_service import MAX_PER_INDUSTRY_OPTIONS, TOP_N_OPTIONS

    return {
        "top_n_options": list(TOP_N_OPTIONS),
        "max_per_industry_options": list(MAX_PER_INDUSTRY_OPTIONS),
    }


@router.get("")
def get_strategy_sm(
    _: None = Depends(require_internal_token),
) -> dict:
    """저장된 설정을 반환한다. 저장된 값이 없거나 깨졌으면 에러다(기본값 대체 없음)."""
    settings = load_settings()
    return {
        "settings": settings,
        # 풀별 저장 설정 맵 — 화면이 풀 셀렉트 전환 시 즉시 그 풀의 값을 채운다.
        "settings_by_pool": load_settings_map()["settings_by_pool"],
        "pool_options": pool_options(),
        "month_options": _month_options(settings),
        "ma_rule": _ma_rule_payload(settings),
        "constraints": _constraints_payload(),
        "picks": None,
    }


@router.put("/settings")
def put_strategy_sm_settings(
    payload: dict = Body(...),
    _: None = Depends(require_internal_token),
) -> dict:
    """설정을 검증 후 저장한다. body: ``{"settings": {...}}``.

    이평선(short/long_ma_days)도 설정의 일부로 **전략 전용으로 저장**한다 —
    종목풀 설정(순위·종목풀 백테스트·알람)에는 영향을 주지 않는다.
    """
    settings = payload.get("settings") if isinstance(payload, dict) else None
    if not isinstance(settings, dict):
        raise ValueError("저장할 'settings' 가 필요합니다.")
    saved = save_settings(settings)

    return {
        "settings": saved,
        "settings_by_pool": load_settings_map()["settings_by_pool"],
        "pool_options": pool_options(),
        "month_options": _month_options(saved),
        "ma_rule": _ma_rule_payload(saved),
        "constraints": _constraints_payload(),
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
    """주간 리밸런싱 백테스트.

    body: ``{"months": 12, "include_daily": false}``.
    ``include_daily`` 는 일간 탭을 볼 때만 참으로 보낸다 — 일별 계산은 응답이
    수천 행으로 커지므로 필요할 때만 만든다.
    """
    from utils.steady_momentum_backtest import run_backtest

    months = payload.get("months") if isinstance(payload, dict) else None
    if not isinstance(months, int) or isinstance(months, bool):
        raise ValueError("'months' 는 정수여야 합니다.")
    include_daily = payload.get("include_daily") if isinstance(payload, dict) else None
    if not isinstance(include_daily, bool):
        raise ValueError("'include_daily' 는 참/거짓이어야 합니다.")
    return run_backtest(months, include_daily=include_daily)
