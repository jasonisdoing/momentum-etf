"""모멘텀 전략(전략-ST) 설정·선정 API."""

from fastapi import APIRouter, Body, Depends, Query

from fastapi_app.dependencies import require_internal_token
from utils.momentum_service import (
    compute_picks,
    load_settings,
    load_settings_map,
    pool_options,
    save_settings,
)

router = APIRouter(prefix="/internal/strategy-momentum", tags=["strategy-momentum"])


def _month_options(settings: dict) -> list[int]:
    """기간 선택지 — 시스템 공용 목록에서 이 전략이 실제로 돌릴 수 있는 개월 수까지만 남긴다.

    상한은 벤치마크 데이터와 전략 장기 이평선이 정한다. 예전에는 그 상한값 자체(예 87개월)를
    선택지 끝에 덧붙였는데, 풀·이평선에 따라 값이 달라져 화면마다 선택지가 어긋났다.
    """
    from utils.momentum_service import available_backtest_months, load_benchmark_close
    from utils.pool_signal_backtest_service import get_month_options

    limit = available_backtest_months(load_benchmark_close(settings["pool"]), int(settings["long_ma_days"]))
    return [month for month in get_month_options() if month <= limit]


def _ma_rule_payload(settings: dict) -> dict:
    """전략 이평선 + 선택지 — 종목풀 설정(`pool_settings`)이 단일 소스다."""
    from utils.ma_options import ma_options_payload
    from utils.momentum_service import pool_info

    return {
        "short_ma_days": int(settings["short_ma_days"]),
        "long_ma_days": int(settings["long_ma_days"]),
        **ma_options_payload(pool_info(settings["pool"])["country"]),
    }


def _constraints_payload() -> dict:
    """화면 셀렉트 선택지 — 백엔드 상수가 단일 소스(프론트 복사본 제거)."""
    from utils.momentum_service import INTRAWEEK_STOP_OPTIONS

    return {
        # 주중 손절선(%) — 주중 이탈이 켜진 풀에서만 화면에 노출한다. None = 손절 없음.
        "intraweek_stop_options": list(INTRAWEEK_STOP_OPTIONS),
    }


@router.get("")
def get_strategy_momentum(
    pool: str | None = Query(default=None),
    _: None = Depends(require_internal_token),
) -> dict:
    """저장된 설정을 반환한다. 저장된 값이 없거나 깨졌으면 에러다(기본값 대체 없음).

    ``pool`` 은 화면이 로컬스토리지에 기억해 둔 선택이다 — 없으면 저장분이 있는 첫 풀.
    기억된 풀에 저장분이 없으면(다른 화면에서 고른 새 풀) 실패 대신 저장분이 있는 첫
    풀의 설정을 돌려주고 ``requested_pool`` 로 알린다 — 화면이 그 풀을 '첫 설정 초안'
    으로 띄운다(풀 셀렉트 전환과 같은 흐름). 새 풀의 값을 지어내는 게 아니다.
    """
    from utils.momentum_service import load_settings_for_view

    requested = str(pool or "").strip().lower() or None
    try:
        settings, coerced = load_settings_for_view(requested)
    except RuntimeError:
        if requested is None:
            raise  # 저장된 풀이 하나도 없음 — 진짜 에러
        settings, coerced = load_settings_for_view(None)
    return {
        "requested_pool": requested or settings["pool"],
        "settings": settings,
        # 선택지 밖 저장값을 보정했으면 화면이 '저장되지 않은 변경'으로 띄운다.
        "coerced": coerced,
        # 풀별 저장 설정 맵 — 화면이 풀 셀렉트 전환 시 즉시 그 풀의 값을 채운다.
        "settings_by_pool": load_settings_map()["settings_by_pool"],
        "pool_options": pool_options(),
        "month_options": _month_options(settings),
        "ma_rule": _ma_rule_payload(settings),
        "constraints": _constraints_payload(),
        "picks": None,
    }


@router.put("/settings")
def put_strategy_momentum_settings(
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
def post_strategy_momentum_picks(
    pool: str | None = Query(default=None),
    _: None = Depends(require_internal_token),
) -> dict:
    """현재 주 확정 포트폴리오 선정을 실행한다 (가격 캐시 기반 — 수 초).

    ``pool`` 은 화면이 고른 종목풀이다 — 없으면 저장분이 있는 첫 풀.
    """
    return compute_picks(load_settings(pool))


@router.post("/backtest")
def post_strategy_momentum_backtest(
    payload: dict = Body(...),
    pool: str | None = Query(default=None),
    _: None = Depends(require_internal_token),
) -> dict:
    """주간 리밸런싱 백테스트.

    body: ``{"months": 12, "include_daily": false}``.
    ``include_daily`` 는 일간 탭을 볼 때만 참으로 보낸다 — 일별 계산은 응답이
    수천 행으로 커지므로 필요할 때만 만든다.
    """
    from utils.momentum_backtest import run_backtest

    months = payload.get("months") if isinstance(payload, dict) else None
    if not isinstance(months, int) or isinstance(months, bool):
        raise ValueError("'months' 는 정수여야 합니다.")
    include_daily = payload.get("include_daily") if isinstance(payload, dict) else None
    if not isinstance(include_daily, bool):
        raise ValueError("'include_daily' 는 참/거짓이어야 합니다.")
    return run_backtest(months, settings=load_settings(pool), include_daily=include_daily)


@router.post("/tuning")
def post_strategy_momentum_tuning(
    payload: dict = Body(...),
    pool: str | None = Query(default=None),
    _: None = Depends(require_internal_token),
) -> dict:
    """튜닝 — 설정 항목 범위의 모든 조합을 백테스트한다 (저장된 설정 기준).

    body: ``{"months": 12, "ranges": {
    "short_ma_days": [...], "long_ma_days": [...], "intraweek": ["off", "none", -5, ...]}}``
    """
    from utils.momentum_tuning import run_tuning

    if not isinstance(payload, dict):
        raise ValueError("요청 형식이 올바르지 않습니다.")
    months = payload.get("months")
    if not isinstance(months, int) or isinstance(months, bool):
        raise ValueError("'months' 는 정수여야 합니다.")
    ranges = payload.get("ranges")
    if not isinstance(ranges, dict) or not all(isinstance(v, list) for v in ranges.values()):
        raise ValueError("'ranges' 는 축별 값 목록이어야 합니다.")
    return run_tuning(months, load_settings(pool), ranges)
