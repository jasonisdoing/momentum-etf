"""Steady Momentum(전략-ST) 설정·선정 API."""

from fastapi import APIRouter, Body, Depends

from fastapi_app.dependencies import require_internal_token
from utils.steady_momentum_service import compute_picks, load_settings, pool_options, save_settings

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


def _ma_rule_payload(pool: str) -> dict:
    """선택 풀의 이평선 규칙 + 선택지 — 종목풀 설정이 단일 소스(순위 화면과 동일)."""
    from utils.pool_settings_store import MA_DAY_OPTIONS, SLOPE_DAY_OPTIONS
    from utils.rankings import get_ticker_type_ma_rules

    rule = get_ticker_type_ma_rules(pool)[0]
    return {
        "short_ma_days": int(rule["short_ma_days"]),
        "long_ma_days": int(rule["long_ma_days"]),
        "slope_days": int(rule["slope_days"]),
        "ma_day_options": list(MA_DAY_OPTIONS),
        "slope_day_options": list(SLOPE_DAY_OPTIONS),
    }


@router.get("")
def get_strategy_sm(
    _: None = Depends(require_internal_token),
) -> dict:
    """저장된 설정을 반환한다. 저장된 값이 없거나 깨졌으면 에러다(기본값 대체 없음)."""
    settings = load_settings()
    return {
        "settings": settings,
        "pool_options": pool_options(),
        "month_options": _month_options(settings),
        "ma_rule": _ma_rule_payload(settings["pool"]),
        "picks": None,
    }


@router.put("/settings")
def put_strategy_sm_settings(
    payload: dict = Body(...),
    _: None = Depends(require_internal_token),
) -> dict:
    """설정을 검증 후 저장한다. body: ``{"settings": {...}, "ma_rule": {...}?}``.

    ``ma_rule`` (단기/장기 이평선·기울기 일수)이 오면 **종목풀 설정에 저장**한다 —
    순위·종목풀 백테스트·자산 헬퍼가 같은 값을 쓰므로 그 화면들도 함께 바뀐다.
    검증·캐시 무효화는 save_pool_settings 한 곳이 담당한다.
    """
    settings = payload.get("settings") if isinstance(payload, dict) else None
    if not isinstance(settings, dict):
        raise ValueError("저장할 'settings' 가 필요합니다.")
    saved = save_settings(settings)

    ma_rule = payload.get("ma_rule")
    if isinstance(ma_rule, dict) and ma_rule:
        from utils.pool_settings_store import save_pool_settings

        save_pool_settings(
            str(saved["pool"]),
            {
                "SHORT_MA_DAYS": ma_rule.get("short_ma_days"),
                "LONG_MA_DAYS": ma_rule.get("long_ma_days"),
                "SLOPE_DAYS": ma_rule.get("slope_days"),
            },
            save_method="Steady Momentum",
        )

    return {
        "settings": saved,
        "pool_options": pool_options(),
        "month_options": _month_options(saved),
        "ma_rule": _ma_rule_payload(saved["pool"]),
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
    """월간 리밸런싱 백테스트.

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
