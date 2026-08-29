"""신고가 돌파 전략 설정·선정 API."""

from fastapi import APIRouter, Body, Depends, Query

from fastapi_app.dependencies import require_internal_token
from utils.new_high_service import (
    DEFAULT_SETTINGS,
    HIGH_WINDOW_WEEKS,
    MIN_VALUE_MULT_OPTIONS,
    STOP_LOSS_OPTIONS,
    load_settings,
    load_settings_for_view,
    load_settings_map,
    pool_options,
    save_settings,
)
from utils.pool_signal_backtest_service import get_month_options

router = APIRouter(prefix="/internal/strategy-new-high", tags=["strategy-new-high"])


def _constraints(pool: str) -> dict:
    """화면 셀렉트 선택지 — 백엔드 상수가 단일 소스(프론트에 복사본을 두지 않는다)."""
    from utils.ma_options import short_ma_options
    from utils.settings_loader import get_ticker_type_settings

    country = str((get_ticker_type_settings(pool) or {}).get("country_code") or "").strip().lower()
    return {
        "stop_loss_options": list(STOP_LOSS_OPTIONS),
        # 이탈 이평선 = 그 풀 국가의 단기 이평 선택지
        "exit_ma_options": list(short_ma_options(country)),
        "min_value_mult_options": list(MIN_VALUE_MULT_OPTIONS),
        # 기간 선택지 — 종목풀 백테스트와 같은 목록이 단일 소스(전략별로 따로 두지 않는다).
        "month_options": get_month_options(),
        # 신고가 창 — 화면 문구("52주 신고가")를 이 값에서 만든다.
        "high_window_weeks": HIGH_WINDOW_WEEKS,
    }


def _view(settings: dict) -> dict:
    return {
        "settings": settings,
        # 저장 이력이 없는 풀로 전환할 때 화면이 채울 값. 직전 풀의 값을 물려받으면
        # 다른 풀의 설정이 섞인다 — 풀별로 따로 보관하는 의미가 없어진다.
        "default_settings": dict(DEFAULT_SETTINGS),
        "settings_by_pool": load_settings_map(),
        "pool_options": pool_options(),
        "constraints": _constraints(settings["pool"]),
    }


@router.get("")
def get_strategy_new_high(
    pool: str | None = Query(default=None),
    _: None = Depends(require_internal_token),
) -> dict:
    """저장된 설정과 화면 선택지를 반환한다.

    ``pool`` 은 화면이 로컬스토리지에 기억해 둔 선택이다 — 없으면 저장분이 있는 첫 풀.
    """
    settings, coerced = load_settings_for_view(pool)
    # 선택지 밖 저장값을 보정했으면 화면이 '저장되지 않은 변경'으로 띄운다.
    return {**_view(settings), "coerced": coerced}


@router.put("/settings")
def put_strategy_new_high_settings(
    payload: dict = Body(...),
    _: None = Depends(require_internal_token),
) -> dict:
    """설정을 검증 후 저장한다. body: ``{"settings": {...}}``."""
    settings = payload.get("settings") if isinstance(payload, dict) else None
    if not isinstance(settings, dict):
        raise ValueError("저장할 'settings' 가 필요합니다.")
    return _view(save_settings(settings))


@router.post("/positions")
def post_strategy_new_high_positions(
    payload: dict = Body(default={}),
    _: None = Depends(require_internal_token),
) -> dict:
    """기준일의 보유·이탈·돌파·후보를 반환한다 (가격 캐시 기반 — 수 초).

    body: ``{"settings": {...}, "as_of": "2026-08-11"}``. ``as_of`` 를 비우면 최신 거래일.
    """
    from utils.new_high_backtest import current_positions

    settings = payload.get("settings") if isinstance(payload, dict) else None
    as_of = payload.get("as_of") if isinstance(payload, dict) else None
    return current_positions(
        settings if isinstance(settings, dict) else None,
        as_of=str(as_of) if as_of else None,
    )


@router.post("/charts")
def post_strategy_new_high_charts(
    payload: dict = Body(default={}),
    _: None = Depends(require_internal_token),
) -> dict:
    """보유 종목 일봉. body: ``{"settings": {...}, "tickers": [...], "as_of": "2026-08-11"}``.

    티커는 화면이 이미 받아둔 보유 목록에서 그대로 넘긴다 — 여기서 보유를 다시 계산하면
    같은 시뮬레이션을 두 번 돌리게 된다.
    """
    from utils.new_high_service import holding_charts, validate_settings

    settings = payload.get("settings") if isinstance(payload, dict) else None
    settings = validate_settings(settings if isinstance(settings, dict) else load_settings())
    tickers = payload.get("tickers") if isinstance(payload, dict) else None
    if not isinstance(tickers, list):
        raise ValueError("'tickers' 는 목록이어야 합니다.")
    as_of = payload.get("as_of") if isinstance(payload, dict) else None
    return {
        "charts": holding_charts(
            settings["pool"],
            [str(ticker) for ticker in tickers],
            int(settings["exit_ma_days"]),
            as_of=str(as_of) if as_of else None,
        ),
        "exit_ma_days": int(settings["exit_ma_days"]),
    }


@router.post("/backtest")
def post_strategy_new_high_backtest(
    payload: dict = Body(...),
    _: None = Depends(require_internal_token),
) -> dict:
    """돌파 전략 백테스트. body: ``{"months": 12, "settings": {...}}``."""
    from utils.new_high_backtest import run_backtest

    months = payload.get("months") if isinstance(payload, dict) else None
    if not isinstance(months, int) or isinstance(months, bool):
        raise ValueError("'months' 는 정수여야 합니다.")
    settings = payload.get("settings") if isinstance(payload, dict) else None
    return run_backtest(months, settings if isinstance(settings, dict) else None)


@router.post("/tuning")
def post_strategy_new_high_tuning(
    payload: dict = Body(...),
    _: None = Depends(require_internal_token),
) -> dict:
    """튜닝 — 설정 항목 범위의 모든 조합을 백테스트한다.

    body: ``{"months": 12, "settings": {...현재 화면 값}, "ranges": {"top_n": [...],
    "stop_loss_pct": [...], "exit_ma_days": [...], "min_value_mult": [...]}}``
    축 밖의 설정은 ``settings`` 값으로 고정한다.
    """
    from utils.new_high_tuning import run_tuning

    if not isinstance(payload, dict):
        raise ValueError("요청 형식이 올바르지 않습니다.")
    months = payload.get("months")
    if not isinstance(months, int) or isinstance(months, bool):
        raise ValueError("'months' 는 정수여야 합니다.")
    settings = payload.get("settings")
    ranges = payload.get("ranges")
    if not isinstance(ranges, dict) or not all(isinstance(v, list) for v in ranges.values()):
        raise ValueError("'ranges' 는 축별 값 목록이어야 합니다.")
    return run_tuning(months, settings if isinstance(settings, dict) else None, ranges)
