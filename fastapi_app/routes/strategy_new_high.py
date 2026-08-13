"""신고가 돌파 전략 설정·선정 API."""

from fastapi import APIRouter, Body, Depends

from fastapi_app.dependencies import require_internal_token
from utils.new_high_service import (
    DEFAULT_SETTINGS,
    ENTRY_PRIORITY_OPTIONS,
    EXIT_MA_OPTIONS,
    HIGH_WINDOW_WEEKS,
    MIN_VALUE_MULT_OPTIONS,
    STOP_LOSS_OPTIONS,
    TOP_N_OPTIONS,
    load_settings,
    load_settings_map,
    pool_options,
    save_settings,
)

router = APIRouter(prefix="/internal/strategy-new-high", tags=["strategy-new-high"])

# 백테스트 기간 선택지 — 짧은 구간은 표본이 모자라 의미가 없다.
_MONTH_OPTIONS = [6, 12, 24, 36, 48, 60]


def _constraints() -> dict:
    """화면 셀렉트 선택지 — 백엔드 상수가 단일 소스(프론트에 복사본을 두지 않는다)."""
    return {
        "top_n_options": list(TOP_N_OPTIONS),
        "stop_loss_options": list(STOP_LOSS_OPTIONS),
        "exit_ma_options": list(EXIT_MA_OPTIONS),
        "entry_priority_options": list(ENTRY_PRIORITY_OPTIONS),
        "min_value_mult_options": list(MIN_VALUE_MULT_OPTIONS),
        "month_options": list(_MONTH_OPTIONS),
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
        "constraints": _constraints(),
    }


@router.get("")
def get_strategy_new_high(_: None = Depends(require_internal_token)) -> dict:
    """저장된 설정과 화면 선택지를 반환한다."""
    return _view(load_settings())


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
