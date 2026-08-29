"""포트폴리오 전략 설정·백테스트 API."""

from fastapi import APIRouter, Body, Depends, Query

from config import HOLDING_CHART_MONTHS, REBALANCE_BAND_PCT_OPTIONS, REBALANCE_LABELS, REBALANCE_OPTIONS
from fastapi_app.dependencies import require_internal_token
from utils.pool_signal_backtest_service import get_month_options
from utils.portfolio_service import (
    DEFAULT_BACKTEST_MONTHS,
    DEFAULT_SETTINGS,
    MAX_HOLDINGS,
    load_settings,
    load_settings_map,
    pool_options,
    save_settings,
    universe_metrics,
)

router = APIRouter(prefix="/internal/strategy-portfolio", tags=["strategy-portfolio"])


def _constraints() -> dict:
    """화면 셀렉트 선택지 — 백엔드 상수가 단일 소스(프론트에 복사본을 두지 않는다)."""
    return {
        "rebalance_options": [{"value": key, "label": REBALANCE_LABELS[key]} for key in REBALANCE_OPTIONS],
        "band_pct_options": list(REBALANCE_BAND_PCT_OPTIONS),
        # 기간 선택지 — 종목풀 백테스트와 같은 목록이 단일 소스(전략별로 따로 두지 않는다).
        "month_options": get_month_options(),
        "default_backtest_months": DEFAULT_BACKTEST_MONTHS,
        "max_holdings": MAX_HOLDINGS,
    }


def _view(settings: dict) -> dict:
    return {
        "settings": settings,
        # 저장 이력이 없는 풀로 전환할 때 화면이 채울 값.
        "default_settings": dict(DEFAULT_SETTINGS),
        "settings_by_pool": load_settings_map(),
        "pool_options": pool_options(),
        # 이 풀에서 담을 수 있는 종목 + 표시 지표(일간·현재가·기간수익률·MDD·소르티노).
        # 화면의 티커 입력이 이 목록으로 검증하고, 표의 지표 컬럼도 여기서 채운다.
        "universe": universe_metrics(settings["pool"]),
        "constraints": _constraints(),
    }


@router.get("")
def get_strategy_portfolio(
    pool: str | None = Query(default=None),
    _: None = Depends(require_internal_token),
) -> dict:
    """저장된 설정과 화면 선택지를 반환한다.

    ``pool`` 은 화면이 로컬스토리지에 기억해 둔 선택이다 — 없으면 저장분이 있는 첫 풀.
    """
    return _view(load_settings(pool))


@router.put("/settings")
def put_strategy_portfolio_settings(
    payload: dict = Body(...),
    _: None = Depends(require_internal_token),
) -> dict:
    """설정을 검증 후 저장한다. body: ``{"settings": {...}}``."""
    settings = payload.get("settings") if isinstance(payload, dict) else None
    if not isinstance(settings, dict):
        raise ValueError("저장할 'settings' 가 필요합니다.")
    return _view(save_settings(settings))


@router.post("/backtest")
def post_strategy_portfolio_backtest(
    payload: dict = Body(default={}),
    _: None = Depends(require_internal_token),
) -> dict:
    """고정 비중 리밸런싱 백테스트. body: ``{"settings": {...}, "months": 12}``."""
    from utils.portfolio_backtest import run_backtest

    settings = payload.get("settings") if isinstance(payload, dict) else None
    months = payload.get("months") if isinstance(payload, dict) else None
    return run_backtest(
        int(months) if months else None,
        settings if isinstance(settings, dict) else None,
    )


@router.post("/charts")
def post_strategy_portfolio_charts(
    payload: dict = Body(default={}),
    _: None = Depends(require_internal_token),
) -> dict:
    """보유 종목 일봉 + 이평선. body: ``{"pool": "...", "tickers": [...]}``.

    포트폴리오에는 판정선이 없어 **풀 설정 이평선**(추세 배지 기준)을 참고로 그린다 —
    합성 화면의 포트폴리오 슬리브와 같은 규칙이다(`strategy_mix_service.holding_charts_for_account`).
    """
    from utils.holding_chart_service import holding_charts
    from utils.settings_loader import get_ticker_type_settings

    pool = str(payload.get("pool") or "") if isinstance(payload, dict) else ""
    settings = load_settings(pool or None)
    tickers = payload.get("tickers") if isinstance(payload, dict) else None
    if not isinstance(tickers, list):
        raise ValueError("'tickers' 는 목록이어야 합니다.")
    pool_settings = get_ticker_type_settings(settings["pool"]) or {}
    ma_days = [int(pool_settings["SHORT_MA_DAYS"]), int(pool_settings["LONG_MA_DAYS"])]
    return {
        "charts": holding_charts(settings["pool"], [str(ticker) for ticker in tickers], ma_days),
        # 화면 안내 문구("최근 N개월 일봉입니다")가 쓰는 값 — 프론트에 복사본을 두지 않는다.
        "months": HOLDING_CHART_MONTHS,
        "short_ma_days": ma_days[0],
        "long_ma_days": ma_days[1],
    }
