"""레버리지 전략 설정·상태 API."""

from fastapi import APIRouter, Body, Depends, Query

from fastapi_app.dependencies import require_internal_token
from utils.leverage_service import (
    load_leverage_settings,
    save_leverage_settings,
)

router = APIRouter(prefix="/internal/leverage", tags=["leverage"])


@router.get("/config")
def get_leverage_config(
    profile: str = Query(default="switch"),
    _: None = Depends(require_internal_token),
) -> dict:
    """레버리지 설정 + 직전 추천 상태를 반환한다."""
    return load_leverage_settings(profile)


@router.post("/config")
def post_leverage_config(
    payload: dict = Body(...),
    profile: str = Query(default="switch"),
    _: None = Depends(require_internal_token),
) -> dict:
    """편집된 설정을 검증 후 저장한다. body: ``{"config": {...}}``. 검증 실패 → 400."""
    config = payload.get("config") if isinstance(payload, dict) else None
    if not isinstance(config, dict):
        raise ValueError("저장할 'config' 가 필요합니다.")
    return save_leverage_settings(profile, config)


@router.get("/candles")
def get_leverage_candles(
    code: str = Query(default="A122630"),
    interval: str = Query(default="min:1"),
    count: int = Query(default=390, ge=1, le=1000),
    feed: str = Query(default="kr-stock"),
    _: None = Depends(require_internal_token),
) -> dict:
    """레버리지 단타용 토스 c-chart 캔들을 과거→최신 순으로 반환한다.

    지표(EMA)는 프런트에서 계산한다. feed: kr-stock(국내주식, 예: KODEX 레버리지) /
    us-futures(미국 선물, 예: 나스닥100 RFU.NQc1).
    """
    from services.toss_market_service import fetch_toss_candles, fetch_toss_stock_candles

    if feed == "kr-stock":
        candles = fetch_toss_stock_candles(code, securities_type="kr-s", interval=interval, count=count)
    elif feed == "us-futures":
        candles = fetch_toss_candles(code, interval=interval, count=count)
    else:
        raise ValueError(f"지원하지 않는 feed 입니다: {feed}")
    return {"code": code, "interval": interval, "feed": feed, "candles": candles}


@router.get("/scalp-settings")
def get_scalp_settings(
    _: None = Depends(require_internal_token),
) -> dict:
    """레버리지 단타 슈퍼트렌드 설정을 반환한다. 저장 전이면 settings=null."""
    from utils.scalp_service import load_scalp_settings

    return {"settings": load_scalp_settings()}


@router.put("/scalp-settings")
def put_scalp_settings(
    payload: dict = Body(...),
    _: None = Depends(require_internal_token),
) -> dict:
    """레버리지 단타 슈퍼트렌드 설정을 검증 후 저장한다. body: ``{"settings": {...}}``."""
    from utils.scalp_service import save_scalp_settings

    settings = payload.get("settings") if isinstance(payload, dict) else None
    if not isinstance(settings, dict):
        raise ValueError("저장할 'settings' 가 필요합니다.")
    return {"settings": save_scalp_settings(settings)}


@router.get("/ma-cross")
def get_ma_cross_view(
    market: str = Query(...),
    _: None = Depends(require_internal_token),
) -> dict:
    """이동평균선 크로스 전략의 현재 판정 + 추천 + 직전 상태(시장별).

    market(kor/us) → 프로필 ma_cross_<market>.
    """
    from leverage.config_store import ma_cross_profile
    from utils.leverage_ma_service import compute_ma_cross_view

    return compute_ma_cross_view(ma_cross_profile(market))


@router.get("/ma-cross/tune")
def get_ma_cross_tune(
    market: str = Query(...),
    months: int = Query(..., ge=1, le=120),
    ma_min: int = Query(..., ge=2),
    ma_max: int = Query(..., ge=2),
    ma_step: int = Query(..., ge=1),
    peak_min: float = Query(..., ge=0),
    peak_max: float = Query(..., ge=0),
    peak_step: float = Query(..., gt=0),
    _: None = Depends(require_internal_token),
) -> dict:
    """지정한 기간·이동선·고점대비 범위로 튜닝 sweep 을 즉시 계산해 반환한다."""
    from leverage.config_store import ma_cross_profile
    from utils.leverage_ma_service import compute_ma_cross_tune

    return compute_ma_cross_tune(
        ma_cross_profile(market),
        months=months,
        ma_min=ma_min,
        ma_max=ma_max,
        ma_step=ma_step,
        peak_min=peak_min,
        peak_max=peak_max,
        peak_step=peak_step,
    )


@router.post("/ma-cross/slack-test")
def post_ma_cross_slack_test(
    market: str = Query(...),
    _: None = Depends(require_internal_token),
) -> dict:
    """현재 판정으로 슬랙 메시지를 즉시(수동) 발송한다. 토글/장 마감 여부와 무관, 테스트 표식."""
    from leverage.config_store import ma_cross_profile
    from leverage.notify import send_slack_ma_cross
    from utils.leverage_ma_service import compute_ma_cross_view

    view = compute_ma_cross_view(ma_cross_profile(market))
    sent = send_slack_ma_cross(view, market_phase="수동 발송", test=True)
    return {"sent": bool(sent)}


@router.get("/resolve-ticker")
def resolve_leverage_ticker(
    ticker: str = Query(...),
    _: None = Depends(require_internal_token),
) -> dict:
    """종목풀(db.stock_meta) 내에서 티커를 조회하여 종목명을 반환합니다."""
    from utils.leverage_service import resolve_pool_ticker
    return resolve_pool_ticker(ticker)
