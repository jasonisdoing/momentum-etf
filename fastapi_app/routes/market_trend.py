from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from fastapi_app.dependencies import require_internal_token
from utils.market_trend_service import compute_index_history, compute_market_trend
from utils.rankings import ALLOWED_MA_TYPES
from utils.settings_loader import get_all_pool_settings

router = APIRouter(prefix="/internal/market-trend", tags=["market-trend"])


def _normalize_ma_type(ma_type: str) -> str:
    normalized = (ma_type or "").strip().upper()
    if normalized not in ALLOWED_MA_TYPES:
        raise ValueError(
            f"지원하지 않는 MA 타입입니다: {ma_type}. 허용 값: {', '.join(ALLOWED_MA_TYPES)}"
        )
    return normalized


def _resolve_default_ma() -> tuple[str, int]:
    """pools.json 의 all.MA_TYPE / all.MA_MONTHS 를 그대로 사용 (설명은 pools.json 참고)."""
    settings = get_all_pool_settings()
    return _normalize_ma_type(str(settings["MA_TYPE"])), int(settings["MA_MONTHS"])


@router.get("/defaults")
def get_market_trend_defaults(
    _: None = Depends(require_internal_token),
) -> dict[str, object]:
    """화면 진입 시 사용할 MA 기본값 + 추세점수 설정 (config.py 가 단일 진실 소스)."""
    import config

    ma_type, ma_months = _resolve_default_ma()
    return {
        "ma_type": ma_type,
        "ma_months": ma_months,
        "ma_types": ALLOWED_MA_TYPES,
        "ma_months_max": config.MARKET_TREND_MA_MONTHS_MAX,
        "score_anchor_percentile": config.MARKET_TREND_SCORE_ANCHOR_PERCENTILE,
    }


@router.get("")
def get_market_trend(
    ma_type: str | None = Query(None, description="이동평균 타입"),
    ma_months: int | None = Query(None, ge=1, le=12, description="이동평균 기간(개월)"),
    _: None = Depends(require_internal_token),
) -> dict[str, object]:
    default_type, default_months = _resolve_default_ma()
    resolved_type = _normalize_ma_type(ma_type) if ma_type else default_type
    resolved_months = int(ma_months) if ma_months is not None else default_months
    return compute_market_trend(resolved_type, resolved_months)


@router.get("/history")
def get_market_trend_history(
    ticker: str = Query(..., description="Yahoo Finance 지수 심볼 (예: ^GSPC)"),
    ma_type: str | None = Query(None, description="이동평균 타입"),
    ma_months: int | None = Query(None, ge=1, le=12, description="이동평균 기간(개월)"),
    _: None = Depends(require_internal_token),
) -> dict[str, object]:
    default_type, default_months = _resolve_default_ma()
    resolved_type = _normalize_ma_type(ma_type) if ma_type else default_type
    resolved_months = int(ma_months) if ma_months is not None else default_months
    return compute_index_history(ticker, resolved_type, resolved_months)
