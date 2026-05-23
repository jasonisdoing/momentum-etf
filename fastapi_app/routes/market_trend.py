from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from config import MARKET_TREND_DEFAULT_MA_MONTHS, MARKET_TREND_DEFAULT_MA_TYPE
from fastapi_app.dependencies import require_internal_token
from utils.market_trend_service import compute_index_history, compute_market_trend
from utils.rankings import ALLOWED_MA_TYPES

router = APIRouter(prefix="/internal/market-trend", tags=["market-trend"])


def _normalize_ma_type(ma_type: str) -> str:
    normalized = (ma_type or "").strip().upper()
    if normalized not in ALLOWED_MA_TYPES:
        raise ValueError(
            f"지원하지 않는 MA 타입입니다: {ma_type}. 허용 값: {', '.join(ALLOWED_MA_TYPES)}"
        )
    return normalized


@router.get("/defaults")
def get_market_trend_defaults(
    _: None = Depends(require_internal_token),
) -> dict[str, object]:
    """화면 진입 시 사용할 MA 기본값 (config.py 가 단일 진실 소스)."""
    return {
        "ma_type": MARKET_TREND_DEFAULT_MA_TYPE,
        "ma_months": MARKET_TREND_DEFAULT_MA_MONTHS,
    }


@router.get("")
def get_market_trend(
    ma_type: str = Query(MARKET_TREND_DEFAULT_MA_TYPE, description="이동평균 타입"),
    ma_months: int = Query(
        MARKET_TREND_DEFAULT_MA_MONTHS, ge=1, le=12, description="이동평균 기간(개월)"
    ),
    _: None = Depends(require_internal_token),
) -> dict[str, object]:
    return compute_market_trend(_normalize_ma_type(ma_type), int(ma_months))


@router.get("/history")
def get_market_trend_history(
    ticker: str = Query(..., description="Yahoo Finance 지수 심볼 (예: ^GSPC)"),
    ma_type: str = Query(MARKET_TREND_DEFAULT_MA_TYPE, description="이동평균 타입"),
    ma_months: int = Query(
        MARKET_TREND_DEFAULT_MA_MONTHS, ge=1, le=12, description="이동평균 기간(개월)"
    ),
    _: None = Depends(require_internal_token),
) -> dict[str, object]:
    return compute_index_history(ticker, _normalize_ma_type(ma_type), int(ma_months))
