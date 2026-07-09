from __future__ import annotations

from fastapi import APIRouter, Depends, Query

import config
from fastapi_app.dependencies import require_internal_token
from utils.market_trend_service import compute_index_history, compute_market_trend

router = APIRouter(prefix="/internal/market-trend", tags=["market-trend"])


@router.get("/defaults")
def get_market_trend_defaults(
    _: None = Depends(require_internal_token),
) -> dict[str, object]:
    """화면 표시용 MA/추세점수 설정 (config.py 가 단일 진실 소스)."""
    return {
        "ma_type": config.MARKET_TREND_REGIME_MA_TYPE,
        "ma_days": config.MARKET_TREND_REGIME_SHORT_MA_DAYS,
        "score_anchor_percentile": config.MARKET_TREND_SCORE_ANCHOR_PERCENTILE,
    }


@router.get("")
def get_market_trend(
    _: None = Depends(require_internal_token),
) -> dict[str, object]:
    """5개 시장지수의 현재가/추세/레짐 — MA 는 SMA {SHORT_MA_DAYS}일 고정."""
    return compute_market_trend()


@router.get("/history")
def get_market_trend_history(
    ticker: str = Query(..., description="Yahoo Finance 지수 심볼 (예: ^GSPC)"),
    _: None = Depends(require_internal_token),
) -> dict[str, object]:
    """단일 지수의 가격/추세/레짐 히스토리 — MA 는 SMA {SHORT_MA_DAYS}일 고정."""
    return compute_index_history(ticker)
