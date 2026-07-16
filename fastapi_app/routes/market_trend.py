from __future__ import annotations

from fastapi import APIRouter, Depends, Query

import config
from fastapi_app.dependencies import require_internal_token
from utils.market_trend_service import (
    INDICES,
    compute_index_history,
    compute_market_trend,
    compute_regime_confirm_backtest,
)

router = APIRouter(prefix="/internal/market-trend", tags=["market-trend"])


@router.get("/indices")
def get_market_trend_indices(
    _: None = Depends(require_internal_token),
) -> dict[str, object]:
    """시장추세 지수 목록 (탑픽 시장 레짐 셀렉터 등에서 사용)."""
    return {"indices": [{"ticker": idx["yf_ticker"], "name": idx["name"]} for idx in INDICES]}


@router.get("/defaults")
def get_market_trend_defaults(
    _: None = Depends(require_internal_token),
) -> dict[str, object]:
    """화면 표시용 MA/추세점수 설정 (config.py 가 단일 진실 소스)."""
    return {
        "ma_days": config.MARKET_TREND_REGIME_MA_SHORT,
        "ma_short": config.MARKET_TREND_REGIME_MA_SHORT,
        "ma_long": config.MARKET_TREND_REGIME_MA_LONG,
        "confirm_days": config.MARKET_TREND_REGIME_CONFIRM_DAYS,
        "score_anchor_percentile": config.MARKET_TREND_SCORE_ANCHOR_PERCENTILE,
    }


@router.get("")
def get_market_trend(
    _: None = Depends(require_internal_token),
) -> dict[str, object]:
    """5개 시장지수의 현재가/추세/레짐 — MA 는 SMA {SHORT_MA_DAYS}일 고정."""
    return compute_market_trend()


@router.get("/regime-backtest")
def get_regime_backtest(
    ticker: str | None = Query(default=None),
    months: int = Query(default=12, ge=1, le=36),
    up_cash: float | None = Query(default=None, ge=0, le=100),
    neutral_cash: float | None = Query(default=None, ge=0, le=100),
    down_cash: float | None = Query(default=None, ge=0, le=100),
    _: None = Depends(require_internal_token),
) -> dict[str, object]:
    """MA20/60 교차 확인일수 백테스트 — 선택 지수의 최근 N개월 비교 (읽기 전용)."""
    return compute_regime_confirm_backtest(
        ticker=ticker, months=months, up_cash=up_cash, neutral_cash=neutral_cash, down_cash=down_cash
    )


@router.get("/history")
def get_market_trend_history(
    ticker: str = Query(..., description="Yahoo Finance 지수 심볼 (예: ^GSPC)"),
    _: None = Depends(require_internal_token),
) -> dict[str, object]:
    """단일 지수의 가격/추세/레짐 히스토리 — MA 는 SMA {SHORT_MA_DAYS}일 고정."""
    return compute_index_history(ticker)
