"""탑픽 설정/비중 계산 API."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from fastapi_app.dependencies import require_internal_token
from utils.top_pick_service import (
    approve_top_pick_weights,
    calculate_top_pick_weights,
    load_top_pick_settings,
    run_top_pick_backtest,
    run_top_pick_weights,
    save_top_pick_settings,
)

router = APIRouter(prefix="/internal/top-pick", tags=["top-pick"])


class TopPickSettingsPayload(BaseModel):
    tickers: list[dict[str, Any]]
    settings: dict[str, Any] | None = None
    backtest_settings: dict[str, Any] | None = None


@router.get("/settings")
def get_settings(_: None = Depends(require_internal_token)) -> dict[str, Any]:
    return load_top_pick_settings()


@router.put("/settings")
def put_settings(payload: TopPickSettingsPayload, _: None = Depends(require_internal_token)) -> dict[str, Any]:
    return save_top_pick_settings(payload.tickers, payload.settings, payload.backtest_settings)


@router.post("/run")
def post_run(payload: TopPickSettingsPayload, _: None = Depends(require_internal_token)) -> dict[str, Any]:
    return run_top_pick_weights(payload.tickers, payload.settings)


@router.post("/backtest")
def post_backtest(payload: TopPickSettingsPayload, _: None = Depends(require_internal_token)) -> dict[str, Any]:
    return run_top_pick_backtest(payload.tickers, payload.settings, payload.backtest_settings)


@router.post("/approve")
def post_approve(payload: TopPickSettingsPayload, _: None = Depends(require_internal_token)) -> dict[str, Any]:
    return approve_top_pick_weights(payload.tickers, payload.settings)


@router.get("/weights")
def get_weights(_: None = Depends(require_internal_token)) -> dict[str, Any]:
    return calculate_top_pick_weights()
