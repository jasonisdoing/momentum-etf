"""탑픽 설정/비중 계산 API."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from fastapi_app.dependencies import require_internal_token
from utils.top_pick_service import (
    approve_top_pick_weights,
    calculate_top_pick_weights,
    list_top_pick_accounts,
    load_top_pick_settings_for_edit,
    run_top_pick_backtest,
    run_top_pick_weights,
    save_top_pick_settings,
)

router = APIRouter(prefix="/internal/top-pick", tags=["top-pick"])


class TopPickSettingsPayload(BaseModel):
    tickers: list[dict[str, Any]]
    settings: dict[str, Any] | None = None
    backtest_settings: dict[str, Any] | None = None
    account_id: str | None = None
    weight_mode: str = "variable"


@router.get("/accounts")
def get_accounts(_: None = Depends(require_internal_token)) -> dict[str, Any]:
    return {"accounts": list_top_pick_accounts()}


@router.get("/settings")
def get_settings(
    account_id: str = Query(...), _: None = Depends(require_internal_token)
) -> dict[str, Any]:
    return load_top_pick_settings_for_edit(account_id)


@router.put("/settings")
def put_settings(payload: TopPickSettingsPayload, _: None = Depends(require_internal_token)) -> dict[str, Any]:
    return save_top_pick_settings(
        payload.tickers, payload.settings, payload.backtest_settings, account_id=payload.account_id
    )


@router.post("/run")
def post_run(payload: TopPickSettingsPayload, _: None = Depends(require_internal_token)) -> dict[str, Any]:
    return run_top_pick_weights(payload.tickers, payload.settings)


@router.post("/backtest")
def post_backtest(payload: TopPickSettingsPayload, _: None = Depends(require_internal_token)) -> dict[str, Any]:
    return run_top_pick_backtest(
        payload.tickers, payload.settings, payload.backtest_settings, weight_mode=payload.weight_mode
    )


@router.post("/approve")
def post_approve(payload: TopPickSettingsPayload, _: None = Depends(require_internal_token)) -> dict[str, Any]:
    return approve_top_pick_weights(payload.tickers, payload.settings, account_id=payload.account_id)


@router.get("/weights")
def get_weights(
    account_id: str | None = Query(default=None), _: None = Depends(require_internal_token)
) -> dict[str, Any]:
    return calculate_top_pick_weights(account_id)
