"""자산 헬퍼 설정/비중 계산 API."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel

from fastapi_app.dependencies import require_internal_token
from utils.asset_helper_service import (
    load_asset_helper_settings_for_edit,
    run_asset_helper_backtest,
    run_asset_helper_weights,
    save_asset_helper_settings,
)

router = APIRouter(prefix="/internal/asset-helper", tags=["asset-helper"])


class AssetHelperSettingsPayload(BaseModel):
    tickers: list[dict[str, Any]]
    settings: dict[str, Any] | None = None
    backtest_settings: dict[str, Any] | None = None
    account_id: str | None = None
    weight_mode: str


@router.get("/settings")
def get_settings(account_id: str = Query(...), _: None = Depends(require_internal_token)) -> dict[str, Any]:
    return load_asset_helper_settings_for_edit(account_id)


@router.put("/settings")
def put_settings(payload: AssetHelperSettingsPayload, _: None = Depends(require_internal_token)) -> dict[str, Any]:
    return save_asset_helper_settings(
        payload.tickers,
        payload.weight_mode,
        payload.settings,
        payload.backtest_settings,
        account_id=payload.account_id,
    )


@router.post("/run")
def post_run(payload: AssetHelperSettingsPayload, _: None = Depends(require_internal_token)) -> dict[str, Any]:
    return run_asset_helper_weights(
        payload.tickers,
        payload.settings,
        payload.backtest_settings,
        weight_mode=payload.weight_mode,
    )


@router.post("/backtest")
def post_backtest(payload: AssetHelperSettingsPayload, _: None = Depends(require_internal_token)) -> dict[str, Any]:
    return run_asset_helper_backtest(
        payload.tickers,
        payload.settings,
        payload.backtest_settings,
        weight_mode=payload.weight_mode,
    )
