"""포트폴리오 실험(portfolio-lab) API — 한국 전용 균등비중 보유 시뮬레이션.

실행/티커조회/저장/목록/삭제. 계산은 utils.portfolio_lab_service 가 담당한다.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from fastapi_app.dependencies import require_internal_token
from utils.portfolio_lab_service import (
    delete_portfolio,
    list_saved_portfolios,
    resolve_kor_name,
    run_portfolio_lab,
    save_portfolio,
)

router = APIRouter(prefix="/internal/portfolio-lab", tags=["portfolio-lab"])


class RunPayload(BaseModel):
    tickers: list[dict[str, Any]]
    months: int = 12
    benchmark: dict[str, Any] | None = None
    rebalance: str = "none"


class SavePayload(BaseModel):
    name: str
    tickers: list[dict[str, Any]]
    months: int = 12
    benchmark: dict[str, Any] | None = None
    rebalance: str = "none"


@router.post("/run")
def post_run(payload: RunPayload, _: None = Depends(require_internal_token)) -> dict[str, Any]:
    try:
        return run_portfolio_lab(payload.tickers, payload.months, payload.benchmark, payload.rebalance)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/resolve")
def get_resolve(ticker: str, _: None = Depends(require_internal_token)) -> dict[str, str]:
    try:
        return {"ticker": ticker.strip().upper(), "name": resolve_kor_name(ticker)}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/saved")
def get_saved(_: None = Depends(require_internal_token)) -> dict[str, Any]:
    return {"portfolios": list_saved_portfolios()}


@router.put("/saved")
def put_saved(payload: SavePayload, _: None = Depends(require_internal_token)) -> dict[str, Any]:
    try:
        save_portfolio(payload.name, payload.tickers, payload.months, payload.benchmark, payload.rebalance)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "portfolios": list_saved_portfolios()}


@router.delete("/saved/{name}")
def delete_saved(name: str, _: None = Depends(require_internal_token)) -> dict[str, Any]:
    try:
        delete_portfolio(name)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return {"ok": True, "portfolios": list_saved_portfolios()}
