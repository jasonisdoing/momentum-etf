from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from fastapi import APIRouter, Body, Depends, Query

from fastapi_app.dependencies import require_internal_token
from services.portfolio_change_service import compute_portfolio_change_bundle
from utils.holdings_detail_service import (
    add_holding,
    delete_holding,
    load_all_holdings_detail,
    reorder_holdings,
    update_holding,
    validate_ticker_for_account,
)

router = APIRouter(prefix="/internal/holdings", tags=["holdings"])


@router.get("")
def get_all_holdings(
    account: str | None = Query(default=None),
    _: None = Depends(require_internal_token),
) -> dict[str, Any]:
    return load_all_holdings_detail(account_id=account)


@router.delete("")
def delete_one_holding(
    account: str = Query(...),
    ticker: str = Query(...),
    _: None = Depends(require_internal_token),
) -> dict[str, Any]:
    return delete_holding(account_id=account, ticker=ticker)


@router.patch("")
def patch_one_holding(
    body: dict[str, Any] = Body(...),
    _: None = Depends(require_internal_token),
) -> dict[str, Any]:
    return update_holding(
        account_id=body.get("account_id", ""),
        ticker=body.get("ticker", ""),
        quantity=body.get("quantity"),
        average_buy_price=body.get("average_buy_price"),
        memo=body.get("memo"),
        target_ratio=body.get("target_ratio"),
    )


@router.patch("/order")
def patch_holdings_order(
    body: dict[str, Any] = Body(...),
    _: None = Depends(require_internal_token),
) -> dict[str, Any]:
    return reorder_holdings(
        account_id=body.get("account_id", ""),
        ordered_tickers=body.get("ordered_tickers", []),
    )


@router.post("")
def post_one_holding(
    body: dict[str, Any] = Body(...),
    _: None = Depends(require_internal_token),
) -> dict[str, Any]:
    return add_holding(
        account_id=body.get("account_id", ""),
        ticker=body.get("ticker", ""),
        quantity=body.get("quantity", 0),
        average_buy_price=body.get("average_buy_price", 0),
        memo=body.get("memo"),
        target_ratio=body.get("target_ratio"),
    )


@router.post("/validate")
def validate_ticker(
    body: dict[str, Any] = Body(...),
    _: None = Depends(require_internal_token),
) -> dict[str, Any]:
    return validate_ticker_for_account(
        account_id=body.get("account_id", ""),
        ticker=body.get("ticker", ""),
    )


@router.post("/portfolio-changes")
def post_portfolio_changes(
    body: dict[str, Any] = Body(...),
    _: None = Depends(require_internal_token),
) -> dict[str, Any]:
    """ETF 보유 종목들의 포트폴리오 변동률을 병렬로 계산해 반환한다."""
    items = body.get("items") or []
    if not isinstance(items, list):
        return {"results": {}}

    targets: list[tuple[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        ticker = str(item.get("ticker") or "").strip().upper()
        ticker_type = str(item.get("ticker_type") or "").strip().lower()
        if not ticker or not ticker_type:
            continue
        targets.append((ticker, ticker_type))

    results: dict[str, Any] = {}
    if not targets:
        return {"results": results}

    def _worker(ticker: str, ticker_type: str) -> tuple[str, dict[str, Any] | None]:
        try:
            bundle = compute_portfolio_change_bundle(ticker, ticker_type)
        except Exception:
            return ticker, None
        if not bundle:
            return ticker, None
        return ticker, {
            "total_pct": bundle.get("total_pct"),
            "coverage_weight": bundle.get("coverage_weight"),
            "base_date": bundle.get("base_date"),
        }

    with ThreadPoolExecutor(max_workers=min(len(targets), 8)) as executor:
        futures = [executor.submit(_worker, t, tt) for (t, tt) in targets]
        for future in as_completed(futures):
            try:
                ticker, payload = future.result()
            except Exception:
                continue
            if payload is not None:
                results[ticker] = payload

    return {"results": results}
