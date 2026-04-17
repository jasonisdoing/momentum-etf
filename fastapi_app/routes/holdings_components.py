"""보유종목 상세(구성종목 통합) API 라우트."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from fastapi_app.dependencies import require_internal_token
from utils.holdings_components_service import load_account_holdings_components
from utils.account_registry import load_account_configs

router = APIRouter(prefix="/internal/holdings-components", tags=["holdings-components"])


@router.get("/accounts")
def get_accounts(
    _: None = Depends(require_internal_token),
) -> list[dict[str, str]]:
    """보유 계좌 목록을 반환한다."""
    configs = load_account_configs()
    return [
        {
            "account_id": str(c["account_id"]),
            "name": str(c.get("name", c["account_id"])),
        }
        for c in configs
    ]


@router.get("")
def get_holdings_components(
    account_id: str = Query(...),
    _: None = Depends(require_internal_token),
) -> dict:
    """선택된 계좌의 보유 ETF 구성종목 통합 데이터를 반환한다."""
    return load_account_holdings_components(account_id)
