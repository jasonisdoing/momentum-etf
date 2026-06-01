"""보유종목 상세(구성종목 통합) API 라우트."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from fastapi_app.dependencies import require_internal_token
from utils.account_registry import load_account_configs
from utils.holdings_components_service import (
    list_holding_country_options,
    load_account_holdings_components,
    load_holding_country_components,
)

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


@router.get("/holding-countries")
def get_holding_countries(
    _: None = Depends(require_internal_token),
) -> list[dict[str, str]]:
    """종목 국가 셀렉터에 사용할 코드/라벨 목록 (미국, 한국, 호주, 기타국가 순)."""
    return list_holding_country_options()


@router.get("")
def get_holdings_components(
    account_id: str = Query(...),
    _: None = Depends(require_internal_token),
) -> dict:
    """선택된 계좌의 보유 ETF 구성종목 통합 데이터를 반환한다."""
    return load_account_holdings_components(account_id)


@router.get("/by-holding-country")
def get_holdings_by_holding_country(
    country_code: str = Query(...),
    _: None = Depends(require_internal_token),
) -> dict:
    """종목 국가별 보유 ETF 구성종목 통합 데이터를 반환한다."""
    return load_holding_country_components(country_code)
