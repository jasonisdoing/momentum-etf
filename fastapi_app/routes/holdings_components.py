"""보유종목 상세(구성종목 통합) API 라우트."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from fastapi_app.dependencies import require_internal_token
from utils.account_registry import load_account_configs
from utils.holdings_components_service import (
    load_account_holdings_components,
    load_exposure_country_holdings_components,
)
from utils.settings_loader import list_available_exposure_countries

router = APIRouter(prefix="/internal/holdings-components", tags=["holdings-components"])

# 노출국가 코드 → 화면 표시 한글 라벨.
_EXPOSURE_COUNTRY_LABELS: dict[str, str] = {
    "us": "미국",
    "kor": "한국",
    "au": "호주",
}


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


@router.get("/exposure-countries")
def get_exposure_countries(
    _: None = Depends(require_internal_token),
) -> list[dict[str, str]]:
    """노출국가 셀렉터에 사용할 코드/라벨 목록을 반환한다 (미국, 한국, 호주 순)."""
    available = set(list_available_exposure_countries())
    preferred_order = ["us", "kor", "au"]
    items: list[dict[str, str]] = []
    for code in preferred_order:
        if code in available:
            items.append({"code": code, "label": _EXPOSURE_COUNTRY_LABELS.get(code, code.upper())})
    # 위 순서에 없는 추가 코드는 정의 순서로 뒤에 붙인다.
    for code in list_available_exposure_countries():
        if code in preferred_order:
            continue
        items.append({"code": code, "label": _EXPOSURE_COUNTRY_LABELS.get(code, code.upper())})
    return items


@router.get("")
def get_holdings_components(
    account_id: str = Query(...),
    _: None = Depends(require_internal_token),
) -> dict:
    """선택된 계좌의 보유 ETF 구성종목 통합 데이터를 반환한다."""
    return load_account_holdings_components(account_id)


@router.get("/by-exposure-country")
def get_holdings_by_exposure_country(
    exposure_country_code: str = Query(...),
    _: None = Depends(require_internal_token),
) -> dict:
    """노출국가별 보유 ETF 구성종목 통합 데이터를 반환한다."""
    return load_exposure_country_holdings_components(exposure_country_code)
