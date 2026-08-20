"""증권사 API 연동 — 커넥터 목록·계좌 나열 (계좌 설정 화면의 '확인' 흐름).

조회 전용이다. 연동 정보 저장은 계좌 설정 저장(`/internal/account-settings`)이 담당한다.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from fastapi_app.dependencies import require_internal_token
from services.broker_api_service import (
    BrokerApiError,
    apply_fetched_balance,
    cached_broker_balance,
    fetch_broker_balance,
    list_broker_accounts,
    list_providers,
)

router = APIRouter(prefix="/internal/broker-api", tags=["broker-api"])


@router.get("/providers")
def get_providers(_: None = Depends(require_internal_token)) -> dict:
    return {"providers": list_providers()}


@router.get("/accounts")
def get_accounts(
    provider: str = Query(...),
    _: None = Depends(require_internal_token),
) -> dict:
    """커넥터 검증(토큰 발급 포함) + 계좌 나열. 실패는 한국어 메시지로 돌려준다."""
    try:
        return {"accounts": list_broker_accounts(provider.strip().upper())}
    except BrokerApiError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _linked_account(account_id: str) -> tuple[str, str]:
    """계좌 설정에서 연동 정보(provider, account_no)를 찾는다 — 없으면 400."""
    from utils.settings_loader import get_account_settings

    try:
        settings = get_account_settings(account_id)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"계좌 설정을 찾을 수 없습니다: {account_id}") from exc
    linked = settings.get("broker_api") or {}
    provider = str(linked.get("provider") or "").strip().upper()
    account_no = str(linked.get("account_no") or "").strip()
    if not provider or not account_no:
        raise HTTPException(status_code=400, detail=f"'{account_id}' 에 증권사 API 연동이 저장돼 있지 않습니다.")
    return provider, account_no


@router.get("/balance")
def get_balance(
    account_id: str = Query(...),
    _: None = Depends(require_internal_token),
) -> dict:
    """연동 계좌의 잔고를 불러와 현재 저장값과 나란히 돌려준다 — 화면이 차이를 보여준다."""
    from utils.portfolio_io import load_portfolio_master

    provider, account_no = _linked_account(account_id)
    try:
        fetched = fetch_broker_balance(provider, account_no)
    except BrokerApiError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    current = load_portfolio_master(account_id) or {"cash_balance": 0.0, "holdings": []}
    # 현금은 자산 화면과 같은 기준 — 다통화 맵의 계좌 통화 값 우선, 없으면 레거시.
    from utils.settings_loader import get_account_settings

    currency = str((get_account_settings(account_id) or {}).get("currency") or "KRW").strip().upper()
    cash_map = current.get("cash") or {}
    current_cash = float(cash_map.get(currency, current.get("cash_balance") or 0))
    return {
        "fetched": fetched,
        "current": {
            "cash": current_cash,
            "holdings": [
                {
                    "ticker": str(row.get("ticker") or ""),
                    "name": str(row.get("name") or ""),
                    "quantity": float(row.get("quantity") or 0),
                    "average_buy_price": float(row.get("average_buy_price") or 0),
                }
                for row in current.get("holdings") or []
            ],
        },
    }


class ApplyPayload(BaseModel):
    account_id: str


@router.post("/apply")
def apply_balance(payload: ApplyPayload, _: None = Depends(require_internal_token)) -> dict:
    """가장 최근 불러온 잔고를 portfolio_master 에 반영한다.

    재호출하지 않고 불러오기 캐시를 쓴다(일일 호출 제한 절약). 캐시가 만료됐으면
    임의로 다시 부르지 않고 다시 불러오라고 안내한다 — 사용자가 본 값과 저장되는
    값이 어긋나면 안 된다.
    기존 보유의 메모·매수일·정렬은 보존하고, 수량·평단·현금만 API 값으로 바꾼다.
    """
    provider, account_no = _linked_account(payload.account_id)
    fetched = cached_broker_balance(provider, account_no)
    if fetched is None:
        raise HTTPException(status_code=409, detail="불러온 잔고가 만료됐습니다 — '잔고 불러오기'를 다시 눌러주세요.")

    try:
        result = apply_fetched_balance(payload.account_id, provider, fetched)
    except BrokerApiError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"ok": True, **result}
