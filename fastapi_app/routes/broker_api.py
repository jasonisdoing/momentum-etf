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
def get_providers(account_id: str = Query(...), _: None = Depends(require_internal_token)) -> dict:
    """커넥터 목록 — 키 등록 여부는 계좌마다 다르다(계좌별 키 커넥터)."""
    return {"providers": list_providers(account_id.strip())}


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


@router.get("/balance")
def get_balance(
    account_id: str = Query(...),
    _: None = Depends(require_internal_token),
) -> dict:
    """연동 계좌의 잔고를 불러와 현재 저장값과 나란히 돌려준다 — 화면이 차이를 보여준다."""
    from utils.broker_balance_sync import stored_cash_map
    from utils.portfolio_io import load_portfolio_master

    try:
        fetched = fetch_broker_balance(account_id)
    except BrokerApiError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    current = load_portfolio_master(account_id) or {"cash_balance": 0.0, "holdings": []}
    return {
        "fetched": fetched,
        "current": {
            # 증권사 값과 같은 통화들의 저장 현금 — 동기화 배치의 비교와 같은 기준.
            "cash_map": stored_cash_map(current, list(fetched["cash_map"])),
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
    fetched = cached_broker_balance(payload.account_id)
    if fetched is None:
        raise HTTPException(status_code=409, detail="불러온 잔고가 만료됐습니다 — '잔고 불러오기'를 다시 눌러주세요.")

    try:
        result = apply_fetched_balance(payload.account_id, fetched["provider"], fetched)
    except BrokerApiError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"ok": True, **result}
