"""증권사 API 연동 — 커넥터 목록·계좌 나열 (계좌 설정 화면의 '확인' 흐름).

조회 전용이다. 연동 정보 저장은 계좌 설정 저장(`/internal/account-settings`)이 담당한다.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query

from fastapi_app.dependencies import require_internal_token
from services.broker_api_service import BrokerApiError, list_broker_accounts, list_providers

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
