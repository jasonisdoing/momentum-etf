"""계좌 API 주문(운영·모의) — 잔고·사고팔기 주문·취소·주문 기록·증권사 주문내역.

계좌 설정에 증권사 연동(나무·메리츠)이 저장된 계좌만 받는다(`services.broker_trade_service.account_env`). 가격·수량은 요청에 없다 —
서버가 전략의 현재가·회차당 수량을 읽는다(`utils.strategy_trade_orders`).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from fastapi_app.dependencies import require_internal_token
from services.broker_api_service import BrokerApiError
from services.broker_trade_service import fetch_order_balance
from utils.slack_mock_orders import send_order_message
from utils.strategy_trade_orders import (
    broker_orders_today,
    cancel_record,
    list_orders,
    place_order,
)

router = APIRouter(prefix="/internal/broker-api/mock", tags=["broker-mock-orders"])


def _bad_request(exc: BrokerApiError) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


@router.get("/balance")
def get_balance(account_id: str = Query(...), _: None = Depends(require_internal_token)) -> dict:
    try:
        return fetch_order_balance(account_id)
    except BrokerApiError as exc:
        raise _bad_request(exc) from exc


@router.get("/orders")
def get_orders(account_id: str = Query(...), _: None = Depends(require_internal_token)) -> dict:
    """우리 주문 기록 — 증권사 체결 조회로 상태를 갱신한 뒤 돌려준다."""
    try:
        return {"orders": list_orders(account_id)}
    except BrokerApiError as exc:
        raise _bad_request(exc) from exc


class PlaceOrderPayload(BaseModel):
    account_id: str
    pool: str
    strategy_id: str
    side: str


@router.post("/orders")
def post_order(payload: PlaceOrderPayload, _: None = Depends(require_internal_token)) -> dict:
    try:
        return {"order": place_order(payload.account_id, payload.pool, payload.strategy_id, payload.side, None, None)}
    except BrokerApiError as exc:
        raise _bad_request(exc) from exc


@router.post("/slack-order")
def post_slack_order(payload: PlaceOrderPayload, _: None = Depends(require_internal_token)) -> dict:
    """주문 메시지(버튼)를 슬랙으로 보낸다 — 주문은 슬랙의 모달 확인 뒤에 나간다."""
    try:
        send_order_message(
            payload.account_id, payload.pool, payload.strategy_id, payload.side, None, None, rebuy=False
        )
    except BrokerApiError as exc:
        raise _bad_request(exc) from exc
    return {"ok": True}


class CancelPayload(BaseModel):
    order_id: str


@router.post("/orders/cancel")
def post_cancel(payload: CancelPayload, _: None = Depends(require_internal_token)) -> dict:
    try:
        return {"order": cancel_record(payload.order_id)}
    except BrokerApiError as exc:
        raise _bad_request(exc) from exc


@router.get("/broker-orders")
def get_broker_orders(account_id: str = Query(...), _: None = Depends(require_internal_token)) -> dict:
    """증권사에 오늘 들어간 주문 전체 — 우리 기록과 대조하는 용도."""
    try:
        return {"orders": broker_orders_today(account_id)}
    except BrokerApiError as exc:
        raise _bad_request(exc) from exc
