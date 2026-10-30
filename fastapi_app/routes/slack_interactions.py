"""슬랙 인터랙션 웹훅 — 버튼 클릭·모달 제출을 받는다.

Node 가 슬랙 요청 원문과 서명 헤더를 그대로 넘기고, 여기서 서명을 검증한다(`utils.slack_mock_orders`).
"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel

from fastapi_app.dependencies import require_internal_token
from services.broker_api_service import BrokerApiError
from utils.slack_mock_orders import (
    open_confirm_modal,
    parse_payload,
    place_submitted_order,
    submitted_request,
    verify_signature,
)

router = APIRouter(prefix="/internal/slack", tags=["slack-interactions"])


class InteractionPayload(BaseModel):
    raw_body: str
    timestamp: str
    signature: str


@router.post("/interactions")
def post_interaction(
    body: InteractionPayload,
    background: BackgroundTasks,
    _: None = Depends(require_internal_token),
) -> dict:
    try:
        verify_signature(body.raw_body, body.timestamp, body.signature)
        payload = parse_payload(body.raw_body)
        if payload["type"] == "block_actions":
            open_confirm_modal(payload)
        elif payload["type"] == "view_submission":
            background.add_task(place_submitted_order, submitted_request(payload))
        else:
            raise BrokerApiError(f"처리하지 않는 슬랙 요청 종류입니다: {payload['type']}")
    except BrokerApiError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {}
