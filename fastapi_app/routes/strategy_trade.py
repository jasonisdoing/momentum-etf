"""전략 사고팔기(분할 매수/매도) 운용 현황 API — 계좌별 종목 전략."""

from fastapi import APIRouter, Body, Depends, Query

from fastapi_app.dependencies import require_internal_token
from utils.strategy_trade_notify import mark_sent_rounds, send_test
from utils.strategy_trade_service import (
    load_strategy_trade_view,
    save_account_trade,
)
from utils.strategy_trade_threads import delete_signal_thread

router = APIRouter(prefix="/internal/strategy-trade", tags=["strategy-trade"])


@router.get("")
def get_strategy_trade(
    account_id: str | None = Query(None),
    _: None = Depends(require_internal_token),
) -> dict:
    """사고팔기 계좌 목록과, 고른 계좌의 전략별 회차 현황·매수/매도 지정가를 반환한다.

    ``account_id`` 를 안 주면 첫 계좌다.
    """
    return mark_sent_rounds(load_strategy_trade_view(account_id))


@router.put("/settings")
def put_strategy_trade_settings(
    payload: dict = Body(...),
    _: None = Depends(require_internal_token),
) -> dict:
    """한 계좌의 슬랙 설정과 전략 목록을 저장하고 갱신된 화면 데이터를 반환한다."""
    account_id = str(payload.get("account_id") or "").strip()
    if not account_id or payload.get("strategies") is None or "slack_enabled" not in payload:
        raise ValueError("account_id·slack_enabled·strategies 가 모두 필요합니다.")
    save_account_trade(account_id, payload["slack_enabled"], payload["strategies"], payload.get("groups"))
    return mark_sent_rounds(load_strategy_trade_view(account_id))


@router.post("/delete-signal")
def post_strategy_trade_delete_signal(
    payload: dict = Body(...),
    _: None = Depends(require_internal_token),
) -> dict:
    """슬랙으로 나간 신호 한 건을 지운다 — 주문이 없을 때만.

    body: ``{"key": 화면이 준 발송 기록 키}``. 슬랙 스레드 전체와 우리 쪽 기록을 함께 지운다.
    """
    key = str(payload.get("key") or "").strip()
    if not key:
        raise ValueError("key 가 필요합니다.")
    return delete_signal_thread(key)


@router.post("/slack-test")
def post_strategy_trade_slack_test(
    payload: dict = Body(...),
    _: None = Depends(require_internal_token),
) -> dict:
    """그 계좌의 사고팔기 전략 현재 상태를 슬랙으로 바로 보낸다(스위치와 무관한 테스트).

    body: ``{"account_id": "…"}``.
    """
    account_id = str(payload.get("account_id") or "").strip()
    if not account_id:
        raise ValueError("account_id 가 필요합니다.")
    result = send_test(account_id)
    if not result["sent"]:
        raise ValueError(result["reason"])
    return {"sent": True, "count": len(result["triggers"])}
