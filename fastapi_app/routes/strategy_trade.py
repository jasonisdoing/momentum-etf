"""전략 사고팔기(분할 매수/매도) 운용 현황 API — 종목 1개씩 운용하는 전략."""

from fastapi import APIRouter, Body, Depends

from fastapi_app.dependencies import require_internal_token
from utils.strategy_trade_service import (
    add_strategy,
    delete_strategy,
    load_strategy_trade_view,
    save_slack_enabled,
    save_strategy_settings,
)

router = APIRouter(prefix="/internal/strategy-trade", tags=["strategy-trade"])


@router.get("")
def get_strategy_trade(
    _: None = Depends(require_internal_token),
) -> dict:
    """전략별로 고른 계좌의 실제 보유 기준 회차 현황·매수/매도 지정가를 반환한다."""
    return load_strategy_trade_view()


@router.put("/settings")
def put_strategy_trade_settings(
    payload: dict = Body(...),
    _: None = Depends(require_internal_token),
) -> dict:
    """슬랙 스위치 / 전략 편집·추가·삭제를 저장하고 갱신된 화면 데이터를 반환한다.

    body 는 넷 중 하나:
      ``{"slack_enabled": true}``                                    — 전역 스위치
      ``{"strategy_id": "…", "config": {account_id·ticker + pct 3종}}`` — 전략 편집
      ``{"add_strategy": {account_id·ticker + pct 3종}}``              — 전략 추가
      ``{"delete_strategy": "…"}``                                    — 전략 삭제
    config 는 모든 값을 담아야 한다 (검증은 `strategy_trade_service`).
    """
    if not isinstance(payload, dict):
        raise ValueError("요청 형식이 올바르지 않습니다.")

    if payload.get("slack_enabled") is not None:
        save_slack_enabled(bool(payload["slack_enabled"]))
    elif payload.get("add_strategy") is not None:
        add_strategy(payload["add_strategy"])
    elif payload.get("delete_strategy") is not None:
        delete_strategy(str(payload["delete_strategy"]))
    elif payload.get("strategy_id") is not None:
        save_strategy_settings(str(payload["strategy_id"]), config=payload.get("config"))
    else:
        raise ValueError("요청에 저장할 항목이 없습니다.")
    return load_strategy_trade_view()


@router.post("/slack-test")
def post_strategy_trade_slack_test(_: None = Depends(require_internal_token)) -> dict:
    """지금 수동 발송(트리거 없으면 현황만). 화면 '지금 발송(테스트)' 버튼용."""
    from utils.strategy_trade_notify import notify_strategy_trade

    result = notify_strategy_trade(force=True)
    return {"sent": result["sent"], "message": result["reason"]}
