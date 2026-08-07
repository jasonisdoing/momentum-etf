"""전략 사고팔기(분할 매수/매도) 운용 현황 API — 전략 2개(코스피200/코스닥150)."""

from fastapi import APIRouter, Body, Depends

from fastapi_app.dependencies import require_internal_token
from utils.strategy_trade_service import (
    load_strategy_trade_view,
    save_slack_enabled,
    save_strategy_settings,
)

router = APIRouter(prefix="/internal/strategy-trade", tags=["strategy-trade"])


@router.get("")
def get_strategy_trade(
    _: None = Depends(require_internal_token),
) -> dict:
    """두 전략의 계좌 실제 보유 기준 회차 현황·매수/매도 지정가를 반환한다."""
    return load_strategy_trade_view()


@router.put("/settings")
def put_strategy_trade_settings(
    payload: dict = Body(...),
    _: None = Depends(require_internal_token),
) -> dict:
    """슬랙 스위치 / 전략별 파라미터를 저장하고 갱신된 화면 데이터를 반환한다.

    body 는 둘 중 하나:
      ``{"slack_enabled": true}``                                 — 전역 스위치
      ``{"strategy_id": "kospi200", "config": {pct 3종}}``        — 전략별 파라미터
    config 는 세 값을 모두 담아야 한다 (검증은 validate_strategy_trade_config).
    """
    if not isinstance(payload, dict):
        raise ValueError("요청 형식이 올바르지 않습니다.")

    slack_enabled = payload.get("slack_enabled")
    strategy_id = payload.get("strategy_id")
    config = payload.get("config")

    if slack_enabled is not None:
        save_slack_enabled(bool(slack_enabled))
    elif strategy_id is not None or config is not None:
        if not strategy_id or not isinstance(config, dict):
            raise ValueError("'strategy_id' 와 'config' 객체가 함께 필요합니다.")
        save_strategy_settings(str(strategy_id), config=config)
    else:
        raise ValueError("'slack_enabled' 또는 'strategy_id'+'config' 가 필요합니다.")
    return load_strategy_trade_view()


@router.post("/slack-test")
def post_strategy_trade_slack_test(_: None = Depends(require_internal_token)) -> dict:
    """지금 수동 발송(트리거 없으면 현황만). 화면 '지금 발송(테스트)' 버튼용."""
    from utils.strategy_trade_notify import notify_strategy_trade

    result = notify_strategy_trade(force=True)
    return {"sent": result["sent"], "message": result["reason"]}
