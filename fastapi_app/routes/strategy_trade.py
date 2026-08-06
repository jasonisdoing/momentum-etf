"""전략 사고팔기(분할 매수/매도) 운용 현황 API."""

from fastapi import APIRouter, Body, Depends

from fastapi_app.dependencies import require_internal_token
from utils.strategy_trade_service import load_strategy_trade_view, save_settings

router = APIRouter(prefix="/internal/strategy-trade", tags=["strategy-trade"])


@router.get("")
def get_strategy_trade(
    _: None = Depends(require_internal_token),
) -> dict:
    """계좌 실제 보유 기준의 회차 현황·매수/매도 지정가를 반환한다."""
    return load_strategy_trade_view()


@router.put("/settings")
def put_strategy_trade_settings(
    payload: dict = Body(...),
    _: None = Depends(require_internal_token),
) -> dict:
    """슬랙 스위치 / 전략 파라미터를 저장하고 갱신된 화면 데이터를 반환한다.

    body 는 둘 중 하나 이상을 담는다 (없는 항목은 미변경):
      ``{"slack_enabled": true}``
      ``{"config": {"entry_drop_pct": 5, "add_drop_pct": 3, "take_profit_pct": 7}}``
    config 는 세 값을 모두 담아야 한다 (검증은 validate_strategy_trade_config).
    """
    if not isinstance(payload, dict):
        raise ValueError("요청 형식이 올바르지 않습니다.")

    slack_enabled = payload.get("slack_enabled")
    config = payload.get("config")
    if slack_enabled is None and config is None:
        raise ValueError("'slack_enabled' 또는 'config' 가 필요합니다.")
    if config is not None and not isinstance(config, dict):
        raise ValueError("'config' 는 객체여야 합니다.")

    save_settings(
        slack_enabled=None if slack_enabled is None else bool(slack_enabled),
        config=config,
    )
    return load_strategy_trade_view()


@router.post("/slack-test")
def post_strategy_trade_slack_test(
    _: None = Depends(require_internal_token),
) -> dict:
    """슬랙 수동 발송 — 스위치·중복 방지를 무시하고 현재 상태를 보낸다.

    신호가 있으면 신호를, 없으면 현재 대기 조건을 담아 보낸다.
    """
    from utils.strategy_trade_notify import notify_strategy_trade

    result = notify_strategy_trade(force=True)
    return {"sent": result["sent"], "message": result["reason"]}
