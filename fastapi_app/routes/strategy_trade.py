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
    """슬랙 스위치를 저장하고 갱신된 화면 데이터를 반환한다.

    body: ``{"slack_enabled": true}``
    """
    if not isinstance(payload, dict):
        raise ValueError("요청 형식이 올바르지 않습니다.")

    slack_enabled = payload.get("slack_enabled")
    if slack_enabled is None:
        raise ValueError("'slack_enabled' 가 필요합니다.")

    save_settings(slack_enabled=bool(slack_enabled))
    return load_strategy_trade_view()


@router.post("/slack-test")
def post_strategy_trade_slack_test(
    _: None = Depends(require_internal_token),
) -> dict:
    """[미구현] 슬랙 수동 발송 테스트.

    알림 본문·발송 조건(거래시간 매시간 판정)은 아직 정하지 않았다. 화면 배선을
    먼저 맞춰두고, 발송기는 다음 단계에서 붙인다.
    """
    return {"sent": False, "message": "슬랙 알림은 아직 구현되지 않았습니다."}
