"""보유종목 알람(이동선 이탈·손절 등) 설정·발송 API. 설정은 계좌별."""

from fastapi import APIRouter, Body, Depends

from fastapi_app.dependencies import require_internal_token

router = APIRouter(prefix="/internal/alarms", tags=["alarms"])


@router.get("")
def get_alarms(_: None = Depends(require_internal_token)) -> dict:
    """계좌별 알람 On/Off + 기준(이평선 일수·손절 %) 목록 + 셀렉트 선택지."""
    from utils.holdings_alarm_service import get_alarm_view

    return get_alarm_view()


@router.post("/account")
def post_account(payload: dict = Body(...), _: None = Depends(require_internal_token)) -> dict:
    """계좌별 알람 저장. body: ``{account_id, alarm_type: 'ma20'|'stoploss', enabled: bool, value: number}``.

    value 는 ma20 이면 이평선 일수(정수), stoploss 면 손절 기준(%, 음수).
    """
    from utils.holdings_alarm_service import set_account_alarm

    account_id = str(payload.get("account_id") or "").strip()
    alarm_type = str(payload.get("alarm_type") or "").strip()
    if not account_id:
        raise ValueError("account_id 가 필요합니다.")
    if alarm_type not in ("ma20", "stoploss"):
        raise ValueError("알 수 없는 알람 종류입니다.")
    value = payload.get("value")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("기준 value 는 숫자여야 합니다.")
    return set_account_alarm(account_id, alarm_type, enabled=bool(payload.get("enabled")), value=float(value))


@router.post("/send")
def post_send(_: None = Depends(require_internal_token)) -> dict:
    """지금 수동 발송(켜진 계좌·종류 중 조건 충족 종목만). 화면 '슬랙 테스트' 버튼용."""
    from utils.holdings_alarm_service import send_holdings_alarms

    return send_holdings_alarms(manual=True)
