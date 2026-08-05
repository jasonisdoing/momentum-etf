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
    """계좌별 알람 저장. body: ``{account_id, alarm_type, enabled: bool, values: {...}, icon?: str}``.

    values 는 알람 종류가 요구하는 기준값 전부다 (부족하면 에러 — 임의로 채우지 않는다).
      ma       : ``{"short_days": int, "long_days": int}`` — 단기·장기 이평선 일수
      stoploss : ``{"threshold_pct": float}`` — 손절 기준(%, 음수)
    icon 은 자산 화면 종목명 배지용 이모지(생략 시 미변경, 빈 문자열 = 배지 끔).
    """
    from utils.holdings_alarm_service import set_account_alarm

    account_id = str(payload.get("account_id") or "").strip()
    alarm_type = str(payload.get("alarm_type") or "").strip()
    if not account_id:
        raise ValueError("account_id 가 필요합니다.")
    if alarm_type not in ("ma", "stoploss"):
        raise ValueError("알 수 없는 알람 종류입니다.")
    values = payload.get("values")
    if not isinstance(values, dict) or not values:
        raise ValueError("기준 values 는 비어있지 않은 객체여야 합니다.")
    for key, value in values.items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"기준 values.{key} 는 숫자여야 합니다.")
    raw_icon = payload.get("icon")
    icon = str(raw_icon) if isinstance(raw_icon, str) else None
    return set_account_alarm(account_id, alarm_type, enabled=bool(payload.get("enabled")), values=values, icon=icon)


@router.get("/badges")
def get_badges(account_id: str, _: None = Depends(require_internal_token)) -> dict:
    """자산 화면 종목명 배지 — 계좌 알람 설정·판정 그대로 티커→아이콘 맵 반환."""
    from utils.holdings_alarm_service import compute_account_alert_badges

    return compute_account_alert_badges(account_id)


@router.post("/send")
def post_send(_: None = Depends(require_internal_token)) -> dict:
    """지금 수동 발송(켜진 계좌·종류 중 조건 충족 종목만). 화면 '슬랙 테스트' 버튼용."""
    from utils.holdings_alarm_service import send_holdings_alarms

    return send_holdings_alarms(manual=True)
