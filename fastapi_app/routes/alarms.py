"""보유종목 알람(이동선 이탈·손절) 배지·발송 API.

On/Off 는 **계좌 설정 화면**(`/account-settings`)에서 계좌 문서에 직접 저장한다.
이평선 일수·손절 기준(%)은 종목이 속한 종목풀 설정에서 온다. 그래서 여기엔 설정 조회/저장
엔드포인트가 없다.
"""

from fastapi import APIRouter, Depends

from fastapi_app.dependencies import require_internal_token

router = APIRouter(prefix="/internal/alarms", tags=["alarms"])


@router.get("/badges")
def get_badges(account_id: str, _: None = Depends(require_internal_token)) -> dict:
    """자산 화면 종목명 배지 — 계좌 알람 설정·판정 그대로 티커→아이콘 맵 반환."""
    from utils.holdings_alarm_service import compute_account_alert_badges

    return compute_account_alert_badges(account_id)


@router.post("/send")
def post_send(_: None = Depends(require_internal_token)) -> dict:
    """지금 수동 발송(켜진 계좌·종류 중 조건 충족 종목만). 계좌 설정의 '슬랙 알람 테스트' 버튼용."""
    from utils.holdings_alarm_service import send_holdings_alarms

    return send_holdings_alarms(manual=True)
