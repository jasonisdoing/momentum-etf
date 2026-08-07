"""보유종목 알람 배치 (무인자 래퍼).

계좌별 마스터 On 계좌에서 전역 On 알람 종류(20일선 이탈·손절 등)의 트리거를 모아
슬랙 1건으로 발송한다. 트리거가 없으면 보내지 않는다. 새 알람 종류는 서비스에만 추가하면
이 배치가 함께 발송한다. 평일 한국 장 개시 직후(09:10 KST) 1회 실행(크론) — 아침 알림.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.holdings_alarm_service import send_holdings_alarms  # noqa: E402


def main() -> None:
    result = send_holdings_alarms(manual=False)
    print(f"[holdings_alarm] {result}")


if __name__ == "__main__":
    main()
