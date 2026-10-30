"""슬랙 모의 주문 결과 알림 배치 (무인자 실행 — 스케줄러용).

슬랙으로 낸 모의 주문 중 체결·취소·거부로 끝난 것을 원 메시지 스레드로 알리고 메시지의 버튼을 걷어낸다.
진행 중인 슬랙 주문이 없으면 증권사를 조회하지 않고 바로 끝난다.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from utils.env import load_env_if_present
from utils.slack_mock_orders import notify_finished_orders


def main() -> int:
    load_env_if_present()
    print(f"[slack_mock_order_notify] 알림 {notify_finished_orders()}건")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
