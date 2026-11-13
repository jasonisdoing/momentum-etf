"""슬랙 주문 결과 알림 배치 (무인자 실행 — 스케줄러용).

슬랙으로 낸 주문(운영·모의) 중 체결·취소·거부로 끝난 것을 원 메시지 스레드로 알리고 메시지의 버튼을 걷어낸다.
장 마감(15:30) 직전(15:20~)까지 체결되지 않은 주문은 한 번 경고하고 취소 버튼을 다시 띄운다.
신호 메시지 반응은 스레드 상태 하나(⏳ 진행 중 · ✅ 완료 · ❌ 실패)로 맞춘다.
진행 중인 슬랙 주문이 없으면 증권사를 조회하지 않고 바로 끝난다.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from utils.env import load_env_if_present
from utils.slack_mock_orders import notify_close_approaching, notify_finished_orders, sync_signal_threads


def main() -> int:
    load_env_if_present()
    finished = notify_finished_orders()
    closing = notify_close_approaching()
    print(
        f"[slack_mock_order_notify] 결과 알림 {finished}건, 마감 경고 {closing}건, 완료 스레드 {sync_signal_threads()}건"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
