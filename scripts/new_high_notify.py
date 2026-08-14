"""신고가 돌파 진입·매도 예정 슬랙 알림 배치 (무인자 실행 — 스케줄러용).

10분마다 실행돼, 슬랙 알람을 켠 종목풀들의 진입·매도 예정 **변화**가 있을 때만
풀 번호순으로 묶어 한 건의 슬랙을 보낸다. 변화가 없으면 아무것도 보내지 않는다.
장이 닫힌 시장의 풀은 계산 없이 건너뛴다 (utils/new_high_notify 참고).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from utils.env import load_env_if_present
from utils.new_high_notify import notify_all


def main() -> int:
    load_env_if_present()
    result = notify_all()
    pools = ", ".join(f"{p['pool']}({'변화' if p.get('changed') else p.get('error', '유지')})" for p in result["pools"])
    print(
        f"[new_high_notify] sent={result['sent']} sections={result['section_count']}"
        + (f" pools=[{pools}]" if pools else " (감시 창 밖 — 대상 없음)")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
