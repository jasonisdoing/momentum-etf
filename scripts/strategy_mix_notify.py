"""합성 오늘의 액션 슬랙 알람 배치 (스케줄러용).

인자: 국가 코드(kor/us) — 그 국가 풀의 계좌만 감시한다(장 시간이 달라 배치를 나눠 돌린다).
인자가 없으면 전체.

계좌 설정에서 mix_pool + mix_slack_enabled 가 켜진 계좌를 감시해, 새 지시나
수량 증가가 있을 때만 슬랙을 보낸다(감소·소멸은 체결 반영이라 조용히 넘긴다).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from utils.env import load_env_if_present
from utils.strategy_mix_notify import notify_all


def main() -> int:
    load_env_if_present()
    country = (sys.argv[1].strip().lower() if len(sys.argv) > 1 else None) or None
    result = notify_all(country)
    for row in result["targets"]:
        if "error" in row:
            print(f"[strategy_mix_notify] {row['pool']} 실패 — {row['error']}")
        elif row.get("changed"):
            print(
                f"[strategy_mix_notify] {row['pool']} 발송 — 신규·증가 {row['new_or_grown']}건 (전체 {row['items']}건)"
            )
        else:
            print(f"[strategy_mix_notify] {row['pool']} 변화 없음 ({row['items']}건 유지)")
    if not result["targets"]:
        print("[strategy_mix_notify] 감시 대상 계좌 없음")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
