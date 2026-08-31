"""신고가 전용 슬랙 알람 배치 (스케줄러용).

인자: 국가 코드(kor) — 그 국가의 풀만 감시한다(장 시간에 맞춰 배치를 나눠 돌린다).
인자가 없으면 전체.

신고가 설정에서 `slack_enabled` 가 켜진 풀을 감시해, 어떤 종목의 거래대금 배수가
새 정수 단(1배·2배·3배…)에 처음 도달했을 때만 전체 내용을 슬랙 1건으로 보낸다.
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
    country = (sys.argv[1].strip().lower() if len(sys.argv) > 1 else None) or None
    result = notify_all(country)
    for row in result["targets"]:
        if row.get("skipped"):
            print(f"[new_high_notify] {row['pool']} 스킵 — {row['reason']}")
        elif "error" in row:
            print(f"[new_high_notify] {row['pool']} 실패 — {row['error']}")
        elif row.get("sent"):
            print(f"[new_high_notify] {row['pool']} 발송 — {row['items']}건, 트리거 {', '.join(row['triggered'])}")
        else:
            print(f"[new_high_notify] {row['pool']} 새 단 도달 없음 ({row['items']}건)")
    if not result["targets"]:
        print("[new_high_notify] 감시 대상 풀 없음")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
