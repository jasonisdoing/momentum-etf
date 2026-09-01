"""증권사 잔고 동기화 배치 (무인자 실행 — 스케줄러용).

계좌 설정에 API 연동(broker_api)이 저장된 계좌들의 잔고·보유를 증권사 값으로
덮어쓴다. 실패는 계좌별로 격리되고, 슬랙은 실패 시작·복구 시 1회씩만 보낸다.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from utils.broker_balance_sync import sync_all
from utils.env import load_env_if_present


def main() -> int:
    load_env_if_present()
    result = sync_all()
    rows = result["accounts"]
    if not rows:
        print("[broker_balance_sync] 연동된 계좌 없음")
        return 0
    for row in rows:
        if row["ok"] and not row.get("changed"):
            print(f"[broker_balance_sync] {row['account_id']} 변화 없음 — 저장·알림 생략")
        elif row["ok"]:
            print(
                f"[broker_balance_sync] {row['account_id']} 반영 — 현금 {row['cash']:,.0f} · {row['holdings_count']}종목"
            )
        else:
            print(f"[broker_balance_sync] {row['account_id']} 실패 — {row['error']}")
    return 0 if all(row["ok"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
