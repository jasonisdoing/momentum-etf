"""leverage_state(switch) 를 07-06 장전 오확정 이전의 '현금 보유' 상태로 되돌린다.

07-06 07:55 장전 실행이 오늘을 새 확정일로 잘못 잡아 CASH→공격(243880), 보유시작일=07-06
으로 덮어썼다. 코드 수정(장전=장중 취급) 후, 잃어버린 '현금 8거래일째' 연속성을 복원한다.

절차: 비교확인(현재값 출력) → 백업(파일) → --apply 시 복원.
기본은 dry-run. 실제 반영은 `python scripts/restore_leverage_state_cash.py --apply`.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.env import load_env_if_present

load_env_if_present()

_PROFILE = "switch"
# 거래일 역산으로 확정한 현금 보유시작일 (07-03=7거래일째 / 07-06=8거래일째 검증 완료).
_CASH_HOLDING_START = "2026-06-25"
_TARGET = {
    "target": "CASH",
    "target_name": "현금",
    "holding_start_date": _CASH_HOLDING_START,
}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="leverage_state 현금 상태 복원")
    parser.add_argument("--apply", action="store_true", help="실제 DB 반영 (없으면 dry-run)")
    args = parser.parse_args(argv[1:])

    from utils.db_manager import get_db_connection

    db = get_db_connection()
    if db is None:
        print("DB 연결 실패", file=sys.stderr)
        return 1

    doc = db.leverage_state.find_one({"_id": _PROFILE})
    current = dict(doc.get("state") or {}) if doc else {}
    print("현재 상태:", json.dumps(current, ensure_ascii=False, default=str))

    # 백업 (항상 남긴다)
    backup_dir = ROOT_DIR / "leverage" / "zresults" / "backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"leverage_state_{_PROFILE}_{stamp}.json"
    backup_path.write_text(json.dumps(current, ensure_ascii=False, default=str, indent=2), encoding="utf-8")
    print(f"백업 저장: {backup_path}")

    # 복원값: 날짜(date)는 마지막 확정일을 유지하지 못하므로 보유시작일 기준으로 둔다.
    restored = {
        "date": _CASH_HOLDING_START,
        "target": _TARGET["target"],
        "target_name": _TARGET["target_name"],
        "holding_start_date": _TARGET["holding_start_date"],
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    print("복원 예정:", json.dumps(restored, ensure_ascii=False))

    if not args.apply:
        print("\ndry-run 완료: 반영하려면 --apply 를 붙여 다시 실행하세요.")
        return 0

    db.leverage_state.update_one(
        {"_id": _PROFILE},
        {"$set": {"state": restored, "updated_at": datetime.now(timezone.utc)}},
        upsert=True,
    )
    print("\n복원 완료. 다음 장 마감 후 정상 재확정됩니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
