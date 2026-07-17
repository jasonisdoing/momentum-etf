"""pool_settings 의 MAIN_MA_DAYS → LONG_MA_DAYS 키 이름 변경 마이그레이션.

앱이 죽지 않도록 expand → (코드 배포) → contract 2단계로 나눈다.

    expand   : LONG_MA_DAYS 를 MAIN_MA_DAYS 값으로 추가한다. 두 키가 공존하므로
               구버전 코드(MAIN_MA_DAYS 사용)도 그대로 동작한다.
    contract : MAIN_MA_DAYS 를 제거한다. 신버전 코드 배포 후에 실행한다.

각 단계는 비교확인(기본) → 백업 → 실행 순서로만 진행한다.

    python scripts/migrate_main_ma_to_long_ma.py expand                    # 비교확인만
    python scripts/migrate_main_ma_to_long_ma.py expand --backup out.json  # 백업
    python scripts/migrate_main_ma_to_long_ma.py expand --apply --backup out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.db_manager import get_db_connection

COLLECTION = "pool_settings"
OLD_KEY = "MAIN_MA_DAYS"
NEW_KEY = "LONG_MA_DAYS"


def _load_docs(db: Any) -> list[dict[str, Any]]:
    return list(db[COLLECTION].find({}).sort("order", 1))


def _print_comparison(docs: list[dict[str, Any]], phase: str) -> int:
    """현재 상태와 변경 후 상태를 나란히 출력하고, 변경 대상 수를 반환한다."""
    print(f"\n=== 비교확인: {phase} · {COLLECTION} {len(docs)}건 ===")
    print(f"{'pool_id':<12} {'현재':<28} {'변경 후':<28}")
    print("-" * 70)

    targets = 0
    for doc in docs:
        pool_id = str(doc.get("_id"))
        old_val = doc.get(OLD_KEY)
        new_val = doc.get(NEW_KEY)
        before = f"{OLD_KEY}={old_val} {NEW_KEY}={new_val}"

        if phase == "expand":
            if old_val is None:
                after = "(대상 아님: MAIN_MA_DAYS 없음)"
            elif new_val is not None:
                after = "(대상 아님: 이미 LONG_MA_DAYS 있음)"
            else:
                after = f"{OLD_KEY}={old_val} {NEW_KEY}={old_val}"
                targets += 1
        else:  # contract
            if old_val is None:
                after = "(대상 아님: 이미 제거됨)"
            elif new_val is None:
                after = "!! 위험: LONG_MA_DAYS 가 없어 제거 불가"
            else:
                after = f"{NEW_KEY}={new_val} ({OLD_KEY} 제거)"
                targets += 1

        print(f"{pool_id:<12} {before:<28} {after:<28}")

    print("-" * 70)
    print(f"변경 대상: {targets}건")
    return targets


def _guard_contract(docs: list[dict[str, Any]]) -> None:
    """LONG_MA_DAYS 없이 MAIN_MA_DAYS 만 있는 문서가 있으면 중단한다."""
    broken = [str(d.get("_id")) for d in docs if d.get(OLD_KEY) is not None and d.get(NEW_KEY) is None]
    if broken:
        raise SystemExit(
            f"중단: {', '.join(broken)} 에 {NEW_KEY} 가 없습니다. expand 를 먼저 실행하세요."
        )

    # 값 불일치는 정상일 수 있다. 신버전 코드 배포 후 LONG_MA_DAYS 를 수정하면
    # 구 키는 옛 값으로 남는다. 이때 정본은 LONG_MA_DAYS 이므로 알리기만 한다.
    mismatched = [
        f"{d.get('_id')}({OLD_KEY}={d.get(OLD_KEY)} → {NEW_KEY}={d.get(NEW_KEY)} 유지)"
        for d in docs
        if d.get(OLD_KEY) is not None and d.get(NEW_KEY) is not None and d.get(OLD_KEY) != d.get(NEW_KEY)
    ]
    if mismatched:
        print(f"알림: 배포 후 값이 바뀐 종목풀 — {', '.join(mismatched)}")


def _write_backup(docs: list[dict[str, Any]], path: Path) -> None:
    payload = json.dumps(docs, ensure_ascii=False, indent=2, default=str)
    path.write_text(payload, encoding="utf-8")
    print(f"백업 완료: {path} ({len(docs)}건)")


def main() -> None:
    parser = argparse.ArgumentParser(description=f"{COLLECTION}: {OLD_KEY} → {NEW_KEY}")
    parser.add_argument("phase", choices=("expand", "contract"))
    parser.add_argument("--apply", action="store_true", help="실제로 DB 를 변경한다 (기본은 비교확인만)")
    parser.add_argument("--backup", type=Path, help="변경 전 문서를 JSON 으로 저장할 경로")
    args = parser.parse_args()

    db = get_db_connection()
    if db is None:
        raise SystemExit("중단: DB 연결에 실패했습니다.")

    docs = _load_docs(db)
    if not docs:
        raise SystemExit(f"중단: {COLLECTION} 컬렉션이 비어 있습니다.")

    if args.phase == "contract":
        _guard_contract(docs)

    targets = _print_comparison(docs, args.phase)

    if args.backup:
        _write_backup(docs, args.backup)

    if not args.apply:
        print("\n비교확인만 수행했습니다. 백업 후 --apply 로 실행하세요.")
        return

    if not args.backup:
        raise SystemExit("중단: --apply 는 --backup 과 함께 써야 합니다.")

    if targets == 0:
        print("\n변경 대상이 없습니다.")
        return

    now = datetime.now()
    if args.phase == "expand":
        result = db[COLLECTION].update_many(
            {OLD_KEY: {"$ne": None}, NEW_KEY: None},
            [{"$set": {NEW_KEY: f"${OLD_KEY}", "updated_at": now}}],
        )
    else:
        result = db[COLLECTION].update_many(
            {OLD_KEY: {"$ne": None}},
            {"$unset": {OLD_KEY: ""}, "$set": {"updated_at": now}},
        )

    print(f"\n실행 완료: {result.modified_count}건 변경")

    print("\n=== 실행 후 상태 ===")
    for doc in _load_docs(db):
        print(f"  {doc.get('_id'):<12} {OLD_KEY}={doc.get(OLD_KEY)} {NEW_KEY}={doc.get(NEW_KEY)}")


if __name__ == "__main__":
    main()
