"""top_pick_settings → asset_helper_settings 컬렉션 개명 마이그레이션.

탑픽 기능이 자산 헬퍼로 개편되면서 코드의 컬렉션명이 ``asset_helper_settings`` 로 바뀌었다.
기존 데이터가 담긴 ``top_pick_settings`` 컬렉션을 새 이름으로 옮긴다.

안전 절차(회사 규칙): **기본은 dry-run(비교 확인)** 이고, ``--apply`` 를 줘야 실제로 백업 파일을
남긴 뒤 개명한다. 실행은 사용자가 직접 한다.

    python scripts/migrate_asset_helper_collection.py            # 비교만(변경 없음)
    python scripts/migrate_asset_helper_collection.py --apply    # 백업 후 실제 개명

동작:
- 새 컬렉션(asset_helper_settings)이 이미 있으면 중단(수동 확인 필요).
- 구 컬렉션이 없으면 개명할 것이 없으므로 종료.
- --apply: 구 컬렉션 문서를 백업 파일로 덤프 → renameCollection 으로 개명.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from utils.db_manager import get_db_connection  # noqa: E402

_OLD = "top_pick_settings"
_NEW = "asset_helper_settings"


def main() -> None:
    parser = argparse.ArgumentParser(description="top_pick_settings → asset_helper_settings 개명")
    parser.add_argument("--apply", action="store_true", help="실제 개명(미지정 시 비교만).")
    args = parser.parse_args()

    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결 실패")

    names = set(db.list_collection_names())
    old_exists = _OLD in names
    new_exists = _NEW in names
    old_count = db[_OLD].count_documents({}) if old_exists else 0
    new_count = db[_NEW].count_documents({}) if new_exists else 0

    print("=== 개명 계획 ===")
    print(f"  {_OLD}: {'있음' if old_exists else '없음'} (문서 {old_count})")
    print(f"  {_NEW}: {'있음' if new_exists else '없음'} (문서 {new_count})")

    if new_exists:
        print(f"\n⚠️ 새 컬렉션 '{_NEW}' 가 이미 존재합니다. 수동 확인 후 진행하세요(자동 개명 중단).")
        return
    if not old_exists:
        print(f"\n구 컬렉션 '{_OLD}' 가 없습니다. 개명할 대상이 없어 종료합니다.")
        return

    if not args.apply:
        print("\n[dry-run] 변경하지 않았습니다. 확인 후 --apply 로 실행하세요.")
        return

    # 백업(구 컬렉션 전체 덤프)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"backup_{_OLD}_{ts}.json")
    docs = list(db[_OLD].find({}))
    with open(backup_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n백업 저장: {backup_path} (문서 {len(docs)})")

    db[_OLD].rename(_NEW)
    print(f"개명 완료: {_OLD} → {_NEW}")


if __name__ == "__main__":
    main()
