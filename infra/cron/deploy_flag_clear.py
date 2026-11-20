"""배포 진행 플래그를 MongoDB batch_locks 에서 제거한다.

deploy.yml 의 trap quote escape 문제를 회피하기 위해 별도 파일로 분리. 맥미니 배포 셸에서 스크립트로 실행되므로
저장소 루트를 import 경로에 넣는다(넣지 않으면 `utils` 를 못 찾아 잠금이 걸리지 않는다).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from utils.db_manager import get_db_connection  # noqa: E402


def main() -> None:
    db = get_db_connection()
    if db is None:
        # 연결 실패는 무시 — deploy 종료 시점이라 다음 deploy 시작에서 덮어쓰면 됨
        return
    result = db.batch_locks.delete_one({"_id": "__deploy__"})
    print(f">>> deploy 플래그 해제 (deleted={result.deleted_count})")


if __name__ == "__main__":
    main()
