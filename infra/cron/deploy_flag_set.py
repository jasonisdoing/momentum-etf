"""배포 진행 플래그를 MongoDB batch_locks 에 기록한다.

deploy.yml 의 trap quote escape 문제를 회피하기 위해 별도 파일로 분리. 맥미니 배포 셸에서 스크립트로 실행되므로
저장소 루트를 import 경로에 넣는다(넣지 않으면 `utils` 를 못 찾아 잠금이 걸리지 않는다).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from datetime import datetime, timedelta

from utils.db_manager import get_db_connection  # noqa: E402


def main() -> None:
    db = get_db_connection()
    if db is None:
        raise SystemExit("DB 연결 실패")
    now = datetime.utcnow()
    db.batch_locks.replace_one(
        {"_id": "__deploy__"},
        {
            "_id": "__deploy__",
            "host": "jason-server",
            "started_at": now,
            "expires_at": now + timedelta(minutes=30),
        },
        upsert=True,
    )
    print(">>> deploy 플래그 설정 (Mongo batch_locks.__deploy__)")


if __name__ == "__main__":
    main()
