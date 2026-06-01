"""배포 진행 플래그를 MongoDB batch_locks 에 기록한다.

deploy.yml 의 trap quote escape 문제를 회피하기 위해 별도 파일로 분리.
fastapi_app 컨테이너 안에서 실행된다.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from utils.db_manager import get_db_connection


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
