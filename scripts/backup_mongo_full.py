"""MongoDB 전체 백업 — `backups/YYYYMMDD/` 에 mongodump(BSON) 로 저장한다.

DB 정리·마이그레이션 같은 위험 작업 전에 스냅샷을 남기는 용도다. BSON 그대로 받으므로
가격 캐시의 parquet 바이너리가 부풀지 않고, 복원은 `mongorestore` 한 줄로 끝난다.

    python scripts/backup_mongo_full.py                 # backups/YYYYMMDD/ 로 백업
    python scripts/backup_mongo_full.py --gzip          # 각 컬렉션을 gzip 압축
    python scripts/backup_mongo_full.py --dry-run       # 실행할 명령만 출력

복원 예시(전체):
    mongorestore --uri="<접속문자열>" --drop backups/YYYYMMDD/<db이름>
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # noqa: E402

from utils.db_manager import _resolve_connection_string  # noqa: E402
from utils.env import load_env_if_present  # noqa: E402

_BACKUP_ROOT = Path("backups")


def _mask_uri(uri: str) -> str:
    """로그·출력용 — 접속 문자열의 자격증명을 가린다."""
    return re.sub(r"://[^@]+@", "://***:***@", uri)


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"환경 변수 {name} 가 필요합니다.")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="MongoDB 전체 백업 (mongodump)")
    parser.add_argument("--gzip", action="store_true", help="컬렉션별 gzip 압축.")
    parser.add_argument("--dry-run", action="store_true", help="실행할 명령만 출력.")
    args = parser.parse_args()

    load_env_if_present()
    uri = _resolve_connection_string()
    db_name = _require_env("MONGO_DB_NAME")

    out_dir = _BACKUP_ROOT / datetime.now().strftime("%Y%m%d")
    command = [
        "mongodump",
        f"--uri={uri}",
        f"--db={db_name}",
        f"--out={out_dir}",
    ]
    if args.gzip:
        command.append("--gzip")

    printable = " ".join(_mask_uri(part) for part in command)
    print(f"실행: {printable}")
    if args.dry_run:
        print("\n--dry-run 이라 실제로 백업하지 않았습니다.")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(command, capture_output=True, text=True)
    # mongodump 는 진행 로그를 stderr 로 보낸다 — 실패 판정은 반환코드로만 한다.
    if result.returncode != 0:
        print(result.stderr.strip() or result.stdout.strip())
        raise SystemExit(f"mongodump 실패 (exit={result.returncode})")

    dumped = out_dir / db_name
    files = sorted(dumped.glob("*.bson")) if dumped.exists() else []
    total = sum(f.stat().st_size for f in files)
    print(f"\n완료 — {dumped}")
    print(f"  컬렉션 {len(files)}개, 합계 {total / 1e6:.1f} MB")
    print(f"\n복원: mongorestore --uri=\"<접속문자열>\" --drop {dumped}")


if __name__ == "__main__":
    main()
