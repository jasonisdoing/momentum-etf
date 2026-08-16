"""MongoDB 전체 백업 — `backups/YYYYMMDD/` 에 mongodump 로 저장한다.

매시 크론(로컬 전용)이 실행해 오늘 폴더를 **항상 최신으로 덮어쓴다** — 임시 폴더에
받은 뒤 성공했을 때만 교체하므로, 받다가 중단돼도 직전 정상 백업이 깨지지 않는다.
복원은 `mongorestore` 한 줄로 끝난다. 보존은 최근 `--keep`(기본 30)개 날짜 폴더.

    python scripts/backup_mongo_full.py --gzip          # 표준 실행 (크론과 동일)
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
# 연결 끊김 같은 일시 장애용 재시도 (첫 시도 포함 횟수).
_MAX_ATTEMPTS = 3
_RETRY_WAIT_SECONDS = 10.0


def _mask_uri(uri: str) -> str:
    """로그·출력용 — 접속 문자열의 자격증명을 가린다."""
    return re.sub(r"://[^@]+@", "://***:***@", uri)


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"환경 변수 {name} 가 필요합니다.")
    return value


def _rotate_backups(keep: int) -> None:
    """날짜 폴더(YYYYMMDD)를 최신 keep 개만 남기고 오래된 것부터 삭제한다."""
    if keep <= 0:
        return
    dated = sorted(
        (p for p in _BACKUP_ROOT.iterdir() if p.is_dir() and re.fullmatch(r"\d{8}", p.name)),
        key=lambda p: p.name,
    )
    for old in dated[:-keep]:
        import shutil

        shutil.rmtree(old)
        print(f"보존 회전: 오래된 백업 삭제 — {old}")


def main() -> None:
    parser = argparse.ArgumentParser(description="MongoDB 전체 백업 (mongodump)")
    parser.add_argument("--gzip", action="store_true", help="컬렉션별 gzip 압축.")
    parser.add_argument("--dry-run", action="store_true", help="실행할 명령만 출력.")
    parser.add_argument(
        "--keep",
        type=int,
        default=30,
        help="보존할 날짜 폴더 수 — 초과분은 오래된 것부터 삭제 (0=회전 안 함).",
    )
    args = parser.parse_args()

    load_env_if_present()
    uri = _resolve_connection_string()
    db_name = _require_env("MONGO_DB_NAME")

    out_dir = _BACKUP_ROOT / datetime.now().strftime("%Y%m%d")
    # 임시 폴더에 받은 뒤 성공 시에만 교체 — 중단된 시도가 정상 백업을 덮지 않게 한다.
    tmp_dir = _BACKUP_ROOT / f".tmp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    command = [
        "mongodump",
        f"--uri={uri}",
        f"--db={db_name}",
        f"--out={tmp_dir}",
    ]
    if args.gzip:
        command.append("--gzip")

    printable = " ".join(_mask_uri(part) for part in command)
    print(f"실행: {printable}")
    if args.dry_run:
        print("\n--dry-run 이라 실제로 백업하지 않았습니다.")
        return

    import shutil
    import time

    # 대용량 컬렉션을 읽는 도중 연결이 끊기는 일이 있다(관측: 2026-08-12,
    # "use of closed network connection" — 로컬 맥 절전/일시 부하). 일시 장애라
    # 다시 돌리면 대개 성공하므로 짧게 재시도한다. 매번 처음부터 다시 받는다.
    last_result = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            last_result = subprocess.run(command, capture_output=True, text=True)
        except BaseException:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
        # mongodump 는 진행 로그를 stderr 로 보낸다 — 실패 판정은 반환코드로만 한다.
        if last_result.returncode == 0:
            break
        if attempt < _MAX_ATTEMPTS:
            print(f"[backup] mongodump 실패 (exit={last_result.returncode}) — 재시도 {attempt}/{_MAX_ATTEMPTS - 1}")
            time.sleep(_RETRY_WAIT_SECONDS * attempt)

    if last_result is None or last_result.returncode != 0:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        if last_result is not None:
            print(last_result.stderr.strip() or last_result.stdout.strip())
        raise SystemExit(f"mongodump 실패 (exit={last_result.returncode if last_result else -1})")

    # 성공 — 오늘 폴더를 원자적으로 교체한다.
    if out_dir.exists():
        shutil.rmtree(out_dir)
    tmp_dir.rename(out_dir)

    dumped = out_dir / db_name
    pattern = "*.bson.gz" if args.gzip else "*.bson"
    files = sorted(dumped.glob(pattern)) if dumped.exists() else []
    total = sum(f.stat().st_size for f in files)
    print(f"\n완료 — {dumped}")
    print(f"  컬렉션 {len(files)}개, 합계 {total / 1e6:.1f} MB")

    _rotate_backups(args.keep)

    restore_gzip = " --gzip" if args.gzip else ""
    print(f'\n복원: mongorestore --uri="<접속문자열>" --drop{restore_gzip} {dumped}')


if __name__ == "__main__":
    main()
