"""로컬에서 서버 MongoDB 를 가져와 로컬 디스크에 백업한다.

run_local_dev.py 진입점에서 호출되어, 마지막 백업으로부터 일정 시간(기본 24h)
이상 지났을 때만 실제 백업을 수행한다. 그 외에는 즉시 종료한다.

동작 방식:
    SSH 로 서버에 접속 → docker compose exec mongodb mongodump --archive --gzip
    → 결과를 SSH stdout 으로 받아 로컬 파일에 저장.

기본 보관 기간: 7일 (그 이상 된 백업 자동 삭제).

환경변수:
    BACKUP_MIN_INTERVAL_HOURS  (기본 24)
    BACKUP_RETENTION_DAYS      (기본 7)
    BACKUP_SSH_HOST            (기본 134.185.109.82)
    BACKUP_SSH_USER            (기본 ubuntu)
    BACKUP_SSH_KEY             (기본 ~/DEV/ssh-key-2025-10-09.key)
    BACKUP_DIR                 (기본 <repo>/backups)
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_BACKUP_DIR = ROOT_DIR / "backups"


def _get_env(name: str, default: str) -> str:
    value = os.environ.get(name, "").strip()
    return value or default


def _backup_dir() -> Path:
    raw = _get_env("BACKUP_DIR", str(DEFAULT_BACKUP_DIR))
    path = Path(raw).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _last_backup_marker(backup_dir: Path) -> Path:
    return backup_dir / ".last_backup"


def _should_run(backup_dir: Path, min_interval_hours: float) -> bool:
    marker = _last_backup_marker(backup_dir)
    if not marker.exists():
        return True
    age_hours = (time.time() - marker.stat().st_mtime) / 3600
    if age_hours >= min_interval_hours:
        return True
    print(
        f"[backup_mongo] 마지막 백업으로부터 {age_hours:.1f}시간 경과 — "
        f"임계({min_interval_hours:.0f}h) 미만이라 스킵"
    )
    return False


def _cleanup_old(backup_dir: Path, retention_days: int) -> None:
    if retention_days <= 0:
        return
    cutoff = time.time() - retention_days * 86400
    for entry in backup_dir.glob("*.archive.gz"):
        try:
            if entry.stat().st_mtime < cutoff:
                entry.unlink()
                print(f"[backup_mongo] 오래된 백업 제거: {entry.name}")
        except OSError as exc:
            print(f"[backup_mongo] 오래된 백업 제거 실패 {entry}: {exc}", file=sys.stderr)


def _read_dotenv_pair(key: str) -> str | None:
    """루트 .env 에서 KEY=VALUE 형태로 한 값 읽어온다 (간이 파서)."""
    env_path = ROOT_DIR / ".env"
    if not env_path.exists():
        return None
    try:
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() == key:
                v = v.strip()
                if v.startswith('"') and v.endswith('"'):
                    v = v[1:-1]
                elif v.startswith("'") and v.endswith("'"):
                    v = v[1:-1]
                return v
    except OSError:
        return None
    return None


def run_backup() -> int:
    backup_dir = _backup_dir()
    min_interval = float(_get_env("BACKUP_MIN_INTERVAL_HOURS", "24"))
    retention_days = int(_get_env("BACKUP_RETENTION_DAYS", "7"))

    if not _should_run(backup_dir, min_interval):
        return 0

    ssh_host = _get_env("BACKUP_SSH_HOST", "134.185.109.82")
    ssh_user = _get_env("BACKUP_SSH_USER", "ubuntu")
    ssh_key = Path(_get_env("BACKUP_SSH_KEY", "~/DEV/ssh-key-2025-10-09.key")).expanduser()
    db_name = _get_env("BACKUP_MONGO_DB_NAME", "momentum_etf_db")
    mongo_user = _get_env("BACKUP_MONGO_USER", _read_dotenv_pair("MONGO_DB_ROOT_USER") or "root")
    mongo_pass = _get_env("BACKUP_MONGO_PASSWORD", _read_dotenv_pair("MONGO_DB_ROOT_PASSWORD") or "")

    if not mongo_pass:
        print("[backup_mongo] MONGO_DB_ROOT_PASSWORD 가 .env 에 없습니다 — 백업 불가", file=sys.stderr)
        return 1
    if not ssh_key.exists():
        print(f"[backup_mongo] SSH 키 없음: {ssh_key}", file=sys.stderr)
        return 1

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = backup_dir / f"{timestamp}.archive.gz"
    print(f"[backup_mongo] 백업 시작 → {target}")

    remote_cmd = (
        f"sudo docker compose -f ~/apps/momentum-etf/docker-compose.yml exec -T mongodb "
        f"mongodump "
        f"--username={mongo_user} --password={mongo_pass} --authenticationDatabase=admin "
        f"--db={db_name} --archive --gzip"
    )
    ssh_cmd = [
        "ssh",
        "-i", str(ssh_key),
        "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=15",
        f"{ssh_user}@{ssh_host}",
        remote_cmd,
    ]

    started = time.monotonic()
    with target.open("wb") as out:
        proc = subprocess.run(ssh_cmd, stdout=out, stderr=subprocess.PIPE, check=False)

    elapsed = time.monotonic() - started
    size_mb = target.stat().st_size / (1024 * 1024) if target.exists() else 0

    if proc.returncode != 0 or size_mb < 0.01:
        stderr_tail = (proc.stderr or b"").decode("utf-8", errors="ignore").strip()[-500:]
        print(f"[backup_mongo] 실패 (exit={proc.returncode}, size={size_mb:.1f}MB): {stderr_tail}", file=sys.stderr)
        # 실패한 0바이트 파일 정리
        try:
            target.unlink(missing_ok=True)
        except OSError:
            pass
        return proc.returncode or 1

    _last_backup_marker(backup_dir).touch()
    print(f"[backup_mongo] 완료 - {size_mb:.1f} MB, {elapsed:.1f}s")
    _cleanup_old(backup_dir, retention_days)
    return 0


if __name__ == "__main__":
    sys.exit(run_backup())
