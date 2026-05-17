"""로컬에서 서버 MongoDB 를 가져와 로컬 디스크에 백업한다.

run_local_dev.py 진입점에서 호출되어, 마지막 백업으로부터 일정 시간(기본 24h)
이상 지났을 때만 실제 백업을 수행한다. 그 외에는 즉시 종료한다.

동작 방식:
    사용자가 미리 띄워둔 autossh 터널(localhost:27017 → 서버 27017)을 사용해
    로컬에서 mongodump 를 직접 실행한다. 스크립트는 SSH 를 별도로 호출하지 않는다.

선행 조건:
    - autossh 터널이 켜져 있어야 함 (localhost:27017 가 서버 MongoDB 로 포워딩)
    - 로컬에 mongodump 설치: `brew install mongodb-database-tools`

기본 보관 기간: 7일 (그 이상 된 백업 자동 삭제).

환경변수:
    BACKUP_MIN_INTERVAL_HOURS  (기본 24)
    BACKUP_RETENTION_DAYS      (기본 7)
    BACKUP_MONGO_HOST          (기본 localhost:27017)
    BACKUP_DIR                 (기본 <repo>/.backups)
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_BACKUP_DIR = ROOT_DIR / ".backups"


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


_MARKER_STARTED_KEY = "마지막 백업 시작:"
_MARKER_ENDED_KEY = "마지막 백업 종료:"
_MARKER_ELAPSED_KEY = "걸린 시간:"
_MARKER_STATUS_KEY = "성공 여부:"
_MARKER_LOG_KEY = "로그:"


def _read_marker_started_at(marker: Path) -> datetime | None:
    """마커 파일에서 '마지막 백업 시작:' 라인을 파싱해 datetime 반환."""
    if not marker.exists():
        return None
    try:
        text = marker.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith(_MARKER_STARTED_KEY):
            value = line[len(_MARKER_STARTED_KEY):].strip()
            if not value:
                return None
            try:
                return datetime.fromisoformat(value)
            except ValueError:
                return None
    return None


def _write_marker(
    marker: Path,
    *,
    started_at: datetime,
    ended_at: datetime,
    elapsed_seconds: float,
    success: bool,
    log: str | None = None,
) -> None:
    """마커 파일에 백업 결과를 사람이 읽을 수 있는 형식으로 저장."""
    lines = [
        f"{_MARKER_STARTED_KEY} {started_at.isoformat(timespec='seconds')}",
        f"{_MARKER_ENDED_KEY} {ended_at.isoformat(timespec='seconds')}",
        f"{_MARKER_ELAPSED_KEY} {elapsed_seconds:.1f}초",
        f"{_MARKER_STATUS_KEY} {'성공' if success else '실패'}",
    ]
    if not success and log:
        lines.append(f"{_MARKER_LOG_KEY}")
        lines.append(log.rstrip())
    marker.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _should_run(backup_dir: Path, min_interval_hours: float) -> bool:
    marker = _last_backup_marker(backup_dir)
    last = _read_marker_started_at(marker)
    if last is None:
        return True
    age_hours = (datetime.now() - last).total_seconds() / 3600
    if age_hours >= min_interval_hours:
        return True
    print(
        f"[backup_mongo] 마지막 백업 시작({last.isoformat(timespec='seconds')})으로부터 "
        f"{age_hours:.1f}시간 경과 — 임계({min_interval_hours:.0f}h) 미만이라 스킵"
    )
    return False


def _notify_failure(message: str) -> None:
    """실패 시에만 슬랙 전송. 성공은 무알림."""
    try:
        sys.path.insert(0, str(ROOT_DIR))
        from utils.notification import send_slack_message_v2

        send_slack_message_v2(message)
    except Exception as exc:  # pragma: no cover - 알림 실패는 표준에러로만
        print(f"[backup_mongo] 슬랙 알림 실패: {exc}", file=sys.stderr)


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

    mongo_host = _get_env("BACKUP_MONGO_HOST", _read_dotenv_pair("MONGO_DB_HOST") or "localhost:27017")
    db_name = _get_env("BACKUP_MONGO_DB_NAME", _read_dotenv_pair("MONGO_DB_NAME") or "momentum_etf_db")
    mongo_user = _get_env("BACKUP_MONGO_USER", _read_dotenv_pair("MONGO_DB_USER") or "")
    mongo_pass = _get_env("BACKUP_MONGO_PASSWORD", _read_dotenv_pair("MONGO_DB_PASSWORD") or "")

    if not mongo_user or not mongo_pass:
        print("[backup_mongo] MONGO_DB_USER/MONGO_DB_PASSWORD 가 .env 에 없습니다 — 백업 불가", file=sys.stderr)
        return 1

    # 로컬에 mongodump 가 설치돼 있는지 확인
    from shutil import which
    if which("mongodump") is None:
        print(
            "[backup_mongo] mongodump 를 찾을 수 없습니다. 설치: brew install mongodb-database-tools",
            file=sys.stderr,
        )
        return 1

    started_at = datetime.now()
    timestamp = started_at.strftime("%Y%m%d-%H%M%S")
    target = backup_dir / f"{timestamp}.archive.gz"
    marker = _last_backup_marker(backup_dir)
    print(f"[backup_mongo] 백업 시작 → {target}")

    # autossh 터널(localhost:27017 → 서버 MongoDB)을 통해 로컬에서 직접 mongodump 실행.
    # 인증은 앱 유저(MONGO_DB_USER)로 충분 — admin DB 가 authenticationDatabase.
    dump_cmd = [
        "mongodump",
        f"--host={mongo_host}",
        f"--username={mongo_user}",
        f"--password={mongo_pass}",
        "--authenticationDatabase=admin",
        f"--db={db_name}",
        "--archive",
        "--gzip",
    ]

    started_mono = time.monotonic()
    with target.open("wb") as out:
        proc = subprocess.run(dump_cmd, stdout=out, stderr=subprocess.PIPE, check=False)

    elapsed = time.monotonic() - started_mono
    ended_at = datetime.now()
    size_mb = target.stat().st_size / (1024 * 1024) if target.exists() else 0
    stderr_text = (proc.stderr or b"").decode("utf-8", errors="ignore").strip()

    success = proc.returncode == 0 and size_mb >= 0.01

    if not success:
        # 실패한 0바이트 파일 정리
        try:
            target.unlink(missing_ok=True)
        except OSError:
            pass
        log_tail = stderr_text[-1000:] if stderr_text else f"(exit={proc.returncode}, size={size_mb:.2f}MB)"
        _write_marker(
            marker,
            started_at=started_at,
            ended_at=ended_at,
            elapsed_seconds=elapsed,
            success=False,
            log=log_tail,
        )
        print(f"[backup_mongo] 실패 (exit={proc.returncode}, size={size_mb:.1f}MB): {log_tail[:300]}", file=sys.stderr)
        _notify_failure(
            "❌ *MongoDB 백업 실패 (로컬)*\n"
            f"• 시작: {started_at.isoformat(timespec='seconds')}\n"
            f"• 소요: {elapsed:.1f}s\n"
            f"• exit: {proc.returncode}\n"
            f"```\n{log_tail[-700:]}\n```"
        )
        return proc.returncode or 1

    _write_marker(
        marker,
        started_at=started_at,
        ended_at=ended_at,
        elapsed_seconds=elapsed,
        success=True,
    )
    print(f"[backup_mongo] 완료 - {size_mb:.1f} MB, {elapsed:.1f}s")
    _cleanup_old(backup_dir, retention_days)
    return 0


if __name__ == "__main__":
    sys.exit(run_backup())
