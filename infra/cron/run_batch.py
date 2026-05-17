#!/usr/bin/env python3
"""VM cron 배치 래퍼.

사용법:
    python infra/cron/run_batch.py <job_name> <command...>

예시:
    python infra/cron/run_batch.py cache_refresh python scripts/stock_price_cache_updater.py

동작:
    1) subprocess 로 <command> 를 실행
    2) 실패 시 종료 코드/소요시간/마지막 로그 꼬리(15줄)를 슬랙으로 전송
    3) 프로세스 종료 코드를 그대로 반환

알림 채널:
    utils.notification.send_slack_message_v2() 를 사용하므로
    환경변수 SLACK_BOT_TOKEN 과 config.SLACK_CHANNEL 을 따릅니다.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

KST = ZoneInfo("Asia/Seoul")

# 프로젝트 루트를 파이썬 경로에 추가 (컨테이너 WORKDIR=/app)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MAX_TAIL_LINES = 15
MAX_TAIL_CHARS = 1500

LOCK_DIR = PROJECT_ROOT / "logs" / "cron"
DEPLOY_LOCK = LOCK_DIR / ".deploy.lock"
SUCCESS_NOTIFICATION_DISABLED_JOBS = {
    "cache_refresh",
    "portfolio_refresh",
    "metadata_updater",
    "asset_summary",
    "market_hours_analysis",
    "us_market_stocks",
    "data_aggregate",
}
EXIT_ALREADY_NOTIFIED = 66


def _append_log_line(job_name: str, text: str) -> None:
    """배치 로그를 logs/cron/<job>.log 에 항상 추가한다."""
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOCK_DIR / f"{job_name}.log"
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(text)
        if not text.endswith("\n"):
            handle.write("\n")


def _acquire_db_lock(job_name: str, ttl_seconds: int = 1800) -> tuple[object, str] | None:
    """MongoDB 에 분산 락을 잡는다. 다른 호스트(로컬/서버)에서 동일 작업 중복 실행 방지.

    반환: (db, job_name) 성공 시 / None: 이미 다른 곳에서 실행 중
    """
    from utils.db_manager import get_db_connection  # 지연 임포트

    db = get_db_connection()
    if db is None:
        raise RuntimeError("DB 연결 실패 — 배치 실행 불가")

    # 만료 인덱스 (1회만 호출되어도 idempotent)
    try:
        db.batch_locks.create_index("expires_at", expireAfterSeconds=0)
    except Exception:
        pass  # 이미 있을 수 있음

    now = datetime.now(KST)
    expires_at = now + timedelta(seconds=ttl_seconds)
    host = socket.gethostname()
    pid = os.getpid()

    # 만료된 락은 미리 제거(클럭 차이로 TTL 인덱스 지연이 있을 수 있음)
    db.batch_locks.delete_many({"_id": job_name, "expires_at": {"$lt": now}})

    try:
        db.batch_locks.insert_one({
            "_id": job_name,
            "host": host,
            "pid": pid,
            "acquired_at": now,
            "expires_at": expires_at,
        })
        return (db, job_name)
    except Exception:
        existing = db.batch_locks.find_one({"_id": job_name}) or {}
        owner = f"host={existing.get('host')} pid={existing.get('pid')} acquired_at={existing.get('acquired_at')}"
        raise RuntimeError(f"다른 곳에서 이미 실행 중: {owner}")


def _release_db_lock(handle: tuple[object, str] | None) -> None:
    if handle is None:
        return
    db, job_name = handle
    try:
        db.batch_locks.delete_one({"_id": job_name})
    except Exception as exc:  # pragma: no cover
        print(f"[run_batch] DB 락 해제 실패: {exc}", file=sys.stderr)


def _notify(text: str) -> None:
    """슬랙 전송. 실패해도 배치 결과에 영향 없도록 방어적으로 처리."""
    try:
        from utils.notification import send_slack_message_v2  # 지연 임포트

        send_slack_message_v2(text)
    except Exception as exc:  # pragma: no cover - 알림 실패는 로그만
        print(f"[run_batch] 슬랙 전송 실패: {exc}", file=sys.stderr)


def _format_tail(stdout: str, stderr: str) -> str:
    combined = (stdout or "") + (stderr or "")
    lines = combined.strip().splitlines()
    if not lines:
        return "(출력 없음)"
    tail = "\n".join(lines[-MAX_TAIL_LINES:])
    if len(tail) > MAX_TAIL_CHARS:
        tail = "…(생략)…\n" + tail[-MAX_TAIL_CHARS:]
    return tail


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(
            "usage: run_batch.py <job_name> <command...>",
            file=sys.stderr,
        )
        return 2

    job_name = argv[1]
    command = argv[2:]

    started_at = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST")
    started_monotonic = time.monotonic()

    # 배포 진행 중이면 즉시 스킵 (deploy.yml 이 .deploy.lock 을 생성/제거)
    if DEPLOY_LOCK.exists():
        skip_line = (
            f"[run_batch] SKIP job={job_name} reason=deploy_in_progress at={started_at}"
        )
        print(skip_line)
        _append_log_line(job_name, skip_line)
        return 0

    # MongoDB 분산 락: 로컬/서버 어디서든 동일 작업 중복 실행 차단
    db_lock = None
    try:
        db_lock = _acquire_db_lock(job_name)
    except RuntimeError as exc:
        skip_line = f"[run_batch] SKIP job={job_name} reason={exc} at={started_at}"
        print(skip_line, file=sys.stderr)
        _append_log_line(job_name, skip_line)
        return 0

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    start_line = f"[run_batch] START job={job_name} cmd={' '.join(command)} at={started_at}"
    print(start_line)
    _append_log_line(job_name, start_line)

    try:
        result = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        elapsed = time.monotonic() - started_monotonic
        fail_line = f"[run_batch] FAIL {exc}"
        _append_log_line(job_name, fail_line)
        app_label = os.environ.get("APP_TYPE", "VM").strip() or "VM"
        _notify(
            f"❌ *[{app_label}] 배치 실행 불가*: `{job_name}`\n"
            f"• 시작: {started_at}\n"
            f"• 소요: {elapsed:.1f}s\n"
            f"• 에러: `{exc}`"
        )
        print(fail_line, file=sys.stderr)
        _release_db_lock(db_lock)
        return 127
    except Exception as exc:
        elapsed = time.monotonic() - started_monotonic
        exception_line = f"[run_batch] EXCEPTION {exc}"
        _append_log_line(job_name, exception_line)
        app_label = os.environ.get("APP_TYPE", "VM").strip() or "VM"
        _notify(
            f"❌ *[{app_label}] 배치 예외*: `{job_name}`\n"
            f"• 시작: {started_at}\n"
            f"• 소요: {elapsed:.1f}s\n"
            f"• 에러: `{exc}`"
        )
        print(exception_line, file=sys.stderr)
        _release_db_lock(db_lock)
        return 1
    finally:
        _release_db_lock(db_lock)

    elapsed = time.monotonic() - started_monotonic
    exit_code = result.returncode
    success = exit_code == 0

    # 원본 출력은 그대로 stdout/stderr 로 흘려서 cron 로그파일에도 남김
    if result.stdout:
        sys.stdout.write(result.stdout)
        _append_log_line(job_name, result.stdout)
    if result.stderr:
        sys.stderr.write(result.stderr)
        _append_log_line(job_name, result.stderr)

    emoji = "✅" if success else "❌"
    status = "성공" if success else "실패"
    tail = _format_tail(result.stdout, result.stderr)
    app_label = os.environ.get("APP_TYPE", "VM").strip() or "VM"

    already_notified_failure = exit_code == EXIT_ALREADY_NOTIFIED
    should_notify = ((not success) and (not already_notified_failure)) or (
        success and (job_name not in SUCCESS_NOTIFICATION_DISABLED_JOBS)
    )
    if should_notify:
        _notify(
            f"{emoji} *[{app_label}] 배치 {status}*: `{job_name}`\n"
            f"• 시작: {started_at}\n"
            f"• 소요: {elapsed:.1f}s\n"
            f"• exit: {exit_code}\n"
            f"```\n{tail}\n```"
        )

    ended_at = datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S KST")
    end_line = (
        f"[run_batch] END job={job_name} status={status} exit={exit_code} elapsed={elapsed:.1f}s at={ended_at}"
    )
    print(end_line)
    _append_log_line(job_name, end_line)
    return exit_code


if __name__ == "__main__":
    sys.exit(main(sys.argv))
