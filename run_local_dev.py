from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
WEB_DIR = ROOT_DIR / "web"
PYTHON_BIN = ROOT_DIR / ".venv" / "bin" / "python"


def _ensure_python_exists() -> None:
    if not PYTHON_BIN.exists():
        raise FileNotFoundError("프로젝트 가상환경 Python을 찾을 수 없습니다: .venv/bin/python")


def _start_fastapi() -> subprocess.Popen[bytes]:
    env = os.environ.copy()
    return subprocess.Popen(
        [
            str(PYTHON_BIN),
            "-m",
            "uvicorn",
            "fastapi_app.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
            "--reload",
        ],
        cwd=ROOT_DIR,
        env=env,
    )


def _start_next_dev() -> subprocess.Popen[bytes]:
    env = os.environ.copy()
    env.setdefault("FASTAPI_INTERNAL_URL", "http://127.0.0.1:8000")
    env.setdefault("APP_BASE_URL", "http://localhost:3000")
    return subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=WEB_DIR,
        env=env,
    )


def _start_queue_worker() -> subprocess.Popen[bytes]:
    """`infra/server_scheduler.py` 를 APP_TYPE=Local 로 띄워 워커만 동작시킨다.

    cron 트리거는 서버 scheduler 컨테이너 (APP_TYPE=Server) 만 활성.
    여기서는 worker thread 만 동작하여 큐의 pending 작업을 atomic 하게 claim 한다.
    """
    env = os.environ.copy()
    env["APP_TYPE"] = "Local"  # 명시적으로 Local 지정 — cron 비활성
    return subprocess.Popen(
        [str(PYTHON_BIN), "-u", "infra/server_scheduler.py"],
        cwd=ROOT_DIR,
        env=env,
    )


def _terminate_process(process: subprocess.Popen[bytes] | None, name: str) -> None:
    if process is None or process.poll() is not None:
        return

    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        print(f"{name} 종료가 지연되어 강제 종료합니다.")
        process.kill()
        process.wait(timeout=5)


def main() -> int:
    _ensure_python_exists()

    fastapi_process: subprocess.Popen[bytes] | None = None
    next_process: subprocess.Popen[bytes] | None = None
    worker_process: subprocess.Popen[bytes] | None = None
    shutting_down = False

    def _shutdown(signum: int, _frame) -> None:
        nonlocal shutting_down
        if shutting_down:
            return
        shutting_down = True
        print(f"\n신호 {signum} 수신. 로컬 개발 서버를 종료합니다.")
        _terminate_process(worker_process, "큐 워커")
        _terminate_process(next_process, "Next dev 서버")
        _terminate_process(fastapi_process, "FastAPI 서버")

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print("FastAPI 서버를 시작합니다: http://127.0.0.1:8000")
    fastapi_process = _start_fastapi()
    time.sleep(1.0)
    if fastapi_process.poll() is not None:
        return fastapi_process.returncode or 1

    print("Next dev 서버를 시작합니다: http://localhost:3000")
    next_process = _start_next_dev()

    # 큐 워커도 함께 띄워 사용자는 명령 하나로 fastapi/next/worker 모두 운영.
    # cron 트리거는 서버 scheduler 컨테이너만 담당하므로 로컬에서는 worker 만 동작.
    print("큐 워커를 시작합니다 (APP_TYPE=Local — cron 비활성, worker 만 동작)")
    worker_process = _start_queue_worker()

    try:
        while True:
            if fastapi_process.poll() is not None:
                print("FastAPI 서버가 먼저 종료되었습니다. 함께 종료합니다.")
                _terminate_process(worker_process, "큐 워커")
                _terminate_process(next_process, "Next dev 서버")
                return fastapi_process.returncode or 1

            if next_process.poll() is not None:
                print("Next dev 서버가 먼저 종료되었습니다. 함께 종료합니다.")
                _terminate_process(worker_process, "큐 워커")
                _terminate_process(fastapi_process, "FastAPI 서버")
                return next_process.returncode or 1

            if worker_process.poll() is not None:
                # 워커는 일시적으로 죽어도 dev 환경 자체를 종료시킬 정도는 아님.
                # 종료 코드를 로깅만 하고 fastapi/next 는 계속 동작하게 둔다.
                print(
                    f"⚠️ 큐 워커가 종료되었습니다 (exit={worker_process.returncode}). "
                    "fastapi/next 는 계속 동작합니다. 워커 재실행이 필요하면 dev 서버를 재시작하세요."
                )
                worker_process = None  # 더 이상 polling 안 함

            time.sleep(0.5)
    finally:
        _terminate_process(worker_process, "큐 워커")
        _terminate_process(next_process, "Next dev 서버")
        _terminate_process(fastapi_process, "FastAPI 서버")


if __name__ == "__main__":
    sys.exit(main())
