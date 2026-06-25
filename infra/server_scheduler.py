#!/usr/bin/env python3
"""서버 docker scheduler 컨테이너의 entrypoint (cron + 큐 워커).

이 파일은 **서버 전용**이다. 사용자가 직접 실행할 일은 없다.
배포 흐름에서 `docker-compose.yml` 의 `scheduler` 서비스가
`command: ["python", "-u", "infra/server_scheduler.py"]` 로 띄운다.

역할:
  1) cron 스케줄러 (enqueue) — `infra/cron/crontab` 을 APScheduler 에 등록해
     정해진 시각마다 batch_queue 에 작업을 추가.
  2) 큐 워커 (claim) — batch_queue 의 pending 을 FIFO 로 한 건씩
     atomic 하게 claim 해서 실제 배치 실행.

환경변수:
  • APP_TYPE: 서버 컨테이너는 `Server` (compose 가 자동 설정).
    Server 일 때 cron 활성. 그 외(Local 등)면 cron 비활성, worker 만.

로컬 환경:
  사용자는 `python run_local_dev.py` 한 명령으로 fastapi + next + worker
  를 한 번에 띄운다. cron 은 서버에서만 동작.

큐 동시성:
  서버 worker + 로컬 worker 가 동시에 큐를 polling 해도 MongoDB
  `find_one_and_update` 가 한 곳에서만 작업을 claim 하도록 보장.

특징:
  • crontab 파일을 그대로 파싱 — 주석(`#`, `# PAUSED`)은 모두 활성으로 간주.
  • 노트북이 꺼져 있는 시각의 미실행 분은 무시(misfire_grace_time=0).
  • Ctrl+C / SIGTERM 으로 깔끔히 종료.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

# 이 파일이 `infra/` 아래에 있으므로 두 단계 위가 프로젝트 루트.
ROOT_DIR = Path(__file__).resolve().parents[1]
# subprocess 로 배치를 띄울 때 사용할 python 인터프리터.
# 로컬(.venv) 와 컨테이너(/opt/venv) 모두 sys.executable 이 정답이라 그것을 기본값으로.
# 명시적으로 다른 인터프리터를 강제하고 싶을 때만 SCHEDULER_PYTHON_BIN 으로 override.
PYTHON_BIN = Path(os.environ.get("SCHEDULER_PYTHON_BIN") or sys.executable)
CRONTAB_FILE = ROOT_DIR / "infra" / "cron" / "crontab"
RUN_BATCH = ROOT_DIR / "infra" / "cron" / "run_batch.py"

KST = ZoneInfo("Asia/Seoul")

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("local_scheduler")

# crontab 의 docker compose exec 호출은 로컬에서는 무의미하므로 제거하고
# 컨테이너 안 경로로 적힌 명령을 호스트 venv python 으로 직접 실행한다.
_DOCKER_EXEC_RE = re.compile(
    r"docker\s+compose\s+exec\s+-T\s+fastapi_app\s+python\s+",
)
# crontab 라인의 변수($APP_DIR 등)와 리다이렉션을 떼어내고
# `python infra/cron/run_batch.py <job> python <script.py>` 형태만 추출하기 위한 마커
_RUN_BATCH_MARKER = "infra/cron/run_batch.py"


def _parse_crontab(path: Path) -> list[tuple[str, str, str]]:
    """crontab 파일에서 배치 라인을 추출한다.

    반환: (cron_expr, job_name, script_path) 튜플 리스트.
        cron_expr: "min hour dom month dow" (5필드)
        job_name:  run_batch.py 의 첫 인자
        script_path: 실행할 스크립트 (예: scripts/stock_price_cache_updater.py)
    """
    if not path.exists():
        raise FileNotFoundError(f"crontab 파일 없음: {path}")

    jobs: list[tuple[str, str, str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        # 주석 형태 모두 활성으로 취급: "# PAUSED <cron...>" 또는 "# <cron...>"
        if line.startswith("#"):
            stripped = line.lstrip("#").strip()
            if stripped.upper().startswith("PAUSED"):
                stripped = stripped[len("PAUSED") :].strip()
            line = stripped
        if not line or line.startswith("#"):
            continue
        # 환경 변수 라인은 스킵 (SHELL=, PATH=, APP_DIR= 등)
        if re.match(r"^[A-Z_][A-Z0-9_]*\s*=", line):
            continue
        # cron 라인인지 — 첫 5 토큰이 시간 표현식인지 대충 검증
        tokens = line.split(None, 5)
        if len(tokens) < 6:
            continue
        cron_expr = " ".join(tokens[:5])
        rest = tokens[5]
        # run_batch.py 호출이 포함된 라인만 처리
        if _RUN_BATCH_MARKER not in rest:
            continue
        # docker compose exec 부분 제거
        rest = _DOCKER_EXEC_RE.sub("", rest)
        # 첫 ">" 이전까지가 명령
        cmd_part = rest.split(">>", 1)[0].split(">", 1)[0].strip()
        # `cd $APP_DIR && mkdir -p logs/cron && python infra/cron/run_batch.py <job> python <script>`
        # 의 형식이라 `run_batch.py` 이후 토큰들을 뽑는다.
        try:
            argv = shlex.split(cmd_part, posix=True)
        except ValueError:
            log.warning("crontab 라인 파싱 실패: %s", line)
            continue
        try:
            idx = next(i for i, t in enumerate(argv) if t.endswith("run_batch.py"))
        except StopIteration:
            continue
        # argv[idx+1] = job_name, argv[idx+2] = python, argv[idx+3] = script_path
        if idx + 3 >= len(argv):
            log.warning("run_batch 인자 부족: %s", line)
            continue
        job_name = argv[idx + 1]
        script_path = argv[idx + 3]
        jobs.append((cron_expr, job_name, script_path))
    return jobs


def _enqueue_from_schedule(job_name: str, script_path: str) -> None:
    """APScheduler 가 호출 — 직접 실행하지 않고 batch_queue 에 enqueue 만."""
    sys.path.insert(0, str(ROOT_DIR))
    from utils.batch_queue import enqueue

    try:
        result = enqueue(job_name, script_path, triggered_by="schedule")
        if result.get("enqueued"):
            log.info("스케줄 → 큐 추가: %s", job_name)
        else:
            log.info("스케줄 → 큐 무시 (중복): %s — %s", job_name, result.get("reason"))
    except Exception as exc:
        log.exception("스케줄 enqueue 실패: %s — %s", job_name, exc)


def _run_subprocess(job_name: str, script_path: str, item_id: Any = None) -> int:
    """run_batch.py 래퍼를 통해 배치 1건 실행하고 exit code 반환."""
    log.info("배치 시작: %s (%s)", job_name, script_path)
    env = os.environ.copy()
    env.setdefault("APP_TYPE", "Local")
    env.setdefault("PYTHONUNBUFFERED", "1")
    sys.path.insert(0, str(ROOT_DIR))
    from utils.batch_queue import is_cancel_requested

    try:
        proc = subprocess.Popen(
            [str(PYTHON_BIN), str(RUN_BATCH), job_name, str(PYTHON_BIN), script_path],
            cwd=str(ROOT_DIR),
            env=env,
        )
    except Exception as exc:
        log.exception("배치 실행 실패: %s — %s", job_name, exc)
        return 1

    # 자식 프로세스가 끝날 때까지 wait — 단 5초마다 batch_queue 의 cancel 플래그를 확인한다.
    # cancel 요청이 들어오면 SIGTERM, 5초 뒤에도 안 죽으면 SIGKILL 로 강제 종료.
    while True:
        try:
            exit_code = proc.wait(timeout=5)
            log.info("배치 종료: %s (exit=%d)", job_name, exit_code)
            return exit_code
        except subprocess.TimeoutExpired:
            if item_id is not None and is_cancel_requested(item_id):
                log.warning("배치 취소 요청 수신 → SIGTERM: %s", job_name)
                proc.terminate()
                try:
                    exit_code = proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    log.warning("SIGTERM 10초 무응답 → SIGKILL: %s", job_name)
                    proc.kill()
                    exit_code = proc.wait()
                log.info("배치 강제 종료 완료: %s (exit=%d)", job_name, exit_code)
                # 130 = bash convention for "terminated by signal" — failed 로 마킹된다
                return exit_code if exit_code != 0 else 130


def _queue_worker_loop(stop_event: threading.Event) -> None:
    """큐 컨슈머 스레드. pending 항목을 FIFO 로 직렬 실행한다."""
    sys.path.insert(0, str(ROOT_DIR))
    from utils.batch_queue import (
        claim_next_pending,
        ensure_indexes,
        mark_done,
        mark_failed,
        reap_stale_running,
        update_heartbeat,
    )

    ensure_indexes()
    # 시작 시 stale running 정리 (워커가 죽었던 경우 회복)
    reaped = reap_stale_running()
    if reaped > 0:
        log.warning("워커 시작 시점 stale running %d건 → failed 마킹", reaped)

    log.info("큐 워커 시작 (1초 polling)")
    last_stale_check = time.monotonic()
    while not stop_event.is_set():
        try:
            item = claim_next_pending()
            if item is None:
                # 주기적으로 stale running 청소 (1분마다)
                if time.monotonic() - last_stale_check > 60:
                    reap_stale_running()
                    last_stale_check = time.monotonic()
                stop_event.wait(1.0)
                continue

            item_id = item["_id"]
            job_name = item["job_name"]
            script_path = item["script_path"]
            log.info("큐 → 실행: %s (id=%s)", job_name, item_id)

            # heartbeat 갱신 스레드 — 30초마다
            hb_stop = threading.Event()

            def _heartbeat() -> None:
                while not hb_stop.is_set():
                    try:
                        update_heartbeat(item_id)
                    except Exception:
                        pass
                    hb_stop.wait(30.0)

            hb_thread = threading.Thread(target=_heartbeat, daemon=True)
            hb_thread.start()

            try:
                exit_code = _run_subprocess(job_name, script_path, item_id)
                mark_done(item_id, exit_code)
            except Exception as exc:
                log.exception("큐 항목 처리 실패: %s — %s", job_name, exc)
                mark_failed(item_id, str(exc))
            finally:
                hb_stop.set()
                hb_thread.join(timeout=2)
        except Exception as exc:
            log.exception("큐 워커 루프 예외: %s", exc)
            stop_event.wait(2.0)

    log.info("큐 워커 종료")


_CRON_DOW_TO_NAME = ("sun", "mon", "tue", "wed", "thu", "fri", "sat", "sun")


def _convert_cron_dow(field: str) -> str:
    """표준 cron 의 요일 필드(0=일,1=월,...,6=토,7=일)를 APScheduler 이름 형식으로 변환.

    APScheduler 의 day_of_week 숫자 규칙(0=월) 과 표준 cron(0=일) 이 다르기 때문에
    숫자 그대로 넘기면 주말 발송 버그가 발생한다. mon/tue/... 같은 이름은 두 시스템이
    동일하게 인식하므로 명시 변환한다.
    """
    if not field or field == "*":
        return field

    def convert_token(token: str) -> str:
        token = token.strip()
        if "-" in token:
            start, end = token.split("-", 1)
            return f"{convert_token(start)}-{convert_token(end)}"
        if "/" in token:
            base, step = token.split("/", 1)
            return f"{convert_token(base)}/{step}"
        if token.isdigit():
            idx = int(token)
            if 0 <= idx <= 7:
                return _CRON_DOW_TO_NAME[idx]
        return token

    return ",".join(convert_token(t) for t in field.split(","))


def _build_trigger(cron_expr: str) -> CronTrigger:
    """5필드 cron 식 → APScheduler CronTrigger.

    중요: APScheduler 의 day_of_week 숫자는 0=월요일이라 표준 cron(0=일)과 어긋난다.
    숫자를 mon/tue 등 이름으로 명시 변환해서 두 체계의 차이를 흡수한다.
    """
    minute, hour, dom, month, dow = cron_expr.split()
    return CronTrigger(
        minute=minute,
        hour=hour,
        day=dom,
        month=month,
        day_of_week=_convert_cron_dow(dow),
        timezone=KST,
    )


def main() -> int:
    if not PYTHON_BIN.exists():
        log.error("프로젝트 가상환경 Python 을 찾을 수 없습니다: %s", PYTHON_BIN)
        return 1
    if not RUN_BATCH.exists():
        log.error("run_batch.py 를 찾을 수 없습니다: %s", RUN_BATCH)
        return 1

    # APP_TYPE 정책:
    #   - Server: cron 트리거 활성 (enqueue) + worker 활성 (claim)
    #   - 그 외(Local 등): cron 트리거 비활성, worker 만 활성
    # 환경변수가 없으면 Local 로 간주해 cron 트리거를 띄우지 않는다.
    # 이렇게 하면 로컬에서 실행해도 cron 중복 트리거 위험이 없다.
    os.environ.setdefault("APP_TYPE", "Local")
    app_type = str(os.environ.get("APP_TYPE") or "Local").strip()
    enable_scheduler = app_type.lower() == "server"

    sched: BackgroundScheduler | None = None
    if enable_scheduler:
        jobs = _parse_crontab(CRONTAB_FILE)
        if not jobs:
            log.warning("등록할 배치가 없습니다. crontab 파일을 확인하세요: %s", CRONTAB_FILE)
            return 1
        sched = BackgroundScheduler(timezone=KST)
        # 같은 배치(job_name)가 서로 다른 시각으로 여러 크론 줄을 가질 수 있으므로
        # (예: asset_summary 09:20·15:35) APScheduler id 는 줄마다 유니크하게 둔다.
        # id 를 job_name 으로 두면 뒷줄이 앞줄을 덮어써 한 시각만 등록된다.
        for index, (cron_expr, job_name, script_path) in enumerate(jobs):
            trigger = _build_trigger(cron_expr)
            sched.add_job(
                _enqueue_from_schedule,
                trigger=trigger,
                args=(job_name, script_path),
                id=f"{job_name}#{index}",
                name=job_name,
                misfire_grace_time=None,  # 노트북 꺼져있던 시간은 따라잡지 않음
                coalesce=True,
                max_instances=1,
                replace_existing=True,
            )
            log.info("등록: %-25s  cron=\"%s\"  script=%s", job_name, cron_expr, script_path)
    else:
        log.info("cron 스케줄러 비활성 — APP_TYPE=%s (worker 만 동작)", app_type)

    stop_event = threading.Event()
    worker_thread = threading.Thread(
        target=_queue_worker_loop, args=(stop_event,), daemon=False, name="batch-queue-worker"
    )

    def _shutdown(signum: int, _frame) -> None:
        log.info("신호 %s 수신 — 스케줄러/워커 종료", signum)
        stop_event.set()
        if sched is not None:
            try:
                sched.shutdown(wait=False)
            except Exception:
                pass

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    if sched is not None:
        log.info("스케줄러 시작 (cron+worker, APP_TYPE=%s)", app_type)
        sched.start()
    else:
        log.info("worker 단독 시작 (APP_TYPE=%s) — 큐만 처리, cron 트리거 없음", app_type)
    log.info("종료: Ctrl+C")
    worker_thread.start()

    try:
        worker_thread.join()
    except (KeyboardInterrupt, SystemExit):
        stop_event.set()
    return 0


if __name__ == "__main__":
    sys.exit(main())
