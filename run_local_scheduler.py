#!/usr/bin/env python3
"""로컬(Mac) 배치 스케줄러.

`run_local_dev.py` 와는 별도 프로세스로 실행하여, VM 의 cron 을 대체한다.
배치 정의는 `infra/cron/crontab` 파일을 단일 진실 소스(single source of truth)
로 읽어들이고, APScheduler 로 등록한다.

사용법:
    터미널1:  python run_local_dev.py        # 웹 (FastAPI + Next dev)
    터미널2:  python run_local_scheduler.py  # 배치

특징:
  • crontab 파일을 그대로 파싱 — 주석(`#`, `# PAUSED`)은 모두 활성으로 간주.
    (VM 에서는 cron 비활성 표시였으나, 로컬에서는 이 파일 자체가 활성 정의로 동작)
  • APP_TYPE 환경변수가 "Local" 로 설정되어 batch_locks 의 owner 식별이 가능.
  • 노트북이 꺼져 있는 시각의 미실행 분은 무시(misfire_grace_time=0).
  • 시작 시 즉시 실행은 하지 않음. 다음 예약 시각부터 동작.
  • Ctrl+C 로 깔끔히 종료.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import signal
import subprocess
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

ROOT_DIR = Path(__file__).resolve().parent
PYTHON_BIN = ROOT_DIR / ".venv" / "bin" / "python"
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


def _run_job(job_name: str, script_path: str) -> None:
    """run_batch.py 래퍼를 통해 배치 1건 실행."""
    log.info("배치 시작: %s (%s)", job_name, script_path)
    env = os.environ.copy()
    env.setdefault("APP_TYPE", "Local")
    env.setdefault("PYTHONUNBUFFERED", "1")
    try:
        result = subprocess.run(
            [str(PYTHON_BIN), str(RUN_BATCH), job_name, str(PYTHON_BIN), script_path],
            cwd=str(ROOT_DIR),
            env=env,
            check=False,
        )
        log.info("배치 종료: %s (exit=%d)", job_name, result.returncode)
    except Exception as exc:  # pragma: no cover
        log.exception("배치 실행 실패: %s — %s", job_name, exc)


def _build_trigger(cron_expr: str) -> CronTrigger:
    """5필드 cron 식 → APScheduler CronTrigger."""
    minute, hour, dom, month, dow = cron_expr.split()
    return CronTrigger(
        minute=minute,
        hour=hour,
        day=dom,
        month=month,
        day_of_week=dow,
        timezone=KST,
    )


def main() -> int:
    if not PYTHON_BIN.exists():
        log.error("프로젝트 가상환경 Python 을 찾을 수 없습니다: %s", PYTHON_BIN)
        return 1
    if not RUN_BATCH.exists():
        log.error("run_batch.py 를 찾을 수 없습니다: %s", RUN_BATCH)
        return 1

    os.environ.setdefault("APP_TYPE", "Local")

    jobs = _parse_crontab(CRONTAB_FILE)
    if not jobs:
        log.warning("등록할 배치가 없습니다. crontab 파일을 확인하세요: %s", CRONTAB_FILE)
        return 1

    sched = BlockingScheduler(timezone=KST)
    for cron_expr, job_name, script_path in jobs:
        trigger = _build_trigger(cron_expr)
        sched.add_job(
            _run_job,
            trigger=trigger,
            args=(job_name, script_path),
            id=job_name,
            name=job_name,
            misfire_grace_time=None,  # 노트북 꺼져있던 시간은 따라잡지 않음
            coalesce=True,
            max_instances=1,
            replace_existing=True,
        )
        log.info("등록: %-25s  cron=\"%s\"  script=%s", job_name, cron_expr, script_path)

    def _shutdown(signum: int, _frame) -> None:
        log.info("신호 %s 수신 — 스케줄러 종료", signum)
        sched.shutdown(wait=False)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    log.info("로컬 스케줄러 시작 (등록 %d건, APP_TYPE=Local)", len(jobs))
    log.info("종료: Ctrl+C")
    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
