from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

from utils.account_registry import load_account_configs
from utils.env import load_env_if_present

load_env_if_present()

SystemAction = Literal[
    "data_aggregate",
    "cache_refresh",
    "market_hours_analysis",
    "metadata_updater",
    "asset_summary",
    "us_market_stocks",
]

# 배치 정의: 키는 infra/cron/crontab 의 job name 과 동일해야 합니다.
SCHEDULE_ROWS = [
    {
        "key": "asset_summary",
        "job": "전체 자산 요약 알림",
        "target": "전체 계좌",
        "cadence": "평일 09:30, 16:30 KST",
        "command": "python scripts/slack_asset_summary.py",
    },
    {
        "key": "data_aggregate",
        "job": "데이터 집계",
        "target": "일별/주별/월별 데이터",
        "cadence": "평일 09:32, 16:32 KST",
        "command": "python scripts/collect_data.py",
    },
    {
        "key": "cache_refresh",
        "job": "가격 캐시 업데이트",
        "target": "모든 종목/포트폴리오 변동",
        "cadence": "월~토 05:20 ~ 17:20 매시 20분 KST",
        "command": "python scripts/stock_price_cache_updater.py",
    },
    {
        "key": "market_hours_analysis",
        "job": "장 시간 분석",
        "target": "시장 스케줄",
        "cadence": "평일 07:00 KST",
        "command": "python scripts/analyze_market_hours.py",
    },
    {
        "key": "metadata_updater",
        "job": "종목 메타데이터 업데이트",
        "target": "모든 종목타입",
        "cadence": "평일 09:30 ~ 17:30 매시 30분 KST",
        "command": "python scripts/stock_meta_cache_updater.py",
    },
    {
        "key": "us_market_stocks",
        "job": "미국 개별주 업데이트",
        "target": "S&P500, NASDAQ100",
        "cadence": "평일 08:00 KST",
        "command": "python scripts/update_us_market_stocks.py",
    },
]

# action 키 → 실행할 스크립트 경로
_SCRIPT_BY_ACTION: dict[str, str] = {
    "data_aggregate": "scripts/collect_data.py",
    "cache_refresh": "scripts/stock_price_cache_updater.py",
    "market_hours_analysis": "scripts/analyze_market_hours.py",
    "metadata_updater": "scripts/stock_meta_cache_updater.py",
    "asset_summary": "scripts/slack_asset_summary.py",
    "us_market_stocks": "scripts/update_us_market_stocks.py",
}

_LABEL_BY_ACTION: dict[str, str] = {
    row["key"]: row["job"] for row in SCHEDULE_ROWS
}

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LOCK_DIR = _PROJECT_ROOT / "logs" / "cron"
_LOG_DIR = _PROJECT_ROOT / "logs" / "cron"
# 락파일이 이 시간(초)보다 오래되면 stale 로 간주하고 삭제한다.
_STALE_LOCK_SECONDS = 60 * 60  # 1시간
_RUN_BATCH_START_PATTERN = re.compile(
    r"^\[run_batch\] START job=(?P<job>[a-z_]+) cmd=.* at=(?P<started_at>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} KST)$"
)
_RUN_BATCH_END_PATTERN = re.compile(
    r"^\[run_batch\] END job=(?P<job>[a-z_]+) status=(?P<status>성공|실패) exit=(?P<exit>\d+) elapsed=(?P<elapsed>[0-9.]+)s(?: at=(?P<ended_at>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} KST))?$"
)

_logger = logging.getLogger(__name__)


def _parse_kst_datetime(value: str) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return datetime.strptime(raw, "%Y-%m-%d %H:%M:%S KST").replace(tzinfo=ZoneInfo("Asia/Seoul"))
    except ValueError:
        return None


def _format_elapsed_since(when: datetime | None) -> str:
    if when is None:
        return "-"
    now = datetime.now(ZoneInfo("Asia/Seoul"))
    delta_seconds = max(0, int((now - when).total_seconds()))
    if delta_seconds < 60:
        elapsed = "방금전"
    elif delta_seconds < 3600:
        elapsed = f"{delta_seconds // 60}분전"
    elif delta_seconds < 86400:
        elapsed = f"{delta_seconds // 3600}시간전"
    else:
        elapsed = f"{delta_seconds // 86400}일전"
    weekday = ["월", "화", "수", "목", "금", "토", "일"][when.weekday()]
    return f"{when.strftime('%Y-%m-%d')}({weekday}) {when.strftime('%H:%M')}({elapsed})"


def _read_last_job_run(job_key: str) -> dict[str, str | None]:
    log_path = _LOG_DIR / f"{job_key}.log"
    if not log_path.exists():
        return {"status": None, "display": "-"}

    try:
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return {"status": None, "display": "-"}

    pending_started_at: str | None = None
    last_started_at: str | None = None
    last_status: str | None = None

    for line in lines:
        start_match = _RUN_BATCH_START_PATTERN.match(line.strip())
        if start_match and start_match.group("job") == job_key:
            pending_started_at = start_match.group("started_at")
            continue

        end_match = _RUN_BATCH_END_PATTERN.match(line.strip())
        if end_match and end_match.group("job") == job_key:
            last_status = "success" if end_match.group("status") == "성공" else "failure"
            last_started_at = pending_started_at
            pending_started_at = None

    if not last_status or not last_started_at:
        return {"status": None, "display": "-"}

    started_at = _parse_kst_datetime(last_started_at)
    icon = "✅" if last_status == "success" else "❌"
    return {
        "status": last_status,
        "display": f"{icon} {_format_elapsed_since(started_at)}",
    }


def get_running_jobs() -> list[str]:
    """현재 실행 중인 배치 키 목록을 반환합니다.

    run_batch.py 가 생성한 `logs/cron/<key>.lock` 파일을 스캔합니다.
    1시간 이상 된 락파일은 비정상 종료로 간주하고 삭제합니다.
    """
    if not _LOCK_DIR.exists():
        return []
    now = time.time()
    running: list[str] = []
    for lock_file in _LOCK_DIR.glob("*.lock"):
        key = lock_file.stem
        if key not in _SCRIPT_BY_ACTION:
            continue
        try:
            mtime = lock_file.stat().st_mtime
        except FileNotFoundError:
            continue
        age = now - mtime
        if age > _STALE_LOCK_SECONDS:
            try:
                lock_file.unlink(missing_ok=True)
                _logger.warning(
                    "stale 락파일 제거: %s (age=%.0fs)", lock_file.name, age
                )
            except Exception as exc:  # pragma: no cover
                _logger.warning("stale 락파일 제거 실패: %s (%s)", lock_file.name, exc)
            continue
        running.append(key)
    return sorted(running)


def load_system_data() -> dict[str, object]:
    accounts = load_account_configs()
    return {
        "summary_rows": [
            {
                "category": f"{account.get('icon', '')} {account['name']}".strip(),
                "count": int(account["order"]),
                "target": account["account_id"],
            }
            for account in accounts
        ],
        "schedule_rows": SCHEDULE_ROWS,
        "schedule_note": (
            "VM 호스트 cron 이 `infra/cron/run_batch.py` 래퍼를 통해 실행하며 "
            "실패 결과만 슬랙으로 알립니다. 개별 스크립트가 보내는 본문 알림은 "
            "별도로 유지됩니다. (정의: `infra/cron/crontab`)"
        ),
        "running_jobs": get_running_jobs(),
        "last_run_by_job": {
            row["key"]: _read_last_job_run(str(row["key"]))
            for row in SCHEDULE_ROWS
        },
    }


class BatchAlreadyRunningError(RuntimeError):
    """다른 배치가 실행 중일 때 트리거 요청을 거부하기 위한 예외."""


def trigger_system_action(action: SystemAction) -> str:
    """배치를 백그라운드로 실행. cron 과 동일하게 run_batch.py 래퍼를 경유해
    실패 결과를 슬랙으로 알립니다. 다른 배치가 실행 중이면 거부합니다."""

    if action not in _SCRIPT_BY_ACTION:
        raise ValueError("지원하지 않는 시스템 작업입니다.")

    running = get_running_jobs()
    if running:
        running_labels = ", ".join(_LABEL_BY_ACTION.get(k, k) for k in running)
        raise BatchAlreadyRunningError(
            f"다른 배치가 실행 중입니다: {running_labels}. 완료 후 다시 시도해주세요."
        )

    project_root = _PROJECT_ROOT
    script_rel = _SCRIPT_BY_ACTION[action]
    wrapper_rel = "infra/cron/run_batch.py"

    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")

    subprocess.Popen(
        [sys.executable, wrapper_rel, action, sys.executable, script_rel],
        cwd=str(project_root),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    label = _LABEL_BY_ACTION.get(action, action)
    return f"[시스템-배치] {label} 백그라운드 실행을 시작했습니다."
