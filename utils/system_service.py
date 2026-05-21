from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
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

# 평일(월~금) / 월~토 weekday 셋. (Python: 0=월 ... 6=일)
_WEEKDAYS_MON_FRI = [0, 1, 2, 3, 4]
_WEEKDAYS_MON_SAT = [0, 1, 2, 3, 4, 5]

# 배치 정의: 키는 infra/cron/crontab 의 job name 과 동일해야 합니다.
# schedule 필드는 infra/cron/crontab 과 동기화해야 합니다.
# 운영 방식: VM cron 은 제거됐고, 로컬(Mac) 의 run_local_scheduler.py 가
# infra/cron/crontab 을 읽어 APScheduler 로 실행합니다.
SCHEDULE_ROWS = [
    {
        "key": "asset_summary",
        "job": "전체 자산 요약 알림",
        "target": "전체 계좌",
        "cadence": "평일 09:40, 16:40 KST",
        "command": "python scripts/slack_asset_summary.py",
        "schedule": {"minutes": [40], "hours": [9, 16], "weekdays": _WEEKDAYS_MON_FRI},
    },
    {
        "key": "data_aggregate",
        "job": "데이터 집계",
        "target": "일별/주별/월별/년별 데이터",
        "cadence": "평일 09:15 ~ 16:15 매시 :15 KST",
        "command": "python scripts/collect_data.py",
        "schedule": {
            "minutes": [15],
            "hours": list(range(9, 17)),
            "weekdays": _WEEKDAYS_MON_FRI,
        },
    },
    {
        "key": "cache_refresh",
        "job": "가격 캐시 업데이트",
        "target": "모든 종목 가격",
        "cadence": "월~토 24시간 매시 0분 KST",
        "command": "python scripts/stock_price_cache_updater.py",
        "schedule": {"minutes": [0], "hours": list(range(24)), "weekdays": _WEEKDAYS_MON_SAT},
    },
    {
        "key": "market_hours_analysis",
        "job": "장 시간 분석",
        "target": "시장 스케줄",
        "cadence": "평일 07:00 KST",
        "command": "python scripts/analyze_market_hours.py",
        "schedule": {"minutes": [0], "hours": [7], "weekdays": _WEEKDAYS_MON_FRI},
    },
    {
        "key": "metadata_updater",
        "job": "종목 메타데이터 업데이트",
        "target": "모든 종목타입",
        "cadence": "평일 09:45 ~ 17:45 매시 45분 KST",
        "command": "python scripts/stock_meta_cache_updater.py",
        "schedule": {"minutes": [45], "hours": list(range(9, 18)), "weekdays": _WEEKDAYS_MON_FRI},
    },
    {
        "key": "us_market_stocks",
        "job": "미국 개별주 업데이트",
        "target": "S&P500, NASDAQ100",
        "cadence": "평일 08:00 KST",
        "command": "python scripts/update_us_market_stocks.py",
        "schedule": {"minutes": [0], "hours": [8], "weekdays": _WEEKDAYS_MON_FRI},
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

_LABEL_BY_ACTION: dict[str, str] = {row["key"]: row["job"] for row in SCHEDULE_ROWS}

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_LOCK_DIR = _PROJECT_ROOT / "logs" / "cron"
_LOG_DIR = _PROJECT_ROOT / "logs" / "cron"
DEPLOY_LOCK_DOC_ID = "__deploy__"


def is_deploying() -> bool:
    """MongoDB batch_locks 에서 배포 진행 플래그 조회."""
    try:
        from utils.db_manager import get_db_connection

        db = get_db_connection()
        if db is None:
            return False
        return db.batch_locks.find_one({"_id": DEPLOY_LOCK_DOC_ID}) is not None
    except Exception:
        return False


# 락파일이 이 시간(초)보다 오래되면 stale 로 간주하고 삭제한다.
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


def _compute_next_run(schedule: dict[str, list[int]] | None) -> datetime | None:
    """주어진 스케줄(minutes/hours/weekdays)에 맞는 가장 가까운 다음 실행 시각(KST)을 반환한다."""
    if not schedule:
        return None
    minutes = sorted(set(int(m) for m in schedule.get("minutes", [])))
    hours = sorted(set(int(h) for h in schedule.get("hours", [])))
    weekdays = sorted(set(int(w) for w in schedule.get("weekdays", [])))
    if not minutes or not hours or not weekdays:
        return None

    tz = ZoneInfo("Asia/Seoul")
    now = datetime.now(tz)
    # 다음 분 단위부터 검사 (현재 분은 이미 지났다고 보수적으로 가정)
    candidate = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)
    # 최대 8일까지 탐색 (어떤 weekday/hour/minute 조합이라도 이 범위 내에 반드시 매치)
    end = candidate + timedelta(days=8)
    while candidate < end:
        if candidate.weekday() in weekdays:
            if candidate.hour in hours:
                if candidate.minute in minutes:
                    return candidate
                # 같은 시간 내에서 다음 후보 분 찾기
                next_minute = next((m for m in minutes if m > candidate.minute), None)
                if next_minute is not None:
                    candidate = candidate.replace(minute=next_minute)
                    continue
                # 시간 내 후보 분이 없으면 다음 시간으로
                candidate = (candidate + timedelta(hours=1)).replace(minute=0)
                continue
            # 같은 날 내에서 다음 후보 시간 찾기
            next_hour = next((h for h in hours if h > candidate.hour), None)
            if next_hour is not None:
                candidate = candidate.replace(hour=next_hour, minute=minutes[0])
                continue
            # 시간 내 후보가 없으면 다음 날로
            candidate = (candidate + timedelta(days=1)).replace(hour=hours[0], minute=minutes[0])
            continue
        # weekday 불일치 → 다음 날 첫 후보로
        candidate = (candidate + timedelta(days=1)).replace(hour=hours[0], minute=minutes[0])
    return None


def _format_until(when: datetime | None) -> str:
    if when is None:
        return "-"
    now = datetime.now(ZoneInfo("Asia/Seoul"))
    delta_seconds = int((when - now).total_seconds())
    if delta_seconds <= 0:
        return "곧"
    if delta_seconds < 60:
        return f"{delta_seconds}초 후"
    if delta_seconds < 3600:
        return f"{delta_seconds // 60}분 후"
    if delta_seconds < 86400:
        hours = delta_seconds // 3600
        minutes = (delta_seconds % 3600) // 60
        return f"{hours}시간 {minutes}분 후" if minutes else f"{hours}시간 후"
    days = delta_seconds // 86400
    hours = (delta_seconds % 86400) // 3600
    return f"{days}일 {hours}시간 후" if hours else f"{days}일 후"


def _format_elapsed_since(when: datetime | None) -> str:
    if when is None:
        return "-"
    now = datetime.now(ZoneInfo("Asia/Seoul"))
    delta_seconds = max(0, int((now - when).total_seconds()))
    if delta_seconds < 60:
        return "방금 전"
    if delta_seconds < 3600:
        return f"{delta_seconds // 60}분 전"
    if delta_seconds < 86400:
        hours = delta_seconds // 3600
        minutes = (delta_seconds % 3600) // 60
        return f"{hours}시간 {minutes}분 전" if minutes else f"{hours}시간 전"
    days = delta_seconds // 86400
    hours = (delta_seconds % 86400) // 3600
    return f"{days}일 {hours}시간 전" if hours else f"{days}일 전"


def _format_duration_seconds(seconds: float | int | None) -> str | None:
    if seconds is None:
        return None
    total_seconds = max(0, int(round(float(seconds))))
    if total_seconds < 60:
        return f"{total_seconds}초"
    minutes = total_seconds // 60
    remain_seconds = total_seconds % 60
    if minutes < 60:
        return f"{minutes}분 {remain_seconds}초" if remain_seconds else f"{minutes}분"
    hours = minutes // 60
    remain_minutes = minutes % 60
    return f"{hours}시간 {remain_minutes}분" if remain_minutes else f"{hours}시간"


def _read_average_job_elapsed_seconds(job_key: str, sample_size: int = 5) -> float | None:
    """최근 성공 실행의 평균 소요시간(초)을 반환한다. 성공 로그가 없으면 최근 종료 로그를 사용한다."""
    log_path = _LOG_DIR / f"{job_key}.log"
    if not log_path.exists():
        return None
    try:
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return None

    successful: list[float] = []
    completed: list[float] = []
    for line in reversed(lines):
        end_match = _RUN_BATCH_END_PATTERN.match(line.strip())
        if not end_match or end_match.group("job") != job_key:
            continue
        try:
            elapsed = float(end_match.group("elapsed"))
        except (TypeError, ValueError):
            continue
        completed.append(elapsed)
        if end_match.group("status") == "성공":
            successful.append(elapsed)
            if len(successful) >= sample_size:
                break

    samples = successful[:sample_size] or completed[:sample_size]
    if not samples:
        return None
    return sum(samples) / len(samples)


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


def _cleanup_stale_locks() -> int:
    """예상 소요시간의 2배를 초과한 락은 stale 로 간주하고 삭제한다.

    - 평균(예상) 소요시간을 모르는 락은 건드리지 않는다.
    - elapsed > estimated * 2 인 락만 삭제한다.

    Returns:
        삭제된 락 개수
    """
    try:
        from utils.db_manager import get_db_connection

        db = get_db_connection()
        if db is None:
            return 0
        now_kst = datetime.now(ZoneInfo("Asia/Seoul"))
        deleted = 0
        for doc in db.batch_locks.find({}, {"_id": 1, "acquired_at": 1}):
            key = str(doc.get("_id") or "")
            if key not in _SCRIPT_BY_ACTION:
                continue
            acquired_at = doc.get("acquired_at")
            if not isinstance(acquired_at, datetime):
                continue
            started_at = (
                acquired_at.astimezone(ZoneInfo("Asia/Seoul"))
                if acquired_at.tzinfo
                else acquired_at.replace(tzinfo=timezone.utc).astimezone(ZoneInfo("Asia/Seoul"))
            )
            elapsed_seconds = int((now_kst - started_at).total_seconds())
            if elapsed_seconds <= 0:
                continue
            estimated_seconds = _read_average_job_elapsed_seconds(key)
            if estimated_seconds is None or estimated_seconds <= 0:
                continue
            if elapsed_seconds > int(estimated_seconds * 2):
                try:
                    db.batch_locks.delete_one({"_id": key})
                    deleted += 1
                    _logger.warning(
                        "Stale lock 제거: job=%s elapsed=%ds estimated=%ds",
                        key, elapsed_seconds, int(estimated_seconds),
                    )
                except Exception as exc:
                    _logger.warning("Stale lock 삭제 실패 (job=%s): %s", key, exc)
        return deleted
    except Exception as exc:
        _logger.warning("Stale lock 정리 실패: %s", exc)
        return 0


def get_running_jobs() -> list[str]:
    """현재 실행 중인 배치 키 목록을 반환합니다.

    MongoDB `batch_locks` 컬렉션을 조회합니다. TTL 인덱스가 만료된 락을
    자동 정리하며, 예상 소요시간 2배를 초과한 락도 stale 로 보고 삭제합니다.
    """
    # 조회 전에 stale 락 정리 (조회/상세 모두에서 효과)
    _cleanup_stale_locks()
    try:
        from utils.db_manager import get_db_connection

        db = get_db_connection()
        if db is None:
            return []
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        running: list[str] = []
        for doc in db.batch_locks.find({}, {"_id": 1, "expires_at": 1}):
            key = str(doc.get("_id") or "")
            if key not in _SCRIPT_BY_ACTION:
                continue
            expires_at = doc.get("expires_at")
            # MongoDB는 datetime을 UTC 기준으로 저장하고 naive datetime으로 반환할 수 있다.
            # 따라서 현재 시각도 UTC naive로 맞춰 비교한다.
            try:
                if isinstance(expires_at, datetime):
                    comparable_expires_at = (
                        expires_at.astimezone(timezone.utc).replace(tzinfo=None) if expires_at.tzinfo else expires_at
                    )
                    if comparable_expires_at < now:
                        continue
            except Exception:
                pass
            running.append(key)
        return sorted(running)
    except Exception as exc:
        _logger.warning("batch_locks 조회 실패: %s", exc)
        return []


def get_running_job_details() -> dict[str, dict[str, object]]:
    """실행 중인 배치별 시작 시각, 예상 소요시간, 락 소유 인스턴스 정보를 반환한다."""
    try:
        from utils.db_manager import get_db_connection

        db = get_db_connection()
        if db is None:
            return {}
        now_utc_naive = datetime.now(timezone.utc).replace(tzinfo=None)
        now_kst = datetime.now(ZoneInfo("Asia/Seoul"))
        my_app_type = (os.environ.get("APP_TYPE") or "PROD").strip() or "PROD"
        details: dict[str, dict[str, object]] = {}
        for doc in db.batch_locks.find(
            {}, {"_id": 1, "expires_at": 1, "acquired_at": 1, "app_type": 1}
        ):
            key = str(doc.get("_id") or "")
            if key not in _SCRIPT_BY_ACTION:
                continue
            expires_at = doc.get("expires_at")
            if isinstance(expires_at, datetime):
                comparable_expires_at = (
                    expires_at.astimezone(timezone.utc).replace(tzinfo=None) if expires_at.tzinfo else expires_at
                )
                if comparable_expires_at < now_utc_naive:
                    continue

            acquired_at = doc.get("acquired_at")
            started_at: datetime | None = None
            if isinstance(acquired_at, datetime):
                started_at = acquired_at.astimezone(ZoneInfo("Asia/Seoul")) if acquired_at.tzinfo else acquired_at.replace(tzinfo=timezone.utc).astimezone(ZoneInfo("Asia/Seoul"))

            estimated_seconds = _read_average_job_elapsed_seconds(key)
            elapsed_seconds = max(0, int((now_kst - started_at).total_seconds())) if started_at else None
            remaining_seconds = (
                max(0, int(round(estimated_seconds)) - int(elapsed_seconds))
                if estimated_seconds is not None and elapsed_seconds is not None
                else None
            )
            owner_app_type = (str(doc.get("app_type") or "") or "PROD").strip()
            is_mine = owner_app_type.lower() == my_app_type.lower()
            details[key] = {
                "started_at": started_at.isoformat() if started_at else None,
                "estimated_seconds": int(round(estimated_seconds)) if estimated_seconds is not None else None,
                "elapsed_seconds": elapsed_seconds,
                "remaining_seconds": remaining_seconds,
                "estimated_display": _format_duration_seconds(estimated_seconds),
                "remaining_display": _format_duration_seconds(remaining_seconds),
                "owner_app_type": owner_app_type,
                "is_mine": is_mine,
            }
        return details
    except Exception as exc:
        _logger.warning("batch_locks 상세 조회 실패: %s", exc)
        return {}


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
            "자동 배치는 로컬(Mac) 의 `run_local_scheduler.py` 에서 실행됩니다. "
            "VM cron 은 사용하지 않으며, `infra/cron/crontab` 파일이 스케줄 정의의 단일 진실 소스입니다."
        ),
        "running_jobs": get_running_jobs(),
        "running_job_details": get_running_job_details(),
        "is_deploying": is_deploying(),
        "last_run_by_job": {row["key"]: _read_last_job_run(str(row["key"])) for row in SCHEDULE_ROWS},
        "next_run_by_job": {row["key"]: _build_next_run_payload(row.get("schedule")) for row in SCHEDULE_ROWS},
    }


def _build_next_run_payload(schedule: dict | None) -> dict[str, str | None]:
    next_run = _compute_next_run(schedule)
    if next_run is None:
        return {"at": None, "display": "-"}
    return {
        "at": next_run.isoformat(),
        "display": _format_until(next_run),
    }


class BatchAlreadyRunningError(RuntimeError):
    """다른 배치가 실행 중일 때 트리거 요청을 거부하기 위한 예외."""


class DeployInProgressError(RuntimeError):
    """배포 진행 중에는 수동 배치를 거부한다."""


def trigger_system_action(action: SystemAction) -> str:
    """배치를 백그라운드로 실행. cron 과 동일하게 run_batch.py 래퍼를 경유해
    실패 결과를 슬랙으로 알립니다. 다른 배치가 실행 중이면 거부합니다."""

    if action not in _SCRIPT_BY_ACTION:
        raise ValueError("지원하지 않는 시스템 작업입니다.")

    running = get_running_jobs()
    if running:
        running_labels = ", ".join(_LABEL_BY_ACTION.get(k, k) for k in running)
        raise BatchAlreadyRunningError(f"다른 배치가 실행 중입니다: {running_labels}. 완료 후 다시 시도해주세요.")

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
