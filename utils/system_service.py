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

from utils.db_manager import get_db_connection
from utils.env import load_env_if_present
from utils.ticker_registry import load_ticker_type_configs

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
# 운영 방식: VM cron 은 제거됐고, 서버 docker scheduler 컨테이너의 infra/server_scheduler.py
# 가 infra/cron/crontab 을 읽어 APScheduler 로 큐에 enqueue 합니다. worker 는
# 서버와 로컬(`python infra/server_scheduler.py`) 양쪽에서 큐를 atomic 하게 claim 합니다.
SCHEDULE_ROWS = [
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
        "key": "asset_summary",
        "job": "전체 자산 요약 알림",
        "target": "전체 계좌",
        "cadence": "평일 09:40, 16:40 KST",
        "command": "python scripts/slack_asset_summary.py",
        "schedule": {"minutes": [40], "hours": [9, 16], "weekdays": _WEEKDAYS_MON_FRI},
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
    {
        "key": "market_hours_analysis",
        "job": "장 시간 분석",
        "target": "시장 스케줄",
        "cadence": "평일 07:00 KST",
        "command": "python scripts/analyze_market_hours.py",
        "schedule": {"minutes": [0], "hours": [7], "weekdays": _WEEKDAYS_MON_FRI},
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
_DAILY_LOG_DIR = _PROJECT_ROOT / "logs"  # logs/YYYY-MM-DD.log (스크립트 stdout/stderr 통합 로그)
_DAILY_LINE_TS_PATTERN = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
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


def _to_float(value: object) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _build_pool_summary_rows() -> list[dict[str, object]]:
    """종목풀별 활성 종목과 최근 성과 요약을 만든다."""
    db = get_db_connection()
    if db is None:
        raise RuntimeError("DB 연결 실패로 종목풀 요약을 조회할 수 없습니다.")

    rows: list[dict[str, object]] = []
    for config in load_ticker_type_configs():
        ticker_type = str(config["ticker_type"])
        docs = list(
            db.stock_meta.find(
                {"ticker_type": ticker_type, "is_deleted": {"$ne": True}},
                {
                    "_id": 0,
                    "ticker": 1,
                    "is_etf": 1,
                    "1_week_earn_rate": 1,
                },
            )
        )

        stock_count = len(docs)
        rising_count = sum(1 for doc in docs if _to_float(doc.get("1_week_earn_rate")) > 0)
        etf_count = sum(1 for doc in docs if bool(doc.get("is_etf")))
        rows.append(
            {
                "id": ticker_type,
                "order": int(config["order"]),
                "pool": str(config["name"]),
                "ticker_type": ticker_type,
                "country_code": str(config.get("country_code") or "").upper(),
                "stock_count": stock_count,
                "rising_count": rising_count,
                "rising_ratio": round((rising_count / stock_count) * 100, 2) if stock_count > 0 else 0.0,
                "etf_count": etf_count,
            }
        )

    return rows


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


def extract_job_logs_for_run(
    job_key: str,
    started_at_iso: str,
    ended_at_iso: str | None = None,
) -> str | None:
    """logs/YYYY-MM-DD.log 에서 시작/종료 시각 사이의 라인을 추출해 반환.

    여러 작업이 같은 시각대에 끼어든 라인도 그대로 포함된다 (실패 진단 시 유용).
    종료 시각이 없으면 시작 + 30분으로 보수적 cap.
    반환: 추출된 텍스트(빈 문자열일 수 있음) 또는 파일을 못 찾으면 None.
    """
    try:
        started_at = datetime.fromisoformat(started_at_iso)
    except (TypeError, ValueError):
        return None
    if started_at.tzinfo is None:
        started_at = started_at.replace(tzinfo=ZoneInfo("Asia/Seoul"))
    started_kst = started_at.astimezone(ZoneInfo("Asia/Seoul"))

    ended_kst: datetime
    if ended_at_iso:
        try:
            ended_at = datetime.fromisoformat(ended_at_iso)
            if ended_at.tzinfo is None:
                ended_at = ended_at.replace(tzinfo=ZoneInfo("Asia/Seoul"))
            ended_kst = ended_at.astimezone(ZoneInfo("Asia/Seoul"))
        except (TypeError, ValueError):
            ended_kst = started_kst + timedelta(minutes=30)
    else:
        ended_kst = started_kst + timedelta(minutes=30)

    collected: list[str] = []
    found_any_file = False
    cur_date = started_kst.date()
    end_date = ended_kst.date()
    while cur_date <= end_date:
        log_path = _DAILY_LOG_DIR / f"{cur_date.isoformat()}.log"
        if log_path.exists():
            found_any_file = True
            try:
                for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                    m = _DAILY_LINE_TS_PATTERN.match(line)
                    if not m:
                        continue
                    try:
                        ts_naive = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        continue
                    ts = ts_naive.replace(tzinfo=ZoneInfo("Asia/Seoul"))
                    if ts < started_kst:
                        continue
                    if ts > ended_kst:
                        break
                    collected.append(line)
            except OSError:
                pass
        cur_date += timedelta(days=1)

    if not found_any_file:
        return None
    header = (
        f"# Job: {job_key}\n"
        f"# Started: {started_kst.isoformat()}\n"
        f"# Ended:   {ended_kst.isoformat()}\n"
        f"# Lines extracted: {len(collected)}\n"
        f"# ─────────────────────────────────────────────────────\n"
    )
    return header + "\n".join(collected) + "\n"


def _read_last_job_run_from_log(job_key: str) -> tuple[str, datetime] | None:
    """logs/cron/{job_key}.log 의 마지막 END 라인을 찾아 (status, started_at) 반환.

    status: "success" | "failure".
    """
    log_path = _LOG_DIR / f"{job_key}.log"
    if not log_path.exists():
        return None
    try:
        lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return None

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
        return None
    started_at = _parse_kst_datetime(last_started_at)
    if started_at is None:
        return None
    return (last_status, started_at)


def _read_last_job_run_from_queue(
    job_key: str,
) -> tuple[str, datetime, str | None, datetime | None] | None:
    """batch_queue 에서 가장 최근 종료된(done/failed) 항목의 (status, started_at, app_type, ended_at) 반환.

    wrapper 가 SIGKILL/배포로 외부 종료되면 logs/cron/{job}.log 에 END 라인이 남지
    않으므로 batch_queue 의 ended_at + status 가 더 정확한 최근 정보를 갖는다.
    """
    try:
        from utils.db_manager import get_db_connection

        db = get_db_connection()
        if db is None:
            return None
        doc = db.batch_queue.find_one(
            {"job_name": job_key, "status": {"$in": ["done", "failed"]}, "ended_at": {"$ne": None}},
            sort=[("ended_at", -1)],
            projection={"_id": 0, "status": 1, "started_at": 1, "ended_at": 1, "app_type": 1},
        )
        if not isinstance(doc, dict):
            return None
        started_raw = doc.get("started_at") or doc.get("ended_at")
        ended_raw = doc.get("ended_at")
        if not isinstance(started_raw, datetime):
            return None
        kst = ZoneInfo("Asia/Seoul")
        started_at = (
            started_raw.astimezone(kst)
            if started_raw.tzinfo
            else started_raw.replace(tzinfo=timezone.utc).astimezone(kst)
        )
        ended_at: datetime | None = None
        if isinstance(ended_raw, datetime):
            ended_at = (
                ended_raw.astimezone(kst) if ended_raw.tzinfo else ended_raw.replace(tzinfo=timezone.utc).astimezone(kst)
            )
        status = "success" if str(doc.get("status")) == "done" else "failure"
        app_type = str(doc.get("app_type") or "").strip() or None
        return (status, started_at, app_type, ended_at)
    except Exception as exc:
        _logger.warning("batch_queue 최근 종료 조회 실패 (job=%s): %s", job_key, exc)
        return None


def _read_last_job_run(job_key: str) -> dict[str, object | None]:
    """마지막 실행 정보를 반환한다.

    두 소스를 보고 **더 최근 시점**의 것을 표시한다:
      1) logs/cron/{job_key}.log 의 [run_batch] END 라인 — 이 환경 worker 가 처리한 기록
      2) batch_queue 의 가장 최근 done/failed — 다른 인스턴스(서버/로컬)에서 처리된 기록도 포함

    반환 필드:
      - status: "success"|"failure"|None
      - display: 화면 표시용 한 줄
      - owner_app_type: 어느 인스턴스가 처리했나 ("Server"|"Local"|None)
      - started_at, ended_at: ISO 문자열 (로그 다운로드 endpoint 호출용)
      - is_clickable: 현재 fastapi 인스턴스의 APP_TYPE 과 owner 가 일치하면 True
    """
    log_result = _read_last_job_run_from_log(job_key)
    queue_result = _read_last_job_run_from_queue(job_key)
    my_app_type = (os.environ.get("APP_TYPE") or "Server").strip() or "Server"

    chosen_source: str | None = None
    chosen_status: str | None = None
    chosen_started_at: datetime | None = None
    chosen_ended_at: datetime | None = None
    chosen_owner: str | None = None

    if log_result and queue_result:
        if log_result[1] >= queue_result[1]:
            chosen_source = "log"
        else:
            chosen_source = "queue"
    elif log_result:
        chosen_source = "log"
    elif queue_result:
        chosen_source = "queue"

    if chosen_source == "log" and log_result:
        chosen_status, chosen_started_at = log_result
        chosen_owner = my_app_type  # logs/cron 에 END 라인이 있다는 건 이 환경의 worker 가 처리했다는 뜻
    elif chosen_source == "queue" and queue_result:
        chosen_status, chosen_started_at, chosen_owner, chosen_ended_at = queue_result
        chosen_owner = chosen_owner or my_app_type

    if not chosen_status or not chosen_started_at:
        return {
            "status": None,
            "display": "-",
            "owner_app_type": None,
            "started_at": None,
            "ended_at": None,
            "is_clickable": False,
        }
    icon = "✅" if chosen_status == "success" else "❌"
    is_clickable = (chosen_owner or "").lower() == my_app_type.lower()
    return {
        "status": chosen_status,
        "display": f"{icon} {_format_elapsed_since(chosen_started_at)}",
        "owner_app_type": chosen_owner,
        "started_at": chosen_started_at.isoformat(),
        "ended_at": chosen_ended_at.isoformat() if chosen_ended_at else None,
        "is_clickable": is_clickable,
    }


# 평균 소요시간을 모르는 락도 이 시간(초)을 넘으면 무조건 stale 로 간주.
# (TTL 인덱스가 동작 안 하는 케이스에 대비한 절대 안전망)
_STALE_LOCK_ABSOLUTE_MAX_SECONDS = 6 * 3600  # 6시간


def _cleanup_stale_locks() -> int:
    """오래된 락을 stale 로 간주하고 삭제한다.

    삭제 조건(우선순위):
        1. expires_at 이 현재 시각보다 과거 → 즉시 삭제 (TTL 인덱스 fallback)
        2. elapsed > 예상 소요시간 × 2 → 삭제
        3. elapsed > 6시간 → 무조건 삭제 (예상값을 모르더라도 절대 상한)

    Returns:
        삭제된 락 개수
    """
    try:
        from utils.db_manager import get_db_connection

        db = get_db_connection()
        if db is None:
            return 0
        kst = ZoneInfo("Asia/Seoul")
        now_kst = datetime.now(kst)
        now_utc = datetime.now(timezone.utc)
        deleted = 0
        for doc in db.batch_locks.find({}, {"_id": 1, "acquired_at": 1, "expires_at": 1}):
            key = str(doc.get("_id") or "")
            if key not in _SCRIPT_BY_ACTION:
                continue

            # (1) expires_at 기반 정리 — TTL 인덱스가 어떤 이유로 안 도는 경우의 안전망
            expires_at = doc.get("expires_at")
            if isinstance(expires_at, datetime):
                expires_aware = (
                    expires_at if expires_at.tzinfo else expires_at.replace(tzinfo=timezone.utc)
                )
                if expires_aware < now_utc:
                    try:
                        db.batch_locks.delete_one({"_id": key})
                        deleted += 1
                        _logger.warning(
                            "Stale lock 제거(expires 초과): job=%s expires_at=%s",
                            key, expires_at,
                        )
                        continue
                    except Exception as exc:
                        _logger.warning("Stale lock 삭제 실패 (job=%s): %s", key, exc)

            # (2)(3) elapsed 기반 정리
            acquired_at = doc.get("acquired_at")
            if not isinstance(acquired_at, datetime):
                continue
            started_at = (
                acquired_at.astimezone(kst)
                if acquired_at.tzinfo
                else acquired_at.replace(tzinfo=timezone.utc).astimezone(kst)
            )
            elapsed_seconds = int((now_kst - started_at).total_seconds())
            if elapsed_seconds <= 0:
                continue

            estimated_seconds = _read_average_job_elapsed_seconds(key)
            should_delete = False
            reason = ""
            if estimated_seconds is not None and estimated_seconds > 0:
                if elapsed_seconds > int(estimated_seconds * 2):
                    should_delete = True
                    reason = f"estimated×2 초과 (estimated={int(estimated_seconds)}s)"
            if not should_delete and elapsed_seconds > _STALE_LOCK_ABSOLUTE_MAX_SECONDS:
                should_delete = True
                reason = f"절대 상한 {_STALE_LOCK_ABSOLUTE_MAX_SECONDS}s 초과"

            if should_delete:
                try:
                    db.batch_locks.delete_one({"_id": key})
                    deleted += 1
                    _logger.warning(
                        "Stale lock 제거: job=%s elapsed=%ds (%s)",
                        key, elapsed_seconds, reason,
                    )
                except Exception as exc:
                    _logger.warning("Stale lock 삭제 실패 (job=%s): %s", key, exc)
        return deleted
    except Exception as exc:
        _logger.warning("Stale lock 정리 실패: %s", exc)
        return 0


def get_running_jobs() -> list[str]:
    """현재 실행 중인 배치 키 목록을 반환합니다.

    두 소스를 합쳐서 본다:
      1) batch_locks — 락이 살아있는 작업
      2) batch_queue status=running 이면서 heartbeat 가 최근 3분 안에 갱신된 작업
         (장시간 작업이라 락이 만료됐어도 worker 가 계속 처리 중인 경우)
    """
    # 조회 전에 stale 락 정리 (조회/상세 모두에서 효과)
    _cleanup_stale_locks()
    try:
        from utils.db_manager import get_db_connection

        db = get_db_connection()
        if db is None:
            return []
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        running: set[str] = set()

        # 1) batch_locks
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
            running.add(key)

        # 2) batch_queue running + heartbeat 살아있음 (3분 안)
        try:
            heartbeat_threshold = now - timedelta(minutes=3)
            for doc in db.batch_queue.find(
                {"status": "running"}, {"_id": 0, "job_name": 1, "last_heartbeat": 1}
            ):
                key = str(doc.get("job_name") or "")
                if not key or key not in _SCRIPT_BY_ACTION:
                    continue
                last_heartbeat = doc.get("last_heartbeat")
                if isinstance(last_heartbeat, datetime):
                    hb_utc = (
                        last_heartbeat.astimezone(timezone.utc).replace(tzinfo=None)
                        if last_heartbeat.tzinfo
                        else last_heartbeat
                    )
                    if hb_utc < heartbeat_threshold:
                        continue
                running.add(key)
        except Exception as exc:
            _logger.warning("batch_queue running 조회 실패: %s", exc)

        return sorted(running)
    except Exception as exc:
        _logger.warning("batch_locks 조회 실패: %s", exc)
        return []


def get_running_job_details() -> dict[str, dict[str, object]]:
    """실행 중인 배치별 시작 시각, 예상 소요시간, 인스턴스 정보를 반환한다.

    두 소스를 합쳐서 본다:
      1) batch_locks — 락이 살아있는 작업 (정상 흐름)
      2) batch_queue status=running — 장시간 작업이라 락이 만료됐지만
         heartbeat 가 살아있는 작업 (락 만료 후에도 worker 가 계속 처리 중)
    같은 key 가 양쪽에 있으면 batch_locks 정보를 우선한다.
    """
    try:
        from utils.db_manager import get_db_connection

        db = get_db_connection()
        if db is None:
            return {}
        now_utc_naive = datetime.now(timezone.utc).replace(tzinfo=None)
        now_kst = datetime.now(ZoneInfo("Asia/Seoul"))
        my_app_type = (os.environ.get("APP_TYPE") or "PROD").strip() or "PROD"
        details: dict[str, dict[str, object]] = {}

        # 1) batch_locks 기반 (락이 살아있는 작업)
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

        # 2) batch_queue 의 running 항목 (락 만료된 장시간 작업도 표시)
        # heartbeat 가 최근 3분 안에 갱신된 것만 살아있다고 간주한다.
        try:
            heartbeat_threshold = now_utc_naive - timedelta(minutes=3)
            for doc in db.batch_queue.find(
                {"status": "running"},
                {
                    "_id": 0,
                    "job_name": 1,
                    "started_at": 1,
                    "last_heartbeat": 1,
                    "app_type": 1,
                    "cancel_requested": 1,
                },
            ):
                key = str(doc.get("job_name") or "")
                if not key or key not in _SCRIPT_BY_ACTION:
                    continue
                if key in details:
                    continue  # batch_locks 정보 우선
                last_heartbeat = doc.get("last_heartbeat")
                if isinstance(last_heartbeat, datetime):
                    hb_utc = (
                        last_heartbeat.astimezone(timezone.utc).replace(tzinfo=None)
                        if last_heartbeat.tzinfo
                        else last_heartbeat
                    )
                    if hb_utc < heartbeat_threshold:
                        continue  # 죽은 heartbeat — 표시하지 않음
                started_raw = doc.get("started_at")
                queue_started_at: datetime | None = None
                if isinstance(started_raw, datetime):
                    queue_started_at = (
                        started_raw.astimezone(ZoneInfo("Asia/Seoul"))
                        if started_raw.tzinfo
                        else started_raw.replace(tzinfo=timezone.utc).astimezone(ZoneInfo("Asia/Seoul"))
                    )
                q_estimated_seconds = _read_average_job_elapsed_seconds(key)
                q_elapsed_seconds = (
                    max(0, int((now_kst - queue_started_at).total_seconds())) if queue_started_at else None
                )
                q_remaining_seconds = (
                    max(0, int(round(q_estimated_seconds)) - int(q_elapsed_seconds))
                    if q_estimated_seconds is not None and q_elapsed_seconds is not None
                    else None
                )
                owner_app_type = (str(doc.get("app_type") or "") or "Server").strip()
                is_mine = owner_app_type.lower() == my_app_type.lower()
                details[key] = {
                    "started_at": queue_started_at.isoformat() if queue_started_at else None,
                    "estimated_seconds": int(round(q_estimated_seconds)) if q_estimated_seconds is not None else None,
                    "elapsed_seconds": q_elapsed_seconds,
                    "remaining_seconds": q_remaining_seconds,
                    "estimated_display": _format_duration_seconds(q_estimated_seconds),
                    "remaining_display": _format_duration_seconds(q_remaining_seconds),
                    "owner_app_type": owner_app_type,
                    "is_mine": is_mine,
                    "cancel_requested": bool(doc.get("cancel_requested")),
                }
        except Exception as exc:
            _logger.warning("batch_queue running 조회 실패: %s", exc)

        return details
    except Exception as exc:
        _logger.warning("batch_locks 상세 조회 실패: %s", exc)
        return {}


def load_system_data() -> dict[str, object]:
    from utils.batch_queue import list_queue

    queue_items = list_queue(limit=30)
    # ObjectId/datetime 직렬화 — 응답 직전 평탄화
    serialized_queue: list[dict[str, object]] = []
    for q in queue_items:
        serialized_queue.append(
            {
                "id": str(q.get("_id")),
                "job_name": q.get("job_name"),
                "status": q.get("status"),
                "triggered_by": q.get("triggered_by"),
                "triggered_at": q.get("triggered_at").isoformat() if q.get("triggered_at") else None,
                "started_at": q.get("started_at").isoformat() if q.get("started_at") else None,
                "ended_at": q.get("ended_at").isoformat() if q.get("ended_at") else None,
                "exit_code": q.get("exit_code"),
                "error": q.get("error"),
            }
        )
    estimated_by_job: dict[str, dict[str, object]] = {}
    for row in SCHEDULE_ROWS:
        key = str(row["key"])
        seconds = _read_average_job_elapsed_seconds(key)
        estimated_by_job[key] = {
            "seconds": int(round(seconds)) if seconds is not None else None,
            "display": _format_duration_seconds(seconds) if seconds is not None else None,
        }

    return {
        "pool_rows": _build_pool_summary_rows(),
        "schedule_rows": SCHEDULE_ROWS,
        "schedule_note": (
            "cron 스케줄 트리거는 서버 scheduler 컨테이너의 APScheduler 가 담당하며, "
            "`infra/cron/crontab` 파일이 단일 진실 소스입니다. "
            "큐 워커는 서버와 로컬(`python run_local_dev.py` 실행 중) 양쪽에서 함께 동작하며 "
            "MongoDB `find_one_and_update` 로 한 곳에서만 atomic 하게 claim 합니다. "
            "트리거(수동 클릭 / 스케줄)는 큐에 추가되어 FIFO 순서로 직렬 처리됩니다."
        ),
        "running_jobs": get_running_jobs(),
        "running_job_details": get_running_job_details(),
        "batch_queue": serialized_queue,
        "is_deploying": is_deploying(),
        "last_run_by_job": {row["key"]: _read_last_job_run(str(row["key"])) for row in SCHEDULE_ROWS},
        "next_run_by_job": {row["key"]: _build_next_run_payload(row.get("schedule")) for row in SCHEDULE_ROWS},
        "estimated_by_job": estimated_by_job,
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


class JobCancelForbiddenError(RuntimeError):
    """현재 인스턴스의 APP_TYPE 과 worker 의 app_type 이 일치하지 않을 때 거부."""


def request_cancel_running_job(job_key: str) -> dict[str, str]:
    """현재 fastapi 인스턴스의 APP_TYPE 과 일치하는 worker 가 처리 중일 때만 취소 요청을 보낸다.

    예외:
      - JobCancelForbiddenError: 다른 인스턴스(서버↔로컬)에서 처리 중인 작업
      - ValueError: 알 수 없는 job_key
      - RuntimeError: running 항목이 없거나 DB 실패
    """
    if job_key not in _SCRIPT_BY_ACTION:
        raise ValueError(f"알 수 없는 작업 키: {job_key}")

    my_app_type = (os.environ.get("APP_TYPE") or "Server").strip() or "Server"

    try:
        from utils.batch_queue import request_cancel_running
        from utils.db_manager import get_db_connection

        db = get_db_connection()
        if db is None:
            raise RuntimeError("DB 연결 실패")
        # 현재 running 인 항목의 app_type 을 먼저 확인 — 권한 검증
        doc = db.batch_queue.find_one(
            {"job_name": job_key, "status": "running"},
            {"_id": 0, "app_type": 1},
        )
        if not isinstance(doc, dict):
            raise RuntimeError("실행 중인 항목이 없습니다.")
        worker_app_type = (str(doc.get("app_type") or "")).strip() or "Server"
        if worker_app_type.lower() != my_app_type.lower():
            raise JobCancelForbiddenError(
                f"이 작업은 [{worker_app_type.upper()}] 에서 실행 중이라 [{my_app_type.upper()}] 페이지에서는 중단할 수 없습니다."
            )

        result = request_cancel_running(job_key, requester_app_type=my_app_type)
        if not result.get("ok"):
            raise RuntimeError(str(result.get("reason") or "취소 요청 실패"))
        return {"message": f"[{worker_app_type.upper()}] 의 {job_key} 중단 요청을 보냈습니다. 곧 종료됩니다."}
    except (JobCancelForbiddenError, ValueError, RuntimeError):
        raise
    except Exception as exc:
        _logger.warning("취소 요청 실패: %s — %s", job_key, exc)
        raise RuntimeError(f"취소 요청 실패: {exc}") from exc


def trigger_system_action(action: SystemAction) -> str:
    """배치 작업을 큐에 추가한다 (직접 실행하지 않음).

    같은 작업이 이미 pending/running 이면 무시. 워커가 FIFO 로 직렬 처리.
    """
    from utils.batch_queue import enqueue

    if action not in _SCRIPT_BY_ACTION:
        raise ValueError("지원하지 않는 시스템 작업입니다.")

    script_rel = _SCRIPT_BY_ACTION[action]
    label = _LABEL_BY_ACTION.get(action, action)
    result = enqueue(action, script_rel, triggered_by="manual")
    if not result.get("enqueued"):
        return f"[시스템-배치] {label} 이미 큐에 있습니다 ({result.get('reason')})."
    return f"[시스템-배치] {label} 큐에 추가됨. 워커가 순서대로 실행합니다."
