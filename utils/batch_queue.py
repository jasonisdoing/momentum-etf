"""배치 작업 큐 (MongoDB `batch_queue`).

설계 결정:
    - 단일 워커가 FIFO 로 직렬 처리 (로컬의 run_local_scheduler 안에서 동작)
    - 중복 enqueue 무시 (같은 job_name 이 pending/running 이면 추가 안 함)
    - 24시간 TTL — 워커가 꺼져 있는 동안 무한 누적되는 것 방지
    - heartbeat: 워커가 30초마다 last_heartbeat 갱신
    - 워커 시작 시 stale running (heartbeat 5분 이상 끊김) → failed 자동 마킹

상태 흐름:
    pending → running → done
                ↓
              failed (워커 중단 / 스크립트 실패 / TTL)
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from utils.db_manager import get_db_connection
from utils.logger import get_app_logger

logger = get_app_logger()

BATCH_QUEUE_COLLECTION = "batch_queue"

STATUS_PENDING = "pending"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"

_TTL_HOURS = 24
_HEARTBEAT_STALE_MINUTES = 5


def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def ensure_indexes() -> None:
    """배치 큐에 필요한 인덱스를 보장한다."""
    db = get_db_connection()
    if db is None:
        return
    coll = db[BATCH_QUEUE_COLLECTION]
    # FIFO 조회용
    coll.create_index([("status", 1), ("triggered_at", 1)])
    # 중복 enqueue 체크용
    coll.create_index([("job_name", 1), ("status", 1)])
    # TTL — expires_at 이 지나면 자동 삭제 (모든 상태에 적용)
    try:
        coll.create_index("expires_at", expireAfterSeconds=0)
    except Exception:
        pass  # 이미 있을 수 있음


def enqueue(
    job_name: str,
    script_path: str,
    triggered_by: str = "manual",
) -> dict[str, Any]:
    """배치 작업을 큐에 추가한다.

    같은 job_name 이 pending/running 이면 추가하지 않고 기존 항목 반환.
    triggered_by: "manual" (사용자 클릭) 또는 "schedule" (스케줄러).
    """
    db = get_db_connection()
    if db is None:
        raise RuntimeError("DB 연결 실패 — 큐 enqueue 불가")
    coll = db[BATCH_QUEUE_COLLECTION]

    existing = coll.find_one(
        {"job_name": job_name, "status": {"$in": [STATUS_PENDING, STATUS_RUNNING]}}
    )
    if existing:
        return {"enqueued": False, "reason": f"이미 큐에 있음 (status={existing.get('status')})", "item": existing}

    now = _now_utc()
    doc = {
        "job_name": job_name,
        "script_path": script_path,
        "triggered_by": triggered_by,
        "triggered_at": now,
        "status": STATUS_PENDING,
        "started_at": None,
        "ended_at": None,
        "last_heartbeat": None,
        "exit_code": None,
        "error": None,
        "expires_at": now + timedelta(hours=_TTL_HOURS),
    }
    result = coll.insert_one(doc)
    doc["_id"] = result.inserted_id
    return {"enqueued": True, "item": doc}


def claim_next_pending() -> dict[str, Any] | None:
    """가장 오래된 pending 1건을 원자적으로 running 으로 변경하고 반환.

    동시 워커 안전 (find_one_and_update 사용).
    """
    db = get_db_connection()
    if db is None:
        return None
    coll = db[BATCH_QUEUE_COLLECTION]
    now = _now_utc()
    return coll.find_one_and_update(
        {"status": STATUS_PENDING},
        {"$set": {"status": STATUS_RUNNING, "started_at": now, "last_heartbeat": now}},
        sort=[("triggered_at", 1)],
        return_document=True,  # type: ignore[arg-type]
    )


def update_heartbeat(item_id: Any) -> None:
    """실행 중 워커가 주기적으로 호출. heartbeat 시각 갱신."""
    db = get_db_connection()
    if db is None:
        return
    db[BATCH_QUEUE_COLLECTION].update_one(
        {"_id": item_id, "status": STATUS_RUNNING},
        {"$set": {"last_heartbeat": _now_utc()}},
    )


def mark_done(item_id: Any, exit_code: int) -> None:
    db = get_db_connection()
    if db is None:
        return
    now = _now_utc()
    db[BATCH_QUEUE_COLLECTION].update_one(
        {"_id": item_id},
        {
            "$set": {
                "status": STATUS_DONE if exit_code == 0 else STATUS_FAILED,
                "ended_at": now,
                "exit_code": int(exit_code),
                # 완료 항목도 TTL 24h 후 정리되도록 expires_at 갱신
                "expires_at": now + timedelta(hours=_TTL_HOURS),
            }
        },
    )


def mark_failed(item_id: Any, error: str) -> None:
    db = get_db_connection()
    if db is None:
        return
    now = _now_utc()
    db[BATCH_QUEUE_COLLECTION].update_one(
        {"_id": item_id},
        {
            "$set": {
                "status": STATUS_FAILED,
                "ended_at": now,
                "error": str(error)[:500],
                "expires_at": now + timedelta(hours=_TTL_HOURS),
            }
        },
    )


def reap_stale_running() -> int:
    """heartbeat 가 끊긴 running 항목을 failed 로 마킹.

    워커 시작 시점과 주기적으로 호출. 반환: 처리한 건수.
    """
    db = get_db_connection()
    if db is None:
        return 0
    coll = db[BATCH_QUEUE_COLLECTION]
    threshold = _now_utc() - timedelta(minutes=_HEARTBEAT_STALE_MINUTES)
    result = coll.update_many(
        {
            "status": STATUS_RUNNING,
            "$or": [
                {"last_heartbeat": {"$lt": threshold}},
                {"last_heartbeat": None},
            ],
        },
        {
            "$set": {
                "status": STATUS_FAILED,
                "ended_at": _now_utc(),
                "error": "워커가 끊긴 것으로 추정 (heartbeat 5분 이상 없음)",
                "expires_at": _now_utc() + timedelta(hours=_TTL_HOURS),
            }
        },
    )
    if result.modified_count > 0:
        logger.warning("Stale running 항목 %d건 → failed 마킹", result.modified_count)
    return result.modified_count


def list_queue(limit: int = 50) -> list[dict[str, Any]]:
    """현재 큐 상태 — 최신순 (pending → running → done/failed)."""
    db = get_db_connection()
    if db is None:
        return []
    return list(
        db[BATCH_QUEUE_COLLECTION]
        .find({})
        .sort("triggered_at", -1)
        .limit(limit)
    )


def get_pending_count() -> int:
    db = get_db_connection()
    if db is None:
        return 0
    return db[BATCH_QUEUE_COLLECTION].count_documents({"status": STATUS_PENDING})


def get_running_item() -> dict[str, Any] | None:
    db = get_db_connection()
    if db is None:
        return None
    return db[BATCH_QUEUE_COLLECTION].find_one({"status": STATUS_RUNNING})


def cancel_pending(item_id: Any) -> bool:
    """pending 항목 취소 (running 은 취소 불가)."""
    db = get_db_connection()
    if db is None:
        return False
    result = db[BATCH_QUEUE_COLLECTION].delete_one(
        {"_id": item_id, "status": STATUS_PENDING}
    )
    return result.deleted_count > 0
