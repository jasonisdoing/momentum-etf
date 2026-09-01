"""배치 작업 큐 (MongoDB `batch_queue`).

설계 결정:
    - 단일 작업 단위 FIFO 직렬 처리. 워커는 서버 scheduler 컨테이너 + 로컬
      (`python run_local_dev.py`) 다중 인스턴스 가능. MongoDB find_one_and_update
      로 동시 claim 안전.
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

import os
from datetime import datetime, timedelta, timezone
from typing import Any

from pymongo.errors import DuplicateKeyError

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

# 무거운 계산이라 서버(약한 VM)에서 돌리면 안 되고, 로컬 워커(APP_TYPE=Local)만 픽하게 하는 잡들.
# (서버 워커는 이 잡들을 claim 하지 않는다 → 로컬이 꺼져 있으면 pending 으로 대기)
# db_backup: 백업 폴더가 로컬 디스크라 서버 워커가 잡으면 안 된다.
# (broker_balance_sync 는 나무증권 토큰을 DB 로 공유하므로 어느 워커든 잡아도 된다 — broker_api_service 참고)
LOCAL_ONLY_JOBS: set[str] = {"db_backup"}


# running 을 1건으로 묶는 부분 유니크 인덱스 이름.
_SINGLE_RUNNING_INDEX = "only_one_running"


def _now_utc() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _ensure_single_running_index(coll: Any) -> None:
    """running 1건 제한 인덱스를 만든다. 이미 2건 이상 running 이면 만들 수 없다.

    워커가 병렬로 돌던 시점에 걸쳐 있으면 생성이 실패한다 — 그때는 stale 정리 뒤
    다음 워커 시작에서 다시 시도한다(실패해도 워커는 계속 떠야 하므로 예외를 삼킨다).
    """
    try:
        coll.create_index(
            [("status", 1)],
            unique=True,
            partialFilterExpression={"status": STATUS_RUNNING},
            name=_SINGLE_RUNNING_INDEX,
        )
    except Exception as exc:
        logger.warning("[배치큐] 직렬화 인덱스 생성 실패 — 병렬 실행이 남을 수 있음: %s", exc)


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
    # 전역 직렬화 — running 상태 문서를 컬렉션 전체에서 1건으로 제한한다.
    # 서버 워커와 로컬 워커가 각자 claim 하면 배치 2건이 동시에 DB 를 때린다. standalone
    # MongoDB 라 트랜잭션을 못 쓰므로, 부분 유니크 인덱스로 DB 가 직접 막게 한다.
    # 두 번째 워커의 claim 은 DuplicateKeyError 로 튕기고 다음 폴링에서 다시 시도한다.
    _ensure_single_running_index(coll)
    # TTL — expires_at 이 지나면 자동 삭제 (모든 상태에 적용)
    try:
        coll.create_index("expires_at", expireAfterSeconds=0)
    except Exception:
        pass  # 이미 있을 수 있음


def enqueue(
    job_name: str,
    script_path: str,
    triggered_by: str = "manual",
    arguments: list[str] | None = None,
) -> dict[str, Any]:
    """배치 작업을 큐에 추가한다.

    같은 job_name 이 pending/running 이면 추가하지 않고 기존 항목 반환.
    triggered_by: "manual" (사용자 클릭) 또는 "schedule" (스케줄러).
    """
    db = get_db_connection()
    if db is None:
        raise RuntimeError("DB 연결 실패 — 큐 enqueue 불가")
    coll = db[BATCH_QUEUE_COLLECTION]

    existing = coll.find_one({"job_name": job_name, "status": {"$in": [STATUS_PENDING, STATUS_RUNNING]}})
    if existing:
        return {"enqueued": False, "reason": f"이미 큐에 있음 (status={existing.get('status')})", "item": existing}

    now = _now_utc()
    doc = {
        "job_name": job_name,
        "script_path": script_path,
        "triggered_by": triggered_by,
        "triggered_at": now,
        "status": STATUS_PENDING,
        "local_only": job_name.split(":")[0] in LOCAL_ONLY_JOBS,
        "arguments": arguments,
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
    워커를 실행 중인 인스턴스의 APP_TYPE 도 함께 기록해 시스템 UI 에서
    "어느 인스턴스가 처리 중인지" 식별 가능하게 한다 (락이 만료된 장시간 작업도 인식).
    """
    db = get_db_connection()
    if db is None:
        return None
    coll = db[BATCH_QUEUE_COLLECTION]
    now = _now_utc()
    worker_app_type = (os.environ.get("APP_TYPE") or "").strip() or "Server"
    # 로컬 워커가 아니면 local_only 잡(튜닝/백테스트)은 픽하지 않는다.
    claim_filter: dict[str, Any] = {"status": STATUS_PENDING}
    if worker_app_type != "Local":
        claim_filter["local_only"] = {"$ne": True}
    try:
        return coll.find_one_and_update(
            claim_filter,
            {
                "$set": {
                    "status": STATUS_RUNNING,
                    "started_at": now,
                    "last_heartbeat": now,
                    "app_type": worker_app_type,
                }
            },
            sort=[("triggered_at", 1)],
            return_document=True,  # type: ignore[arg-type]
        )
    except DuplicateKeyError:
        # 다른 워커가 이미 1건을 돌리고 있다 — 전역 직렬화가 막은 정상 경로다.
        # 아무것도 집지 않고 다음 폴링(1초)에서 다시 시도한다.
        return None


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
    return list(db[BATCH_QUEUE_COLLECTION].find({}).sort("triggered_at", -1).limit(limit))


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


def get_latest_item(job_name: str) -> dict[str, Any] | None:
    """해당 job_name 의 가장 최근 큐 항목 1건 (상태 조회용)."""
    db = get_db_connection()
    if db is None:
        return None
    return db[BATCH_QUEUE_COLLECTION].find_one(
        {"job_name": job_name},
        sort=[("triggered_at", -1)],
    )


def cancel_pending(item_id: Any) -> bool:
    """pending 항목 취소 (running 은 취소 불가)."""
    db = get_db_connection()
    if db is None:
        return False
    result = db[BATCH_QUEUE_COLLECTION].delete_one({"_id": item_id, "status": STATUS_PENDING})
    return result.deleted_count > 0


def request_cancel_running(job_name: str, requester_app_type: str | None = None) -> dict[str, Any]:
    """현재 running 인 작업에 취소 요청 플래그를 세운다.

    worker 가 polling 으로 이 플래그를 확인하고 자식 프로세스에 SIGTERM 을 보낸다.
    같은 app_type 의 worker 만 자기 작업을 중단할 수 있는 책임은 호출 측에서 검증한다.
    Returns: {"ok": bool, "reason": str, "item": dict | None}
    """
    db = get_db_connection()
    if db is None:
        return {"ok": False, "reason": "DB 연결 실패", "item": None}
    coll = db[BATCH_QUEUE_COLLECTION]
    now = _now_utc()
    doc = coll.find_one_and_update(
        {"job_name": job_name, "status": STATUS_RUNNING},
        {
            "$set": {
                "cancel_requested": True,
                "cancel_requested_at": now,
                "cancel_requested_by": requester_app_type or "unknown",
            }
        },
        return_document=True,  # type: ignore[arg-type]
    )
    if not doc:
        return {"ok": False, "reason": "실행 중인 항목이 없습니다.", "item": None}
    return {"ok": True, "reason": "취소 요청 완료", "item": doc}


def is_cancel_requested(item_id: Any) -> bool:
    """worker 가 자식 프로세스 중단 여부를 빠르게 polling 확인."""
    db = get_db_connection()
    if db is None:
        return False
    doc = db[BATCH_QUEUE_COLLECTION].find_one({"_id": item_id}, {"_id": 0, "cancel_requested": 1})
    return bool(isinstance(doc, dict) and doc.get("cancel_requested"))
