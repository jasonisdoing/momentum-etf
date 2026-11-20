"""자동 배치·수동 발송·회수가 공유하는 DB 선점 — 발송 이력의 덮어쓰기를 막는다."""

from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from time import monotonic, sleep
from uuid import uuid4

from pymongo.errors import DuplicateKeyError

from utils.db_manager import get_db_connection

_LOCK_WAIT_SECONDS = 30
_LOCK_RETRY_SECONDS = 0.2
_STALE_LOCK_SECONDS = 120


@contextmanager
def _db_lock(key: str, busy_message: str):
    """잠금 획득만 짧게 재시도하고 작업 자체는 한 번 실행한다."""
    db = get_db_connection()
    if db is None:
        raise RuntimeError("DB 연결에 실패했습니다.")
    collection = db["system_config"]
    token = uuid4().hex
    deadline = monotonic() + _LOCK_WAIT_SECONDS
    while True:
        try:
            collection.insert_one({"_id": key, "token": token, "created_at": datetime.now(UTC)})
            break
        except DuplicateKeyError as exc:
            stale_before = datetime.now(UTC) - timedelta(seconds=_STALE_LOCK_SECONDS)
            if collection.delete_one({"_id": key, "created_at": {"$lt": stale_before}}).deleted_count:
                continue
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise ValueError(busy_message) from exc
        sleep(min(_LOCK_RETRY_SECONDS, remaining))
    try:
        yield
    finally:
        collection.delete_one({"_id": key, "token": token})


def signal_lock():
    """신호 발송·주문·회수 전체를 한 번에 하나만 돌게 한다."""
    return _db_lock("strategy_trade_signal_lock", "다른 신호 작업이 아직 진행 중입니다. 잠시 후 다시 시도하세요.")


def signal_thread_lock(thread_ts: str):
    """신호 스레드 하나의 반응 맞추기를 프로세스 사이에서 한 번에 하나만 돌게 한다 — 다른 스레드는 기다리지 않는다."""
    return _db_lock(f"strategy_trade_signal_thread_lock:{thread_ts}", "같은 신호 스레드의 반응을 맞추는 중입니다.")
