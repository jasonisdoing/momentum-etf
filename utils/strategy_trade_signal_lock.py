"""자동 배치·수동 발송·회수가 공유하는 DB 선점 — 발송 이력의 덮어쓰기를 막는다."""

from contextlib import contextmanager
from datetime import UTC, datetime
from uuid import uuid4

from pymongo.errors import DuplicateKeyError

from utils.db_manager import get_db_connection


@contextmanager
def signal_lock():
    """프로세스 간 단일 실행. 중단된 발송은 자동 재시도하지 않고 잠금을 남겨 확인하게 한다."""
    db = get_db_connection()
    if db is None:
        raise RuntimeError("DB 연결에 실패했습니다.")
    collection = db["system_config"]
    token = uuid4().hex
    key = "strategy_trade_signal_lock"
    try:
        collection.insert_one({"_id": key, "token": token, "created_at": datetime.now(UTC)})
    except DuplicateKeyError as exc:
        raise ValueError(
            "다른 신호 작업이 진행 중입니다. 잠시 후 다시 시도하세요. 계속되면 발송 기록과 서버 잠금을 확인하세요."
        ) from exc
    try:
        yield
    finally:
        collection.delete_one({"_id": key, "token": token})
