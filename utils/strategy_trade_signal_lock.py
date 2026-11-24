"""계좌·종목별 작업 예약 — 서로 다른 종목의 외부 API 대기는 분리한다."""

import os
import socket
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from threading import Event, Thread
from time import monotonic, sleep
from uuid import uuid4

from pymongo.errors import DuplicateKeyError

from utils.db_manager import get_db_connection
from utils.logger import get_app_logger

_LOCK_WAIT_SECONDS = 5
_LOCK_RETRY_SECONDS = 0.2
_STALE_LOCK_SECONDS = 120
_LOCK_RENEW_SECONDS = 20
logger = get_app_logger()


class SignalBusyError(ValueError):
    """같은 대상의 작업 예약을 정해진 시간 안에 얻지 못했다."""


@contextmanager
def _db_lock(key: str, busy_message: str, *, wait_seconds: float):
    """짧게 예약을 재시도하고 실행 중에는 소유 토큰으로 임대를 갱신한다."""
    db = get_db_connection()
    if db is None:
        raise RuntimeError("DB 연결에 실패했습니다.")
    collection = db["system_config"]
    token = uuid4().hex
    owner = {"host": socket.gethostname(), "pid": os.getpid()}
    deadline = monotonic() + wait_seconds
    while True:
        now = datetime.now(UTC)
        try:
            collection.insert_one({"_id": key, "token": token, "created_at": now, "renewed_at": now, **owner})
            break
        except DuplicateKeyError as exc:
            stale_before = now - timedelta(seconds=_STALE_LOCK_SECONDS)
            # 기존 스레드 예약은 갱신 시각 없이 생성 시각만 저장했다.
            stale = {
                "_id": key,
                "$or": [
                    {"renewed_at": {"$lt": stale_before}},
                    {"renewed_at": {"$exists": False}, "created_at": {"$lt": stale_before}},
                ],
            }
            if collection.delete_one(stale).deleted_count:
                continue
            remaining = deadline - monotonic()
            if remaining <= 0:
                holder = collection.find_one({"_id": key}, {"host": 1, "pid": 1, "created_at": 1, "renewed_at": 1})
                logger.info("신호 작업 예약 경합: %s, 점유=%s", busy_message, holder)
                raise SignalBusyError(busy_message) from exc
        sleep(min(_LOCK_RETRY_SECONDS, remaining))
    stop, lost = Event(), Event()

    def renew():
        while not stop.wait(_LOCK_RENEW_SECONDS):
            try:
                result = collection.update_one(
                    {"_id": key, "token": token}, {"$set": {"renewed_at": datetime.now(UTC)}}
                )
                if not result.matched_count:
                    raise RuntimeError("신호 작업 예약의 소유권을 잃었습니다.")
            except Exception:
                lost.set()
                logger.exception("신호 작업 예약 갱신 실패: %s", key)
                return

    heartbeat = Thread(target=renew, name="signal-reservation", daemon=True)
    heartbeat.start()
    try:
        yield
        if lost.is_set():
            raise RuntimeError("신호 작업 예약을 유지하지 못했습니다. 발송·주문 기록을 확인하세요.")
    finally:
        stop.set()
        heartbeat.join(timeout=_LOCK_RENEW_SECONDS)
        collection.delete_one({"_id": key, "token": token})


def _signal_key(account_id: str, ticker: str) -> str:
    if not account_id or not ticker:
        raise ValueError("신호 작업 예약에 계좌와 종목이 필요합니다.")
    digest = sha256(f"{account_id}\0{ticker}".encode()).hexdigest()
    return f"strategy_trade_signal_lock:{digest}"


def signal_lock(account_id: str, ticker: str):
    """동일 계좌·종목의 발송·주문·회수만 직렬화하고 잠깐 경합하면 재시도한다."""
    return _db_lock(
        _signal_key(account_id, ticker),
        f"{account_id} · {ticker} 신호 작업이 진행 중입니다. 잠시 후 다시 시도하세요.",
        wait_seconds=_LOCK_WAIT_SECONDS,
    )


def batch_signal_lock(account_id: str, ticker: str):
    """배치는 예약 중인 종목만 다음 실행으로 미루고 나머지 종목을 계속 처리한다."""
    return _db_lock(
        _signal_key(account_id, ticker), f"{account_id} · {ticker} 신호 작업이 진행 중입니다.", wait_seconds=0
    )


def settings_signal_lock(account_id: str):
    """같은 계좌의 설정 저장끼리만 직렬화한다. 종목 예약은 호출자가 정렬해 얻는다."""
    return _db_lock(
        f"strategy_trade_settings_lock:{account_id}",
        "같은 계좌의 설정을 저장 중입니다.",
        wait_seconds=_LOCK_WAIT_SECONDS,
    )


def order_limit_lock(account_id: str):
    """계좌 한도 확인과 주문 기록 삽입만 묶는다. 증권사 요청은 예약 밖에서 한다."""
    return _db_lock(
        f"strategy_trade_order_limit_lock:{account_id}",
        "같은 계좌의 주문 한도를 예약 중입니다. 잠시 후 다시 시도하세요.",
        wait_seconds=_LOCK_WAIT_SECONDS,
    )


def broker_token_lock(account_id: str):
    """계좌별 인증 갱신만 선점한다. 종목의 신호·주문 예약과는 별개다."""
    return _db_lock(
        f"meritz_token_lock:{account_id}",
        f"{account_id} 인증 토큰을 갱신 중입니다. 잠시 후 다시 조회하세요.",
        wait_seconds=_LOCK_WAIT_SECONDS,
    )


def signal_thread_lock(thread_ts: str):
    """한 스레드의 반응 동기화만 직렬화한다."""
    return _db_lock(
        f"strategy_trade_signal_thread_lock:{thread_ts}",
        "같은 신호 스레드의 반응을 맞추는 중입니다.",
        wait_seconds=_LOCK_WAIT_SECONDS,
    )
