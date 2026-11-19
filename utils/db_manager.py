import os
import threading
import time
from urllib.parse import quote_plus

from dotenv import load_dotenv
from pymongo import MongoClient

from utils.logger import get_app_logger

load_dotenv()

_db_connection = None
_mongo_client: MongoClient | None = None
_last_connect_failure_at: float = 0.0
_CONNECT_FAILURE_COOLDOWN_SEC = 5.0
_conn_lock = threading.Lock()
logger = get_app_logger()


def _reset_connection_unlocked() -> None:
    global _db_connection, _mongo_client
    _db_connection = None
    if _mongo_client is not None:
        try:
            _mongo_client.close()
        except Exception:
            pass
    _mongo_client = None


def _reset_connection() -> None:
    """캐시된 Mongo 연결을 닫고 초기화한다."""
    with _conn_lock:
        _reset_connection_unlocked()


def _get_required_env(name: str) -> str:
    value = (os.environ.get(name) or "").strip()
    if not value:
        raise RuntimeError(f"{name} 환경변수가 필요합니다.")
    return value


def _resolve_connection_string() -> str:
    user = quote_plus(_get_required_env("MONGO_DB_USER"))
    password = quote_plus(_get_required_env("MONGO_DB_PASSWORD"))
    host = _get_required_env("MONGO_DB_HOST")
    auth_source = _get_required_env("MONGO_DB_AUTH_SOURCE")
    return f"mongodb://{user}:{password}@{host}/?authSource={quote_plus(auth_source)}"


def _build_client(connection_string: str) -> MongoClient:
    """MongoClient 를 생성한다 — 튜닝값은 상수다(환경변수 override 폐기, 2026-09 미사용 정리)."""
    return MongoClient(
        connection_string,
        maxPoolSize=10,
        minPoolSize=1,
        retryWrites=True,
        retryReads=True,
        serverSelectionTimeoutMS=2_500,
        connectTimeoutMS=3_000,
        socketTimeoutMS=10_000,
        heartbeatFrequencyMS=10_000,
        maxIdleTimeMS=300_000,
        waitQueueTimeoutMS=15_000,
        compressors="zlib",
        appname="momentum-etf",
    )


def get_db_connection():
    """
    MongoDB 클라이언트 연결을 생성하고, 전역 변수에 저장하여 재사용합니다.
    """
    global _db_connection, _mongo_client, _last_connect_failure_at

    # 이미 연결이 설정되어 있으면, 락 없이 기존 연결을 바로 반환합니다.
    if _db_connection is not None:
        return _db_connection

    # 직전에 연결 실패했다면 쿨다운 동안 락 대기·재시도를 생략해 화면/헬스체크가 밀리지 않게 한다.
    if time.monotonic() - _last_connect_failure_at < _CONNECT_FAILURE_COOLDOWN_SEC:
        return None

    with _conn_lock:
        if _db_connection is not None:
            return _db_connection
        if time.monotonic() - _last_connect_failure_at < _CONNECT_FAILURE_COOLDOWN_SEC:
            return None

        try:
            connection_string = _resolve_connection_string()
        except RuntimeError as exc:
            logger.error("오류: MongoDB 연결 정보가 설정되지 않았습니다. %s", exc)
            return None
        db_name = _get_required_env("MONGO_DB_NAME")

        try:
            if _mongo_client is None:
                _mongo_client = _build_client(connection_string)

            client = _mongo_client
            client.admin.command("ping")
            _db_connection = client[db_name]
            _last_connect_failure_at = 0.0
            logger.debug("MongoDB에 성공적으로 연결되었습니다.")
            return _db_connection
        except Exception as exc:
            _last_connect_failure_at = time.monotonic()
            logger.error("오류: MongoDB 연결에 실패했습니다: %s", exc)
            _reset_connection_unlocked()
            return None
