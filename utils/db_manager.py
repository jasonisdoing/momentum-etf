import os
import time
from urllib.parse import quote_plus

from dotenv import load_dotenv
from pymongo import MongoClient

from utils.logger import get_app_logger

load_dotenv()

_db_connection = None
_mongo_client: MongoClient | None = None
logger = get_app_logger()


def _reset_connection() -> None:
    """캐시된 Mongo 연결을 닫고 초기화한다."""
    global _db_connection, _mongo_client
    _db_connection = None
    if _mongo_client is not None:
        try:
            _mongo_client.close()
        except Exception:
            pass
    _mongo_client = None


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
        minPoolSize=0,
        retryWrites=True,
        retryReads=True,
        serverSelectionTimeoutMS=10_000,
        connectTimeoutMS=10_000,
        socketTimeoutMS=10_000,
        heartbeatFrequencyMS=10_000,
        maxIdleTimeMS=60_000,
        waitQueueTimeoutMS=15_000,
        appname="momentum-etf",
    )


def get_db_connection():
    """
    MongoDB 클라이언트 연결을 생성하고, 전역 변수에 저장하여 재사용합니다.
    """
    global _db_connection, _mongo_client

    # 이미 연결이 설정되어 있으면, 기존 연결을 반환합니다.
    if _db_connection is not None:
        return _db_connection

    try:
        connection_string = _resolve_connection_string()
    except RuntimeError as exc:
        logger.error("오류: MongoDB 연결 정보가 설정되지 않았습니다. %s", exc)
        return None
    db_name = _get_required_env("MONGO_DB_NAME")

    last_error: Exception | None = None
    for attempt, wait_seconds in enumerate((0, 1.5, 4.0), start=1):
        try:
            if wait_seconds > 0:
                time.sleep(wait_seconds)

            if _mongo_client is None:
                _mongo_client = _build_client(connection_string)

            client = _mongo_client
            client.admin.command("ping")
            _db_connection = client[db_name]
            logger.debug("MongoDB에 성공적으로 연결되었습니다. (attempt=%s)", attempt)
            return _db_connection
        except Exception as exc:
            last_error = exc
            logger.warning("MongoDB 연결 재시도 %s/3 실패: %s", attempt, exc)
            _reset_connection()

    logger.error("오류: MongoDB 연결에 실패했습니다: %s", last_error)
    return None
