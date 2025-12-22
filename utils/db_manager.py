import os

from dotenv import load_dotenv
from pymongo import MongoClient

try:
    import data.settings as global_settings  # type: ignore
except ImportError:  # pragma: no cover - 하위 호환
    global_settings = object()
from utils.logger import get_app_logger

# .env 파일이 있다면 로드합니다.
load_dotenv()

# --- 전역 변수로 DB 연결 관리 ---
_db_connection = None
_mongo_client: MongoClient | None = None
logger = get_app_logger()


def get_db_connection():
    """
    MongoDB 클라이언트 연결을 생성하고, 전역 변수에 저장하여 재사용합니다.
    """
    global _db_connection, _mongo_client

    # 이미 연결이 설정되어 있으면, 기존 연결을 반환합니다.
    if _db_connection is not None:
        return _db_connection

    try:
        connection_string = os.environ.get("MONGO_DB_CONNECTION_STRING") or getattr(
            global_settings, "MONGO_DB_CONNECTION_STRING", None
        )
        db_name = os.environ.get("MONGO_DB_NAME") or getattr(global_settings, "MONGO_DB_NAME", "momentum_etf_db")
        # 연결 풀 관련 환경 변수(선택 사항)를 반영한다.
        max_pool = int(os.environ.get("MONGO_DB_MAX_POOL_SIZE", "20"))
        min_pool = int(os.environ.get("MONGO_DB_MIN_POOL_SIZE", "0"))
        max_idle = int(os.environ.get("MONGO_DB_MAX_IDLE_TIME_MS", "0"))  # 0 = driver default
        wait_q_timeout = int(os.environ.get("MONGO_DB_WAIT_QUEUE_TIMEOUT_MS", "0"))  # 0 = driver default

        if not connection_string:
            raise ValueError("MongoDB 연결 문자열이 설정되지 않았습니다. (MONGO_DB_CONNECTION_STRING)")

        if _mongo_client is None:
            client_kwargs = dict(
                maxPoolSize=max_pool,
                minPoolSize=min_pool,
                retryWrites=True,
                serverSelectionTimeoutMS=30000,
                connectTimeoutMS=30000,
                socketTimeoutMS=30000,
            )
            if max_idle > 0:
                client_kwargs["maxIdleTimeMS"] = max_idle
            if wait_q_timeout > 0:
                client_kwargs["waitQueueTimeoutMS"] = wait_q_timeout
            _mongo_client = MongoClient(connection_string, **client_kwargs)
        client = _mongo_client
        # 서버 상태를 확인해 연결 성공 여부를 검증한다.
        client.server_info()

        # 성공적으로 연결되면, 전역 변수에 DB 객체를 저장합니다.
        _db_connection = client[db_name]

        # 서버 연결 수 정보를 함께 출력한다.
        try:
            status = client.admin.command("serverStatus")  # Atlas 환경에서는 clusterMonitor 권한 필요
            conn = status.get("connections", {}) if isinstance(status, dict) else {}
            current = conn.get("current")
            available = conn.get("available")
            total_created = conn.get("totalCreated")
            if current is not None:
                logger.debug(
                    "MongoDB 연결 성공 (connections: current=%s, available=%s, totalCreated=%s)",
                    current,
                    available,
                    total_created,
                )
            else:
                logger.debug("MongoDB에 성공적으로 연결되었습니다.")
        except Exception:
            # 권한 부족 등으로 serverStatus 실패 시, 기본 메시지만 기록
            logger.debug("MongoDB에 성공적으로 연결되었습니다.")
        return _db_connection
    except Exception as e:
        error_message = f"오류: MongoDB 연결에 실패했습니다: {e}"
        logger.error(error_message)
        return None
