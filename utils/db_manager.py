import ipaddress
import os
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import quote_plus

from dotenv import load_dotenv
from pymongo import MongoClient

from utils.logger import get_app_logger

load_dotenv()

_db_connection = None
_mongo_client: MongoClient | None = None
_last_connect_failure_at: float = 0.0
_last_connect_error_detail: str | None = None
_detected_mongo_host_hint: str | None = None
_last_scan_at: float = 0.0
_CONNECT_FAILURE_COOLDOWN_SEC = 5.0
_SCAN_COOLDOWN_SEC = 30.0
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


def _build_connection_string(host: str) -> str:
    user = quote_plus(_get_required_env("MONGO_DB_USER"))
    password = quote_plus(_get_required_env("MONGO_DB_PASSWORD"))
    auth_source = _get_required_env("MONGO_DB_AUTH_SOURCE")
    return f"mongodb://{user}:{password}@{host}/?authSource={quote_plus(auth_source)}"


def resolve_connection_string() -> str:
    """외부 도구(mongodump 등)용 MongoDB 접속 URI 를 반환한다."""
    configured_host = (os.environ.get("MONGO_DB_HOST") or "").strip()
    if configured_host:
        return _build_connection_string(configured_host)
    db = get_db_connection()
    if db is None or db.client.address is None:
        raise RuntimeError("MongoDB 호스트를 찾지 못했습니다.")
    host, port = db.client.address
    return _build_connection_string(f"{host}:{port}")


def _local_lan_prefix() -> str | None:
    """현재 머신의 사설 IPv4 /24 prefix(예: '192.168.68')를 반환한다."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("192.0.2.1", 80))
        ip = sock.getsockname()[0]
    except OSError:
        return None
    finally:
        sock.close()
    try:
        ip_obj = ipaddress.ip_address(ip)
    except ValueError:
        return None
    if not ip_obj.is_private or ip_obj.is_loopback:
        return None
    return ".".join(ip.split(".")[:3])


def _scan_lan_mongo_hosts(prefix: str, port: int = 27017, *, exclude_ip: str | None = None) -> list[str]:
    """같은 /24 대역에서 MongoDB 포트가 열린 호스트 목록(`ip:port`)을 찾는다."""

    def _probe(ip: str) -> str | None:
        if exclude_ip and ip == exclude_ip:
            return None
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.35)
        try:
            if sock.connect_ex((ip, port)) == 0:
                return f"{ip}:{port}"
        except OSError:
            pass
        finally:
            sock.close()
        return None

    with ThreadPoolExecutor(max_workers=64) as executor:
        return [res for res in executor.map(_probe, [f"{prefix}.{i}" for i in range(1, 255)]) if res]


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


def _detect_lan_mongo_host(configured_host: str) -> str | None:
    """로컬 환경에서 사설 IP 접속 실패 시 같은 /24 대역에서 27017이 열린 다른 호스트를 찾는다."""
    global _last_scan_at, _detected_mongo_host_hint

    if (os.environ.get("APP_TYPE") or "").strip().lower() != "local":
        return None
    host_only = configured_host.split(":", 1)[0].strip()
    port = (
        int(configured_host.split(":", 1)[1])
        if ":" in configured_host and configured_host.split(":", 1)[1].isdigit()
        else 27017
    )
    try:
        ip_obj = ipaddress.ip_address(host_only)
    except ValueError:
        return None
    if not ip_obj.is_private:
        return None

    now = time.monotonic()
    if now - _last_scan_at < _SCAN_COOLDOWN_SEC:
        return _detected_mongo_host_hint
    _last_scan_at = now

    prefix = ".".join(host_only.split(".")[:3])
    found = _scan_lan_mongo_hosts(prefix, port, exclude_ip=host_only)
    _detected_mongo_host_hint = found[0] if found else None
    return _detected_mongo_host_hint


def _discover_local_mongo_hosts() -> list[str]:
    """`MONGO_DB_HOST` 미설정 시 로컬 머신이면 LAN 스캔, 서버면 127.0.0.1:27017 을 반환한다."""
    if (os.environ.get("APP_TYPE") or "").strip().lower() != "local":
        return ["127.0.0.1:27017"]
    prefix = _local_lan_prefix()
    if not prefix:
        return []
    return _scan_lan_mongo_hosts(prefix, 27017)


def get_db_connection_diagnostic() -> dict[str, str | None]:
    """최근 MongoDB 연결 실패 원인과 로컬 IP 변경 감지 힌트를 반환한다."""
    hint = _detected_mongo_host_hint
    detail = _last_connect_error_detail or "DB 연결 실패"
    if hint:
        detail = f"{detail} (감지된 호스트: {hint})"
    return {
        "detail": detail,
        "suggested_env": f"MONGO_DB_HOST={hint}" if hint else None,
    }


def get_db_connection():
    """
    MongoDB 클라이언트 연결을 생성하고, 전역 변수에 저장하여 재사용합니다.
    """
    global \
        _db_connection, \
        _mongo_client, \
        _last_connect_failure_at, \
        _last_connect_error_detail, \
        _detected_mongo_host_hint

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

        configured_host = (os.environ.get("MONGO_DB_HOST") or "").strip()
        try:
            db_name = _get_required_env("MONGO_DB_NAME")
            candidates = [configured_host] if configured_host else _discover_local_mongo_hosts()
        except RuntimeError as exc:
            _last_connect_failure_at = time.monotonic()
            _last_connect_error_detail = str(exc)
            logger.error("오류: MongoDB 연결 정보가 설정되지 않았습니다. %s", exc)
            return None

        if not candidates:
            _last_connect_failure_at = time.monotonic()
            _last_connect_error_detail = "내부망에서 MongoDB(27017) 호스트를 찾지 못했습니다."
            logger.error("오류: %s", _last_connect_error_detail)
            return None

        last_exc: Exception | None = None
        for host in candidates:
            try:
                connection_string = _build_connection_string(host)
                client = _build_client(connection_string)
                client.admin.command("ping")
                _mongo_client = client
                _db_connection = client[db_name]
                _last_connect_failure_at = 0.0
                _last_connect_error_detail = None
                _detected_mongo_host_hint = None
                logger.info("MongoDB 연결 완료 (%s)", host)
                return _db_connection
            except Exception as exc:
                last_exc = exc
                _reset_connection_unlocked()

        _last_connect_failure_at = time.monotonic()
        _last_connect_error_detail = str(last_exc) if last_exc else "DB 연결 실패"
        if configured_host:
            _detect_lan_mongo_host(configured_host)
        logger.error("오류: MongoDB 연결에 실패했습니다: %s", _last_connect_error_detail)
        return None
