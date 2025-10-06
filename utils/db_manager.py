import os
from datetime import datetime
import math
from typing import Any, Dict, Optional, Tuple

from bson import ObjectId
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
_mongo_client: Optional[MongoClient] = None
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
        db_name = os.environ.get("MONGO_DB_NAME") or getattr(
            global_settings, "MONGO_DB_NAME", "momentum_etf_db"
        )
        # 연결 풀 관련 환경 변수(선택 사항)를 반영한다.
        max_pool = int(os.environ.get("MONGO_DB_MAX_POOL_SIZE", "20"))
        min_pool = int(os.environ.get("MONGO_DB_MIN_POOL_SIZE", "0"))
        max_idle = int(os.environ.get("MONGO_DB_MAX_IDLE_TIME_MS", "0"))  # 0 = driver default
        wait_q_timeout = int(
            os.environ.get("MONGO_DB_WAIT_QUEUE_TIMEOUT_MS", "0")
        )  # 0 = driver default

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
                logger.info(
                    "MongoDB 연결 성공 (connections: current=%s, available=%s, totalCreated=%s)",
                    current,
                    available,
                    total_created,
                )
            else:
                logger.info("MongoDB에 성공적으로 연결되었습니다.")
        except Exception:
            # 권한 부족 등으로 serverStatus 실패 시, 기본 메시지만 기록
            logger.info("MongoDB에 성공적으로 연결되었습니다.")
        return _db_connection
    except Exception as e:
        error_message = f"오류: MongoDB 연결에 실패했습니다: {e}"
        logger.error(error_message)
        return None


def save_trade(trade_data: Dict) -> bool:
    """단일 거래 내역을 'trades' 컬렉션에 저장합니다."""
    db = get_db_connection()
    if db is None:
        return False

    try:
        trade_data["created_at"] = datetime.now()
        db.trades.insert_one(trade_data)
        account = str(trade_data.get("account") or "").upper()
        logger.info("%s 계정의 거래를 저장했습니다: %s", account or "UNKNOWN", trade_data)
        return True
    except Exception as e:
        logger.error("거래 내역 저장 중 오류 발생: %s", e)
        return False


def delete_trade_by_id(trade_id: str) -> bool:
    """ID를 기준으로 'trades' 컬렉션에서 단일 거래 내역을 삭제합니다."""
    db = get_db_connection()
    if db is None:
        return False

    try:
        obj_id = ObjectId(trade_id)
        result = db.trades.delete_one({"_id": obj_id})
        if result.deleted_count > 0:
            logger.info("거래 ID %s 를 삭제했습니다.", trade_id)
            return True
        else:
            logger.warning("삭제할 거래 ID %s 를 찾지 못했습니다.", trade_id)
            return False  # 삭제할 문서가 없으면 실패로 간주
    except Exception as e:
        logger.error("거래 내역 삭제 중 오류 발생: %s", e)
        return False


def update_trade_by_id(trade_id: str, update_data: Dict) -> bool:
    """ID를 기준으로 'trades' 컬렉션에서 단일 거래 내역을 업데이트합니다."""
    db = get_db_connection()
    if db is None:
        return False

    try:
        obj_id = ObjectId(trade_id)
        # 업데이트할 데이터에 수정 시간을 추가합니다.
        update_doc = {"$set": {**update_data, "updated_at": datetime.now()}}
        result = db.trades.update_one({"_id": obj_id}, update_doc)
        if result.modified_count > 0:
            logger.info("거래 ID %s 를 업데이트했습니다.", trade_id)
            return True
        else:
            logger.warning("업데이트할 거래 ID %s 를 찾지 못했거나 변경된 내용이 없습니다.", trade_id)
            return False
    except Exception as e:
        logger.error("거래 내역 업데이트 중 오류 발생: %s", e)
        return False


def infer_trade_from_state_change(
    q_old: float,
    avg_old: float,
    q_new: float,
    avg_new: float,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    보유 수량 및 평균 단가 변경 전/후 상태를 기반으로 발생한 거래(BUY 또는 SELL)를 추론합니다.

    Args:
        q_old: 기존 보유 수량
        avg_old: 기존 평균 단가
        q_new: 새로운 보유 수량
        avg_new: 새로운 평균 단가

    Returns:
        Tuple: (추론된 거래 정보 딕셔너리, 메시지)
        - 성공 시: ({"action": "BUY/SELL", ...}, None)
        - 수량 변경 없을 시: (None, "변경 사항이 없습니다.")
        - 오류 발생 시: (None, "오류 메시지")
    """
    delta_q = q_new - q_old

    if abs(delta_q) < 1e-9:  # 수량 변화가 거의 없으면 거래 없음
        return None, "변경 사항이 없습니다."

    try:
        if delta_q > 0:  # 수량 증가 -> BUY
            # 매수 가격 추론: p = (new_total_cost - old_total_cost) / delta_q
            numerator = (q_new * avg_new) - (q_old * avg_old)
            price = numerator / delta_q
            if math.isfinite(price) and price >= 0:  # 0원 매수도 허용 (증여 등)
                return {"action": "BUY", "shares": delta_q, "price": price}, None

        elif delta_q < 0:  # 수량 감소 -> SELL
            # 매도 가격 추론: p = (old_total_cost - new_total_cost) / abs(delta_q)
            # 매도 후 남은 주식의 평균 단가는 변하지 않아야 하므로, avg_new는 avg_old와 같아야 합니다.
            # 하지만 사용자가 다른 값을 입력할 수 있으므로, 이를 기반으로 매도 가격을 역산합니다.
            numerator = (q_old * avg_old) - (q_new * avg_new)
            price = numerator / abs(delta_q)
            if math.isfinite(price) and price >= 0:
                return {"action": "SELL", "shares": abs(delta_q), "price": price}, None

    except (ValueError, TypeError, ZeroDivisionError) as e:
        return None, f"거래 계산 중 오류가 발생했습니다: {e}"

    return None, "알 수 없는 오류로 거래를 추론할 수 없습니다."
