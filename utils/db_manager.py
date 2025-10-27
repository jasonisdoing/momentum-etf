import os
from datetime import datetime
import math
from typing import Any, Dict, List, Optional, Tuple

from bson import ObjectId
from dotenv import load_dotenv
from pymongo import DESCENDING, MongoClient

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
        connection_string = os.environ.get("MONGO_DB_CONNECTION_STRING") or getattr(global_settings, "MONGO_DB_CONNECTION_STRING", None)
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


def insert_trade_event(
    *,
    account_id: str,
    ticker: str,
    name: str | None = None,
    action: str,
    executed_at: datetime,
    memo: str | None,
    created_by: str,
    source: str = "streamlit",
    country_code: str | None = None,
) -> str:
    """`trades` 컬렉션에 매수/매도 이벤트를 기록합니다."""

    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결을 초기화할 수 없습니다.")

    account_norm = (account_id or "").strip().lower()
    country_norm = (country_code or account_norm).strip().lower()
    if not account_norm:
        raise ValueError("계정 ID를 지정해야 합니다.")

    doc = {
        "account": account_norm,
        "country_code": country_norm,
        "ticker": ticker,
        "action": action.upper(),
        "name": (name or "").strip(),
        "executed_at": executed_at,
        "memo": memo or "",
        "created_by": created_by,
        "source": source,
        "created_at": datetime.utcnow(),
    }

    if not doc["name"] and action.upper() == "SELL":
        doc.pop("name")

    result = db.trades.insert_one(doc)
    return str(result.inserted_id)


def migrate_account_id(old_account_id: str, new_account_id: str) -> dict[str, int]:
    """`trades` 컬렉션에서 계정 ID를 일괄 변경합니다."""

    old_norm = (old_account_id or "").strip().lower()
    new_norm = (new_account_id or "").strip().lower()
    if not old_norm or not new_norm:
        raise ValueError("old_account_id와 new_account_id는 비어 있을 수 없습니다.")
    if old_norm == new_norm:
        return {"matched": 0, "modified": 0}

    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결을 초기화할 수 없습니다.")

    result_account = db.trades.update_many(
        {"account": old_norm},
        {"$set": {"account": new_norm}},
    )

    return {
        "matched": int(result_account.matched_count),
        "modified": int(result_account.modified_count),
    }


def delete_account_trades(account_id: str) -> dict[str, int]:
    """지정한 계정 ID의 거래 이력을 모두 삭제합니다."""

    account_norm = (account_id or "").strip().lower()
    if not account_norm:
        raise ValueError("account_id는 비어 있을 수 없습니다.")

    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결을 초기화할 수 없습니다.")

    result_account = db.trades.delete_many({"account": account_norm})
    deleted_count = int(result_account.deleted_count)

    logger.info(
        "trades 컬렉션에서 계정 '%s' 데이터를 삭제했습니다. deleted=%d",
        account_norm,
        deleted_count,
    )

    return {
        "deleted": deleted_count,
    }


def fetch_recent_trades(account_id: str | None = None, *, limit: int = 100, include_deleted: bool = False) -> List[dict[str, Any]]:
    """최근 트레이드 목록을 반환합니다."""
    db = get_db_connection()
    if db is None:
        return []

    query: dict[str, Any] = {}
    if not include_deleted:
        query["deleted_at"] = {"$exists": False}
    else:
        query["deleted_at"] = {"$exists": True}

    if account_id:
        query["account"] = account_id.strip().lower()

    cursor = db.trades.find(query).sort([("executed_at", DESCENDING), ("_id", DESCENDING)]).limit(int(limit))

    trades: List[dict[str, Any]] = []
    for doc in cursor:
        trades.append(
            {
                "id": str(doc.get("_id")),
                "account": str(doc.get("account") or ""),
                "country_code": str(doc.get("country_code") or ""),
                "ticker": str(doc.get("ticker") or ""),
                "action": str(doc.get("action") or ""),
                "name": str(doc.get("name") or ""),
                "executed_at": doc.get("executed_at"),
                "memo": doc.get("memo", ""),
                "created_by": doc.get("created_by"),
                "deleted_at": doc.get("deleted_at"),
                "is_deleted": "deleted_at" in doc,
            }
        )
    return trades


def list_open_positions(account_id: str) -> List[dict[str, Any]]:
    """특정 계정의 최신 매수 상태(미매도) 종목 목록을 반환합니다."""

    account_norm = (account_id or "").strip().lower()
    if not account_norm:
        return []

    db = get_db_connection()
    if db is None:
        return []

    # 모든 거래를 가져와서 Python에서 처리
    try:
        all_trades = list(
            db.trades.find(
                {
                    "account": account_norm,
                    "ticker": {"$ne": None},
                    "deleted_at": {"$exists": False},
                },
                projection={"ticker": 1, "action": 1, "executed_at": 1, "created_at": 1, "name": 1, "memo": 1, "_id": 1},
            ).sort([("ticker", 1), ("executed_at", 1), ("created_at", 1), ("_id", 1)])
        )
    except Exception:
        return []

    # 티커별로 마지막 거래 찾기
    from collections import defaultdict

    ticker_trades = defaultdict(list)
    for trade in all_trades:
        ticker = str(trade.get("ticker") or "").upper()
        if ticker:
            ticker_trades[ticker].append(trade)

    holdings: List[dict[str, Any]] = []
    for ticker, trades in ticker_trades.items():
        if not trades:
            continue
        # 마지막 거래 (이미 정렬됨)
        last_trade = trades[-1]
        last_action = str(last_trade.get("action") or "").upper()

        # 마지막 거래가 BUY인 경우만 포함
        if last_action == "BUY":
            holdings.append(
                {
                    "id": str(last_trade.get("_id")) if last_trade.get("_id") else "",
                    "ticker": ticker,
                    "last_action": last_action,
                    "executed_at": last_trade.get("executed_at"),
                    "name": str(last_trade.get("name", "")),
                    "memo": last_trade.get("memo", ""),
                }
            )

    # 티커 순으로 정렬
    holdings.sort(key=lambda x: x.get("ticker", ""))
    return holdings


def update_trade_event(
    trade_id: str,
    *,
    account_id: Optional[str] = None,
    ticker: Optional[str] = None,
    action: Optional[str] = None,
    executed_at: Optional[datetime] = None,
    memo: Optional[str] = None,
) -> bool:
    """트레이드 문서를 업데이트합니다."""

    trade_id = (trade_id or "").strip()
    if not trade_id:
        return False

    db = get_db_connection()
    if db is None:
        return False

    try:
        object_id = ObjectId(trade_id)
    except Exception:
        return False

    fields: dict[str, Any] = {}
    if account_id is not None:
        fields["account"] = account_id.strip().lower()
    if ticker is not None:
        fields["ticker"] = ticker.strip()
    if action is not None:
        fields["action"] = action.upper()
    if executed_at is not None:
        fields["executed_at"] = executed_at
    if memo is not None:
        fields["memo"] = memo

    if not fields:
        return False

    result = db.trades.update_one({"_id": object_id}, {"$set": fields})
    return result.modified_count > 0


def soft_delete_trade(trade_id: str) -> bool:
    """트레이드 문서를 소프트 삭제합니다."""
    db = get_db_connection()
    if not db:
        return False

    try:
        result = db.trades.update_one(
            {"_id": ObjectId(trade_id)},
            {"$set": {"deleted_at": datetime.utcnow()}},
        )
        return result.modified_count > 0
    except Exception as e:
        logger.error("소프트 삭제 실패 (trade_id=%s): %s", trade_id, e)
        return False


def delete_trade(trade_id: str) -> bool:
    """트레이드 문서를 완전히 삭제합니다.

    Args:
        trade_id: 삭제할 트레이드의 ID

    Returns:
        삭제 성공 여부 (True/False)
    """
    db = get_db_connection()
    if db is None:
        return False

    try:
        result = db.trades.delete_one({"_id": ObjectId(trade_id)})
        return result.deleted_count > 0
    except Exception as e:
        logger.error("거래 삭제 실패 (trade_id=%s): %s", trade_id, e)
        return False
