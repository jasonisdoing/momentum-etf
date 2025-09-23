"""자본 추가 및 현금 인출('transactions' 컬렉션) 관련 DB 작업을 처리하는 모듈입니다."""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from bson import ObjectId
from pymongo import DESCENDING

from utils.db_manager import get_db_connection


def get_transactions_collection():
    """'transactions' 컬렉션 객체를 반환합니다."""
    db = get_db_connection()
    if db is None:
        raise ConnectionError("MongoDB에 연결할 수 없습니다.")
    return db.transactions


def save_transaction(transaction_data: Dict[str, Any]) -> bool:
    """거래(자본추가/현금인출) 내역을 'transactions' 컬렉션에 저장합니다."""
    if not all(k in transaction_data for k in ["country", "account", "date", "type", "amount"]):
        logging.error("Transaction data is missing required fields.")
        return False
    try:
        collection = get_transactions_collection()
        transaction_data["updated_at"] = datetime.now()
        transaction_data["is_deleted"] = False
        result = collection.insert_one(transaction_data)
        return result.acknowledged
    except Exception as e:
        logging.error(f"Failed to save transaction: {e}", exc_info=True)
        return False


def get_all_transactions(
    country: str,
    account: str,
    transaction_type: Optional[str] = None,
    include_deleted: bool = False,
) -> List[Dict[str, Any]]:
    """특정 계좌의 모든 거래(자본추가/현금인출) 내역을 조회합니다."""
    try:
        collection = get_transactions_collection()
        query = {"country": country, "account": account}
        if transaction_type:
            query["type"] = transaction_type
        if not include_deleted:
            query["is_deleted"] = {"$ne": True}

        transactions = []
        for t in collection.find(query).sort("date", DESCENDING):
            t["id"] = str(t["_id"])
            transactions.append(t)
        return transactions
    except Exception as e:
        logging.error(f"Failed to get transactions for {country}/{account}: {e}", exc_info=True)
        return []


def get_transactions_up_to_date(
    country: str, account: str, base_date: datetime, transaction_type: str
) -> List[Dict[str, Any]]:
    """특정 날짜까지의 특정 유형의 거래 내역을 조회합니다."""
    try:
        collection = get_transactions_collection()
        query = {
            "country": country,
            "account": account,
            "type": transaction_type,
            "date": {"$lte": base_date},
            "is_deleted": {"$ne": True},
        }
        return list(collection.find(query))
    except Exception as e:
        logging.error(
            f"Failed to get {transaction_type} for {country}/{account}: {e}", exc_info=True
        )
        return []


def update_transaction_by_id(transaction_id: str, update_data: Dict[str, Any]) -> bool:
    """ID로 거래(자본/인출)를 업데이트합니다."""
    try:
        collection = get_transactions_collection()
        update_data["updated_at"] = datetime.now()
        result = collection.update_one({"_id": ObjectId(transaction_id)}, {"$set": update_data})
        return result.modified_count > 0
    except Exception as e:
        logging.error(f"Failed to update transaction {transaction_id}: {e}", exc_info=True)
        return False


def delete_transaction_by_id(transaction_id: str) -> bool:
    """ID로 거래(자본/인출)를 논리적으로 삭제합니다."""
    return update_transaction_by_id(transaction_id, {"is_deleted": True})


def restore_transaction_by_id(transaction_id: str) -> bool:
    """ID로 거래(자본/인출)를 복구합니다."""
    return update_transaction_by_id(transaction_id, {"is_deleted": False})
