"""사고팔기를 계좌 단위 설정으로 옮기고 옛 데이터를 정리한다 (1회성).

1. strategy_trade_settings.accounts.{계좌} = {pools, slack_enabled} — 전략이 쓰던 종목풀을 그 계좌에 붙이고
   (전략 순서대로), 슬랙은 옛 `slack_accounts` 값을 옮긴다. 그 뒤 `slack_accounts` 를 지운다.
2. pool_settings 의 사고팔기 사용 토글(STRATEGY_USE.strategy_trade)을 지운다 — 토글을 없앴다.
3. 계좌 ID 전환 때 남겨 둔 옛 키를 지운다 — 주문 기록의 account_no 필드·인덱스, 나무 계좌번호로 묶인
   옛 회차 원장·지난 체결 캐시 문서(파생 데이터라 백업하지 않는다).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from utils.account_settings_store import load_account_docs  # noqa: E402
from utils.db_manager import get_db_connection  # noqa: E402
from utils.env import load_env_if_present  # noqa: E402

SETTINGS_KEY = "strategy_trade_settings"


def _move_to_accounts(db) -> None:
    doc = db.system_config.find_one({"_id": SETTINGS_KEY}) or {}
    slack = doc.get("slack_accounts") or {}
    pools: dict[str, list[str]] = {}
    for entry in (doc.get("strategies") or {}).values():
        account_pools = pools.setdefault(str(entry["account_id"]), [])
        if entry["pool"] not in account_pools:
            account_pools.append(entry["pool"])
    for account_id in slack:
        pools.setdefault(str(account_id), [])
    accounts = {
        account_id: {"pools": account_pools, "slack_enabled": bool(slack.get(account_id, False))}
        for account_id, account_pools in pools.items()
    }
    db.system_config.update_one(
        {"_id": SETTINGS_KEY},
        {"$set": {f"accounts.{key}": value for key, value in accounts.items()}, "$unset": {"slack_accounts": ""}},
    )
    print(f"[migrate] 계좌별 사고팔기 설정: {accounts}")


def _drop_pool_toggle(db) -> None:
    result = db.pool_settings.update_many(
        {"STRATEGY_USE.strategy_trade": {"$exists": True}}, {"$unset": {"STRATEGY_USE.strategy_trade": ""}}
    )
    print(f"[migrate] 종목풀 사고팔기 토글 삭제: {result.modified_count}건")


def _drop_account_no_keys(db) -> None:
    account_nos = {
        str(doc["broker_api"]["account_no"])
        for doc in load_account_docs()
        if (doc.get("broker_api") or {}).get("account_no")
    }
    orders = db.strategy_trade_orders
    missing = orders.count_documents({"account_id": {"$exists": False}})
    if missing:
        raise RuntimeError(f"account_id 가 없는 주문 기록이 {missing}건 있어 account_no 를 지우지 않습니다.")
    result = orders.update_many({"account_no": {"$exists": True}}, {"$unset": {"account_no": ""}})
    print(f"[migrate] 주문 기록 account_no 삭제: {result.modified_count}건")
    if "account_no_1_created_at_-1" in orders.index_information():
        orders.drop_index("account_no_1_created_at_-1")
        print("[migrate] 주문 기록 account_no 인덱스 삭제")
    result = db.strategy_trade_ledgers.delete_many({"account_no": {"$exists": True}})
    print(f"[migrate] 옛 회차 원장 삭제: {result.deleted_count}건")
    old_ids = [
        doc["_id"]
        for doc in db.strategy_trade_day_fills.find({}, {"_id": 1})
        if str(doc["_id"]).split(":", 1)[0] in account_nos
    ]
    result = db.strategy_trade_day_fills.delete_many({"_id": {"$in": old_ids}})
    print(f"[migrate] 옛 지난 체결 캐시 삭제: {result.deleted_count}건")


def main() -> int:
    load_env_if_present()
    db = get_db_connection()
    if db is None:
        raise RuntimeError("DB 연결에 실패했습니다.")
    _move_to_accounts(db)
    _drop_pool_toggle(db)
    _drop_account_no_keys(db)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
