"""사고팔기 주문 기록·회차 원장·지난 체결 캐시를 나무 계좌번호(account_no) 대신 계좌 ID 로 묶기 (1회성).

증권사가 나무·메리츠 둘이 되면서(메리츠는 계좌번호가 없다) 주문 경로를 계좌 ID 로 바꿨다.
새 키를 **더하기만** 한다 — 옛 키(account_no 필드·옛 _id 문서)는 새 코드 배포·동작 확인 뒤 따로 지운다.

- strategy_trade_orders: account_id 필드 추가
- strategy_trade_ledgers: `{account_id}:{ticker}` 문서 복사(account_id 필드)
- strategy_trade_day_fills: `{account_id}:{YYYYMMDD}` 문서 복사
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from utils.account_settings_store import load_account_docs  # noqa: E402
from utils.db_manager import get_db_connection  # noqa: E402
from utils.env import load_env_if_present  # noqa: E402


def main() -> int:
    load_env_if_present()
    db = get_db_connection()
    if db is None:
        raise RuntimeError("DB 연결에 실패했습니다.")
    by_no = {
        str(doc["broker_api"]["account_no"]): doc["account_id"]
        for doc in load_account_docs()
        if (doc.get("broker_api") or {}).get("account_no")
    }
    print(f"[migrate] 계좌번호 → 계좌 ID: {by_no}")

    unknown = set(db.strategy_trade_orders.distinct("account_no")) - set(by_no)
    unknown |= set(db.strategy_trade_ledgers.distinct("account_no")) - set(by_no)
    if unknown:
        print(f"[migrate] 계좌 설정에 없는 계좌번호가 있어 중단합니다: {sorted(unknown)}")
        return 1

    for account_no, account_id in by_no.items():
        result = db.strategy_trade_orders.update_many(
            {"account_no": account_no, "account_id": {"$exists": False}}, {"$set": {"account_id": account_id}}
        )
        print(f"[migrate] 주문 기록 {account_no} → {account_id}: {result.modified_count}건")

    ledgers = 0
    for doc in db.strategy_trade_ledgers.find({"account_no": {"$exists": True}}):
        account_id = by_no[doc["account_no"]]
        new = {key: value for key, value in doc.items() if key != "account_no"}
        new.update({"_id": f"{account_id}:{doc['ticker']}", "account_id": account_id})
        db.strategy_trade_ledgers.replace_one({"_id": new["_id"]}, new, upsert=True)
        ledgers += 1
    print(f"[migrate] 회차 원장 복사: {ledgers}건")

    fills = 0
    for doc in db.strategy_trade_day_fills.find({}):
        account_no, day = str(doc["_id"]).split(":", 1)
        if account_no not in by_no:
            continue
        new_id = f"{by_no[account_no]}:{day}"
        db.strategy_trade_day_fills.replace_one({"_id": new_id}, {**doc, "_id": new_id}, upsert=True)
        fills += 1
    print(f"[migrate] 지난 체결 캐시 복사: {fills}건")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
