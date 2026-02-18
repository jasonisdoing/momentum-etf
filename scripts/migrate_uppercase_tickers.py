import os
import sys


def migrate_tickers_to_uppercase():
    db = get_db_connection()
    if db is None:
        print("MongoDB Connection Failed")
        return

    coll = db["stock_meta"]

    # 1. 소문자 티커 찾기 (정규식 사용)
    # [a-z]가 포함된 티커 검색
    cursor = coll.find({"ticker": {"$regex": "[a-z]"}})

    count = 0
    migrated_count = 0
    error_count = 0

    print("Checking for lowercase tickers...")

    for doc in cursor:
        count += 1
        old_ticker = doc["ticker"]
        new_ticker = old_ticker.upper()

        if old_ticker == new_ticker:
            continue

        account_id = doc["account_id"]
        print(f"Migrating {account_id}: {old_ticker} -> {new_ticker}")

        try:
            # 1. Check if uppercase ticker already exists
            existing = coll.find_one({"account_id": account_id, "ticker": new_ticker})

            if existing:
                # If already exists, we might need to merge or just delete the old one.
                # Here, we'll delete the old one if the new one is active.
                # If the new one is deleted, we might want to "undelete" it and update properties.
                # For safety, let's just delete the lowercase one if uppercase exists.
                print(f"  Uppercase {new_ticker} already exists. Removing lowercase entry.")
                coll.delete_one({"_id": doc["_id"]})
            else:
                # 2. Update the ticker field
                coll.update_one({"_id": doc["_id"]}, {"$set": {"ticker": new_ticker}})

            migrated_count += 1

        except Exception as e:
            print(f"  Error migrating {old_ticker}: {e}")
            error_count += 1

    print("Migration completed.")
    print(f"Total checked: {count}")
    print(f"Migrated: {migrated_count}")
    print(f"Errors: {error_count}")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    sys.path.append(project_root)

    from utils.db_manager import get_db_connection

    migrate_tickers_to_uppercase()
