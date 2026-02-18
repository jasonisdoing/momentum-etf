import os
import sys

# 프로젝트 루트 경로 추가
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.db_manager import get_db_connection


def migrate_buckets():
    """모든 계좌의 stock_meta에 bucket 필드를 추가하고 기본값 1로 초기화합니다."""
    db = get_db_connection()
    collection = db["stock_meta"]

    # 1. bucket 필드가 없는 문서 찾기
    query = {"bucket": {"$exists": False}}
    update = {"$set": {"bucket": 1}}

    result = collection.update_many(query, update)

    print(f"[Migration] Matched: {result.matched_count}, Modified: {result.modified_count}")

    # 2. 검증
    sample = collection.find_one()
    print(f"[Verification] Sample doc: {sample}")


if __name__ == "__main__":
    print("Starting migration to 5-Bucket Strategy...")
    migrate_buckets()
    print("Migration completed.")
