import os
import sys

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.db_manager import get_db_connection


def delete_soft_deleted_documents():
    db = get_db_connection()
    if db is None:
        print("MongoDB 연결에 실패했습니다. 스크립트를 실행할 수 없습니다.")
        return

    collections_to_check = ["trades", "daily_equities", "transactions"]

    print("이전에 soft delete 처리되었을 수 있는 데이터를 영구 삭제합니다.")
    print("-" * 50)

    for collection_name in collections_to_check:
        try:
            collection = db[collection_name]
            # is_deleted 필드가 True인 문서를 찾아서 삭제합니다.
            # is_deleted 필드가 없는 경우에도 오류 없이 작동합니다.
            result = collection.delete_many({"is_deleted": True})

            if result.deleted_count > 0:
                print(f"컬렉션 '{collection_name}': {result.deleted_count}개의 soft-deleted 문서 삭제 완료.")
            else:
                print(f"컬렉션 '{collection_name}': soft-deleted 문서가 없거나 이미 삭제되었습니다.")
        except Exception as e:
            print(f"컬렉션 '{collection_name}' 처리 중 오류 발생: {e}")

    print("-" * 50)
    print("soft-deleted 데이터 삭제 스크립트 실행 완료.")


if __name__ == "__main__":
    delete_soft_deleted_documents()
