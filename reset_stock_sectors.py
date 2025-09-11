import os
import sys
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.db_manager import get_db_connection, _get_stock_collection_names


def reset_all_stock_sectors():
    """
    모든 종목 컬렉션에 있는 모든 종목의 'sector' 필드를 공백으로 초기화합니다.
    또한 'last_modified' 타임스탬프를 업데이트하여 UI에서 쉽게 찾을 수 있도록 합니다.
    """
    db = get_db_connection()
    if db is None:
        print("DB에 연결할 수 없습니다.")
        return

    stock_collections = _get_stock_collection_names(db)
    if not stock_collections:
        print("처리할 종목 컬렉션이 없습니다.")
        return

    now = datetime.now()
    total_modified_count = 0

    print("모든 종목의 업종을 초기화합니다...")

    for coll_name in stock_collections:
        print(f"-> '{coll_name}' 컬렉션 처리 중...")
        result = db[coll_name].update_many(
            {"is_deleted": {"$ne": True}},
            {"$set": {"sector": "", "last_modified": now}}
        )
        modified_count = result.modified_count
        total_modified_count += modified_count
        print(f"   {modified_count}개 종목의 업종이 초기화되었습니다.")

    print(f"\n완료: 총 {total_modified_count}개 종목의 업종이 초기화되었습니다.")
    print("이제 웹 앱에서 종목 정보를 다시 분류할 수 있습니다.")

if __name__ == "__main__":
    # 사용자에게 확인을 받습니다.
    confirm = input("정말로 모든 종목의 업종을 초기화하시겠습니까? 이 작업은 되돌릴 수 없습니다. (yes/no): ")
    if confirm.lower() == 'yes':
        reset_all_stock_sectors()
    else:
        print("작업이 취소되었습니다.")