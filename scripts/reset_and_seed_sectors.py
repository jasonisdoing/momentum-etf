import os
import sys
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.db_manager import get_db_connection, _get_stock_collection_names, save_sectors

# 사용자가 제공한 새로운 업종 목록
NEW_SECTORS = [
    "지수",
    "AI·반도체",
    "IT·소프트웨어",
    "금융",
    "고배당",
    "인컴·부동산",
    "에너지·원자재",
    "산업재·인프라",
    "의약·헬스케어",
    "소비재 (생활·필수재)",
    "농업·식량",
    "레저·서비스",
    "미디어·엔터테인먼트",
    "자동차·운송·물류",
    "친환경·신재생에너지",
    "방산·우주항공",
]

def reset_and_seed_sectors():
    """
    기존의 모든 업종을 삭제하고, 새로운 업종 목록으로 교체합니다.
    또한 모든 종목의 업종을 공백으로 초기화하여 재분류할 수 있도록 합니다.
    """
    db = get_db_connection()
    if db is None:
        print("DB에 연결할 수 없습니다.")
        return

    now = datetime.now()
    new_sector_docs = [{"name": name, "added_date": now} for name in NEW_SECTORS]

    print("기존 업종을 모두 삭제하고 새로운 업종 목록으로 교체합니다...")
    if save_sectors(new_sector_docs):
        print(f"-> {len(new_sector_docs)}개의 새로운 업종이 성공적으로 저장되었습니다.")
    else:
        print("오류: 새로운 업종 저장에 실패했습니다.")
        return

    stock_collections = _get_stock_collection_names(db)
    print("\n모든 종목의 업종을 초기화합니다...")
    for coll_name in stock_collections:
        db[coll_name].update_many({"is_deleted": {"$ne": True}}, {"$set": {"sector": "", "last_modified": now}})
        print(f"-> '{coll_name}' 컬렉션의 모든 종목 업종이 초기화되었습니다.")

    print("\n작업 완료. 이제 웹 앱에서 종목 정보를 다시 분류할 수 있습니다.")

if __name__ == "__main__":
    confirm = input("정말로 모든 업종을 삭제하고 새 목록으로 교체하시겠습니까? 모든 종목의 업종 분류가 초기화됩니다. (yes/no): ")
    if confirm.lower() == 'yes':
        reset_and_seed_sectors()
    else:
        print("작업이 취소되었습니다.")