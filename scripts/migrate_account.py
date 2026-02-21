import os
import sys

# 프로젝트 루트 경로를 sys.path에 추가 (utils 모듈을 로드하기 위함)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.db_manager import get_db_connection


def migrate_account(old_id: str, new_id: str):
    print(f"[{old_id}] -> [{new_id}] 마이그레이션을 시작합니다...")

    try:
        db = get_db_connection()
        if db is None:
            print("DB 연결 실패")
            return

        # 1. stock_meta 컬렉션 업데이트
        stock_meta_result = db.stock_meta.update_many({"account_id": old_id}, {"$set": {"account_id": new_id}})
        print(f"  - `stock_meta`: {stock_meta_result.modified_count}개의 도큐먼트가 변경되었습니다.")

        # 2. recommend_snap_v2 컬렉션 업데이트
        recommend_result = db.recommend_snap_v2.update_many({"account_id": old_id}, {"$set": {"account_id": new_id}})
        print(f"  - `recommend_snap_v2`: {recommend_result.modified_count}개의 도큐먼트가 변경되었습니다.")

        # 3. daily_prices_v2 컬렉션 업데이트 (계좌별 가격 캐시)
        # cache_utils.py에 따르면 리스트 필드가 아니라 metadata 딕셔너리에 account_id가 저장됨.
        prices_result = db.daily_prices_v2.update_many(
            {"metadata.account_id": old_id}, {"$set": {"metadata.account_id": new_id}}
        )
        print(f"  - `daily_prices_v2`: {prices_result.modified_count}개의 도큐먼트가 변경되었습니다.")

        print("마이그레이션이 성공적으로 완료되었습니다!")

    except Exception as e:
        print(f"마이그레이션 도중 에러가 발생했습니다: {e}")


if __name__ == "__main__":
    migrate_account("kor_us", "kor_pension")
