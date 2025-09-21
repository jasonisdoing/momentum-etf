import argparse
import os
import sys

from pymongo.errors import OperationFailure

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.db_manager import get_db_connection


def main():
    """
    'status_reports' 컬렉션의 데이터를 새로운 'signals' 컬렉션으로 이전합니다.
    """
    parser = argparse.ArgumentParser(
        description="MongoDB 'status_reports'를 'signals' 컬렉션으로 이전합니다."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="실제 DB에 데이터를 쓰지 않고 마이그레이션을 시뮬레이션합니다.",
    )
    parser.add_argument(
        "--drop-old",
        action="store_true",
        help="성공적인 마이그레이션 후 기존 'status_reports' 컬렉션을 삭제합니다. (--dry-run과 함께 사용할 수 없습니다)",
    )
    args = parser.parse_args()

    if args.dry_run:
        print("--- DRY RUN 모드 ---")
        print("데이터베이스에 어떠한 변경도 적용되지 않습니다.")

    db = get_db_connection()
    if db is None:
        print("오류: 데이터베이스에 연결할 수 없습니다.")
        return

    old_collection_name = "status_reports"
    new_collection_name = "signals"

    old_collection = db[old_collection_name]
    new_collection = db[new_collection_name]

    try:
        count = old_collection.count_documents({})
        if count == 0:
            print(
                f"'{old_collection_name}' 컬렉션이 비어있거나 존재하지 않습니다. 이전할 데이터가 없습니다."
            )
            return
    except OperationFailure as e:
        print(f"'{old_collection_name}' 컬렉션 확인 중 오류: {e}")
        print("컬렉션이 존재하지 않을 수 있습니다. 이 경우 이전할 데이터가 없습니다.")
        return

    print(
        f"'{old_collection_name}'에서 {count}개의 문서를 찾았습니다. '{new_collection_name}'으로 이전을 시작합니다..."
    )

    migrated_count = 0
    updated_count = 0
    for doc in old_collection.find():
        try:
            # 각 문서를 고유하게 식별하기 위한 쿼리 정의
            query = {
                "country": doc.get("country"),
                "account": doc.get("account"),
                "date": doc.get("date"),
            }

            # MongoDB가 새로운 _id를 생성하도록 기존 _id 필드를 제거
            doc.pop("_id", None)

            if args.dry_run:
                print(
                    f"  [DRY RUN] {query['country']}/{query['account']} ({query['date'].strftime('%Y-%m-%d')}) 문서를 이전할 것입니다."
                )
            else:
                # upsert=True를 사용하여 문서를 삽입하거나 업데이트
                result = new_collection.update_one(query, {"$set": doc}, upsert=True)
                if result.upserted_id:
                    migrated_count += 1
                elif result.matched_count > 0:
                    updated_count += 1

        except Exception as e:
            print(f"오류: 문서 {doc.get('_id', 'N/A')} 이전 중 오류 발생: {e}")

    print("\n--- 이전 요약 ---")
    print(f"처리된 총 문서 수: {count}")
    if not args.dry_run:
        print(f"새로 이전된 문서 수: {migrated_count}")
        print(f"기존 문서 업데이트 수: {updated_count}")

    if args.drop_old and not args.dry_run:
        print(f"\n요청에 따라 기존 '{old_collection_name}' 컬렉션을 삭제합니다...")
        old_collection.drop()
        print(f"'{old_collection_name}' 컬렉션을 성공적으로 삭제했습니다.")

    print("\n마이그레이션 완료.")


if __name__ == "__main__":
    main()
