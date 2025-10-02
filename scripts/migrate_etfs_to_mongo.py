import os
import sys
from datetime import datetime

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.db_manager import get_db_connection


def migrate_etfs_from_json_to_mongo(country: str):
    """
    data/stocks/{country}.json 파일 데이터를 읽어 MongoDB의 'stocks' 컬렉션으로 마이그레이션합니다.
    """
    # JSON 파일 로딩을 위해 기존 함수를 임시로 가져옵니다.
    # 이 스크립트 실행 후에는 이 함수는 더 이상 사용되지 않습니다.
    from utils.stock_list_io import _get_data_dir
    import json

    file_path = os.path.join(_get_data_dir(), "stocks", f"{country}.json")
    if not os.path.exists(file_path):
        print(f"'{file_path}' 파일이 없어 마이그레이션을 건너뜁니다.")
        return

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"JSON 파일 읽기 실패: {e}")
        return

    db = get_db_connection()
    if db is None:
        print("DB 연결 실패. 마이그레이션을 중단합니다.")
        return

    stocks_collection = db.stocks
    migrated_count = 0
    for category_block in data:
        category_name = category_block.get("category", "Uncategorized")
        for item in category_block.get("tickers", []):
            ticker = item.get("ticker")
            if not ticker:
                continue

            # DB에 이미 존재하는지 확인 (멱등성 보장)
            if stocks_collection.find_one({"ticker": ticker, "country": country}):
                continue

            name = item.get("name", "")
            doc = {
                "ticker": ticker,
                "name": name,
                "country": country,
                "category": category_name,
                "last_modified": datetime.utcnow(),
            }
            stocks_collection.insert_one(doc)
            migrated_count += 1
            print(f"  -> Migrated: {ticker} ({name})")

    print(f"\n{country.upper()} 국가의 {migrated_count}개 종목을 성공적으로 마이그레이션했습니다.")


if __name__ == "__main__":
    countries_to_migrate = ["kor", "aus"]
    for c in countries_to_migrate:
        print(f"\n--- {c.upper()} 마이그레이션 시작 ---")
        migrate_etfs_from_json_to_mongo(c)
    print("\n모든 마이그레이션 작업이 완료되었습니다.")
