import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from bson import ObjectId
import pandas as pd
from pymongo import MongoClient, DESCENDING

import settings as global_settings


# --- 전역 변수로 DB 연결 관리 ---
_db_connection = None

def get_db_connection():
    """
    MongoDB 클라이언트 연결을 생성하고, 전역 변수에 저장하여 재사용합니다.
    """
    global _db_connection

    # 이미 연결이 설정되어 있으면, 기존 연결을 반환합니다.
    if _db_connection is not None:
        return _db_connection

    try:
        connection_string = os.environ.get("MONGO_DB_CONNECTION_STRING") or getattr(
            global_settings, "MONGO_DB_CONNECTION_STRING", None
        )
        db_name = os.environ.get("MONGO_DB_NAME") or getattr(
            global_settings, "MONGO_DB_NAME", "momentum_pilot_db"
        )

        if not connection_string:
            raise ValueError("MongoDB 연결 문자열이 설정되지 않았습니다. (MONGO_DB_CONNECTION_STRING)")

        client = MongoClient(connection_string)
        # 서버에 연결하여 연결 성공 여부 확인
        client.server_info()
        
        # 성공적으로 연결되면, 전역 변수에 DB 객체를 저장합니다.
        _db_connection = client[db_name]
        print("-> MongoDB에 성공적으로 연결되었습니다.")
        return _db_connection
    except Exception as e:
        print(f"오류: MongoDB 연결에 실패했습니다: {e}")
        return None


def get_portfolio_snapshot(
    country: str, date_str: Optional[str] = None
) -> Optional[Dict]:
    """
    'trades'와 'daily_equities' 컬렉션에서 데이터를 재구성하여 특정 날짜의 포트폴리오 스냅샷을 생성합니다.
    """
    db = get_db_connection()
    if db is None:
        return None

    # 1. 대상 날짜 결정
    target_date = None
    if date_str:
        try:
            target_date = pd.to_datetime(date_str).to_pydatetime()
        except ValueError:
            print(f"오류: 잘못된 날짜 형식입니다: {date_str}")
            return None
    else:
        # daily_equities에서 가장 최근 날짜를 찾습니다.
        latest_equity = db.daily_equities.find_one({"country": country, "is_deleted": {"$ne": True}}, sort=[("date", DESCENDING)])
        if latest_equity:
            target_date = latest_equity["date"]

    if not target_date:
        # 거래 내역은 있지만 평가금액이 없을 수도 있으므로, trades에서 날짜를 찾아봅니다.
        latest_trade = db.trades.find_one({"country": country, "is_deleted": {"$ne": True}}, sort=[("date", DESCENDING)])
        if latest_trade:
            target_date = latest_trade["date"]
        else:
            return None # 데이터가 전혀 없음

    # 2. 대상 날짜의 총 평가금액 조회
    equity_data = db.daily_equities.find_one({"country": country, "date": target_date, "is_deleted": {"$ne": True}})
    # 기본값 초기화
    is_equity_stale = False
    
    # 해당 날짜의 평가금액이 유효한지 확인합니다.
    is_equity_valid = equity_data and equity_data.get("total_equity", 0) > 0

    # 평가금액이 없거나 0이면, 가장 최근의 유효한 평가금액을 찾습니다.
    if not is_equity_valid:
        latest_valid_equity = db.daily_equities.find_one(
            {"country": country, "date": {"$lt": target_date}, "total_equity": {"$gt": 0}, "is_deleted": {"$ne": True}},
            sort=[("date", DESCENDING)]
        )
        if latest_valid_equity:
            equity_data = latest_valid_equity

    # 최종적으로 사용되는 평가금액의 날짜를 결정합니다.
    equity_date = equity_data.get("date") if equity_data else None
    if equity_date and equity_date != target_date:
        is_equity_stale = True

    total_equity = equity_data.get("total_equity", 0) if equity_data else 0

    # 3. 'trades' 컬렉션에서 보유 종목 재구성
    # 날짜 오름차순, 그리고 같은 날짜 내에서는 생성 순서(ObjectId) 오름차순으로 정렬합니다.
    # 이를 통해 동일한 날짜에 발생한 매도 후 매수 등의 거래 순서를 정확히 반영합니다.
    trades_cursor = db.trades.find(
        {"country": country, "date": {"$lte": target_date}, "is_deleted": {"$ne": True}}
    ).sort(
        [("date", 1), ("_id", 1)]
    )
    
    holdings_agg = {}
    for trade in trades_cursor:
        ticker = trade["ticker"]
        if ticker not in holdings_agg:
            holdings_agg[ticker] = {"shares": 0.0, "total_cost": 0.0, "name": trade.get("name", "")}
        
        if trade["action"] == "BUY":
            holdings_agg[ticker]["shares"] += trade["shares"]
            holdings_agg[ticker]["total_cost"] += trade["shares"] * trade["price"]
        elif trade["action"] == "SELL":
            original_shares = holdings_agg[ticker]["shares"]
            if original_shares > 0:
                cost_per_share = holdings_agg[ticker]["total_cost"] / original_shares
                holdings_agg[ticker]["total_cost"] -= trade["shares"] * cost_per_share
            holdings_agg[ticker]["shares"] -= trade["shares"]

    holdings_list = []
    for ticker, data in holdings_agg.items():
        if data["shares"] > 0:
            avg_cost = data["total_cost"] / data["shares"] if data["shares"] > 0 else 0
            holdings_list.append({
                "ticker": ticker, "name": data["name"],
                "shares": data["shares"], "avg_cost": avg_cost
            })

    # 4. 최종 스냅샷 조립
    snapshot = {
        "date": target_date, "country": country,
        "total_equity": total_equity, "holdings": holdings_list,
        "is_equity_stale": is_equity_stale
    }
    # 'equity_date'를 항상 포함하여 헤더에서 비교/표시가 가능하도록 합니다.
    if equity_date:
        snapshot["equity_date"] = equity_date

    if country == "aus" and equity_data and "international_shares" in equity_data:
        snapshot["international_shares"] = equity_data["international_shares"]

    return snapshot


def get_previous_portfolio_snapshot(
    country: str, as_of_date: datetime
) -> Optional[Dict]:
    """
    주어진 날짜 이전의 가장 최근 포트폴리오 스냅샷을 가져옵니다.
    """
    db = get_db_connection()
    if db is None:
        return None

    # 'daily_equities'와 'trades'에서 as_of_date 이전의 날짜들을 찾습니다.
    equity_dates = db.daily_equities.distinct("date", {"country": country, "date": {"$lt": as_of_date}, "is_deleted": {"$ne": True}})
    trade_dates = db.trades.distinct("date", {"country": country, "date": {"$lt": as_of_date}, "is_deleted": {"$ne": True}})
    
    all_prev_dates = set(equity_dates).union(set(trade_dates))
    
    if not all_prev_dates:
        return None
        
    # 가장 최근 날짜를 찾습니다.
    prev_date = max(all_prev_dates)
    prev_date_str = prev_date.strftime("%Y-%m-%d")
    
    return get_portfolio_snapshot(country, prev_date_str)

def get_available_snapshot_dates(country: str, as_of_date: Optional[datetime] = None, include_as_of_date: bool = False) -> List[str]:
    """
    지정된 국가에 대해 DB에 저장된 모든 스냅샷의 날짜 목록을 반환합니다.
    'daily_equities'와 'trades' 양쪽의 날짜를 모두 고려합니다.
    """
    db = get_db_connection()
    if db is None:
        return []

    query = {"country": country}
    if as_of_date:
        query["date"] = {"$lte" if include_as_of_date else "$lt": as_of_date}

    equity_dates = set(db.daily_equities.distinct("date", {**query, "is_deleted": {"$ne": True}}))
    trade_dates = set(db.trades.distinct("date", {**query, "is_deleted": {"$ne": True}}))
    
    all_dates = equity_dates.union(trade_dates)

    # 날짜를 'YYYY-MM-DD' 형식의 문자열로 변환하고 내림차순으로 정렬합니다.
    sorted_dates = sorted([d.strftime("%Y-%m-%d") for d in all_dates], reverse=True)
    return sorted_dates


def get_app_settings(country: str) -> Optional[Dict]:
    """지정된 국가의 앱 설정을 DB에서 가져옵니다."""
    db = get_db_connection()
    if db is None:
        return None
    
    settings = db.app_settings.find_one({"country": country})
    if settings:
        settings.pop("_id", None)
    return settings


def save_app_settings(country: str, settings_data: Dict) -> bool:
    """지정된 국가의 앱 설정을 DB에 저장합니다."""
    db = get_db_connection()
    if db is None:
        return False
    
    try:
        query = {"country": country}
        db.app_settings.update_one(query, {"$set": settings_data}, upsert=True)
        print(f"성공: {country.upper()} 국가의 설정을 저장했습니다.")
        return True
    except Exception as e:
        print(f"오류: 앱 설정 저장 중 오류 발생: {e}")
        return False

def get_common_settings() -> Optional[Dict]:
    """공통(전역) 설정을 DB에서 가져옵니다. 모든 국가가 공유합니다."""
    db = get_db_connection()
    if db is None:
        return None
    settings = db.app_settings.find_one({"country": "common"})
    if settings:
        settings.pop("_id", None)
    return settings

def save_common_settings(settings_data: Dict) -> bool:
    """공통(전역) 설정을 DB에 저장합니다."""
    db = get_db_connection()
    if db is None:
        return False
    try:
        query = {"country": "common"}
        db.app_settings.update_one(query, {"$set": {**settings_data, "country": "common"}}, upsert=True)
        print("성공: 공통 설정을 저장했습니다.")
        return True
    except Exception as e:
        print(f"오류: 공통 설정 저장 중 오류 발생: {e}")
        return False

def get_status_report_from_db(country: str, date: datetime) -> Optional[Dict]:
    """
    지정된 조건에 맞는 현황 리포트를 DB에서 가져옵니다.
    """
    db = get_db_connection()
    if db is None:
        return None

    query = {
        "country": country,
        "date": date
    }
    report_doc = db.status_reports.find_one(query)
    if report_doc and "report" in report_doc:
        # 조용한 읽기: 과거 탭 렌더링 등에서 대량 호출될 수 있으므로 콘솔 로그는 생략합니다.
        return report_doc["report"]
    return None

def save_status_report_to_db(country: str, date: datetime, report_data: Tuple[str, List[str], List[List[str]]]) -> bool:
    """
    계산된 현황 리포트를 DB에 저장합니다.
    """
    db = get_db_connection()
    if db is None:
        return False

    try:
        header_line, headers, rows = report_data
        doc_to_save = {
            "country": country, "date": date,
            "report": { "header_line": header_line, "headers": headers, "rows": rows },
            "created_at": datetime.now()
        }
        query = {"country": country, "date": date}
        # upsert=True 이므로, 문서가 존재하면 업데이트하고, 없으면 새로 생성합니다.
        # $unset을 사용하여 과거에 있었을 수 있는 'strategy' 필드를 명시적으로 제거합니다.
        update_operation = {
            "$set": doc_to_save,
            "$unset": {"strategy": ""}
        }
        db.status_reports.update_one(query, update_operation, upsert=True)
        print(f"-> 현황 리포트가 DB에 저장되었습니다: {country}/{date.strftime('%Y-%m-%d')}")
        return True
    except Exception as e:
        print(f"오류: 현황 리포트 DB 저장 중 오류 발생: {e}")
        return False

def get_sectors() -> List[Dict]:
    """
    전역 업종 목록을 DB에서 가져옵니다.
    필요 시, 구버전 데이터 모델에서 신규 모델로의 일회성 마이그레이션을 수행합니다.
    """
    db = get_db_connection()
    if db is None:
        return []
    
    # 신규 모델: _id가 "master_list"인 단일 문서 사용
    master_list_doc = db.sectors.find_one({"_id": "master_list"})

    # 마이그레이션 조건: master_list가 없고, 구버전 데이터(country 필드가 있는 문서)가 존재할 때
    if not master_list_doc and db.sectors.count_documents({"country": {"$exists": True}}) > 0:
        print("-> [DB] 구버전 업종 데이터 모델을 감지했습니다. 신규 모델로 마이그레이션을 시작합니다.")
        
        # 1. 모든 국가의 모든 업종을 수집하여 유일한 업종명 목록 생성
        all_old_docs = db.sectors.find({"country": {"$exists": True}})
        unique_sectors_map = {}
        for doc in all_old_docs:
            for sector in doc.get("sectors", []):
                name = sector.get("name")
                if not name: continue
                
                added_date_raw = sector.get("added_date")
                if name not in unique_sectors_map:
                    unique_sectors_map[name] = sector
                else:
                    # 중복 시, 더 오래된 added_date를 유지
                    try:
                        added_date = pd.to_datetime(added_date_raw) if added_date_raw else None
                        existing_date = pd.to_datetime(unique_sectors_map[name].get("added_date")) if unique_sectors_map[name].get("added_date") else None
                        if added_date and existing_date and added_date < existing_date:
                            unique_sectors_map[name] = sector
                    except (ValueError, TypeError):
                        continue
        
        migrated_sectors = []
        for name, data in unique_sectors_map.items():
            migrated_sectors.append({"name": name, "added_date": data.get("added_date")})
        
        # 2. 모든 종목의 업종을 '기타'로 초기화
        print("-> [DB] 모든 종목의 업종을 '글로벌'로 초기화합니다.")
        db.stocks.update_many(
            {"is_deleted": {"$ne": True}},
            {"$set": {"sector": "글로벌"}}
        )

        # 3. 새로운 마스터 목록을 저장하고 이전 데이터 삭제
        print("-> [DB] 새로운 업종 마스터 목록을 생성하고 이전 데이터를 삭제합니다.")
        db.sectors.delete_many({"country": {"$exists": True}}) # 모든 구버전 문서 삭제
        
        new_master_doc = {
            "_id": "master_list",
            "sectors": sorted(migrated_sectors, key=lambda x: x.get("name", ""))
        }
        db.sectors.insert_one(new_master_doc)
        print("-> [DB] 업종 데이터 마이그레이션 완료.")
        
        return new_master_doc["sectors"]

    elif master_list_doc:
        return master_list_doc.get("sectors", [])
    
    # 호환성: 아주 오래된 문자열 리스트 형식의 데이터 처리
    doc = db.sectors.find_one({}) # 아무 문서나 하나 가져옴
    if doc and "sectors" in doc and isinstance(doc["sectors"][0], str):
        print("-> [DB] 레거시 업종 데이터(문자열 리스트)를 감지했습니다. 마이그레이션합니다.")
        sectors_data = doc["sectors"]
        migrated_sectors = [{"name": s, "added_date": datetime.now()} for s in sectors_data]
        db.sectors.delete_many({})
        db.sectors.insert_one({
            "_id": "master_list",
            "sectors": sorted(migrated_sectors, key=lambda x: x.get("name", ""))
        })
        return migrated_sectors

    return []

def save_sectors(sectors: List[Dict]) -> bool:
    """전역 업종 목록을 DB에 저장(덮어쓰기)합니다."""
    db = get_db_connection()
    if db is None:
        return False
    
    try:
        # 이름 기준 중복 제거
        unique_sectors = []
        seen_names = set()
        for sector in sectors:
            name = sector.get("name")
            if name and name not in seen_names:
                unique_sectors.append(sector)
                seen_names.add(name)

        # 이름으로 정렬하여 저장
        sorted_sectors = sorted(unique_sectors, key=lambda x: x.get("name", ""))
        db.sectors.update_one(
            {"_id": "master_list"},
            {"$set": {"sectors": sorted_sectors}},
            upsert=True
        )
        print("성공: 업종 마스터 목록을 저장했습니다.")
        return True
    except Exception as e:
        print(f"오류: 업종 목록 저장 중 오류 발생: {e}")
        return False


def save_sector_changes(new_sectors_list: List[Dict], name_changes: Dict[str, str]) -> bool:
    """
    업종 목록의 변경사항을 저장하고, 이름이 변경된 경우 관련 종목 정보도 업데이트합니다.
    
    Args:
        new_sectors_list (List[Dict]): 저장할 전체 업종 목록
        name_changes (Dict[str, str]): {old_name: new_name} 형식의 이름 변경 맵
    """
    db = get_db_connection()
    if db is None:
        return False
    
    try:
        # 1. 새로운 업종 목록 전체를 저장 (덮어쓰기)
        save_sectors(new_sectors_list)
        
        # 2. 이름이 변경된 경우, 'stocks' 컬렉션에서 해당 업종을 사용하는 종목들을 업데이트합니다.
        if name_changes:
            for old_name, new_name in name_changes.items():
                db.stocks.update_many(
                    {"sector": old_name}, {"$set": {"sector": new_name}}
                )
            print(f"성공: {len(name_changes)}개의 업종명 변경에 따라 종목 정보를 업데이트했습니다.")
        
        return True
        
    except Exception as e:
        print(f"오류: 업종 변경사항 저장 중 오류 발생: {e}")
        return False

def delete_sectors_and_reset_stocks(names_to_delete: List[str]) -> bool:
    """
    지정된 이름의 업종을 마스터 목록에서 삭제하고,
    해당 업종을 사용하던 모든 종목의 업종을 공백으로 초기화합니다.
    """
    db = get_db_connection()
    if db is None or not names_to_delete:
        return False

    try:
        # 1. 업종 마스터 목록에서 해당 업종 삭제
        db.sectors.update_one(
            {"_id": "master_list"},
            {"$pull": {"sectors": {"name": {"$in": names_to_delete}}}}
        )
        print(f"성공: 업종 마스터에서 {len(names_to_delete)}개의 업종을 삭제했습니다.")

        # 2. 모든 종목 컬렉션에서 해당 업종을 사용하는 종목들의 업종을 초기화
        stock_collections = _get_stock_collection_names(db)
        total_modified_count = 0
        for coll_name in stock_collections:
            result = db[coll_name].update_many(
                {"sector": {"$in": names_to_delete}},
                {"$set": {"sector": ""}}
            )
            total_modified_count += result.modified_count
        
        if total_modified_count > 0:
            print(f"성공: 총 {total_modified_count}개 종목의 업종을 초기화했습니다.")

        return True
    except Exception as e:
        print(f"오류: 업종 삭제 중 오류 발생: {e}")
        return False

def _get_stock_collection_names(db) -> List[str]:
    """'_stocks'로 끝나는 모든 컬렉션 이름을 찾아 반환합니다. (예: 'kor_stocks')"""
    # 더 명시적으로 하려면 ['kor_stocks', 'aus_stocks']와 같이 하드코딩할 수 있습니다.
    return [name for name in db.list_collection_names() if name.endswith("_stocks")]

def get_sector_stock_counts() -> Dict[str, Dict[str, int]]:
    """
    각 업종에 속한 종목의 개수를 포트폴리오 국가별로 분류하여 반환합니다.
    예: {'AI·반도체': {'kor': 5, 'aus': 1}}
    """
    db = get_db_connection()
    if db is None:
        return {}
    from collections import defaultdict
    
    # { 'sector_name': { 'kor': count, 'aus': count } }
    total_counts = defaultdict(lambda: defaultdict(int))
    stock_collections = _get_stock_collection_names(db)

    for coll_name in stock_collections:
        # 'kor_stocks' -> 'kor'
        portfolio_country = coll_name.replace("_stocks", "")
        
        pipeline = [
            {"$match": {"is_deleted": {"$ne": True}}},
            {"$group": {"_id": "$sector", "count": {"$sum": 1}}},
        ]
        results = db[coll_name].aggregate(pipeline)
        for item in results:
            sector_name = item.get("_id")
            if sector_name:
                total_counts[sector_name][portfolio_country] = item["count"]
    
    return dict(total_counts)

def get_all_stock_tickers() -> set:
    """DB에 있는 모든 활성 종목의 티커를 집합으로 반환합니다."""
    db = get_db_connection()
    if db is None:
        return set()
    
    all_tickers = set()
    stock_collections = _get_stock_collection_names(db)
    for coll_name in stock_collections:
        cursor = db[coll_name].find({"is_deleted": {"$ne": True}}, {"ticker": 1, "_id": 0})
        all_tickers.update({item['ticker'] for item in cursor})
    return all_tickers


def get_stocks(portfolio_country: str) -> List[Dict]:
    """지정된 포트폴리오 국가의 모든 종목 목록을 DB에서 가져옵니다."""
    db = get_db_connection()
    if db is None:
        return []
    
    collection_name = f"{portfolio_country}_stocks"
    stocks = list(db[collection_name].find(
        {"is_deleted": {"$ne": True}}, 
        {"_id": 0, "ticker": 1, "name": 1, "sector": 1, "type": 1, "country": 1, "last_modified": 1}
    ).sort("ticker", 1))
    return stocks

def save_stocks(portfolio_country: str, stocks: List[Dict], edited_tickers: Optional[List[str]] = None) -> bool:
    """지정된 포트폴리오 국가의 종목 목록 변경사항을 DB에 저장합니다."""
    db = get_db_connection()
    if db is None:
        return False
    
    try:
        from pymongo import UpdateOne

        collection_name = f"{portfolio_country}_stocks"
        collection = db[collection_name]
        requests = []
        now = datetime.now()
        for stock in stocks:
            # 티커를 기준으로 문서를 찾습니다. 티커는 고유하다고 가정합니다.
            query = {"ticker": stock["ticker"]}
            update_doc = {
                "country": stock.get("country"),
                "ticker": stock.get("ticker"),
                "name": stock.get("name", ""), 
                "type": stock.get("type"),
                "sector": stock.get("sector"),
                "is_deleted": False
            }
            
            # 업데이트 연산을 조건부로 구성하여 '$set'과 '$setOnInsert' 간의 필드 충돌을 방지합니다.
            set_on_insert_op = {"created_at": now}
            set_op = {"$set": update_doc, "$unset": {"deleted_at": ""}}
            
            # 편집된 행의 경우, last_modified 타임스탬프를 업데이트합니다.
            if edited_tickers and stock["ticker"] in edited_tickers:
                set_op["$set"]["last_modified"] = now
                set_op["$setOnInsert"] = set_on_insert_op
            else:
                # 새로 추가되는 문서에만 last_modified를 설정합니다.
                set_on_insert_op["last_modified"] = now
                set_op["$setOnInsert"] = set_on_insert_op
            
            requests.append(UpdateOne(query, set_op, upsert=True))
        
        if requests:
            collection.bulk_write(requests, ordered=False)

        # 현재 뷰에 없는 해당 국가의 다른 종목들을 소프트 삭제합니다.
        tickers_in_payload = {s['ticker'] for s in stocks}
        collection.update_many(
            {"ticker": {"$nin": list(tickers_in_payload)}},
            {"$set": {"is_deleted": True, "deleted_at": datetime.now()}}
        )

        print(f"성공: {portfolio_country.upper()} 포트폴리오의 종목 목록을 저장했습니다.")
        return True
    except Exception as e:
        print(f"오류: 종목 목록 저장 중 오류 발생: {e}")
        return False

def add_stocks(portfolio_country: str, stocks_to_add: List[Dict]) -> bool:
    """주어진 종목 목록을 지정된 포트폴리오 국가의 컬렉션에 추가/업데이트합니다."""
    db = get_db_connection()
    if db is None or not stocks_to_add:
        return False
    
    try:
        from pymongo import UpdateOne
        requests = []
        now = datetime.now()
        
        collection_name = f"{portfolio_country}_stocks"
        collection = db[collection_name]

        for stock in stocks_to_add:
            query = {"ticker": stock["ticker"]}
            # is_deleted: False와 $unset deleted_at은 소프트 삭제된 종목을 다시 활성화하는 데 중요합니다.
            update_doc = {
                "country": stock.get("country", "글로벌"),
                "ticker": stock.get("ticker"),
                "name": stock.get("name", ""), 
                "type": stock.get("type"),
                "sector": stock.get("sector"),
                "is_deleted": False,
                "last_modified": now,
            }
            set_on_insert_doc = {"created_at": now}
            requests.append(UpdateOne(query, {"$set": update_doc, "$setOnInsert": set_on_insert_doc, "$unset": {"deleted_at": ""}}, upsert=True))
        
        if requests:
            collection.bulk_write(requests, ordered=False)
        
        print(f"성공: {portfolio_country.upper()} 포트폴리오에 {len(stocks_to_add)}개의 종목을 추가/업데이트했습니다.")
        return True
    except Exception as e:
        print(f"오류: {portfolio_country.upper()} 포트폴리오에 종목 추가 중 오류 발생: {e}")
        return False

def delete_stocks_by_ticker(portfolio_country: str, tickers_to_delete: List[str]) -> bool:
    """
    지정된 티커 목록에 해당하는 종목들을 소프트 삭제합니다.
    """
    db = get_db_connection()
    if db is None or not tickers_to_delete:
        return False

    try:
        collection_name = f"{portfolio_country}_stocks"
        collection = db[collection_name]
        
        result = collection.update_many(
            {"ticker": {"$in": tickers_to_delete}},
            {"$set": {"is_deleted": True, "deleted_at": datetime.now()}}
        )
        
        print(f"성공: {portfolio_country.upper()} 포트폴리오에서 {result.modified_count}개의 종목을 삭제 처리했습니다.")
        return True
    except Exception as e:
        print(f"오류: 종목 삭제 중 오류 발생: {e}")
        return False



def get_all_daily_equities(country: str, start_date: datetime, end_date: datetime) -> List[Dict]:
    """지정된 기간 내의 모든 일별 평가금액 데이터를 DB에서 가져옵니다."""
    db = get_db_connection()
    if db is None:
        return []
    
    query = {
        "country": country,
        "date": {
            "$gte": start_date,
            "$lte": end_date,
        },
        "is_deleted": {"$ne": True}
    }
    equities = list(db.daily_equities.find(query).sort("date", 1))
    for equity in equities:
        equity.pop("_id", None)
    return equities


def get_all_trades(country: str) -> List[Dict]:
    """지정된 국가의 모든 거래 내역을 DB에서 가져옵니다."""
    db = get_db_connection()
    if db is None:
        return []
    
    # 최신 거래가 위로 오도록 날짜와 생성 순서(_id)로 정렬합니다.
    trades = list(db.trades.find(
        {"country": country, "is_deleted": {"$ne": True}}
    ).sort([("date", DESCENDING), ("_id", DESCENDING)]))

    for trade in trades:
        # ObjectId를 웹 앱에서 사용하기 쉽도록 문자열 ID로 변환합니다.
        trade["id"] = str(trade.pop("_id"))
    return trades


def get_trades_on_date(country: str, target_date: datetime) -> List[Dict]:
    """지정된 날짜에 발생한 모든 거래 내역을 DB에서 가져옵니다."""
    db = get_db_connection()
    if db is None:
        return []
    
    # target_date의 시작과 끝을 정의합니다.
    start_of_day = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
    end_of_day = target_date.replace(hour=23, minute=59, second=59, microsecond=999999)

    query = {
        "country": country,
        "date": {
            "$gte": start_of_day,
            "$lte": end_of_day
        },
        "is_deleted": {"$ne": True}
    }
    # 생성 순서대로 정렬
    trades = list(db.trades.find(query).sort("_id", 1))
    for trade in trades:
        trade["id"] = str(trade.pop("_id"))
    return trades


def save_trade(trade_data: Dict) -> bool:
    """단일 거래 내역을 'trades' 컬렉션에 저장합니다."""
    db = get_db_connection()
    if db is None:
        return False

    try:
        trade_data["is_deleted"] = False
        trade_data["created_at"] = datetime.now()
        db.trades.insert_one(trade_data)
        print(f"성공: {trade_data['country'].upper()} 국가의 거래를 저장했습니다: {trade_data}")
        return True
    except Exception as e:
        print(f"오류: 거래 내역 저장 중 오류 발생: {e}")
        return False


def delete_trade_by_id(trade_id: str) -> bool:
    """ID를 기준으로 'trades' 컬렉션에서 단일 거래 내역을 삭제합니다."""
    db = get_db_connection()
    if db is None:
        return False

    try:
        obj_id = ObjectId(trade_id)
        result = db.trades.update_one(
            {"_id": obj_id},
            {"$set": {"is_deleted": True, "deleted_at": datetime.now()}}
        )
        if result.modified_count > 0:
            print(f"성공: 거래 ID {trade_id} 를 삭제했습니다.")
            return True
        else:
            print(f"경고: 삭제할 거래 ID {trade_id} 를 찾지 못했습니다.")
            return True # 이미 삭제되었거나 없는 경우도 성공으로 간주
    except Exception as e:
        print(f"오류: 거래 내역 삭제 중 오류 발생: {e}")
        return False


def save_daily_equity(country: str, date: datetime, total_equity: float, international_shares: Optional[Dict] = None) -> bool:
    """Saves or updates the total equity for a given date."""
    db = get_db_connection()
    if db is None: return False
    
    try:
        query = {"country": country, "date": date}
        set_data = {"country": country, "date": date, "total_equity": total_equity, "is_deleted": False}
        if international_shares is not None:
            set_data["international_shares"] = international_shares
        
        update_operation = {
            "$set": set_data,
            "$unset": {"deleted_at": ""}
        }
        
        db.daily_equities.update_one(query, update_operation, upsert=True)
        return True
    except Exception as e:
        print(f"오류: 일별 평가금액 저장 중 오류 발생: {e}")
        return False
