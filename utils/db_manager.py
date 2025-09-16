import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
from bson import ObjectId
from pymongo import DESCENDING, MongoClient

import settings as global_settings

# --- 전역 변수로 DB 연결 관리 ---
_db_connection = None
_mongo_client: Optional[MongoClient] = None


def get_db_connection():
    """
    MongoDB 클라이언트 연결을 생성하고, 전역 변수에 저장하여 재사용합니다.
    """
    global _db_connection, _mongo_client

    # 이미 연결이 설정되어 있으면, 기존 연결을 반환합니다.
    if _db_connection is not None:
        return _db_connection

    try:
        connection_string = os.environ.get("MONGO_DB_CONNECTION_STRING") or getattr(
            global_settings, "MONGO_DB_CONNECTION_STRING", None
        )
        db_name = os.environ.get("MONGO_DB_NAME") or getattr(
            global_settings, "MONGO_DB_NAME", "momentum_etf_db"
        )
        # Connection pool tuning (env optional)
        max_pool = int(os.environ.get("MONGO_DB_MAX_POOL_SIZE", "20"))
        min_pool = int(os.environ.get("MONGO_DB_MIN_POOL_SIZE", "0"))
        max_idle = int(os.environ.get("MONGO_DB_MAX_IDLE_TIME_MS", "0"))  # 0 = driver default
        wait_q_timeout = int(
            os.environ.get("MONGO_DB_WAIT_QUEUE_TIMEOUT_MS", "0")
        )  # 0 = driver default

        if not connection_string:
            raise ValueError(
                "MongoDB 연결 문자열이 설정되지 않았습니다. (MONGO_DB_CONNECTION_STRING)"
            )

        if _mongo_client is None:
            client_kwargs = dict(
                maxPoolSize=max_pool,
                minPoolSize=min_pool,
                retryWrites=True,
                serverSelectionTimeoutMS=8000,
                connectTimeoutMS=5000,
                socketTimeoutMS=10000,
            )
            if max_idle > 0:
                client_kwargs["maxIdleTimeMS"] = max_idle
            if wait_q_timeout > 0:
                client_kwargs["waitQueueTimeoutMS"] = wait_q_timeout
            _mongo_client = MongoClient(connection_string, **client_kwargs)
        client = _mongo_client
        # 서버에 연결하여 연결 성공 여부 확인
        client.server_info()

        # 성공적으로 연결되면, 전역 변수에 DB 객체를 저장합니다.
        _db_connection = client[db_name]

        # 연결 수(서버 전체) 정보 출력 시도
        try:
            status = client.admin.command("serverStatus")  # requires clusterMonitor on Atlas
            conn = status.get("connections", {}) if isinstance(status, dict) else {}
            current = conn.get("current")
            available = conn.get("available")
            total_created = conn.get("totalCreated")
            if current is not None:
                print(
                    f"-> MongoDB에 성공적으로 연결되었습니다. (connections: current={current}, available={available}, totalCreated={total_created})"
                )
            else:
                print("-> MongoDB에 성공적으로 연결되었습니다.")
        except Exception:
            # 권한 부족 등으로 serverStatus 실패 시, 기본 메시지만 출력
            print("-> MongoDB에 성공적으로 연결되었습니다.")
        return _db_connection
    except Exception as e:
        print(f"오류: MongoDB 연결에 실패했습니다: {e}")
        return None


def get_portfolio_snapshot(country: str, date_str: Optional[str] = None) -> Optional[Dict]:
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
        latest_equity = db.daily_equities.find_one(
            {"country": country, "is_deleted": {"$ne": True}}, sort=[("date", DESCENDING)]
        )
        if latest_equity:
            target_date = latest_equity["date"]

    if not target_date:
        # 거래 내역은 있지만 평가금액이 없을 수도 있으므로, trades에서 날짜를 찾아봅니다.
        latest_trade = db.trades.find_one(
            {"country": country, "is_deleted": {"$ne": True}}, sort=[("date", DESCENDING)]
        )
        if latest_trade:
            target_date = latest_trade["date"]
        else:
            return None  # 데이터가 전혀 없음

    # 2. 대상 날짜의 총 평가금액 조회
    equity_data = db.daily_equities.find_one(
        {"country": country, "date": target_date, "is_deleted": {"$ne": True}}
    )
    # 기본값 초기화
    is_equity_stale = False

    # 해당 날짜의 평가금액이 유효한지 확인합니다.
    is_equity_valid = equity_data and equity_data.get("total_equity", 0) > 0

    # 평가금액이 없거나 0이면, 가장 최근의 유효한 평가금액을 찾습니다.
    if not is_equity_valid:
        latest_valid_equity = db.daily_equities.find_one(
            {
                "country": country,
                "date": {"$lt": target_date},
                "total_equity": {"$gt": 0},
                "is_deleted": {"$ne": True},
            },
            sort=[("date", DESCENDING)],
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
    # 코인의 경우, daily_equities가 자정(00:00)으로 저장되고 트레이드는 시각 포함으로 저장되므로
    # 동일한 달력일에 발생한 모든 트레이드를 포함하도록 상한을 '해당일의 23:59:59.999999'로 확장합니다.
    if country == "coin":
        upper_bound = target_date.replace(hour=23, minute=59, second=59, microsecond=999999)
    else:
        upper_bound = target_date
    trades_cursor = db.trades.find(
        {"country": country, "date": {"$lte": upper_bound}, "is_deleted": {"$ne": True}}
    ).sort([("date", 1), ("_id", 1)])

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
            holdings_list.append(
                {
                    "ticker": ticker,
                    "name": data["name"],
                    "shares": data["shares"],
                    "avg_cost": avg_cost,
                }
            )

    # 4. 최종 스냅샷 조립
    snapshot = {
        "date": target_date,
        "country": country,
        "total_equity": total_equity,
        "holdings": holdings_list,
        "is_equity_stale": is_equity_stale,
    }
    # 'equity_date'를 항상 포함하여 헤더에서 비교/표시가 가능하도록 합니다.
    if equity_date:
        snapshot["equity_date"] = equity_date

    if country == "aus" and equity_data and "international_shares" in equity_data:
        snapshot["international_shares"] = equity_data["international_shares"]

    return snapshot


def get_previous_portfolio_snapshot(country: str, as_of_date: datetime) -> Optional[Dict]:
    """
    주어진 날짜 이전의 가장 최근 포트폴리오 스냅샷을 가져옵니다.
    """
    db = get_db_connection()
    if db is None:
        return None

    # 'daily_equities'와 'trades'에서 as_of_date 이전의 날짜들을 찾습니다.
    equity_dates = db.daily_equities.distinct(
        "date", {"country": country, "date": {"$lt": as_of_date}, "is_deleted": {"$ne": True}}
    )
    trade_dates = db.trades.distinct(
        "date", {"country": country, "date": {"$lt": as_of_date}, "is_deleted": {"$ne": True}}
    )

    all_prev_dates = set(equity_dates).union(set(trade_dates))

    if not all_prev_dates:
        return None

    # 가장 최근 날짜를 찾습니다.
    prev_date = max(all_prev_dates)
    prev_date_str = prev_date.strftime("%Y-%m-%d")

    return get_portfolio_snapshot(country, prev_date_str)


def get_available_snapshot_dates(
    country: str, as_of_date: Optional[datetime] = None, include_as_of_date: bool = False
) -> List[str]:
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

    equity_dates = list(db.daily_equities.distinct("date", {**query, "is_deleted": {"$ne": True}}))
    trade_dates = list(db.trades.distinct("date", {**query, "is_deleted": {"$ne": True}}))

    # 날짜를 'YYYY-MM-DD' 문자열로 변환한 뒤, 중복(같은 날 서로 다른 시각)을 제거합니다.
    def to_day_str_list(dt_list):
        day_strs = []
        for d in dt_list or []:
            try:
                day_strs.append(d.strftime("%Y-%m-%d"))
            except Exception:
                # 일부 드라이버가 naive/aware 차이가 있을 수 있으므로 방어적으로 처리
                try:
                    day_strs.append(pd.to_datetime(d).strftime("%Y-%m-%d"))  # type: ignore
                except Exception:
                    continue
        return day_strs

    equity_days = set(to_day_str_list(equity_dates))
    trade_days = set(to_day_str_list(trade_dates))
    all_days = equity_days.union(trade_days)

    # 내림차순 정렬
    return sorted(all_days, reverse=True)


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
        db.app_settings.update_one(
            query, {"$set": {**settings_data, "country": "common"}}, upsert=True
        )
        print("성공: 공통 설정을 저장했습니다.")
        return True
    except Exception as e:
        print(f"오류: 공통 설정 저장 중 오류 발생: {e}")
        return False


# --- Import checkpoints (incremental sync) ---
def get_import_checkpoint(source: str, country: str, key: Optional[str] = None) -> Optional[int]:
    """Return last processed timestamp (ms) for a given import source/country/key.

    - source: e.g., 'bithumb_v1_trades'
    - country: e.g., 'coin'
    - key: optional sub-key (e.g., base ticker like 'BTC')
    """
    db = get_db_connection()
    if db is None:
        return None
    q: Dict[str, object] = {"source": source, "country": country}
    if key is not None:
        q["key"] = key
    doc = db.import_checkpoints.find_one(q)
    if not doc:
        return None
    try:
        return int(doc.get("last_ts_ms"))
    except Exception:
        return None


def save_import_checkpoint(
    source: str, country: str, last_ts_ms: int, key: Optional[str] = None
) -> bool:
    """Upsert last processed timestamp (ms) for a given import source/country/key."""
    db = get_db_connection()
    if db is None:
        return False
    q: Dict[str, object] = {"source": source, "country": country}
    if key is not None:
        q["key"] = key
    try:
        db.import_checkpoints.update_one(
            q,
            {"$set": {**q, "last_ts_ms": int(last_ts_ms), "updated_at": datetime.now()}},
            upsert=True,
        )
        return True
    except Exception as e:
        print(f"오류: 체크포인트 저장 실패 ({source}/{country}/{key}): {e}")
        return False


def get_status_report_from_db(country: str, date: datetime) -> Optional[Dict]:
    """
    지정된 조건에 맞는 현황 리포트를 DB에서 가져옵니다.
    """
    db = get_db_connection()
    if db is None:
        return None

    query = {"country": country, "date": date}
    report_doc = db.status_reports.find_one(query)
    if report_doc and "report" in report_doc:
        # 조용한 읽기: 과거 탭 렌더링 등에서 대량 호출될 수 있으므로 콘솔 로그는 생략합니다.
        return report_doc["report"]
    return None


def save_status_report_to_db(
    country: str, date: datetime, report_data: Tuple[str, List[str], List[List[str]]]
) -> bool:
    """
    계산된 현황 리포트를 DB에 저장합니다.
    """
    db = get_db_connection()
    if db is None:
        return False

    try:
        header_line, headers, rows = report_data
        doc_to_save = {
            "country": country,
            "date": date,
            "report": {"header_line": header_line, "headers": headers, "rows": rows},
            "created_at": datetime.now(),
        }
        query = {"country": country, "date": date}
        # upsert=True 이므로, 문서가 존재하면 업데이트하고, 없으면 새로 생성합니다.
        # $unset을 사용하여 과거에 있었을 수 있는 'strategy' 필드를 명시적으로 제거합니다.
        update_operation = {"$set": doc_to_save, "$unset": {"strategy": ""}}
        db.status_reports.update_one(query, update_operation, upsert=True)
        print(f"-> 현황 리포트가 DB에 저장되었습니다: {country}/{date.strftime('%Y-%m-%d')}")
        return True
    except Exception as e:
        print(f"오류: 현황 리포트 DB 저장 중 오류 발생: {e}")
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
        "is_deleted": {"$ne": True},
    }
    equities = list(db.daily_equities.find(query).sort("date", 1))
    for equity in equities:
        equity.pop("_id", None)
    return equities


def get_all_trades(country: str) -> List[Dict]:
    """지정된 국가의 모든 거래 내역을 DB에서 가져옵니다.

    country 필드의 대소문자 불일치를 허용하기 위해 정규식(대소문자 무시)으로 조회합니다.
    """
    db = get_db_connection()
    if db is None:
        return []

    query = {
        "country": {"$regex": f"^{country}$", "$options": "i"},
        "is_deleted": {"$ne": True},
    }

    # 최신 거래가 위로 오도록 날짜와 생성 순서(_id)로 정렬합니다.
    trades = list(db.trades.find(query).sort([("date", DESCENDING), ("_id", DESCENDING)]))

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
        "date": {"$gte": start_of_day, "$lte": end_of_day},
        "is_deleted": {"$ne": True},
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
            {"_id": obj_id}, {"$set": {"is_deleted": True, "deleted_at": datetime.now()}}
        )
        if result.modified_count > 0:
            print(f"성공: 거래 ID {trade_id} 를 삭제했습니다.")
            return True
        else:
            print(f"경고: 삭제할 거래 ID {trade_id} 를 찾지 못했습니다.")
            return True  # 이미 삭제되었거나 없는 경우도 성공으로 간주
    except Exception as e:
        print(f"오류: 거래 내역 삭제 중 오류 발생: {e}")
    return False


def save_daily_equity(
    country: str, date: datetime, total_equity: float, international_shares: Optional[Dict] = None
) -> bool:
    """Saves or updates the total equity for a given date."""
    db = get_db_connection()
    if db is None:
        return False

    try:
        query = {"country": country, "date": date}
        set_data = {
            "country": country,
            "date": date,
            "total_equity": total_equity,
            "is_deleted": False,
        }
        if international_shares is not None:
            set_data["international_shares"] = international_shares

        update_operation = {"$set": set_data, "$unset": {"deleted_at": ""}}

        db.daily_equities.update_one(query, update_operation, upsert=True)
        return True
    except Exception as e:
        print(f"오류: 일별 평가금액 저장 중 오류 발생: {e}")
        return False


def soft_delete_all_trades(country: str) -> int:
    """Soft-delete all trades for the specified country by setting is_deleted=True.

    Returns the number of modified documents.
    """
    db = get_db_connection()
    if db is None:
        return 0
    try:
        res = db.trades.update_many(
            {"country": country, "is_deleted": {"$ne": True}},
            {"$set": {"is_deleted": True, "deleted_at": datetime.now()}},
        )
        print(f"성공: {country.upper()} 국가의 거래 {res.modified_count}건을 삭제 처리했습니다.")
        return int(res.modified_count)
    except Exception as e:
        print(f"오류: {country.upper()} 국가 거래 일괄 삭제 중 오류: {e}")
        return 0
