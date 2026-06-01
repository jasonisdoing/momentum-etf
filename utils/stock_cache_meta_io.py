"""종목 메타 캐시(stock_cache_meta) 컬렉션을 읽고 쓰는 유틸리티."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Any

from utils.db_manager import get_db_connection
from utils.logger import get_app_logger

logger = get_app_logger()

_COLLECTION_NAME = "stock_cache_meta"
_INDEX_ENSURED = False


def _get_collection():
    """stock_cache_meta 컬렉션 핸들을 반환하고, 최초 호출 시 인덱스를 보장한다."""
    global _INDEX_ENSURED
    db = get_db_connection()
    if db is None:
        return None

    coll = db[_COLLECTION_NAME]
    if not _INDEX_ENSURED:
        try:
            coll.create_index(
                [("ticker_type", 1), ("ticker", 1)],
                unique=True,
                name="ticker_type_ticker_unique",
                background=True,
            )
            coll.create_index(
                [("country_code", 1), ("ticker", 1)],
                name="country_code_ticker_lookup",
                background=True,
            )
            _INDEX_ENSURED = True
        except Exception:
            pass
    return coll


def ensure_stock_cache_meta_readable() -> None:
    """stock_cache_meta 컬렉션을 읽을 수 없으면 즉시 예외를 발생시킨다."""
    coll = _get_collection()
    if coll is None:
        raise RuntimeError("MongoDB 연결 실패 — stock_cache_meta 컬렉션을 읽을 수 없습니다.")


def get_stock_cache_meta_doc(ticker_type: str, ticker: str) -> dict[str, Any] | None:
    """종목 메타 캐시 문서 1건을 반환한다."""
    type_norm = (ticker_type or "").strip().lower()
    ticker_norm = str(ticker or "").strip().upper()
    if not type_norm:
        raise ValueError("ticker_type must be provided")
    if not ticker_norm:
        raise ValueError("ticker must be provided")

    coll = _get_collection()
    if coll is None:
        raise RuntimeError("MongoDB 연결 실패 — stock_cache_meta 컬렉션을 읽을 수 없습니다.")

    doc = coll.find_one({"ticker_type": type_norm, "ticker": ticker_norm}, {"_id": 0})
    return dict(doc) if isinstance(doc, dict) else None


def get_stock_cache_meta_docs(ticker_type: str, tickers: list[str]) -> dict[str, dict[str, Any]]:
    """종목 메타 캐시 문서를 티커 기준 맵으로 반환한다."""
    type_norm = (ticker_type or "").strip().lower()
    normalized_tickers = sorted({str(ticker or "").strip().upper() for ticker in tickers if str(ticker or "").strip()})
    if not type_norm:
        raise ValueError("ticker_type must be provided")
    if not normalized_tickers:
        return {}

    coll = _get_collection()
    if coll is None:
        raise RuntimeError("MongoDB 연결 실패 — stock_cache_meta 컬렉션을 읽을 수 없습니다.")

    docs = coll.find(
        {"ticker_type": type_norm, "ticker": {"$in": normalized_tickers}},
        {"_id": 0},
    )
    result: dict[str, dict[str, Any]] = {}
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        ticker_norm = str(doc.get("ticker") or "").strip().upper()
        if ticker_norm:
            result[ticker_norm] = dict(doc)
    return result


def get_previous_stock_cache_meta_history(
    ticker_type: str, 
    ticker: str, 
    before_date: str
) -> dict[str, Any] | None:
    """특정 날짜(YYYY-MM-DD) 이전의 가장 최근 히스토리 스냅샷을 반환한다."""
    type_norm = (ticker_type or "").strip().lower()
    ticker_norm = str(ticker or "").strip().upper()
    
    db = get_db_connection()
    if db is None:
        return None
        
    coll = db["stock_cache_meta_history"]
    # before_date보다 작은 날짜 중 가장 최근 것 하나 조회
    doc = coll.find_one(
        {
            "ticker_type": type_norm, 
            "ticker": ticker_norm, 
            "date": {"$lt": before_date}
        },
        sort=[("date", -1)],
        projection={"_id": 0}
    )
    return dict(doc) if isinstance(doc, dict) else None


def upsert_stock_cache_meta_doc(
    ticker_type: str,
    ticker: str,
    *,
    country_code: str,
    name: str,
    meta_cache: dict[str, Any] | None = None,
    holdings_cache: dict[str, Any] | None = None,
) -> None:
    """종목 메타 캐시 문서를 upsert한다."""
    type_norm = (ticker_type or "").strip().lower()
    ticker_norm = str(ticker or "").strip().upper()
    country_norm = str(country_code or "").strip().lower()
    name_norm = str(name or "").strip()
    if not type_norm:
        raise ValueError("ticker_type must be provided")
    if not ticker_norm:
        raise ValueError("ticker must be provided")
    if not country_norm:
        raise ValueError("country_code must be provided")
    if not name_norm:
        raise ValueError("name must be provided")

    coll = _get_collection()
    if coll is None:
        raise RuntimeError("MongoDB 연결 실패 — stock_cache_meta 컬렉션에 쓸 수 없습니다.")

    now = datetime.now(timezone.utc)
    payload: dict[str, Any] = {
        "ticker_type": type_norm,
        "ticker": ticker_norm,
        "country_code": country_norm,
        "name": name_norm,
        "updated_at": now,
    }
    if meta_cache is not None:
        payload["meta_cache"] = meta_cache
    if holdings_cache is not None:
        payload["holdings_cache"] = holdings_cache

    # 1. 최신 정보 업데이트 (기존 로직)
    coll.update_one(
        {"ticker_type": type_norm, "ticker": ticker_norm},
        {
            "$set": payload,
            "$setOnInsert": {"created_at": now},
        },
        upsert=True,
    )

    # 2. 일자별 히스토리 스냅샷 저장
    db = get_db_connection()
    if db is not None:
        history_coll = db["stock_cache_meta_history"]
        
        # 한국 시간 기준으로 귀속 날짜 결정
        now_kst = datetime.now(ZoneInfo("Asia/Seoul"))
        from utils.data_loader import get_trading_days
        
        # 최근 7일간 거래일 조회
        start_search = (now_kst - timedelta(days=7)).strftime("%Y-%m-%d")
        end_search = now_kst.strftime("%Y-%m-%d")
        trading_days = get_trading_days(start_search, end_search, "kor")
        trading_days_str = [d.strftime("%Y-%m-%d") for d in trading_days]
        
        today_str = now_kst.strftime("%Y-%m-%d")
        
        # 9시 이후이고 오늘이 거래일이면 오늘 날짜 사용, 아니면 직전 거래일 사용
        if now_kst.hour >= 9 and today_str in trading_days_str:
            snapshot_date = today_str
        else:
            # 오늘이 거래일이어도 9시 전이면 직전 거래일, 오늘이 휴장일이면 가장 최근 거래일
            past_days = [d for d in trading_days_str if d < today_str]
            snapshot_date = past_days[-1] if past_days else today_str

        history_payload = payload.copy()
        history_payload["date"] = snapshot_date
        
        # 히스토리 컬렉션 인덱스 (최초 1회)
        history_coll.create_index(
            [("ticker_type", 1), ("ticker", 1), ("date", -1)],
            unique=True,
            name="ticker_date_history_unique",
            background=True
        )
        
        history_coll.update_one(
            {"ticker_type": type_norm, "ticker": ticker_norm, "date": snapshot_date},
            {"$set": history_payload},
            upsert=True
        )


def update_stock_portfolio_change_cache_doc(
    ticker_type: str,
    ticker: str,
    portfolio_change_cache: dict[str, Any],
) -> None:
    """종목 메타 캐시 문서에 포트폴리오 변동 계산 캐시만 갱신한다."""
    type_norm = (ticker_type or "").strip().lower()
    ticker_norm = str(ticker or "").strip().upper()
    if not type_norm:
        raise ValueError("ticker_type must be provided")
    if not ticker_norm:
        raise ValueError("ticker must be provided")
    if not isinstance(portfolio_change_cache, dict):
        raise ValueError("portfolio_change_cache must be a dict")

    coll = _get_collection()
    if coll is None:
        raise RuntimeError("MongoDB 연결 실패 — stock_cache_meta 컬렉션에 쓸 수 없습니다.")

    now = datetime.now(timezone.utc)
    result = coll.update_one(
        {"ticker_type": type_norm, "ticker": ticker_norm},
        {
            "$set": {
                "portfolio_change_cache": portfolio_change_cache,
                "portfolio_change_cache_updated_at": now,
                "updated_at": now,
            },
        },
    )
    if result.matched_count == 0:
        raise RuntimeError(f"[{type_norm}/{ticker_norm}] 포트폴리오 변동 캐시를 저장할 메타 문서가 없습니다.")


def delete_stock_cache_meta_doc(ticker_type: str, ticker: str) -> None:
    """종목 메타 캐시 문서 1건을 삭제한다."""
    type_norm = (ticker_type or "").strip().lower()
    ticker_norm = str(ticker or "").strip().upper()
    if not type_norm:
        raise ValueError("ticker_type must be provided")
    if not ticker_norm:
        raise ValueError("ticker must be provided")

    coll = _get_collection()
    if coll is None:
        raise RuntimeError("MongoDB 연결 실패 — stock_cache_meta 컬렉션에 쓸 수 없습니다.")

    coll.delete_one({"ticker_type": type_norm, "ticker": ticker_norm})
