"""종목 메타데이터를 MongoDB stock_meta 컬렉션에서 읽고 쓰는 유틸리티."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

from utils.db_manager import get_db_connection
from utils.logger import get_app_logger

logger = get_app_logger()

# ---------------------------------------------------------------------------
# 컬렉션 / 인덱스
# ---------------------------------------------------------------------------
_COLLECTION_NAME = "stock_meta"
_INDEX_ENSURED = False


def _get_collection():
    """stock_meta 컬렉션 핸들을 반환하고, 최초 호출 시 인덱스를 보장한다."""
    global _INDEX_ENSURED
    db = get_db_connection()
    if db is None:
        return None
    coll = db[_COLLECTION_NAME]
    if not _INDEX_ENSURED:
        try:
            coll.create_index(
                [("account_id", 1), ("ticker", 1)],
                unique=True,
                name="account_ticker_unique",
                background=True,
            )
            _INDEX_ENSURED = True
        except Exception:
            pass
    return coll


# ---------------------------------------------------------------------------
# 인메모리 캐시 (기존 호환)
# ---------------------------------------------------------------------------
_ACCOUNT_STOCKS_CACHE: dict[str, list[dict]] = {}
_LISTING_CACHE: dict[tuple[str, str], str | None] = {}


def _invalidate_cache(account_id: str | None = None) -> None:
    """캐시를 무효화한다. account_id가 None이면 전체 캐시를 초기화한다."""
    if account_id is None:
        _ACCOUNT_STOCKS_CACHE.clear()
        _LISTING_CACHE.clear()
    else:
        norm = (account_id or "").strip().lower()
        _ACCOUNT_STOCKS_CACHE.pop(norm, None)
        # listing 캐시에서 해당 계좌 관련 항목 제거는 비용이 크므로 전체 초기화
        _LISTING_CACHE.clear()


# ---------------------------------------------------------------------------
# 내부 로드 (캐시 적용)
# ---------------------------------------------------------------------------


def _load_account_stocks_raw(account_id: str) -> list[dict]:
    """DB에서 해당 계좌의 활성(is_deleted!=True) 종목 메타데이터를 로드한다 (캐시 적용)."""
    account_norm = (account_id or "").strip().lower()
    if account_norm in _ACCOUNT_STOCKS_CACHE:
        return _ACCOUNT_STOCKS_CACHE[account_norm]

    coll = _get_collection()
    if coll is None:
        logger.error("MongoDB 연결 실패 — stock_meta 컬렉션을 읽을 수 없습니다.")
        _ACCOUNT_STOCKS_CACHE[account_norm] = []
        return []

    try:
        # Soft Delete: is_deleted가 True가 아닌 것만 조회
        query = {
            "account_id": account_norm,
            "is_deleted": {"$ne": True},
        }
        docs = list(coll.find(query, {"_id": 0}))
        # account_id 필드는 내부 관리용이므로 반환 데이터에서 제거
        for doc in docs:
            doc.pop("account_id", None)
            doc.pop("created_at", None)
            doc.pop("updated_at", None)
            doc.pop("is_deleted", None)
            doc.pop("deleted_at", None)
        _ACCOUNT_STOCKS_CACHE[account_norm] = docs
        return docs
    except Exception as exc:
        logger.error("stock_meta 컬렉션 조회 실패 (account=%s): %s", account_norm, exc)
        _ACCOUNT_STOCKS_CACHE[account_norm] = []
        return []


# ---------------------------------------------------------------------------
# 공개 API — 읽기
# ---------------------------------------------------------------------------

# 하위 호환용 import
from utils.settings_loader import get_account_settings, list_available_accounts  # noqa: E402


def get_etfs(account_id: str, include_extra_tickers: Iterable[str] | None = None) -> list[dict[str, str]]:
    """
    MongoDB stock_meta 컬렉션에서 활성 종목 목록을 반환합니다.
    플랫 리스트 형태: [{ticker, name, listing_date, ...}, ...]
    """
    account_norm = (account_id or "").strip().lower()
    if not account_norm:
        raise ValueError("account_id must be provided")

    all_etfs: list[dict[str, Any]] = []
    seen_tickers: set[str] = set()

    data = _load_account_stocks_raw(account_norm)
    if not data:
        return []

    for item in data:
        if not isinstance(item, dict) or not item.get("ticker"):
            continue

        ticker = str(item["ticker"]).strip()
        if not ticker or ticker in seen_tickers:
            continue

        seen_tickers.add(ticker)

        new_item = dict(item)
        new_item["ticker"] = ticker
        new_item["type"] = "etf"
        all_etfs.append(new_item)

    logger.info(
        "[%s] 전체 ETF 유니버스 로딩: %d개 종목",
        account_norm.upper(),
        len(all_etfs),
    )

    return all_etfs


def get_etfs_by_country(country: str) -> list[dict[str, Any]]:
    """
    (Legacy Helper) 해당 country_code를 가진 모든 계좌의 종목을 합산하여 반환합니다.
    이름 해석 등 계좌 컨텍스트가 없는 경우에 사용합니다.
    """
    country_norm = (country or "").strip().lower()
    accounts = list_available_accounts()

    unique_tickers: dict[str, dict] = {}
    for account in accounts:
        try:
            settings = get_account_settings(account)
            if settings.get("country_code", "").lower() == country_norm:
                etfs = get_etfs(account)
                for etf in etfs:
                    tkr = etf.get("ticker")
                    if tkr and tkr not in unique_tickers:
                        unique_tickers[tkr] = etf
        except Exception:
            pass

    return list(unique_tickers.values())


def get_all_etfs(account_id: str) -> list[dict[str, Any]]:
    """해당 계좌의 전체 ETF 항목을 반환합니다."""

    raw_data = _load_account_stocks_raw(account_id)
    if not raw_data:
        return []

    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw_data:
        if not isinstance(item, dict):
            continue
        ticker = str(item.get("ticker") or "").strip()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        entry = dict(item)
        entry["ticker"] = ticker
        entry.setdefault("type", "etf")
        results.append(entry)
    return results


# ---------------------------------------------------------------------------
# 공개 API — 쓰기
# ---------------------------------------------------------------------------


def save_etfs(account_id: str, data: list[dict]) -> None:
    """
    종목 메타데이터를 MongoDB stock_meta 컬렉션에 저장합니다 (upsert).
    Soft Delete 방식: 기존에 있지만 새 데이터에 없는 종목은 is_deleted=True 처리합니다.
    """
    account_norm = (account_id or "").strip().lower()
    if not account_norm:
        raise ValueError("account_id required")

    coll = _get_collection()
    if coll is None:
        raise RuntimeError("MongoDB 연결 실패 — stock_meta 컬렉션에 쓸 수 없습니다.")

    now = datetime.now(timezone.utc)
    new_tickers: set[str] = set()

    from pymongo import UpdateOne

    operations = []
    for item in data:
        if not isinstance(item, dict):
            continue
        ticker = str(item.get("ticker") or "").strip()
        if not ticker:
            continue
        new_tickers.add(ticker)

        doc = dict(item)
        doc.pop("type", None)
        doc["account_id"] = account_norm
        doc["ticker"] = ticker
        doc["updated_at"] = now
        doc["is_deleted"] = False  # 활성 상태로 설정
        doc["deleted_at"] = None

        operations.append(
            UpdateOne(
                {"account_id": account_norm, "ticker": ticker},
                {"$set": doc, "$setOnInsert": {"created_at": now}},
                upsert=True,
            )
        )

    if operations:
        try:
            coll.bulk_write(operations, ordered=False)
        except Exception as exc:
            logger.error("stock_meta bulk_write 실패 (account=%s): %s", account_norm, exc)
            raise

    # DB에는 있지만 새 데이터에는 없는 종목 → Soft Delete
    try:
        coll.update_many(
            {"account_id": account_norm, "ticker": {"$nin": list(new_tickers)}},
            {
                "$set": {
                    "is_deleted": True,
                    "deleted_at": now,
                    "updated_at": now,
                }
            },
        )
    except Exception as exc:
        logger.warning("stock_meta 잔여 종목 soft delete 실패 (account=%s): %s", account_norm, exc)

    logger.info("%d개 종목 정보가 stock_meta 컬렉션에 저장되었습니다. (account=%s)", len(new_tickers), account_norm)
    _invalidate_cache(account_norm)


def add_stock(account_id: str, ticker: str, name: str = "", **extra_fields: Any) -> bool:
    """
    단일 종목을 stock_meta 컬렉션에 추가한다.
    이미 존재하면(삭제된 경우 포함) 활성 상태로 복구한다.
    """
    account_norm = (account_id or "").strip().lower()
    ticker_norm = str(ticker or "").strip()
    if not account_norm or not ticker_norm:
        return False

    coll = _get_collection()
    if coll is None:
        return False

    now = datetime.now(timezone.utc)

    # 1. 먼저 존재 여부 확인 (삭제된 것 포함)
    existing = coll.find_one({"account_id": account_norm, "ticker": ticker_norm})

    if existing:
        if not existing.get("is_deleted"):
            # 이미 활성 상태로 존재하면 False 반환 (또는 업데이트? 현재는 기각)
            logger.info("이미 존재하는 종목입니다: %s (account=%s)", ticker_norm, account_norm)
            return False

        # 삭제된 상태라면 복구
        try:
            update_doc = {
                "is_deleted": False,
                "deleted_at": None,
                "updated_at": now,
            }
            if name:
                update_doc["name"] = name
            update_doc.update(extra_fields)

            coll.update_one({"_id": existing["_id"]}, {"$set": update_doc})
            _invalidate_cache(account_norm)
            logger.info("삭제된 종목 복구: %s (account=%s)", ticker_norm, account_norm)
            return True
        except Exception as exc:
            logger.warning("종목 복구 실패 %s: %s", ticker_norm, exc)
            return False

    # 2. 없으면 신규 삽입
    doc: dict[str, Any] = {
        "account_id": account_norm,
        "ticker": ticker_norm,
        "name": name or "",
        "created_at": now,
        "updated_at": now,
        "added_date": now.strftime("%Y-%m-%d"),
        "is_deleted": False,
        "deleted_at": None,
    }
    doc.update(extra_fields)

    try:
        coll.insert_one(doc)
        _invalidate_cache(account_norm)
        logger.info("종목 추가: %s (account=%s)", ticker_norm, account_norm)
        return True
    except Exception as exc:
        logger.warning("종목 추가 실패 %s (account=%s): %s", ticker_norm, account_norm, exc)
        return False


def remove_stock(account_id: str, ticker: str) -> bool:
    """단일 종목을 Soft Delete 처리한다."""
    account_norm = (account_id or "").strip().lower()
    ticker_norm = str(ticker or "").strip()
    if not account_norm or not ticker_norm:
        return False

    coll = _get_collection()
    if coll is None:
        return False

    now = datetime.now(timezone.utc)
    try:
        result = coll.update_one(
            {"account_id": account_norm, "ticker": ticker_norm},
            {
                "$set": {
                    "is_deleted": True,
                    "deleted_at": now,
                    "updated_at": now,
                }
            },
        )
        if result.modified_count > 0:
            _invalidate_cache(account_norm)
            logger.info("종목 삭제(Soft): %s (account=%s)", ticker_norm, account_norm)
            return True
        return False
    except Exception as exc:
        logger.warning("종목 삭제 실패 %s (account=%s): %s", ticker_norm, account_norm, exc)
        return False


# ---------------------------------------------------------------------------
# 공개 API — listing_date 관련
# ---------------------------------------------------------------------------


def check_stock_status(account_id: str, ticker: str) -> str | None:
    """
    종목의 상태를 확인한다.
    Returns:
        "ACTIVE": 이미 활성 상태로 존재함
        "DELETED": 삭제된 상태로 존재함 (복구 가능)
        None: 존재하지 않음 (신규)
    """
    account_norm = (account_id or "").strip().lower()
    ticker_norm = str(ticker or "").strip()
    if not account_norm or not ticker_norm:
        return None

    coll = _get_collection()
    if coll is None:
        return None

    existing = coll.find_one({"account_id": account_norm, "ticker": ticker_norm})
    if existing:
        if existing.get("is_deleted"):
            return "DELETED"
        return "ACTIVE"
    return None


def get_listing_date(country: str, ticker: str) -> str | None:
    """country_code 기준으로 listing_date를 조회한다."""
    country_norm = (country or "").strip().lower()
    ticker_norm = str(ticker or "").strip()
    cache_key = (country_norm, ticker_norm)
    if cache_key in _LISTING_CACHE:
        return _LISTING_CACHE[cache_key]

    accounts = list_available_accounts()
    for account in accounts:
        try:
            settings = get_account_settings(account)
            if settings.get("country_code", "").lower() == country_norm:
                data = _load_account_stocks_raw(account)
                for item in data:
                    if not isinstance(item, dict):
                        continue
                    item_ticker = str(item.get("ticker") or "").strip()
                    if item_ticker == ticker_norm:
                        listing_date = item.get("listing_date")
                        if listing_date:
                            _LISTING_CACHE[cache_key] = listing_date
                            return listing_date
        except Exception:
            continue

    _LISTING_CACHE[cache_key] = None
    return None


def set_listing_date(country: str, ticker: str, listing_date: str) -> None:
    """해당 country_code를 공유하는 모든 계좌에서 티커의 listing_date를 설정한다."""
    country_norm = (country or "").strip().lower()
    ticker_norm = str(ticker or "").strip()
    if not ticker_norm:
        return

    coll = _get_collection()
    if coll is None:
        return

    accounts = list_available_accounts()
    updated_any = False

    for account in accounts:
        try:
            settings = get_account_settings(account)
            if settings.get("country_code", "").lower() != country_norm:
                continue

            account_norm = account.strip().lower()
            result = coll.update_one(
                {"account_id": account_norm, "ticker": ticker_norm},
                {"$set": {"listing_date": listing_date, "updated_at": datetime.now(timezone.utc)}},
            )
            if result.modified_count > 0 or result.matched_count > 0:
                updated_any = True
                _invalidate_cache(account_norm)
        except Exception:
            continue

    if updated_any:
        _LISTING_CACHE[(country_norm, ticker_norm)] = listing_date


# ---------------------------------------------------------------------------
# 파일 기반 레거시 헬퍼 (마이그레이션용)
# ---------------------------------------------------------------------------


def _get_data_dir() -> str:
    """zaccounts 디렉토리 절대경로를 반환한다."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, "zaccounts")


def load_stocks_from_file(account_id: str) -> list[dict]:
    """stocks.json 파일에서 종목 데이터를 로드한다 (마이그레이션 전용)."""
    account_norm = (account_id or "").strip().lower()
    file_path = os.path.join(_get_data_dir(), account_norm, "stocks.json")
    if not os.path.exists(file_path):
        return []
    try:
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
    except Exception as exc:
        logger.error("stocks.json 로드 실패 (%s): %s", file_path, exc)
    return []
