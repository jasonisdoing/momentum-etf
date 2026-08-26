"""OHLCV 데이터를 MongoDB에 캐싱하고 관리하기 위한 헬퍼 함수 모음입니다."""

from __future__ import annotations

import io
import re
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

import pandas as pd
from bson.binary import Binary
from pymongo.errors import PyMongoError

from utils.db_manager import get_db_connection
from utils.logger import get_app_logger

logger = get_app_logger()

_CLOSE_SERIES_MEMORY_CACHE: dict[tuple[str, str], tuple[datetime | None, pd.Series]] = {}

_REFRESH_STATUS_COLLECTION = "cache_refresh_status"

_TEMP_SUFFIX_SANITIZE = re.compile(r"[^a-z0-9_-]", re.IGNORECASE)


def _get_close_series_memory_cache(
    collection_name: str,
    ticker: str,
    updated_at: datetime | None,
) -> pd.Series | None:
    cached_entry = _CLOSE_SERIES_MEMORY_CACHE.get((collection_name, ticker))
    if cached_entry is None:
        return None

    cached_updated_at, cached_series = cached_entry
    if cached_updated_at != updated_at:
        return None

    return cached_series.copy()


def _set_close_series_memory_cache(
    collection_name: str,
    ticker: str,
    updated_at: datetime | None,
    close_series: pd.Series,
) -> None:
    _CLOSE_SERIES_MEMORY_CACHE[(collection_name, ticker)] = (updated_at, close_series.copy())


def _resolve_close_column(columns: Iterable[str] | None) -> str | None:
    if columns is None:
        return None

    normalized = [str(column) for column in columns]
    for candidate in ["unadjusted_close", "Close", "close"]:
        if candidate in normalized:
            return candidate
    return None


def _serialize_close_series_payload(close_series: pd.Series, column_name: str) -> Binary | None:
    if close_series is None or close_series.empty:
        return None

    close_df = close_series.to_frame(name=column_name)
    buf = io.BytesIO()
    try:
        close_df.to_parquet(buf, engine="pyarrow", compression="snappy")
    except Exception:
        return None
    return Binary(buf.getvalue())


def _backfill_close_series_payload(collection, ticker: str, close_series: pd.Series, column_name: str) -> None:
    close_payload = _serialize_close_series_payload(close_series, column_name)
    if close_payload is None:
        return

    try:
        collection.update_one(
            {"ticker": ticker},
            {
                "$set": {
                    "close_data": close_payload,
                    "close_column": column_name,
                    "close_row_count": int(len(close_series)),
                }
            },
        )
    except Exception:
        return


def _get_cache_start_date() -> pd.Timestamp | None:
    """config.py에서 CACHE_START_DATE를 로드하여 Timestamp로 반환합니다."""
    try:
        from utils.settings_loader import load_common_settings

        common_settings = load_common_settings()
        raw = common_settings.get("CACHE_START_DATE")
    except Exception:
        return None

    if not raw:
        return None

    try:
        ts = pd.to_datetime(raw).normalize()
        if isinstance(ts, pd.DatetimeIndex):
            ts = ts[0]
        if isinstance(ts, pd.Timestamp):
            if ts.tzinfo is not None:
                ts = ts.tz_localize(None)
            return ts.normalize()
    except Exception:
        return None

    return None


# 종목풀·계좌가 아닌 **참조 시세**의 저장 위치.
# 이들은 OHLCV 저장 코드를 재사용하려고 같은 함수에 토큰만 다르게 넘겨 쓴다. 그런데 수명이
# 정반대다 — 소유자 캐시(`cache_<소유자>_stocks`)는 종목풀·계좌가 사라지면 함께 지워야 하고,
# 참조 시세는 소유자가 없어 절대 지우면 안 된다. 이름 형식이 같으면 구분이 불가능해서
# 고아 점검이 환율을 '주인 없는 데이터' 로 잡았다(예외 목록으로 막다가 형식을 분리했다).
_REFERENCE_COLLECTIONS = {
    "fx": "reference_fx_prices",
    "etf": "reference_index_prices",
}


def _resolve_collection_name(account_id: str) -> str:
    token = (account_id or "global").strip().lower() or "global"

    # 참조 시세는 소유자 캐시와 다른 이름 형식을 쓴다(위 주석 참고).
    if token in _REFERENCE_COLLECTIONS:
        return _REFERENCE_COLLECTIONS[token]

    # Temporary collection handling
    if "_tmp_" in token:
        base, _, suffix = token.partition("_tmp_")
        # Recursively resolve base collection name
        base_collection = _resolve_collection_name(base)
        suffix_clean = _TEMP_SUFFIX_SANITIZE.sub("", suffix)
        if not suffix_clean:
            raise ValueError(f"잘못된 임시 컬렉션 토큰: {account_id}")
        return f"{base_collection}_tmp_{suffix_clean}"

    # Default to generic cache_{token}_stocks pattern for accounts
    return f"cache_{token}_stocks"


def _get_collection(account_id: str):
    db = get_db_connection()
    if db is None:
        return None
    collection_name = _resolve_collection_name(account_id)
    collection = db[collection_name]
    # 보조 인덱스 생성 (존재 시 무시)
    try:
        collection.create_index("ticker", unique=True, name="ticker_unique", background=True)
    except Exception:
        pass
    return collection


def _get_refresh_status_collection():
    db = get_db_connection()
    if db is None:
        return None
    collection = db[_REFRESH_STATUS_COLLECTION]
    try:
        collection.create_index("target_id", unique=True, name="target_id_unique", background=True)
    except Exception:
        pass
    return collection


def get_cache_lookup_keys(account_id: str) -> list[str]:
    """캐시 조회 키를 반환한다.

    암묵적인 fallback 없이, 전달된 account_id/ticker_type 자신만 조회한다.
    """
    token = (account_id or "").strip().lower()
    if not token:
        return []
    return [token]


def get_all_ticker_type_lookup_keys() -> list[str]:
    """모든 종목풀 캐시 키를 명시적으로 반환한다."""
    from utils.settings_loader import list_available_ticker_types

    return [
        str(ticker_type).strip().lower() for ticker_type in list_available_ticker_types() if str(ticker_type).strip()
    ]


def _load_cached_frames_bulk_from_keys(cache_keys: Iterable[str], tickers: Iterable[str]) -> dict[str, pd.DataFrame]:
    normalized = []
    for ticker in tickers:
        norm = (ticker or "").strip().upper()
        if norm:
            normalized.append(norm)
    if not normalized:
        return {}

    frames: dict[str, pd.DataFrame] = {}
    missing = set(normalized)

    for cache_key in cache_keys:
        key_norm = str(cache_key or "").strip().lower()
        if not key_norm or not missing:
            continue
        fetched = load_cached_frames_bulk(key_norm, missing)
        if not fetched:
            continue
        frames.update(fetched)
        missing -= set(fetched.keys())

    return frames


def _deserialize_cached_doc(doc: dict[str, Any], collection=None) -> pd.DataFrame | None:
    """공통 캐시 문서 역직렬화 로직."""
    if not doc:
        return None

    payload = doc.get("data")
    if payload is None:
        return None

    ticker_name = doc.get("ticker", "UNKNOWN")
    df = None
    try:
        buf = io.BytesIO(payload)
        df = pd.read_parquet(buf, engine="pyarrow")
    except Exception as e:
        logger.warning(
            "캐시 역직렬화 실패 (%s). 구버전 캐시이거나 데이터가 손상되었습니다. 재생성합니다. (Error: %s)",
            ticker_name,
            e,
        )
        if collection is not None and ticker_name != "UNKNOWN":
            try:
                collection.delete_one({"ticker": ticker_name})
                logger.info("손상된 캐시 문서 삭제됨: %s (ticker: %s)", collection.name, ticker_name)
            except Exception as e_delete:
                logger.error("손상된 캐시 문서 삭제 실패 (ticker: %s): %s", ticker_name, e_delete)
        return None

    if df is None or df.empty:
        return None
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            return None
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="first")]

    cache_start = _get_cache_start_date()
    if cache_start is not None:
        df = df[df.index >= cache_start]

    if df.empty:
        return None
    return df


def _deserialize_cached_close_series_doc(doc: dict[str, Any], collection=None) -> pd.Series | None:
    """캐시 문서에서 종가 시리즈만 역직렬화한다."""
    if not doc:
        return None

    payload = doc.get("close_data")
    close_column = str(doc.get("close_column") or "").strip() or None
    if payload is None:
        payload = doc.get("data")
    if payload is None:
        return None

    ticker_name = doc.get("ticker", "UNKNOWN")
    columns = doc.get("columns")
    candidate_columns: list[str] = []
    if close_column:
        candidate_columns.append(close_column)
    resolved_column = _resolve_close_column(columns if isinstance(columns, list) else None)
    if resolved_column and resolved_column not in candidate_columns:
        candidate_columns.append(resolved_column)
    if not candidate_columns:
        candidate_columns = ["unadjusted_close", "Close", "close"]

    close_df = None
    last_error = None
    for index, candidate in enumerate(candidate_columns):
        try:
            buf = io.BytesIO(payload)
            if doc.get("close_data") is not None and index == 0:
                close_df = pd.read_parquet(buf, engine="pyarrow")
            else:
                close_df = pd.read_parquet(buf, engine="pyarrow", columns=[candidate])
            if close_df is not None and not close_df.empty and candidate in close_df.columns:
                break
            if doc.get("close_data") is not None and close_df is not None and not close_df.empty:
                break
        except Exception as exc:
            last_error = exc
            close_df = None

    if close_df is None or close_df.empty:
        if collection is not None and last_error is not None and ticker_name != "UNKNOWN":
            logger.warning("종가 시리즈 역직렬화 실패 (%s): %s", ticker_name, last_error)
        return None

    if not isinstance(close_df.index, pd.DatetimeIndex):
        try:
            close_df.index = pd.to_datetime(close_df.index)
        except Exception:
            return None

    close_df = close_df.sort_index()
    close_df = close_df[~close_df.index.duplicated(keep="first")]

    close_series = pd.to_numeric(close_df.iloc[:, 0], errors="coerce").dropna()
    if close_series.empty:
        return None

    cache_start = _get_cache_start_date()
    if cache_start is not None:
        close_series = close_series[close_series.index >= cache_start]

    if close_series.empty:
        return None
    return close_series.astype(float)


def load_cached_frame(account_id: str, ticker: str) -> pd.DataFrame | None:
    """저장된 캐시 DataFrame을 로드하고, CACHE_START_DATE 이전 데이터를 필터링합니다."""
    collection = _get_collection(account_id)
    if collection is None:
        return None

    try:
        doc = collection.find_one({"ticker": (ticker or "").strip().upper()})
    except Exception:
        return None

    return _deserialize_cached_doc(doc, collection)


def load_cached_frame_with_fallback(account_id: str, ticker: str) -> pd.DataFrame | None:
    """계좌 캐시를 조회한다."""
    for cache_key in get_cache_lookup_keys(account_id):
        df = load_cached_frame(cache_key, ticker)
        if df is not None and not df.empty:
            return df
    return None


def load_cached_frames_bulk(account_id: str, tickers: Iterable[str]) -> dict[str, pd.DataFrame]:
    """다수의 티커를 한 번의 질의로 가져와 역직렬화합니다."""
    normalized = []
    for t in tickers:
        norm = (t or "").strip().upper()
        if norm:
            normalized.append(norm)
    if not normalized:
        return {}

    collection = _get_collection(account_id)
    if collection is None:
        return {}

    frames: dict[str, pd.DataFrame] = {}
    try:
        cursor = collection.find({"ticker": {"$in": list(set(normalized))}})
    except Exception:
        return {}

    for doc in cursor:
        ticker = (doc.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        df = _deserialize_cached_doc(doc, collection)  # Pass collection for potential cleanup
        if df is None:
            continue
        frames[ticker] = df

    return frames


def load_cached_close_series_bulk(account_id: str, tickers: Iterable[str]) -> dict[str, pd.Series]:
    """다수의 티커에 대한 종가 시리즈만 한 번에 가져온다."""
    normalized = []
    for t in tickers:
        norm = (t or "").strip().upper()
        if norm:
            normalized.append(norm)
    if not normalized:
        return {}

    collection = _get_collection(account_id)
    if collection is None:
        return {}

    series_map: dict[str, pd.Series] = {}
    collection_name = collection.name

    try:
        metadata_cursor = collection.find(
            {"ticker": {"$in": list(set(normalized))}}, {"_id": 0, "ticker": 1, "updated_at": 1}
        )
    except Exception:
        return {}

    pending_tickers: list[str] = []
    for doc in metadata_cursor:
        ticker = (doc.get("ticker") or "").strip().upper()
        if not ticker:
            continue

        updated_at = doc.get("updated_at") if isinstance(doc.get("updated_at"), datetime) else None
        cached_series = _get_close_series_memory_cache(collection_name, ticker, updated_at)
        if cached_series is not None:
            series_map[ticker] = cached_series
            continue
        pending_tickers.append(ticker)

    if not pending_tickers:
        return series_map

    try:
        cursor = collection.find(
            {"ticker": {"$in": pending_tickers}},
            {"_id": 0, "ticker": 1, "updated_at": 1, "close_data": 1, "close_column": 1},
        )
    except Exception:
        return series_map

    fallback_tickers: list[str] = []
    for doc in cursor:
        ticker = (doc.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        close_series = _deserialize_cached_close_series_doc(doc, collection)
        if close_series is None:
            fallback_tickers.append(ticker)
            continue
        updated_at = doc.get("updated_at") if isinstance(doc.get("updated_at"), datetime) else None
        _set_close_series_memory_cache(collection_name, ticker, updated_at, close_series)
        series_map[ticker] = close_series

    if not fallback_tickers:
        return series_map

    try:
        fallback_cursor = collection.find(
            {"ticker": {"$in": fallback_tickers}},
            {"_id": 0, "ticker": 1, "updated_at": 1, "data": 1, "columns": 1},
        )
    except Exception:
        return series_map

    for doc in fallback_cursor:
        ticker = (doc.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        close_series = _deserialize_cached_close_series_doc(doc, collection)
        if close_series is None:
            continue
        updated_at = doc.get("updated_at") if isinstance(doc.get("updated_at"), datetime) else None
        close_column = (
            _resolve_close_column(doc.get("columns") if isinstance(doc.get("columns"), list) else None) or "Close"
        )
        _backfill_close_series_payload(collection, ticker, close_series, close_column)
        _set_close_series_memory_cache(collection_name, ticker, updated_at, close_series)
        series_map[ticker] = close_series

    return series_map


def load_cached_close_series_bulk_before_or_at(
    account_id: str,
    tickers: Iterable[str],
    completed_at: datetime,
) -> dict[str, pd.Series]:
    """완료 시각 이하로 저장된 종가 시리즈만 조회한다."""
    normalized = []
    for t in tickers:
        norm = (t or "").strip().upper()
        if norm:
            normalized.append(norm)
    if not normalized:
        return {}

    collection = _get_collection(account_id)
    if collection is None:
        return {}

    series_map: dict[str, pd.Series] = {}
    collection_name = collection.name

    try:
        metadata_cursor = collection.find(
            {
                "ticker": {"$in": list(set(normalized))},
                "updated_at": {"$lte": completed_at},
            },
            {"_id": 0, "ticker": 1, "updated_at": 1},
        )
    except Exception:
        return {}

    pending_tickers: list[str] = []
    for doc in metadata_cursor:
        ticker = (doc.get("ticker") or "").strip().upper()
        if not ticker:
            continue

        updated_at = doc.get("updated_at") if isinstance(doc.get("updated_at"), datetime) else None
        cached_series = _get_close_series_memory_cache(collection_name, ticker, updated_at)
        if cached_series is not None:
            series_map[ticker] = cached_series
            continue
        pending_tickers.append(ticker)

    if not pending_tickers:
        return series_map

    try:
        cursor = collection.find(
            {
                "ticker": {"$in": pending_tickers},
                "updated_at": {"$lte": completed_at},
            },
            {"_id": 0, "ticker": 1, "updated_at": 1, "close_data": 1, "close_column": 1},
        )
    except Exception:
        return series_map

    fallback_tickers: list[str] = []
    for doc in cursor:
        ticker = (doc.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        close_series = _deserialize_cached_close_series_doc(doc, collection)
        if close_series is None:
            fallback_tickers.append(ticker)
            continue
        updated_at = doc.get("updated_at") if isinstance(doc.get("updated_at"), datetime) else None
        _set_close_series_memory_cache(collection_name, ticker, updated_at, close_series)
        series_map[ticker] = close_series

    if not fallback_tickers:
        return series_map

    try:
        fallback_cursor = collection.find(
            {
                "ticker": {"$in": fallback_tickers},
                "updated_at": {"$lte": completed_at},
            },
            {"_id": 0, "ticker": 1, "updated_at": 1, "data": 1, "columns": 1},
        )
    except Exception:
        return series_map

    for doc in fallback_cursor:
        ticker = (doc.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        close_series = _deserialize_cached_close_series_doc(doc, collection)
        if close_series is None:
            continue
        updated_at = doc.get("updated_at") if isinstance(doc.get("updated_at"), datetime) else None
        close_column = (
            _resolve_close_column(doc.get("columns") if isinstance(doc.get("columns"), list) else None) or "Close"
        )
        _backfill_close_series_payload(collection, ticker, close_series, close_column)
        _set_close_series_memory_cache(collection_name, ticker, updated_at, close_series)
        series_map[ticker] = close_series

    return series_map


def load_cached_frames_bulk_with_fallback(account_id: str, tickers: Iterable[str]) -> dict[str, pd.DataFrame]:
    """계좌 캐시를 조회한다."""
    return _load_cached_frames_bulk_from_keys(get_cache_lookup_keys(account_id), tickers)


def load_cached_frames_bulk_from_all_ticker_types(tickers: Iterable[str]) -> dict[str, pd.DataFrame]:
    """모든 종목풀 캐시에서 순서대로 조회한다.

    계좌 보유 화면처럼 종목풀이 섞일 수 있는 경우에만 명시적으로 사용한다.
    """
    return _load_cached_frames_bulk_from_keys(get_all_ticker_type_lookup_keys(), tickers)


def load_cached_frames_bulk_from_ticker_types(
    ticker_types: Iterable[str],
    tickers: Iterable[str],
) -> dict[str, pd.DataFrame]:
    """지정한 종목풀 캐시에서만 순서대로 OHLCV 프레임을 조회한다."""
    cache_keys = [str(ticker_type or "").strip().lower() for ticker_type in ticker_types]
    cache_keys = [ticker_type for ticker_type in cache_keys if ticker_type]
    if not cache_keys:
        raise ValueError("조회할 ticker_types가 필요합니다.")
    return _load_cached_frames_bulk_from_keys(cache_keys, tickers)


def load_cached_close_series_bulk_with_fallback(account_id: str, tickers: Iterable[str]) -> dict[str, pd.Series]:
    """계좌 캐시에서 종가 시리즈만 조회한다."""
    normalized = []
    for ticker in tickers:
        norm = (ticker or "").strip().upper()
        if norm:
            normalized.append(norm)
    if not normalized:
        return {}

    series_map: dict[str, pd.Series] = {}
    missing = set(normalized)

    for cache_key in get_cache_lookup_keys(account_id):
        if not missing:
            break
        fetched = load_cached_close_series_bulk(cache_key, missing)
        if not fetched:
            continue
        series_map.update(fetched)
        missing -= set(fetched.keys())

    return series_map


def load_cached_updated_at_bulk(account_id: str, tickers: Iterable[str]) -> dict[str, datetime]:
    """다수의 티커에 대한 캐시 updated_at 시각을 한 번에 조회합니다."""
    normalized = []
    for t in tickers:
        norm = (t or "").strip().upper()
        if norm:
            normalized.append(norm)
    if not normalized:
        return {}

    collection = _get_collection(account_id)
    if collection is None:
        return {}

    results: dict[str, datetime] = {}
    try:
        cursor = collection.find({"ticker": {"$in": list(set(normalized))}}, {"_id": 0, "ticker": 1, "updated_at": 1})
    except Exception:
        return {}

    for doc in cursor:
        ticker = (doc.get("ticker") or "").strip().upper()
        updated_at = doc.get("updated_at")
        if ticker and isinstance(updated_at, datetime):
            results[ticker] = updated_at

    return results


def load_cached_updated_at_bulk_before_or_at(
    account_id: str,
    tickers: Iterable[str],
    completed_at: datetime,
) -> dict[str, datetime]:
    """완료 시각 이하로 저장된 캐시 updated_at 시각을 조회한다."""
    normalized = []
    for t in tickers:
        norm = (t or "").strip().upper()
        if norm:
            normalized.append(norm)
    if not normalized:
        return {}

    collection = _get_collection(account_id)
    if collection is None:
        return {}

    results: dict[str, datetime] = {}
    try:
        cursor = collection.find(
            {
                "ticker": {"$in": list(set(normalized))},
                "updated_at": {"$lte": completed_at},
            },
            {"_id": 0, "ticker": 1, "updated_at": 1},
        )
    except Exception:
        return {}

    for doc in cursor:
        ticker = (doc.get("ticker") or "").strip().upper()
        updated_at = doc.get("updated_at")
        if ticker and isinstance(updated_at, datetime):
            results[ticker] = updated_at

    return results


def load_cached_updated_at_bulk_with_fallback(account_id: str, tickers: Iterable[str]) -> dict[str, datetime]:
    """계좌 캐시의 updated_at 시각을 조회합니다."""
    normalized = []
    for ticker in tickers:
        norm = (ticker or "").strip().upper()
        if norm:
            normalized.append(norm)
    if not normalized:
        return {}

    updated_map: dict[str, datetime] = {}
    missing = set(normalized)

    for cache_key in get_cache_lookup_keys(account_id):
        if not missing:
            break
        fetched = load_cached_updated_at_bulk(cache_key, missing)
        if not fetched:
            continue
        updated_map.update(fetched)
        missing -= set(fetched.keys())

    return updated_map


def load_cached_updated_at_bulk_before_or_at_with_fallback(
    account_id: str,
    tickers: Iterable[str],
    completed_at: datetime,
) -> dict[str, datetime]:
    """계좌 캐시에서 완료 시각 이하의 updated_at만 조회한다."""
    normalized = []
    for ticker in tickers:
        norm = (ticker or "").strip().upper()
        if norm:
            normalized.append(norm)
    if not normalized:
        return {}

    updated_map: dict[str, datetime] = {}
    missing = set(normalized)

    for cache_key in get_cache_lookup_keys(account_id):
        if not missing:
            break
        fetched = load_cached_updated_at_bulk_before_or_at(cache_key, missing, completed_at)
        if not fetched:
            continue
        updated_map.update(fetched)
        missing -= set(fetched.keys())

    return updated_map


def load_cached_close_series_bulk_before_or_at_with_fallback(
    account_id: str,
    tickers: Iterable[str],
    completed_at: datetime,
) -> dict[str, pd.Series]:
    """계좌 캐시에서 완료 시각 이하의 종가 시리즈만 조회한다."""
    normalized = []
    for ticker in tickers:
        norm = (ticker or "").strip().upper()
        if norm:
            normalized.append(norm)
    if not normalized:
        return {}

    series_map: dict[str, pd.Series] = {}
    missing = set(normalized)

    for cache_key in get_cache_lookup_keys(account_id):
        if not missing:
            break
        fetched = load_cached_close_series_bulk_before_or_at(cache_key, missing, completed_at)
        if not fetched:
            continue
        series_map.update(fetched)
        missing -= set(fetched.keys())

    return series_map


def get_cache_refresh_completed_at(target_id: str) -> datetime | None:
    """지정 대상의 마지막 가격 캐시 완료 시각을 조회한다."""
    target_norm = (target_id or "").strip().lower()
    if not target_norm:
        return None

    collection = _get_refresh_status_collection()
    if collection is None:
        return None

    try:
        doc = collection.find_one({"target_id": target_norm}, {"_id": 0, "completed_at": 1})
    except Exception:
        return None

    completed_at = (doc or {}).get("completed_at")
    return completed_at if isinstance(completed_at, datetime) else None


def set_cache_refresh_completed_at(target_id: str, completed_at: datetime) -> None:
    """지정 대상의 마지막 가격 캐시 완료 시각을 저장한다."""
    target_norm = (target_id or "").strip().lower()
    if not target_norm:
        raise ValueError("target_id가 필요합니다.")

    collection = _get_refresh_status_collection()
    if collection is None:
        raise RuntimeError("캐시 완료 시각 컬렉션을 열 수 없습니다.")

    completed_at_utc = (
        completed_at.replace(tzinfo=None)
        if completed_at.tzinfo is None
        else completed_at.astimezone(timezone.utc).replace(tzinfo=None)
    )

    collection.update_one(
        {"target_id": target_norm},
        {
            "$set": {
                "target_id": target_norm,
                "completed_at": completed_at_utc,
                "updated_at": datetime.utcnow(),
            }
        },
        upsert=True,
    )


# 이상치 경고 — 최근 이 일수 안에서 하루 등락이 한도를 넘으면 저장은 하되 슬랙으로 알린다.
# 차단하지 않는 이유: 유닛 병합·액면분할 직후처럼 정상적으로 큰 변동도 있다(호주 IOO 6.35:1).
# 한도 60% 초과: 개별주 상하한가(±30%)의 2배 = 단일종목 2배 레버리지 ETF 의 이론상 최대가
# 정확히 ±60%다 (2026-07-31 SK하이닉스 상한가 날 +58% 실측). 그 위는 정상 시장에서
# 나올 수 없어 데이터 사고(+542%, +1998% 등)만 걸린다.
_ANOMALY_ALERT_WINDOW_DAYS = 40
_ANOMALY_ALERT_PCT = 60.0
# 이상치 기록도 시세 캐시와 같은 기준으로 나눈다 — 소유자(종목풀·계좌) 것은 소유자가
# 사라질 때 함께 지워야 하고, 참조 시세 것은 지울 주인이 없다. 한 표에 섞으면 삭제·고아
# 점검이 값을 보고 예외를 판정해야 한다(그 예외가 환율 기록을 지울 뻔했다).
_ANOMALY_ALERT_COLLECTION = "price_anomaly_alerts"
_REFERENCE_ANOMALY_COLLECTION = "reference_price_anomalies"


def _lookup_ticker_name(cache_owner: str, ticker: str) -> str:
    """알림 표시용 종목명 — 풀 문서에서 찾고 없으면 다른 풀, 그래도 없으면 빈 값.

    `cache_owner` 가 종목풀이 아니면(환율 `fx` 등) stock_meta 에 없으므로 빈 값이 된다.
    """
    try:
        db = get_db_connection()
        if db is None:
            return ""
        doc = db.stock_meta.find_one(
            {"ticker_type": str(cache_owner).strip().lower(), "ticker": ticker}, {"name": 1}
        ) or db.stock_meta.find_one({"ticker": ticker, "name": {"$nin": [None, ""]}}, {"name": 1})
        return str((doc or {}).get("name") or "").strip()
    except Exception:
        return ""


def _alert_price_anomalies(cache_owner: str, ticker: str, df: pd.DataFrame) -> None:
    """저장 직전 종가의 하루 등락을 검사해 한도 초과를 슬랙으로 경고한다.

    같은 (캐시 소유자, 티커, 날짜) 조합은 한 번만 보낸다 — 오염된 프레임이 매시 재저장돼도
    슬랙이 도배되지 않게 DB 에 알림 이력을 남긴다. 검사·발송 실패는 저장을 막지 않는다.

    기록 위치는 시세 캐시와 같은 기준으로 나뉜다 — 종목풀·계좌 것은 `price_anomaly_alerts`,
    참조 시세(환율·레버리지 지수) 것은 `reference_price_anomalies`. 소유자가 사라질 때
    함께 지워야 하는지가 정반대라서다.
    """
    try:
        close_column = _resolve_close_column(df.columns.astype(str).tolist())
        if close_column is None:
            return
        close = pd.to_numeric(df[close_column], errors="coerce").dropna().astype(float)
        if len(close) < 2:
            return
        recent = close.iloc[-_ANOMALY_ALERT_WINDOW_DAYS:]
        # 부호를 살려 둔다 — 하락 사고를 (+60%) 로 보여주면 오독한다.
        change = recent.pct_change() * 100
        jumps = change[change.abs() > _ANOMALY_ALERT_PCT]
        if jumps.empty:
            return

        db = get_db_connection()
        if db is None:
            return
        alerts = db[
            _REFERENCE_ANOMALY_COLLECTION if cache_owner in _REFERENCE_COLLECTIONS else _ANOMALY_ALERT_COLLECTION
        ]
        lines = []
        for stamp, pct in jumps.items():
            key = {
                "cache_owner": str(cache_owner).strip().lower(),
                "ticker": (ticker or "").strip().upper(),
                "date": pd.Timestamp(stamp).strftime("%Y-%m-%d"),
            }
            marked = alerts.update_one(
                key,
                {"$setOnInsert": {**key, "pct": round(float(pct), 1), "created_at": datetime.utcnow()}},
                upsert=True,
            )
            if marked.upserted_id is not None:  # 처음 보는 (캐시 소유자, 티커, 날짜)만 알린다
                position = close.index.get_loc(stamp)
                before = float(close.iloc[position - 1])
                lines.append(f"· {key['date']} {before:,.2f} → {float(close.loc[stamp]):,.2f} ({pct:+.0f}%)")
        if not lines:
            return

        from utils.notification import send_slack_message_v2

        name = _lookup_ticker_name(cache_owner, key["ticker"])
        title = f"{key['cache_owner'].upper()}/{key['ticker']}" + (f" {name}" if name else "")
        send_slack_message_v2(
            f":rotating_light: 가격 캐시 이상치 감지 — {title}\n"
            + "\n".join(lines)
            + f"\n하루 등락 {_ANOMALY_ALERT_PCT:.0f}% 초과 (단일종목 2배 레버리지의 이론상 최대). "
            "분할·병합이면 정상이지만, yfinance 동시 호출 오염이나 원천 데이터 사고일 수 있어 "
            "확인이 필요합니다. 저장은 그대로 진행됐습니다."
        )
        logger.warning("[CACHE] 가격 이상치 감지 %s/%s: %s", cache_owner, ticker, "; ".join(lines))
    except Exception as exc:  # 경고 실패가 저장을 막으면 안 된다
        logger.warning("[CACHE] 가격 이상치 검사 실패 (%s/%s): %s", cache_owner, ticker, exc)


def save_cached_frame(account_id: str, ticker: str, df: pd.DataFrame) -> None:
    """캐시 DataFrame을 저장합니다. CACHE_START_DATE 이전 데이터는 제외합니다."""
    if df is None or df.empty:
        raise ValueError("저장할 캐시 데이터가 비어 있습니다.")

    collection = _get_collection(account_id)
    if collection is None:
        raise RuntimeError(f"캐시 컬렉션을 열 수 없습니다: {account_id}")

    df_to_save = df.copy()
    df_to_save.sort_index(inplace=True)
    df_to_save = df_to_save[~df_to_save.index.duplicated(keep="first")]

    # CACHE_START_DATE 이전 데이터 필터링
    cache_start = _get_cache_start_date()
    if cache_start is not None:
        df_to_save = df_to_save[df_to_save.index >= cache_start]

    if df_to_save.empty:
        raise ValueError("CACHE_START_DATE 적용 후 저장할 캐시 데이터가 비어 있습니다.")

    ticker_norm = (ticker or "").strip().upper()

    # 이상치 경고(차단 아님) — 뒤섞인 가격이 조용히 저장되는 것을 그날 알아채기 위한 그물.
    _alert_price_anomalies(account_id, ticker_norm, df_to_save)

    # [HOTFIX] 직렬화 오류 방지를 위한 정규화 및 중복 컬럼 제거
    df_to_save.columns = [str(c) for c in df_to_save.columns]
    df_to_save = df_to_save.loc[:, ~df_to_save.columns.duplicated()]
    if hasattr(df_to_save.index, "name"):
        df_to_save.index.name = None

    buf = io.BytesIO()
    try:
        df_to_save.to_parquet(buf, engine="pyarrow", compression="snappy")
    except Exception as exc:
        logger.error(f"캐시 직렬화 오류 발생 ({ticker_norm}): {exc}")
        raise RuntimeError(f"캐시 직렬화 실패 ({ticker_norm}): {exc}") from exc

    payload = Binary(buf.getvalue())

    try:
        result = collection.update_one(
            {"ticker": ticker_norm},
            {
                "$set": {
                    "ticker": ticker_norm,
                    "data": payload,
                    "updated_at": datetime.utcnow(),
                    "row_count": int(df_to_save.shape[0]),
                    "columns": df_to_save.columns.astype(str).tolist(),
                }
            },
            upsert=True,
        )
    except Exception as exc:
        raise RuntimeError(f"캐시 저장 실패 ({ticker_norm})") from exc

    if not result.acknowledged:
        raise RuntimeError(f"캐시 저장이 확인되지 않았습니다 ({ticker_norm})")

    close_column = _resolve_close_column(df_to_save.columns.astype(str).tolist())
    if close_column is not None:
        close_series = pd.to_numeric(df_to_save[close_column], errors="coerce").dropna().astype(float)
        if not close_series.empty:
            _backfill_close_series_payload(collection, ticker_norm, close_series, close_column)

    saved_doc = collection.find_one({"ticker": ticker_norm}, {"_id": 0, "row_count": 1})
    if not saved_doc:
        raise RuntimeError(f"저장 후 캐시 문서를 찾을 수 없습니다 ({ticker_norm})")

    saved_count = int(saved_doc.get("row_count") or 0)
    expected_count = int(df_to_save.shape[0])
    if saved_count != expected_count:
        raise RuntimeError(
            f"저장된 캐시 행 수가 다릅니다 ({ticker_norm}): expected={expected_count}, actual={saved_count}"
        )


def delete_cached_frame(account_id: str, ticker: str) -> None:
    collection = _get_collection(account_id)
    if collection is None:
        return
    try:
        collection.delete_one({"ticker": (ticker or "").strip().upper()})
    except Exception:
        return


def prune_cache_to_tickers(account_id: str, keep_tickers: Iterable[str]) -> list[str]:
    """``keep_tickers`` 에 없는 캐시 문서를 지우고 지운 티커 목록을 돌려준다.

    종목풀에서 빠진 종목의 캐시는 아무도 갱신하지 않아 그 시점에 얼어붙는다. 그대로 두면
    한 컬렉션 안에 최신 종목과 멈춘 종목이 섞이고, 나중에 컬렉션을 통째로 읽는 쪽이
    서로 다른 날짜의 값을 한 표본으로 쓰게 된다.

    보유 중인 종목은 호출자가 ``keep_tickers`` 에 포함시켜야 한다 — 풀에서 빠져도
    자산 화면이 그 가격을 계속 쓴다 (`utils/stocks_service` 의 종목 삭제 규칙과 같다).
    """
    collection = _get_collection(account_id)
    if collection is None:
        return []

    keep = {str(ticker or "").strip().upper() for ticker in keep_tickers}
    keep.discard("")
    if not keep:
        # 지킬 목록을 못 만든 상태에서 지우면 캐시를 통째로 날린다.
        logger.warning("[%s] 캐시 정리 대상 목록이 비어 있어 정리를 건너뜁니다.", account_id)
        return []

    try:
        orphans = [
            str(doc.get("ticker") or "").strip().upper()
            for doc in collection.find({"ticker": {"$nin": list(keep)}}, {"ticker": 1})
        ]
        if orphans:
            collection.delete_many({"ticker": {"$in": orphans}})
    except Exception as exc:
        logger.warning("[%s] 고아 캐시 정리 실패: %s", account_id, exc)
        return []
    return orphans


def drop_cache_collection(account_id: str) -> None:
    db = get_db_connection()
    if db is None:
        return
    collection_name = _resolve_collection_name(account_id)
    try:
        db[collection_name].drop()
    except Exception:
        return


def clean_temp_cache_collections(account_id: str, *, max_age_seconds: int | None = None) -> int:
    """남아 있는 임시 캐시 컬렉션을 조건에 맞게 삭제합니다."""
    db = get_db_connection()
    if db is None:
        return 0
    base_name = _resolve_collection_name(account_id)
    removed = 0
    try:
        threshold = None
        if max_age_seconds is not None and max_age_seconds > 0:
            threshold = datetime.utcnow().timestamp() - max_age_seconds

        for coll_name in db.list_collection_names():
            if not coll_name.startswith(f"{base_name}_tmp_"):
                continue
            if threshold is not None:
                parts = coll_name.rsplit("_", 2)
                if len(parts) >= 2:
                    try:
                        ts_val = int(parts[-2])
                        if ts_val >= threshold:
                            continue
                    except ValueError:
                        pass
            db[coll_name].drop()
            removed += 1
    except Exception:
        return removed
    return removed


def swap_cache_collection(account_id: str, temp_token: str) -> None:
    """임시 컬렉션을 메인 캐시 컬렉션으로 원자적으로 교체합니다."""
    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결을 초기화할 수 없습니다.")

    client = db.client
    db_name = db.name
    main_collection_name = _resolve_collection_name(account_id)
    temp_collection_name = _resolve_collection_name(temp_token)

    if temp_collection_name not in db.list_collection_names():
        raise ValueError(f"임시 컬렉션 '{temp_collection_name}'을 찾을 수 없습니다.")

    try:
        client.admin.command(
            {
                "renameCollection": f"{db_name}.{temp_collection_name}",
                "to": f"{db_name}.{main_collection_name}",
                "dropTarget": True,
            }
        )
        # rename 후 새 컬렉션 핸들을 초기화하여 (재)인덱스 보장
        _get_collection(account_id)
    except PyMongoError as exc:
        logger.error("캐시 컬렉션 교체 실패 (%s <- %s): %s", main_collection_name, temp_collection_name, exc)
        raise
    except Exception as exc:
        logger.error("캐시 컬렉션 교체 중 예외 발생: %s", exc)
        raise


def get_cached_date_range(account_id: str, ticker: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    df = load_cached_frame(account_id, ticker)
    if df is None or df.empty:
        return None
    return df.index.min(), df.index.max()


def list_cached_tickers(account_id: str) -> list[str]:
    collection = _get_collection(account_id)
    if collection is None:
        return []
    try:
        tickers = collection.distinct("ticker")
    except Exception:
        return []
    return sorted(str(ticker or "").upper() for ticker in tickers if ticker)
