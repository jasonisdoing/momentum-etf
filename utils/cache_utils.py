"""OHLCV 데이터를 MongoDB에 캐싱하고 관리하기 위한 헬퍼 함수 모음입니다."""

from __future__ import annotations

import pickle
import re
from collections.abc import Iterable
from datetime import datetime
from typing import Any

import pandas as pd
from bson.binary import Binary
from pymongo.errors import PyMongoError

from utils.db_manager import get_db_connection
from utils.logger import get_app_logger

logger = get_app_logger()

_COLLECTION_NAME_MAP = {
    "kor": "cache_kor_stocks",
    "us": "cache_us_stocks",
}

_TEMP_SUFFIX_SANITIZE = re.compile(r"[^a-z0-9_-]", re.IGNORECASE)


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


def _resolve_collection_name(country_token: str) -> str:
    token = (country_token or "global").strip().lower() or "global"
    if token in _COLLECTION_NAME_MAP:
        return _COLLECTION_NAME_MAP[token]
    if "_tmp_" in token:
        base, _, suffix = token.partition("_tmp_")
        base_collection = _COLLECTION_NAME_MAP.get(base, f"cache_{base}_stocks")
        suffix_clean = _TEMP_SUFFIX_SANITIZE.sub("", suffix)
        if not suffix_clean:
            raise ValueError(f"잘못된 임시 컬렉션 토큰: {country_token}")
        return f"{base_collection}_tmp_{suffix_clean}"
    return f"cache_{token}_stocks"


def _get_collection(country: str):
    db = get_db_connection()
    if db is None:
        return None
    collection_name = _resolve_collection_name(country)
    collection = db[collection_name]
    # 보조 인덱스 생성 (존재 시 무시)
    try:
        collection.create_index("ticker", unique=True, name="ticker_unique", background=True)
    except Exception:
        pass
    return collection


def _deserialize_cached_doc(doc: dict[str, Any]) -> pd.DataFrame | None:
    """공통 캐시 문서 역직렬화 로직."""
    if not doc:
        return None

    payload = doc.get("data")
    if payload is None:
        return None

    try:
        df = pickle.loads(payload)
    except Exception:
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


def load_cached_frame(country: str, ticker: str) -> pd.DataFrame | None:
    """저장된 캐시 DataFrame을 로드하고, CACHE_START_DATE 이전 데이터를 필터링합니다."""
    collection = _get_collection(country)
    if collection is None:
        return None

    try:
        doc = collection.find_one({"ticker": (ticker or "").strip().upper()})
    except Exception:
        return None

    return _deserialize_cached_doc(doc)


def load_cached_frames_bulk(country: str, tickers: Iterable[str]) -> dict[str, pd.DataFrame]:
    """다수의 티커를 한 번의 질의로 가져와 역직렬화합니다."""
    normalized = []
    for t in tickers:
        norm = (t or "").strip().upper()
        if norm:
            normalized.append(norm)
    if not normalized:
        return {}

    collection = _get_collection(country)
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
        df = _deserialize_cached_doc(doc)
        if df is None:
            continue
        frames[ticker] = df

    return frames


def save_cached_frame(country: str, ticker: str, df: pd.DataFrame) -> None:
    """캐시 DataFrame을 저장합니다. CACHE_START_DATE 이전 데이터는 제외합니다."""
    if df is None or df.empty:
        return

    collection = _get_collection(country)
    if collection is None:
        return

    df_to_save = df.copy()
    df_to_save.sort_index(inplace=True)
    df_to_save = df_to_save[~df_to_save.index.duplicated(keep="first")]

    # CACHE_START_DATE 이전 데이터 필터링
    cache_start = _get_cache_start_date()
    if cache_start is not None:
        df_to_save = df_to_save[df_to_save.index >= cache_start]

    if df_to_save.empty:
        return

    try:
        payload = Binary(pickle.dumps(df_to_save, protocol=pickle.HIGHEST_PROTOCOL))
        collection.update_one(
            {"ticker": (ticker or "").strip().upper()},
            {
                "$set": {
                    "ticker": (ticker or "").strip().upper(),
                    "data": payload,
                    "updated_at": datetime.utcnow(),
                    "row_count": int(df_to_save.shape[0]),
                    "columns": df_to_save.columns.astype(str).tolist(),
                }
            },
            upsert=True,
        )
    except Exception:
        return


def delete_cached_frame(country: str, ticker: str) -> None:
    collection = _get_collection(country)
    if collection is None:
        return
    try:
        collection.delete_one({"ticker": (ticker or "").strip().upper()})
    except Exception:
        return


def drop_cache_collection(country: str) -> None:
    db = get_db_connection()
    if db is None:
        return
    collection_name = _resolve_collection_name(country)
    try:
        db[collection_name].drop()
    except Exception:
        return


def clean_temp_cache_collections(country: str, *, max_age_seconds: int | None = None) -> int:
    """남아 있는 임시 캐시 컬렉션을 조건에 맞게 삭제합니다."""
    db = get_db_connection()
    if db is None:
        return 0
    base_name = _COLLECTION_NAME_MAP.get(country.strip().lower(), f"cache_{country}_stocks")
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


def swap_cache_collection(country: str, temp_country_token: str) -> None:
    """임시 컬렉션을 메인 캐시 컬렉션으로 원자적으로 교체합니다."""
    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결을 초기화할 수 없습니다.")

    client = db.client
    db_name = db.name
    main_collection_name = _resolve_collection_name(country)
    temp_collection_name = _resolve_collection_name(temp_country_token)

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
        _get_collection(country)
    except PyMongoError as exc:
        logger.error("캐시 컬렉션 교체 실패 (%s <- %s): %s", main_collection_name, temp_collection_name, exc)
        raise
    except Exception as exc:
        logger.error("캐시 컬렉션 교체 중 예외 발생: %s", exc)
        raise


def get_cached_date_range(country: str, ticker: str) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    df = load_cached_frame(country, ticker)
    if df is None or df.empty:
        return None
    return df.index.min(), df.index.max()


def list_cached_countries() -> list[str]:
    db = get_db_connection()
    if db is None:
        return []

    available: list[str] = []
    try:
        existing = set(db.list_collection_names())
    except Exception:
        existing = set()

    for country, coll in _COLLECTION_NAME_MAP.items():
        if coll in existing:
            try:
                if db[coll].estimated_document_count() > 0:
                    available.append(country)
            except Exception:
                continue
    if not available:
        available = sorted(_COLLECTION_NAME_MAP.keys())
    return available


def list_cached_tickers(country: str) -> list[str]:
    collection = _get_collection(country)
    if collection is None:
        return []
    try:
        tickers = collection.distinct("ticker")
    except Exception:
        return []
    return sorted(str(ticker or "").upper() for ticker in tickers if ticker)
