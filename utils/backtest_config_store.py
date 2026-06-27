"""백테스트 탐색공간(BACKTEST_CONFIG)의 단일 소스(MongoDB) 저장/조회.

기존 `config.py` 의 하드코딩 `BACKTEST_CONFIG` 를 DB 로 이관한다. 풀별로 1개 문서.

    {_id: <pool_id>, BENCHMARK:{ticker,name},
     HOLDING_BONUS_SCORE:[...], MA_TYPE:[...], MA_MONTHS:[...], RSI_LIMIT:[...], updated_at}

DB 가 유일한 소스다. 설정이 없으면 임의 기본값으로 보정하지 않고 **명확히 에러**를 낸다.
멀티프로세스(fastapi/scheduler/worker) 반영을 위해 짧은 TTL 캐시 + 저장 시 무효화를 쓴다.

라이브 단일 적용값은 `pool_settings`(utils/pool_settings_store.py), 백테스트 탐색공간은 여기.
같은 파라미터명이지만 역할이 다르다(라이브 단일값 vs 백테스트 리스트).
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from time import monotonic
from typing import Any

from config import ALLOWED_MA_TYPES

_COLLECTION = "backtest_config"
_CACHE_TTL_SECONDS = 30.0

# 풀별 문서 필수 키
_REQUIRED_LIST_KEYS = ("HOLDING_BONUS_SCORE", "MA_TYPE", "MA_MONTHS", "RSI_LIMIT")

_lock = threading.Lock()
_cache: dict[str, tuple[dict, float]] = {}  # pool_id -> (config, cached_at)


def _db():
    from utils.db_manager import get_db_connection

    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결 실패 (backtest_config)")
    return db


def invalidate_cache(pool_id: str | None = None) -> None:
    with _lock:
        if pool_id is None:
            _cache.clear()
        else:
            _cache.pop(pool_id, None)


def validate_backtest_config(config: Any) -> None:
    """백테스트 탐색공간 형식 검증(실패 시 ValueError). UI 저장·시드 공용."""
    if not isinstance(config, dict):
        raise ValueError("백테스트 설정 형식이 올바르지 않습니다.")

    benchmark = config.get("BENCHMARK")
    if not isinstance(benchmark, dict) or not str(benchmark.get("ticker") or "").strip() or not str(benchmark.get("name") or "").strip():
        raise ValueError("'BENCHMARK' 에는 ticker/name 이 모두 필요합니다.")

    for key in _REQUIRED_LIST_KEYS:
        values = config.get(key)
        if not isinstance(values, list) or len(values) == 0:
            raise ValueError(f"'{key}' 는 1개 이상의 값을 가진 리스트여야 합니다.")

    for v in config["HOLDING_BONUS_SCORE"]:
        if not isinstance(v, (int, float)) or isinstance(v, bool) or v < 0:
            raise ValueError("'HOLDING_BONUS_SCORE' 는 0 이상의 숫자여야 합니다.")
    for v in config["MA_MONTHS"]:
        if not isinstance(v, (int, float)) or isinstance(v, bool) or int(v) <= 0:
            raise ValueError("'MA_MONTHS' 는 0보다 큰 정수여야 합니다.")
    for v in config["RSI_LIMIT"]:
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not (0 <= v <= 100):
            raise ValueError("'RSI_LIMIT' 는 0~100 범위의 숫자여야 합니다.")
    for v in config["MA_TYPE"]:
        if str(v).upper() not in ALLOWED_MA_TYPES:
            raise ValueError(f"'MA_TYPE' 값이 허용되지 않습니다: {v} (허용: {', '.join(ALLOWED_MA_TYPES)})")


def load_backtest_config(pool_id: str) -> dict:
    """풀의 백테스트 탐색공간 dict. 없으면 임의 기본값 없이 에러."""
    now = monotonic()
    with _lock:
        cached = _cache.get(pool_id)
        if cached and now - cached[1] < _CACHE_TTL_SECONDS:
            return dict(cached[0])

    db = _db()
    doc = db[_COLLECTION].find_one({"_id": pool_id})
    if doc is None:
        raise ValueError(
            f"DB(backtest_config)에 '{pool_id}' 백테스트 설정이 없습니다. "
            "모멘텀-설정 화면에서 저장하거나 시드해야 합니다."
        )
    config = {k: v for k, v in doc.items() if k not in ("_id", "updated_at")}

    with _lock:
        _cache[pool_id] = (config, monotonic())
    return dict(config)


def save_backtest_config(pool_id: str, config: dict) -> None:
    """검증 후 풀의 백테스트 탐색공간을 DB 에 저장하고 캐시를 무효화한다."""
    validate_backtest_config(config)
    payload = {k: v for k, v in config.items() if k not in ("_id", "updated_at")}
    db = _db()
    db[_COLLECTION].update_one(
        {"_id": pool_id},
        {"$set": {**payload, "updated_at": datetime.now(timezone.utc)}},
        upsert=True,
    )
    invalidate_cache(pool_id)


def list_backtest_pools() -> list[str]:
    """백테스트 설정이 등록된 풀 id 목록."""
    db = _db()
    return [str(doc["_id"]) for doc in db[_COLLECTION].find({}, {"_id": 1})]
