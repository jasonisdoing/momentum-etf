"""leverage 전략 설정·상태의 단일 소스(MongoDB) 저장/조회.

- `leverage_config` 컬렉션: 사용자/튜닝이 갱신하는 **설정**(_id=profile, config=raw dict)
- `leverage_state`  컬렉션: 추천 배치가 매시 덮어쓰는 **런타임 상태**(_id=profile, state=dict)

DB 가 유일한 소스다. 설정이 없으면 임의 기본값으로 보정하지 않고 **명확히 에러**를 낸다
(파일 fallback·silent default 없음). 멀티프로세스(fastapi/scheduler/worker) 반영을 위해
설정은 짧은 TTL 캐시 + 저장 시 무효화를 쓰고, 상태는 매번 최신을 읽는다.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from time import monotonic

from leverage.engine.backtest.settings import normalize_settings

_CONFIG_COLLECTION = "leverage_config"
_STATE_COLLECTION = "leverage_state"
_CACHE_TTL_SECONDS = 30.0

_lock = threading.Lock()
_config_cache: dict[str, tuple[dict, float]] = {}  # profile -> (raw_config, cached_at)


def _db():
    from utils.db_manager import get_db_connection

    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결 실패 (leverage 설정/상태)")
    return db


def invalidate_config_cache(profile: str | None = None) -> None:
    with _lock:
        if profile is None:
            _config_cache.clear()
        else:
            _config_cache.pop(profile, None)


def load_leverage_config_raw(profile: str = "switch") -> dict:
    """DB 의 원본(정규화 전) 설정 dict. 없으면 임의 기본값 없이 에러를 낸다."""
    now = monotonic()
    with _lock:
        cached = _config_cache.get(profile)
        if cached and now - cached[1] < _CACHE_TTL_SECONDS:
            return dict(cached[0])

    db = _db()
    doc = db[_CONFIG_COLLECTION].find_one({"_id": profile})
    if doc is None:
        raise RuntimeError(
            f"DB 에 레버리지 설정이 없습니다 (profile={profile}). "
            "레버리지-설정 화면에서 저장하거나 직접 시드해야 합니다."
        )
    config = dict(doc.get("config") or {})

    with _lock:
        _config_cache[profile] = (config, monotonic())
    return dict(config)


def load_config(profile: str = "switch") -> dict:
    """검증·정규화된 설정 dict (엔진/추천/튜닝에서 사용)."""
    return normalize_settings(load_leverage_config_raw(profile))


def save_leverage_config_raw(profile: str, config: dict) -> None:
    """원본 설정 dict 를 DB 에 저장하고 캐시를 무효화한다."""
    db = _db()
    db[_CONFIG_COLLECTION].update_one(
        {"_id": profile},
        {"$set": {"config": config, "updated_at": datetime.now(timezone.utc)}},
        upsert=True,
    )
    invalidate_config_cache(profile)


def load_leverage_state(profile: str = "switch") -> dict:
    """직전 추천 상태. 없으면 빈 dict(= 직전 추천 없음, 임의 기본값 아님)."""
    db = _db()
    doc = db[_STATE_COLLECTION].find_one({"_id": profile})
    return dict(doc.get("state") or {}) if doc is not None else {}


def save_leverage_state(profile: str, state: dict) -> None:
    db = _db()
    db[_STATE_COLLECTION].update_one(
        {"_id": profile},
        {"$set": {"state": state, "updated_at": datetime.now(timezone.utc)}},
        upsert=True,
    )
