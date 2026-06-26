"""leverage 전략 설정·상태의 단일 소스(MongoDB) 저장/조회.

기존 파일(`leverage/config/switch.json`, `leverage/state/last_recommendation_switch.json`)을
DB 로 이관한다. DB 문서가 없으면 기존 JSON 파일에서 1회 시드한다(무손실 이관).

- `leverage_config` 컬렉션: 사용자/튜닝이 갱신하는 **설정**(_id=profile, config=raw dict)
- `leverage_state`  컬렉션: 추천 배치가 매시 덮어쓰는 **런타임 상태**(_id=profile, state=dict)

멀티프로세스(fastapi/scheduler/worker)에서 변경이 반영되도록 설정은 짧은 TTL 캐시 +
저장 시 무효화를 쓴다. 상태는 매 실행마다 최신을 읽어야 하므로 캐시하지 않는다.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from time import monotonic

from leverage.constants import CONFIG_DIR, STATE_DIR
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


def _seed_config_from_file(profile: str) -> dict | None:
    path = CONFIG_DIR / f"{profile}.json"
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_leverage_config_raw(profile: str = "switch") -> dict:
    """DB 의 원본(정규화 전) 설정 dict. 없으면 파일에서 시드 후 반환."""
    now = monotonic()
    with _lock:
        cached = _config_cache.get(profile)
        if cached and now - cached[1] < _CACHE_TTL_SECONDS:
            return dict(cached[0])

    db = _db()
    doc = db[_CONFIG_COLLECTION].find_one({"_id": profile})
    if doc is None:
        seed = _seed_config_from_file(profile)
        if seed is None:
            raise RuntimeError(f"leverage 설정이 DB·파일 모두 없습니다: {profile}")
        db[_CONFIG_COLLECTION].update_one(
            {"_id": profile},
            {"$set": {"config": seed, "updated_at": datetime.now(timezone.utc)}},
            upsert=True,
        )
        config = seed
    else:
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
    """직전 추천 상태. DB 없으면 파일에서 시드."""
    db = _db()
    doc = db[_STATE_COLLECTION].find_one({"_id": profile})
    if doc is not None:
        return dict(doc.get("state") or {})

    path = STATE_DIR / f"last_recommendation_{profile}.json"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        state = json.load(f)
    db[_STATE_COLLECTION].update_one(
        {"_id": profile},
        {"$set": {"state": state, "updated_at": datetime.now(timezone.utc)}},
        upsert=True,
    )
    return state


def save_leverage_state(profile: str, state: dict) -> None:
    db = _db()
    db[_STATE_COLLECTION].update_one(
        {"_id": profile},
        {"$set": {"state": state, "updated_at": datetime.now(timezone.utc)}},
        upsert=True,
    )
