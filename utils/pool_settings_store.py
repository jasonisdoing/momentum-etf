"""종목풀 편집 가능 설정의 DB 오버라이드 레이어.

pools.json 은 종목풀의 구조(존재/order/icon/name/country_code/type_source 등)를 정의하는
단일 진실 소스로 유지하고, 자주 바뀌는 아래 5개 값만 MongoDB `pool_settings` 컬렉션에
저장해 화면에서 수정한다 (값 변경 때마다 커밋하던 번거로움 제거).

    TOP_N_HOLD, HOLDING_BONUS_SCORE, MA_TYPE, MA_MONTHS, RSI_LIMIT

읽기: settings_loader 가 pools.json 값 위에 DB 오버라이드를 덮어쓴다. DB 문서/키가 없으면
      pools.json 값을 그대로 사용한다 (silent fallback 아님 — 기본값이 명시적 소스).
캐시: 멀티프로세스(fastapi/scheduler/worker)에서 변경이 반영되도록 짧은 TTL(30초) 캐시를
      쓴다. 저장한 프로세스는 즉시 무효화하고, 나머지는 TTL 내 자동 반영된다.

컬렉션 문서 형태:
    {_id: "__all__",  TOP_N_HOLD, HOLDING_BONUS_SCORE, MA_TYPE, MA_MONTHS, RSI_LIMIT, updated_at}
    {_id: <ticker_type>, ...동일...}
"""

from __future__ import annotations

import threading
from datetime import datetime
from time import monotonic
from typing import Any

from utils.logger import get_app_logger

logger = get_app_logger()

# 전체 가상 종목풀(all) 의 문서 id
ALL_POOL_ID = "__all__"
COLLECTION = "pool_settings"

# DB 오버라이드 대상 키 (이 5개만 수정 가능)
OVERRIDABLE_KEYS: tuple[str, ...] = (
    "TOP_N_HOLD",
    "HOLDING_BONUS_SCORE",
    "MA_TYPE",
    "MA_MONTHS",
    "RSI_LIMIT",
)

_INT_KEYS = ("TOP_N_HOLD", "HOLDING_BONUS_SCORE", "MA_MONTHS", "RSI_LIMIT")

_CACHE_TTL_SECONDS = 30.0
_overlay_cache: dict[str, dict[str, Any]] | None = None
_overlay_cached_at: float = 0.0
_overlay_lock = threading.Lock()


class PoolSettingsError(ValueError):
    """종목풀 설정 검증/저장 오류."""


def invalidate_overlay_cache() -> None:
    """오버라이드 캐시를 비운다 (저장 직후 호출)."""
    global _overlay_cache, _overlay_cached_at
    with _overlay_lock:
        _overlay_cache = None
        _overlay_cached_at = 0.0


def _load_overrides_from_db() -> dict[str, dict[str, Any]]:
    """pool_settings 컬렉션 전체를 {pool_id: {key: value}} 로 읽는다. 실패 시 {}."""
    try:
        from utils.db_manager import get_db_connection

        db = get_db_connection()
        if db is None:
            return {}
        result: dict[str, dict[str, Any]] = {}
        for doc in db[COLLECTION].find({}):
            pool_id = str(doc.get("_id") or "").strip()
            if not pool_id:
                continue
            overrides = {k: doc[k] for k in OVERRIDABLE_KEYS if k in doc and doc[k] is not None}
            if overrides:
                result[pool_id] = overrides
        return result
    except Exception as exc:
        logger.warning("pool_settings 오버라이드 조회 실패: %s", exc)
        return {}


def get_overrides() -> dict[str, dict[str, Any]]:
    """TTL 캐시된 전체 오버라이드 맵을 반환한다 ({pool_id: {key: value}})."""
    global _overlay_cache, _overlay_cached_at
    now = monotonic()
    with _overlay_lock:
        if _overlay_cache is not None and (now - _overlay_cached_at) < _CACHE_TTL_SECONDS:
            return _overlay_cache
    # 락 밖에서 DB 조회 (느린 I/O 동안 락 보유 방지)
    loaded = _load_overrides_from_db()
    with _overlay_lock:
        _overlay_cache = loaded
        _overlay_cached_at = monotonic()
        return loaded


def resolve_pool_values(pool_id: str, base: dict[str, Any]) -> dict[str, Any]:
    """편집 가능한 5개 값을 DB(pool_settings)에서 가져와 base 에 덮어써 반환한다.

    DB 가 이 값들의 단일 소스다. pools.json 값으로의 silent fallback 은 하지 않는다 —
    DB 에 값이 없으면 명시적으로 에러를 낸다(시드 필요). pools.json 의 5개 값은
    최초 시드 소스로만 쓰이고, 읽기 시점에는 DB 값이 항상 우선한다.
    """
    pid = str(pool_id or "").strip()
    overrides = get_overrides().get(pid) or {}
    missing = [key for key in OVERRIDABLE_KEYS if key not in overrides]
    if missing:
        from utils.settings_loader import AccountSettingsError

        raise AccountSettingsError(
            f"종목풀 '{pid}' 의 설정이 DB(pool_settings)에 없습니다: {', '.join(missing)}. "
            f"`python scripts/seed_pool_settings.py` 로 시드가 필요합니다."
        )
    merged = dict(base)
    for key in OVERRIDABLE_KEYS:
        merged[key] = overrides[key]
    return merged


def seed_from_pools_json(*, overwrite: bool = False) -> dict[str, Any]:
    """pools.json 의 현재 5개 값을 pool_settings 컬렉션에 시드한다.

    overwrite=False: 이미 존재하는 문서/값은 건드리지 않고 빠진 것만 채운다($setOnInsert).
    overwrite=True : pools.json 값으로 전부 덮어쓴다.

    반환: {"seeded": [...], "skipped": [...], "overwritten": [...]} 요약.
    """
    from utils.db_manager import get_db_connection
    from utils.settings_loader import _get_all_pool_settings_raw, _load_pool_configs

    db = get_db_connection()
    if db is None:
        raise PoolSettingsError("DB 연결 실패로 시드할 수 없습니다.")

    # 시드 대상: (_id, {5개 값}) 목록 구성
    targets: list[tuple[str, dict[str, Any]]] = []
    all_raw = _get_all_pool_settings_raw()
    targets.append((ALL_POOL_ID, {k: all_raw[k] for k in OVERRIDABLE_KEYS if k in all_raw}))
    for config in _load_pool_configs():
        pid = str(config["ticker_type"])
        targets.append((pid, {k: config[k] for k in OVERRIDABLE_KEYS if k in config}))

    summary: dict[str, list[str]] = {"seeded": [], "skipped": [], "overwritten": []}
    now = datetime.utcnow()
    for pid, values in targets:
        # pools.json 에 5개 값이 없으면(= 이미 DB 전용으로 옮긴 경우) 시드할 게 없다.
        if not values:
            summary["skipped"].append(pid)
            continue
        existing = db[COLLECTION].find_one({"_id": pid})
        if existing and not overwrite:
            summary["skipped"].append(pid)
            continue
        db[COLLECTION].update_one(
            {"_id": pid},
            {"$set": {**values, "updated_at": now}},
            upsert=True,
        )
        summary["overwritten" if existing else "seeded"].append(pid)

    invalidate_overlay_cache()
    return summary


def _validate_values(values: dict[str, Any]) -> dict[str, Any]:
    """저장 입력값을 검증/정규화한다. 잘못된 값은 PoolSettingsError."""
    from utils.rankings import ALLOWED_MA_TYPES, get_rank_months_max

    cleaned: dict[str, Any] = {}
    months_max = get_rank_months_max()

    for key in OVERRIDABLE_KEYS:
        if key not in values:
            continue
        raw = values[key]

        if key == "MA_TYPE":
            ma_type = str(raw or "").strip().upper()
            if ma_type not in ALLOWED_MA_TYPES:
                raise PoolSettingsError(
                    f"MA_TYPE 은 {', '.join(ALLOWED_MA_TYPES)} 중 하나여야 합니다: {raw}"
                )
            cleaned[key] = ma_type
            continue

        try:
            num = int(raw)
        except (TypeError, ValueError) as exc:
            raise PoolSettingsError(f"{key} 은 정수여야 합니다: {raw}") from exc

        if key == "MA_MONTHS":
            if not (1 <= num <= months_max):
                raise PoolSettingsError(f"MA_MONTHS 는 1 ~ {months_max} 범위여야 합니다: {num}")
        elif key == "RSI_LIMIT":
            if not (1 <= num <= 100):
                raise PoolSettingsError(f"RSI_LIMIT 은 1 ~ 100 범위여야 합니다: {num}")
        elif key == "TOP_N_HOLD":
            if not (1 <= num <= 100):
                raise PoolSettingsError(f"TOP_N_HOLD 는 1 ~ 100 범위여야 합니다: {num}")
        elif key == "HOLDING_BONUS_SCORE":
            if not (0 <= num <= 1000):
                raise PoolSettingsError(f"HOLDING_BONUS_SCORE 는 0 ~ 1000 범위여야 합니다: {num}")

        cleaned[key] = num

    if not cleaned:
        raise PoolSettingsError("저장할 값이 없습니다.")
    return cleaned


def save_pool_settings(pool_id: str, values: dict[str, Any]) -> dict[str, Any]:
    """편집한 5개 값을 pool_settings 에 upsert 하고 캐시를 무효화한다.

    pool_id 는 ALL_POOL_ID("__all__") 또는 유효한 ticker_type.
    반환: 저장된(정규화된) 값.
    """
    from utils.db_manager import get_db_connection
    from utils.settings_loader import list_available_ticker_types

    norm_id = str(pool_id or "").strip()
    if norm_id != ALL_POOL_ID:
        norm_id = norm_id.lower()
        if norm_id not in list_available_ticker_types():
            raise PoolSettingsError(f"알 수 없는 종목풀입니다: {pool_id}")

    cleaned = _validate_values(values)

    db = get_db_connection()
    if db is None:
        raise PoolSettingsError("DB 연결 실패로 설정을 저장할 수 없습니다.")
    db[COLLECTION].update_one(
        {"_id": norm_id},
        {"$set": {**cleaned, "updated_at": datetime.utcnow()}},
        upsert=True,
    )

    # 오버라이드 캐시 + 이 값에 의존하는 랭킹 캐시 무효화
    invalidate_overlay_cache()
    try:
        from utils.rank_service import invalidate_rank_data_cache

        invalidate_rank_data_cache()
    except Exception as exc:
        logger.warning("랭킹 캐시 무효화 실패(설정 저장 후): %s", exc)

    return cleaned
