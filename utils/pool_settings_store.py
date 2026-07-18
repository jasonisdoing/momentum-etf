"""종목풀 설정의 DB 단일 소스 레이어.

MongoDB `pool_settings` 컬렉션이 종목풀의 구조와 편집값을 모두 보관한다.

    구조: ticker_type, name, icon, order, country_code, currency, is_active
    편집: TOP_N_HOLD, SHORT_MA_DAYS, LONG_MA_DAYS, SLOPE_DAYS,
          BUY_SLIPPAGE_PCT, SELL_SLIPPAGE_PCT (필수)
          BENCHMARK (선택 — 비우면 미설정)

런타임 로딩은 DB 문서가 없거나 필수 키가 누락되면 명확히 에러를 낸다.
캐시: 멀티프로세스(fastapi/scheduler/worker)에서 변경이 반영되도록 짧은 TTL(30초) 캐시를
      쓴다. 저장한 프로세스는 즉시 무효화하고, 나머지는 TTL 내 자동 반영된다.

컬렉션 문서 형태:
    {
      _id: <ticker_type>, name, icon, order, country_code, currency,
      is_active, TOP_N_HOLD, SHORT_MA_DAYS, LONG_MA_DAYS, SLOPE_DAYS,
      BUY_SLIPPAGE_PCT, SELL_SLIPPAGE_PCT, BENCHMARK, updated_at
    }
"""

from __future__ import annotations

import threading
from datetime import datetime
from time import monotonic
from typing import Any

from utils.logger import get_app_logger

logger = get_app_logger()

COLLECTION = "pool_settings"
INTERNAL_POOL_ID_PREFIX = "__"

# DB 오버라이드 대상 키 — 전부 필수이며 비어 있으면 로딩 자체가 실패한다.
OVERRIDABLE_KEYS: tuple[str, ...] = (
    "TOP_N_HOLD",
    "SHORT_MA_DAYS",
    "LONG_MA_DAYS",
    "SLOPE_DAYS",
)

# 종목풀별 거래비용 설정. 기존 문서에 없어도 설정 화면은 열려야 하므로 로딩 필수값은 아니다.
# 단, 슬리피지를 실제로 사용하는 백테스트/계산 로직은 누락 시 명시적으로 실패한다.
SLIPPAGE_KEYS: tuple[str, ...] = (
    "BUY_SLIPPAGE_PCT",
    "SELL_SLIPPAGE_PCT",
)

# 편집 가능하지만 비워둘 수 있는 키. 필수 검사에서 제외한다.
# 빈 문자열은 '미설정'을 뜻하며, 읽는 쪽이 그 상태를 명시적으로 처리해야 한다(임의 대체 금지).
OPTIONAL_EDITABLE_KEYS: tuple[str, ...] = ("BENCHMARK",)

# 종목풀 설정 화면에서 편집하는 전체 키(순서 = 화면 표시 순서).
POOL_EDITABLE_KEYS: tuple[str, ...] = (*OVERRIDABLE_KEYS, *SLIPPAGE_KEYS, *OPTIONAL_EDITABLE_KEYS)

STRUCTURAL_KEYS: tuple[str, ...] = (
    "name",
    "icon",
    "order",
    "country_code",
    "currency",
    "is_active",
)

MA_DAY_OPTIONS: tuple[int, ...] = (5, 10, 20, 40, 60, 120, 240)
# 기울기 측정 일수(k): 단기 이평선의 k일 전 대비 변화율. 1일은 노이즈가 커 권장하지 않는다.
SLOPE_DAY_OPTIONS: tuple[int, ...] = (1, 2, 3, 5, 10, 20, 40, 60)
# 편도 슬리피지(%) 선택지: 0.05 ~ 0.50, 0.05 단위. 필수라 빈 값 불가.
SLIPPAGE_PCT_OPTIONS: tuple[float, ...] = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5)

_INT_KEYS = ("TOP_N_HOLD", "SHORT_MA_DAYS", "LONG_MA_DAYS", "SLOPE_DAYS")
_FLOAT_KEYS = ("BUY_SLIPPAGE_PCT", "SELL_SLIPPAGE_PCT")
_ALLOWED_COUNTRY_CODES = {"kor", "au", "us"}
_ALLOWED_CURRENCIES = {"KRW", "AUD", "USD"}

# 나중에 추가된 선택 키 → 기본값. DB 문서에 없어도 에러 없이 이 값으로 채운다(하위 호환).
_OPTIONAL_DEFAULTS: dict[str, Any] = {"SLOPE_DAYS": 5}

_CACHE_TTL_SECONDS = 30.0
_overlay_cache: dict[str, dict[str, Any]] | None = None
_overlay_cached_at: float = 0.0
_overlay_lock = threading.Lock()
_pool_docs_cache: tuple[float, list[dict[str, Any]]] | None = None
_pool_docs_lock = threading.Lock()


class PoolSettingsError(ValueError):
    """종목풀 설정 검증/저장 오류."""


def get_pool_benchmark_ticker(settings: dict[str, Any]) -> str:
    """종목풀 설정에서 벤치마크 티커를 꺼낸다. 미설정이면 빈 문자열.

    ``BENCHMARK`` 는 ``{ticker, name}`` 이며 선택 항목이라 없을 수 있다.
    읽는 쪽이 제각각 파싱하면 형태가 어긋나므로 여기서만 해석한다.
    """
    benchmark = settings.get("BENCHMARK")
    if not isinstance(benchmark, dict):
        return ""
    return str(benchmark.get("ticker") or "").strip().upper()


def invalidate_overlay_cache() -> None:
    """오버라이드 캐시를 비운다 (저장 직후 호출)."""
    global _overlay_cache, _overlay_cached_at, _pool_docs_cache
    with _overlay_lock:
        _overlay_cache = None
        _overlay_cached_at = 0.0
    with _pool_docs_lock:
        _pool_docs_cache = None
    _invalidate_dependent_caches()


def _invalidate_dependent_caches() -> None:
    """종목풀 정의 변경에 의존하는 프로세스 내 캐시를 비운다."""
    try:
        from utils import settings_loader

        settings_loader._load_pool_configs.cache_clear()
    except Exception as exc:
        logger.warning("종목풀 로더 캐시 무효화 실패: %s", exc)
    try:
        from utils.rank_service import invalidate_rank_data_cache

        invalidate_rank_data_cache()
    except Exception as exc:
        logger.warning("랭킹 캐시 무효화 실패: %s", exc)


def _db():
    from utils.db_manager import get_db_connection

    db = get_db_connection()
    if db is None:
        raise PoolSettingsError("MongoDB 연결 실패 (pool_settings)")
    return db


def _normalize_ticker_type(value: Any) -> str:
    ticker_type = str(value or "").strip().lower()
    if not ticker_type:
        raise PoolSettingsError("ticker_type 은 필수입니다.")
    if ticker_type.startswith(INTERNAL_POOL_ID_PREFIX):
        raise PoolSettingsError(f"내부 예약 종목풀 ID는 사용할 수 없습니다: {ticker_type}")
    if not ticker_type.replace("_", "").isalnum():
        raise PoolSettingsError(f"ticker_type 은 영문/숫자/밑줄만 허용합니다: {ticker_type}")
    return ticker_type


def _is_internal_pool_id(value: Any) -> bool:
    return str(value or "").strip().lower().startswith(INTERNAL_POOL_ID_PREFIX)


def _normalize_pool_values(values: dict[str, Any], *, require_ticker_type: bool) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    if require_ticker_type or "ticker_type" in values:
        cleaned["ticker_type"] = _normalize_ticker_type(values.get("ticker_type"))

    if "name" in values:
        name = str(values.get("name") or "").strip()
        if not name:
            raise PoolSettingsError("종목풀 이름은 비울 수 없습니다.")
        cleaned["name"] = name
    if "icon" in values:
        icon = str(values.get("icon") or "").strip()
        if not icon:
            raise PoolSettingsError("종목풀 아이콘은 비울 수 없습니다.")
        cleaned["icon"] = icon
    if "order" in values:
        try:
            cleaned["order"] = int(values.get("order"))
        except (TypeError, ValueError) as exc:
            raise PoolSettingsError(f"order 는 정수여야 합니다: {values.get('order')}") from exc
    if "country_code" in values:
        country_code = str(values.get("country_code") or "").strip().lower()
        if country_code not in _ALLOWED_COUNTRY_CODES:
            allowed = ", ".join(sorted(_ALLOWED_COUNTRY_CODES))
            raise PoolSettingsError(f"country_code 는 {allowed} 중 하나여야 합니다: {country_code}")
        cleaned["country_code"] = country_code
    if "currency" in values:
        currency = str(values.get("currency") or "").strip().upper()
        if currency not in _ALLOWED_CURRENCIES:
            allowed = ", ".join(sorted(_ALLOWED_CURRENCIES))
            raise PoolSettingsError(f"currency 는 {allowed} 중 하나여야 합니다: {currency}")
        cleaned["currency"] = currency
    if "is_active" in values:
        cleaned["is_active"] = bool(values.get("is_active"))

    editable_input = {k: values[k] for k in POOL_EDITABLE_KEYS if k in values}
    cleaned.update(_validate_values(editable_input) if editable_input else {})

    return cleaned


def _normalize_pool_doc(doc: dict[str, Any]) -> dict[str, Any]:
    pool_id = _normalize_ticker_type(doc.get("_id") or doc.get("ticker_type"))
    # 나중에 추가된 선택 키는 기존 문서에 없을 수 있으므로 기본값으로 채운다(하위 호환).
    doc = dict(doc)
    for optional_key, default_value in _OPTIONAL_DEFAULTS.items():
        if doc.get(optional_key) in (None, ""):
            doc[optional_key] = default_value
    required_keys = ("name", "icon", "order", "country_code", "currency", *OVERRIDABLE_KEYS)
    missing = [key for key in required_keys if key not in doc or doc[key] in (None, "")]
    if missing:
        raise PoolSettingsError(
            f"종목풀 '{pool_id}' 의 DB 설정에 필수 값이 없습니다: {', '.join(missing)}. "
            f"`/pools-settings` 화면에서 값을 수정하세요."
        )

    normalized = _normalize_pool_values({**doc, "ticker_type": pool_id}, require_ticker_type=True)
    normalized["ticker_type"] = pool_id
    normalized["is_active"] = bool(doc.get("is_active", True))
    if doc.get("updated_at") is not None:
        normalized["updated_at"] = doc["updated_at"]
    if doc.get("save_method") is not None:
        normalized["save_method"] = doc["save_method"]
    if doc.get("default") is not None:
        normalized["default"] = bool(doc.get("default"))
    return normalized


def load_pool_definitions(*, include_inactive: bool = False) -> list[dict[str, Any]]:
    """DB에서 종목풀 정의를 읽는다. 기본은 활성 풀만 반환한다."""
    global _pool_docs_cache
    now = monotonic()
    if not include_inactive:
        with _pool_docs_lock:
            if _pool_docs_cache is not None and now - _pool_docs_cache[0] < _CACHE_TTL_SECONDS:
                return [dict(doc) for doc in _pool_docs_cache[1]]

    query: dict[str, Any] = {} if include_inactive else {"is_active": {"$ne": False}}
    docs = [
        _normalize_pool_doc(dict(doc))
        for doc in _db()[COLLECTION].find(query)
        if not _is_internal_pool_id(doc.get("_id"))
    ]
    if not docs:
        raise PoolSettingsError(
            "종목풀 설정이 DB(pool_settings)에 없습니다. "
            "`/pools-settings` 화면에서 종목풀을 생성하세요."
        )
    docs.sort(key=lambda item: (int(item["order"]), str(item["ticker_type"])))

    if not include_inactive:
        with _pool_docs_lock:
            _pool_docs_cache = (now, [dict(doc) for doc in docs])
    return docs


def _load_overrides_from_db() -> dict[str, dict[str, Any]]:
    """pool_settings 컬렉션 전체를 {pool_id: {key: value}} 로 읽는다. 실패 시 {}."""
    try:
        result: dict[str, dict[str, Any]] = {}
        for doc in _db()[COLLECTION].find({"is_active": {"$ne": False}}):
            if _is_internal_pool_id(doc.get("_id")):
                continue
            pool_id = str(doc.get("_id") or "").strip()
            if not pool_id:
                continue
            overrides = {k: doc[k] for k in POOL_EDITABLE_KEYS if k in doc and doc[k] is not None}
            if "updated_at" in doc and doc["updated_at"] is not None:
                overrides["updated_at"] = doc["updated_at"]
            if "save_method" in doc and doc["save_method"] is not None:
                overrides["save_method"] = doc["save_method"]
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


def _validate_values(values: dict[str, Any]) -> dict[str, Any]:
    """저장 입력값을 검증/정규화한다. 잘못된 값은 PoolSettingsError."""
    cleaned: dict[str, Any] = {}

    for key in _INT_KEYS:
        if key not in values:
            continue
        raw = values[key]

        try:
            num = int(raw)
        except (TypeError, ValueError) as exc:
            raise PoolSettingsError(f"{key} 은 정수여야 합니다: {raw}") from exc

        if key in ("SHORT_MA_DAYS", "LONG_MA_DAYS"):
            if num not in MA_DAY_OPTIONS:
                options = ", ".join(str(day) for day in MA_DAY_OPTIONS)
                raise PoolSettingsError(f"{key} 는 다음 값 중 하나여야 합니다: {options}. 입력값: {num}")
        elif key == "SLOPE_DAYS":
            if num not in SLOPE_DAY_OPTIONS:
                options = ", ".join(str(day) for day in SLOPE_DAY_OPTIONS)
                raise PoolSettingsError(f"SLOPE_DAYS 는 다음 값 중 하나여야 합니다: {options}. 입력값: {num}")
        elif key == "TOP_N_HOLD":
            if not (1 <= num <= 100):
                raise PoolSettingsError(f"TOP_N_HOLD 는 1 ~ 100 범위여야 합니다: {num}")
        cleaned[key] = num

    for key in _FLOAT_KEYS:
        if key not in values:
            continue
        raw = values[key]
        try:
            num = round(float(raw), 2)
        except (TypeError, ValueError) as exc:
            raise PoolSettingsError(f"{key} 은 숫자여야 합니다: {raw}") from exc
        if num not in {round(option, 2) for option in SLIPPAGE_PCT_OPTIONS}:
            options = ", ".join(f"{option:g}" for option in SLIPPAGE_PCT_OPTIONS)
            raise PoolSettingsError(f"{key} 는 다음 값 중 하나여야 합니다: {options}. 입력값: {num}")
        cleaned[key] = num

    if "BENCHMARK" in values:
        # 계좌 설정(account_settings.benchmark)과 같은 {ticker, name} 형태를 쓴다.
        # 계좌 쪽과 달리 종목풀 벤치마크는 선택이라 None/빈 값이면 '미설정'으로 저장한다.
        raw = values["BENCHMARK"]
        if raw in (None, ""):
            cleaned["BENCHMARK"] = None
        elif not isinstance(raw, dict):
            raise PoolSettingsError("BENCHMARK 는 {ticker, name} 객체여야 합니다.")
        else:
            ticker = str(raw.get("ticker") or "").strip().upper()
            bench_name = str(raw.get("name") or "").strip()
            if not ticker and not bench_name:
                cleaned["BENCHMARK"] = None
            elif not ticker or not bench_name:
                raise PoolSettingsError("BENCHMARK 에는 ticker/name 이 모두 필요합니다. 티커를 조회해 이름을 채우세요.")
            else:
                cleaned["BENCHMARK"] = {"ticker": ticker, "name": bench_name}

    if not cleaned:
        raise PoolSettingsError("저장할 값이 없습니다.")
    return cleaned


def save_pool_settings(pool_id: str, values: dict[str, Any], save_method: str = "수동") -> dict[str, Any]:
    """편집한 값을 pool_settings 에 upsert 하고 캐시를 무효화한다.

    pool_id 는 유효한 ticker_type.
    반환: 저장된(정규화된) 값.
    """
    from utils.settings_loader import list_available_ticker_types

    norm_id = str(pool_id or "").strip().lower()
    if norm_id not in list_available_ticker_types():
        raise PoolSettingsError(f"알 수 없는 종목풀입니다: {pool_id}")

    cleaned = _validate_values(values)

    # 벤치마크는 이 종목풀에 실제로 등록된 종목이어야 한다(순위/백테스트에서 매수 후보에서
    # 제외하고 종가를 이 풀 캐시에서 읽기 때문). 없는 종목이면 저장을 거부한다.
    benchmark = cleaned.get("BENCHMARK")
    if isinstance(benchmark, dict) and benchmark.get("ticker"):
        from utils.stock_list_io import get_etfs

        def _norm_ticker(value: Any) -> str:
            return str(value or "").strip().upper().removeprefix("ASX:")

        pool_tickers = {_norm_ticker(item.get("ticker")) for item in get_etfs(norm_id)}
        if _norm_ticker(benchmark["ticker"]) not in pool_tickers:
            raise PoolSettingsError(
                f"벤치마크 '{benchmark['ticker']}' 는 이 종목풀에 등록된 종목이 아닙니다. "
                "종목풀에 있는 종목만 벤치마크로 지정할 수 있습니다."
            )

    db = _db()
    db[COLLECTION].update_one(
        {"_id": norm_id},
        {
            "$set": {**cleaned, "updated_at": datetime.utcnow(), "save_method": save_method},
            "$unset": {"type_source": ""},
        },
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


def create_pool(values: dict[str, Any], save_method: str = "사용자") -> dict[str, Any]:
    """신규 종목풀을 생성한다. 생성 후 ticker_type 은 변경할 수 없다."""
    required = ("ticker_type", "name", "icon", "order", "country_code", "currency", *OVERRIDABLE_KEYS)
    missing = [key for key in required if key not in values or values[key] in (None, "")]
    if missing:
        raise PoolSettingsError(f"신규 종목풀에 필수 값이 없습니다: {', '.join(missing)}")
    cleaned = _normalize_pool_values(values, require_ticker_type=True)
    pool_id = cleaned.pop("ticker_type")
    cleaned["is_active"] = True

    db = _db()
    if db[COLLECTION].find_one({"_id": pool_id}, {"_id": 1}) is not None:
        raise PoolSettingsError(f"이미 존재하는 종목풀입니다: {pool_id}")
    db[COLLECTION].insert_one(
        {"_id": pool_id, **cleaned, "created_at": datetime.utcnow(), "updated_at": datetime.utcnow(), "save_method": save_method}
    )
    invalidate_overlay_cache()
    return {"ticker_type": pool_id, **cleaned}


def update_pool(pool_id: str, values: dict[str, Any], save_method: str = "사용자") -> dict[str, Any]:
    """기존 종목풀의 메타/설정 값을 수정한다. ticker_type 은 불변이다."""
    norm_id = _normalize_ticker_type(pool_id)
    if "ticker_type" in values and _normalize_ticker_type(values["ticker_type"]) != norm_id:
        raise PoolSettingsError("ticker_type 은 변경할 수 없습니다.")
    cleaned = _normalize_pool_values({k: v for k, v in values.items() if k != "ticker_type"}, require_ticker_type=False)
    if not cleaned:
        raise PoolSettingsError("저장할 값이 없습니다.")
    db = _db()
    if db[COLLECTION].find_one({"_id": norm_id}, {"_id": 1}) is None:
        raise PoolSettingsError(f"알 수 없는 종목풀입니다: {pool_id}")
    db[COLLECTION].update_one(
        {"_id": norm_id},
        {
            "$set": {**cleaned, "updated_at": datetime.utcnow(), "save_method": save_method},
            "$unset": {"type_source": ""},
        },
    )
    invalidate_overlay_cache()
    return cleaned


def get_pool_delete_impact(pool_id: str) -> dict[str, Any]:
    """종목풀 삭제 전 영향도를 반환한다."""
    norm_id = _normalize_ticker_type(pool_id)
    db = _db()
    account_count = db["account_settings"].count_documents({"ticker_types": norm_id})
    stock_count = db["stock_meta"].count_documents({"ticker_type": norm_id})
    exists = db[COLLECTION].find_one({"_id": norm_id}, {"_id": 1}) is not None
    return {
        "ticker_type": norm_id,
        "exists": exists,
        "account_count": account_count,
        "stock_count": stock_count,
        "can_delete": exists and account_count == 0,
    }


def delete_pool(pool_id: str) -> dict[str, Any]:
    """계좌 연결이 없는 종목풀을 하드 삭제한다. 연결 종목 메타도 함께 제거한다."""
    impact = get_pool_delete_impact(pool_id)
    if not impact["exists"]:
        raise PoolSettingsError(f"알 수 없는 종목풀입니다: {pool_id}")
    if int(impact["account_count"]) > 0:
        raise PoolSettingsError(
            f"계좌 {impact['account_count']}개에 연결된 종목풀은 삭제할 수 없습니다. 계좌 연결을 먼저 해제하세요."
        )
    norm_id = str(impact["ticker_type"])
    db = _db()
    deleted_pool = db[COLLECTION].delete_one({"_id": norm_id}).deleted_count
    deleted_stocks = db["stock_meta"].delete_many({"ticker_type": norm_id}).deleted_count
    invalidate_overlay_cache()
    try:
        from utils.stock_list_io import invalidate_ticker_type_cache

        invalidate_ticker_type_cache(norm_id)
    except Exception as exc:
        logger.warning("종목 메타 캐시 무효화 실패(종목풀 삭제 후): %s", exc)
    return {
        "ticker_type": norm_id,
        "deleted_pool": deleted_pool,
        "deleted_stocks": deleted_stocks,
    }
