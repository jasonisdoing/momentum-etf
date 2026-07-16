"""계좌 설정의 DB 단일 소스 레이어 (MongoDB `account_settings`).

(구)accounts.json 을 대체한다. 계좌별 1개 문서:

    {_id: <account_id>, name, icon, order, country_code, currency,
     benchmark: {ticker, name}, memo?, top_pick_start_amount_manwon?,
     top_pick_start_date?, URL?, updated_at, save_method}

DB 가 유일한 소스다. 문서가 없으면 임의 기본값 없이 **명확히 에러**를 낸다.
계좌 추가/삭제는 화면에서 지원하지 않는다 — 값 수정만 허용 (account_id 는 불변 키).
멀티프로세스 반영을 위해 짧은 TTL 캐시 + 저장 시 무효화를 쓴다.
"""

from __future__ import annotations

import threading
from datetime import datetime, timezone
from time import monotonic
from typing import Any

from utils.logger import get_app_logger

logger = get_app_logger()

COLLECTION = "account_settings"

# 화면에서 수정 가능한 키 (account_id 는 원장/포트폴리오의 FK 라 불변)
EDITABLE_KEYS: tuple[str, ...] = (
    "name",
    "icon",
    "order",
    "country_code",
    "currency",
    "benchmark",
    "ticker_types",
    "memo",
    "top_pick_start_amount_manwon",
    "top_pick_start_date",
    "URL",
)

_ALLOWED_COUNTRY_CODES = {"kor", "au", "us"}

_CACHE_TTL_SECONDS = 30.0
_cache: tuple[float, list[dict[str, Any]]] | None = None
_cache_lock = threading.Lock()


class AccountSettingsStoreError(ValueError):
    """계좌 설정 검증/저장 오류."""


def invalidate_account_settings_cache() -> None:
    global _cache
    with _cache_lock:
        _cache = None


def _db():
    from utils.db_manager import get_db_connection

    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결 실패 (account_settings)")
    return db


def load_account_docs() -> list[dict[str, Any]]:
    """전체 계좌 문서를 order 순으로 반환한다 (TTL 캐시). 비어 있으면 명시적 에러."""
    global _cache
    now = monotonic()
    with _cache_lock:
        if _cache is not None and now - _cache[0] < _CACHE_TTL_SECONDS:
            return [dict(doc) for doc in _cache[1]]

    docs: list[dict[str, Any]] = []
    for doc in _db()[COLLECTION].find({}):
        entry = dict(doc)
        entry["account_id"] = str(entry.pop("_id"))
        docs.append(entry)
    if not docs:
        raise AccountSettingsStoreError(
            "계좌 설정이 DB(account_settings)에 없습니다. 계좌 문서를 먼저 등록해주세요."
        )
    docs.sort(key=lambda item: (int(item.get("order") or 0), str(item["account_id"])))

    with _cache_lock:
        _cache = (now, [dict(d) for d in docs])
    return docs


def _validate_values(account_id: str, values: dict[str, Any]) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key in EDITABLE_KEYS:
        if key not in values:
            continue
        raw = values[key]
        if key == "name":
            name = str(raw or "").strip()
            if not name:
                raise AccountSettingsStoreError(f"'{account_id}' 의 name 은 비울 수 없습니다.")
            cleaned[key] = name
        elif key == "icon":
            cleaned[key] = str(raw or "").strip()
        elif key == "order":
            try:
                cleaned[key] = int(raw)
            except (TypeError, ValueError) as exc:
                raise AccountSettingsStoreError(f"'{account_id}' 의 order 는 정수여야 합니다: {raw}") from exc
        elif key == "country_code":
            code = str(raw or "").strip().lower()
            if code not in _ALLOWED_COUNTRY_CODES:
                raise AccountSettingsStoreError(
                    f"'{account_id}' 의 country_code 는 {', '.join(sorted(_ALLOWED_COUNTRY_CODES))} 중 하나여야 합니다: {raw}"
                )
            cleaned[key] = code
        elif key == "currency":
            currency = str(raw or "").strip().upper()
            if len(currency) != 3:
                raise AccountSettingsStoreError(f"'{account_id}' 의 currency 는 3자리 코드여야 합니다: {raw}")
            cleaned[key] = currency
        elif key == "benchmark":
            if not isinstance(raw, dict):
                raise AccountSettingsStoreError(f"'{account_id}' 의 benchmark 는 객체여야 합니다.")
            ticker = str(raw.get("ticker") or "").strip().upper()
            bench_name = str(raw.get("name") or "").strip()
            if not ticker or not bench_name:
                raise AccountSettingsStoreError(f"'{account_id}' 의 benchmark 에는 ticker/name 이 모두 필요합니다.")
            cleaned[key] = {"ticker": ticker, "name": bench_name}
        elif key == "ticker_types":
            if not isinstance(raw, (list, tuple)):
                raise AccountSettingsStoreError(f"'{account_id}' 의 ticker_types 는 목록이어야 합니다.")
            from utils.settings_loader import list_available_ticker_types

            available = set(list_available_ticker_types())
            selected: list[str] = []
            for item in raw:
                pool_id = str(item or "").strip()
                if not pool_id:
                    continue
                if pool_id not in available:
                    raise AccountSettingsStoreError(
                        f"'{account_id}' 의 ticker_types 에 알 수 없는 종목풀이 있습니다: {item}"
                    )
                if pool_id not in selected:
                    selected.append(pool_id)
            if len(selected) > 1:
                raise AccountSettingsStoreError(f"'{account_id}' 의 ticker_types 는 1개만 선택할 수 있습니다.")
            cleaned[key] = selected
        elif key == "memo":
            cleaned[key] = str(raw or "").replace("\r", " ").replace("\n", " ").strip()
        elif key == "top_pick_start_amount_manwon":
            if raw in (None, ""):
                cleaned[key] = None
                continue
            try:
                amount = round(float(raw), 2)
            except (TypeError, ValueError) as exc:
                raise AccountSettingsStoreError(f"'{account_id}' 의 top_pick_start_amount_manwon 은 숫자여야 합니다: {raw}") from exc
            if not (1 <= amount <= 1_000_000_000):
                raise AccountSettingsStoreError(
                    f"'{account_id}' 의 top_pick_start_amount_manwon 은 1 ~ 1000000000 범위여야 합니다: {amount}"
                )
            cleaned[key] = amount
        elif key == "top_pick_start_date":
            start_date = str(raw or "").strip()
            if not start_date:
                cleaned[key] = None
                continue
            try:
                datetime.strptime(start_date, "%Y-%m-%d")
            except ValueError as exc:
                raise AccountSettingsStoreError(
                    f"'{account_id}' 의 top_pick_start_date 는 YYYY-MM-DD 형식이어야 합니다: {start_date}"
                ) from exc
            cleaned[key] = start_date
        elif key == "URL":
            cleaned[key] = str(raw or "").strip()
    if not cleaned:
        raise AccountSettingsStoreError("저장할 값이 없습니다.")
    return cleaned


def save_account_settings(account_id: str, values: dict[str, Any], save_method: str = "사용자") -> dict[str, Any]:
    """기존 계좌의 편집값을 검증 후 저장한다 (신규 계좌 생성은 지원하지 않음)."""
    norm_id = str(account_id or "").strip().lower()
    db = _db()
    if db[COLLECTION].find_one({"_id": norm_id}, {"_id": 1}) is None:
        raise AccountSettingsStoreError(f"알 수 없는 계좌입니다: {account_id}")

    cleaned = _validate_values(norm_id, values)
    db[COLLECTION].update_one(
        {"_id": norm_id},
        {"$set": {**cleaned, "updated_at": datetime.now(timezone.utc), "save_method": save_method}},
    )
    invalidate_account_settings_cache()

    # 계좌 메타를 쓰는 파생 캐시 무효화 (registry 는 settings_loader 캐시를 쓰지 않음 — TTL 로 자동 반영)
    return cleaned


def get_account_settings_updated_at(account_id: str) -> str | None:
    doc = _db()[COLLECTION].find_one({"_id": str(account_id or "").strip().lower()}, {"updated_at": 1, "save_method": 1})
    if not doc or doc.get("updated_at") is None:
        return None
    ua = doc["updated_at"]
    if getattr(ua, "tzinfo", None) is None:
        ua = ua.replace(tzinfo=timezone.utc)
    return ua.isoformat()
