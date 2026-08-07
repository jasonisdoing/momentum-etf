"""계좌 설정의 DB 단일 소스 레이어 (MongoDB `account_settings`).

(구)accounts.json 을 대체한다. 계좌별 1개 문서:

    {_id: <account_id>, name, icon, order, country_code, currency,
     benchmark: {ticker, name}, ticker_types?, market_regime_index?, URL?,
     ma_*/stoploss_* 알람 설정, updated_at, save_method}

DB 가 유일한 소스다. 문서가 없으면 임의 기본값 없이 **명확히 에러**를 낸다.
계좌 추가/삭제/값 수정 모두 지원한다 (account_id 는 불변 키).
멀티프로세스 반영을 위해 짧은 TTL 캐시 + 저장 시 무효화를 쓴다.
"""

from __future__ import annotations

import re
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
    "cash_currencies",
    "benchmark",
    "ticker_types",
    "market_regime_index",
    "URL",
    "ma_alarm_enabled",
    "ma_short_days",
    "ma_long_days",
    "ma_alarm_icon",
    "stoploss_alarm_enabled",
    "stoploss_threshold_pct",
    "stoploss_alarm_icon",
)

_ALLOWED_COUNTRY_CODES = {"kor", "au", "us"}
_ALLOWED_CASH_CURRENCIES = {"KRW", "USD", "AUD"}

_CACHE_TTL_SECONDS = 30.0
_cache: tuple[float, list[dict[str, Any]]] | None = None
_cache_lock = threading.Lock()


class AccountSettingsStoreError(ValueError):
    """계좌 설정 검증/저장 오류."""


def invalidate_account_settings_cache() -> None:
    """계좌 설정 변경에 의존하는 프로세스 내 캐시를 모두 비운다.

    이 모듈의 TTL 캐시만 비우면 부족하다. `settings_loader.get_account_settings` 는
    만료 없는 `functools.cache` 라, 여기서 함께 비우지 않으면 순서(`order`) 를 바꿔도
    프로세스를 재시작할 때까지 옛 값이 계속 나온다(종목풀은
    `pool_settings_store._invalidate_dependent_caches` 가 같은 일을 한다).
    """
    global _cache
    with _cache_lock:
        _cache = None

    try:
        from utils import settings_loader

        settings_loader.get_account_settings.cache_clear()
    except Exception as exc:  # 캐시 무효화 실패가 저장 자체를 막지는 않는다
        logger.warning("계좌 설정 로더 캐시 무효화 실패: %s", exc)


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


def _validate_values(account_id: str, values: dict[str, Any], existing_doc: dict[str, Any]) -> dict[str, Any]:
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
        elif key == "cash_currencies":
            # 보유 현금 통화 목록. 허용 통화·최소 1개·주 통화 포함을 강제한다.
            if not isinstance(raw, list):
                raise AccountSettingsStoreError(f"'{account_id}' 의 cash_currencies 는 통화 코드 목록이어야 합니다.")
            normalized: list[str] = []
            for item in raw:
                code = str(item or "").strip().upper()
                if not code:
                    continue
                if code not in _ALLOWED_CASH_CURRENCIES:
                    raise AccountSettingsStoreError(
                        f"'{account_id}' 의 cash_currencies 는 {', '.join(sorted(_ALLOWED_CASH_CURRENCIES))} 중에서만 선택할 수 있습니다: {code}"
                    )
                if code not in normalized:
                    normalized.append(code)
            if not normalized:
                raise AccountSettingsStoreError(f"'{account_id}' 의 cash_currencies 는 최소 1개 이상이어야 합니다.")
            base_currency = str(
                values.get("currency") or existing_doc.get("currency") or ""
            ).strip().upper()
            if base_currency and base_currency not in normalized:
                raise AccountSettingsStoreError(
                    f"'{account_id}' 의 주 통화({base_currency})는 외화 잔액에 반드시 포함되어야 합니다."
                )
            cleaned[key] = normalized
        elif key == "benchmark":
            if not isinstance(raw, dict):
                raise AccountSettingsStoreError(f"'{account_id}' 의 benchmark 는 객체여야 합니다.")
            ticker = str(raw.get("ticker") or "").strip().upper()
            bench_name = str(raw.get("name") or "").strip()
            if not ticker or not bench_name:
                raise AccountSettingsStoreError(f"'{account_id}' 의 benchmark 에는 ticker/name 이 모두 필요합니다.")
            cleaned[key] = {"ticker": ticker, "name": bench_name}
        elif key == "ticker_types":
            # 후보 출처 종목풀 연결(trend). 유효한 종목풀 id 목록이어야 한다.
            if not isinstance(raw, list) or not all(isinstance(item, str) and item.strip() for item in raw):
                raise AccountSettingsStoreError(f"'{account_id}' 의 ticker_types 는 종목풀 id 목록이어야 합니다.")
            from utils.settings_loader import list_available_ticker_types

            available = set(list_available_ticker_types())
            normalized = [item.strip().lower() for item in raw]
            unknown = [item for item in normalized if item not in available]
            if unknown:
                raise AccountSettingsStoreError(
                    f"'{account_id}' 의 ticker_types 에 알 수 없는 종목풀이 있습니다: {', '.join(unknown)}"
                )
            cleaned[key] = normalized
        elif key == "market_regime_index":
            # 계좌별 시장 레짐 판정 지수 — 시장추세 지수(INDICES) 중 하나(필수, {ticker, name}).
            if not isinstance(raw, dict):
                raise AccountSettingsStoreError(f"'{account_id}' 의 market_regime_index 는 {{ticker, name}} 객체여야 합니다.")
            ticker = str(raw.get("ticker") or "").strip()
            if not ticker:
                raise AccountSettingsStoreError(f"'{account_id}' 의 market_regime_index 는 필수입니다(시장 레짐 지수를 선택하세요).")
            from utils.market_trend_service import INDICES

            allowed = {str(item["yf_ticker"]): str(item["name"]) for item in INDICES}
            if ticker not in allowed:
                options = ", ".join(f"{label}({code})" for code, label in allowed.items())
                raise AccountSettingsStoreError(
                    f"'{account_id}' 의 market_regime_index 는 시장추세 지수 중 하나여야 합니다: {options}. 입력값: {ticker}"
                )
            cleaned[key] = {"ticker": ticker, "name": allowed[ticker]}
        elif key == "URL":
            cleaned[key] = str(raw or "").strip()
        elif key == "ma_alarm_enabled":
            cleaned[key] = bool(raw)
        elif key in ("ma_short_days", "ma_long_days"):
            try:
                days = int(raw)
            except (TypeError, ValueError) as exc:
                raise AccountSettingsStoreError(f"'{account_id}' 의 {key} 는 정수여야 합니다: {raw}") from exc
            if days < 2:
                raise AccountSettingsStoreError(f"'{account_id}' 의 {key} 는 2 이상이어야 합니다: {days}")
            cleaned[key] = days
        elif key == "stoploss_alarm_enabled":
            cleaned[key] = bool(raw)
        elif key == "stoploss_threshold_pct":
            try:
                pct = round(float(raw), 2)
            except (TypeError, ValueError) as exc:
                raise AccountSettingsStoreError(f"'{account_id}' 의 stoploss_threshold_pct 는 숫자여야 합니다: {raw}") from exc
            if pct >= 0:
                raise AccountSettingsStoreError(f"'{account_id}' 의 stoploss_threshold_pct 는 음수여야 합니다(예: -7): {pct}")
            cleaned[key] = pct
        elif key in ("ma_alarm_icon", "stoploss_alarm_icon"):
            # 화면 배지용 아이콘(이모지). 빈 문자열 = 배지 미표시(명시적 미설정).
            icon = str(raw or "").strip()
            if len(icon) > 8:
                raise AccountSettingsStoreError(f"'{account_id}' 의 {key} 는 8자 이하여야 합니다: {icon}")
            cleaned[key] = icon
    if not cleaned:
        raise AccountSettingsStoreError("저장할 값이 없습니다.")
    return cleaned


def save_account_settings(account_id: str, values: dict[str, Any], save_method: str = "사용자") -> dict[str, Any]:
    """기존 계좌의 편집값을 검증 후 저장한다 (신규 계좌 생성은 지원하지 않음)."""
    norm_id = str(account_id or "").strip().lower()
    db = _db()
    existing_doc = db[COLLECTION].find_one({"_id": norm_id})
    if existing_doc is None:
        raise AccountSettingsStoreError(f"알 수 없는 계좌입니다: {account_id}")

    cleaned = _validate_values(norm_id, values, existing_doc)
    db[COLLECTION].update_one(
        {"_id": norm_id},
        {"$set": {**cleaned, "updated_at": datetime.now(timezone.utc), "save_method": save_method}},
    )
    invalidate_account_settings_cache()

    # 계좌 메타를 쓰는 파생 캐시 무효화 (registry 는 settings_loader 캐시를 쓰지 않음 — TTL 로 자동 반영)
    return cleaned


_ACCOUNT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def create_account(
    account_id: str,
    name: str,
    *,
    icon: str = "",
    order: int = 0,
    country_code: str = "kor",
    currency: str = "KRW",
) -> dict[str, Any]:
    """새 계좌 문서를 생성한다. account_id(원장 FK)는 사용자 입력이며 중복 불가.

    벤치마크·레짐지수 등 상세는 생성 후 계좌 설정 화면에서 편집한다(여기선 최소 문서만 만든다).
    """
    norm_id = str(account_id or "").strip().lower()
    if not norm_id or not _ACCOUNT_ID_RE.match(norm_id):
        raise AccountSettingsStoreError("계좌 ID는 영문 소문자·숫자로 시작하고 소문자·숫자·-·_ 만 쓸 수 있습니다.")
    name_norm = str(name or "").strip()
    if not name_norm:
        raise AccountSettingsStoreError("계좌 이름을 입력하세요.")
    ccode = str(country_code or "").strip().lower()
    if ccode not in _ALLOWED_COUNTRY_CODES:
        raise AccountSettingsStoreError(f"country_code 는 {', '.join(sorted(_ALLOWED_COUNTRY_CODES))} 중 하나여야 합니다: {country_code}")
    curr = str(currency or "").strip().upper()
    if len(curr) != 3:
        raise AccountSettingsStoreError(f"currency 는 3자리 코드여야 합니다: {currency}")
    try:
        order_int = int(order)
    except (TypeError, ValueError) as exc:
        raise AccountSettingsStoreError(f"order 는 정수여야 합니다: {order}") from exc

    db = _db()
    if db[COLLECTION].find_one({"_id": norm_id}) is not None:
        raise AccountSettingsStoreError(f"이미 존재하는 계좌 ID 입니다: {norm_id}")

    doc = {
        "_id": norm_id,
        "name": name_norm,
        "icon": str(icon or "").strip(),
        "order": order_int,
        "country_code": ccode,
        "currency": curr,
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
        "save_method": "계좌 추가",
    }
    db[COLLECTION].insert_one(doc)
    invalidate_account_settings_cache()

    # 보유 원장(portfolio_master)에도 빈 항목을 만들어 자산 헬퍼에서 바로 종목을 추가할 수 있게 한다.
    # (없으면 "계좌 데이터를 찾을 수 없습니다" 에러가 난다.)
    try:
        from utils.portfolio_io import save_portfolio_master

        save_portfolio_master(norm_id, [], cash_currency=curr)
    except Exception as exc:
        get_app_logger().warning("[계좌 추가] portfolio_master 시드 실패 (%s): %s", norm_id, exc)

    return {"account_id": norm_id, "name": name_norm}


def _account_holding_count(account_id: str) -> int:
    """portfolio_master(GLOBAL) 에서 해당 계좌의 보유종목 수(티커가 있는 항목)를 센다."""
    doc = _db()["portfolio_master"].find_one({"master_id": "GLOBAL"}, {"accounts": 1}) or {}
    for acc in doc.get("accounts") or []:
        if str(acc.get("account_id") or "").strip().lower() == account_id:
            holdings = acc.get("holdings") or []
            return len([h for h in holdings if str((h or {}).get("ticker") or "").strip()])
    return 0


def delete_account(account_id: str) -> dict[str, Any]:
    """계좌를 삭제한다. 보유종목이 남아 있으면 차단(원장 데이터 보호)."""
    norm_id = str(account_id or "").strip().lower()
    db = _db()
    if db[COLLECTION].find_one({"_id": norm_id}) is None:
        raise AccountSettingsStoreError(f"알 수 없는 계좌입니다: {account_id}")
    holding_count = _account_holding_count(norm_id)
    if holding_count > 0:
        raise AccountSettingsStoreError(
            f"보유종목이 {holding_count}건 있어 삭제할 수 없습니다. 먼저 보유를 비운 뒤 삭제하세요."
        )
    db[COLLECTION].delete_one({"_id": norm_id})
    invalidate_account_settings_cache()

    # 보유 원장의 빈 항목도 함께 제거(보유가 있으면 위에서 이미 차단됨).
    try:
        db["portfolio_master"].update_one({"master_id": "GLOBAL"}, {"$pull": {"accounts": {"account_id": norm_id}}})
    except Exception as exc:
        get_app_logger().warning("[계좌 삭제] portfolio_master 정리 실패 (%s): %s", norm_id, exc)

    return {"account_id": norm_id, "deleted": True}


def get_account_settings_updated_at(account_id: str) -> str | None:
    doc = _db()[COLLECTION].find_one({"_id": str(account_id or "").strip().lower()}, {"updated_at": 1, "save_method": 1})
    if not doc or doc.get("updated_at") is None:
        return None
    ua = doc["updated_at"]
    if getattr(ua, "tzinfo", None) is None:
        ua = ua.replace(tzinfo=timezone.utc)
    return ua.isoformat()
