"""계좌 설정의 DB 단일 소스 레이어 (MongoDB `account_settings`).

(구)accounts.json 을 대체한다. 계좌별 1개 문서:

    {_id: <account_id>, name, icon, order, country_code, currency,
     benchmark: {ticker, name}, URL?, updated_at, save_method,
     account_type?: "fixed"|"trend"|"regime", strategy?: {...}}

DB 가 유일한 소스다. 문서가 없으면 임의 기본값 없이 **명확히 에러**를 낸다.
계좌 추가/삭제는 화면에서 지원하지 않는다 — 값 수정만 허용 (account_id 는 불변 키).
멀티프로세스 반영을 위해 짧은 TTL 캐시 + 저장 시 무효화를 쓴다.

계좌 전략(account_type/strategy)은 docs/account_types_plan.md 참고. 계좌가 종목풀을
참조(ticker_types)하는 대신 **자기 종목 리스트를 직접 소유**한다:

    fixed  : strategy.holdings        [{ticker, name, ticker_type?, country_code?, weight_pct}]
    trend  : strategy.universe        [{ticker, name, ticker_type?, country_code?}]
             strategy.long_ma_days / short_ma_days / hold_count
    regime : strategy.regime_index    {ticker, name}
             strategy.up_universe / down_universe  (fixed 없이 trend 와 동일한 종목 항목 형태)
             strategy.long_ma_days / short_ma_days / hold_count
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
    "benchmark",
    "account_type",
    "ticker_types",
    "strategy",
    "market_regime_index",
    "URL",
    "ma20_alarm_enabled",
    "ma20_ma_days",
    "stoploss_alarm_enabled",
    "stoploss_threshold_pct",
)

_ALLOWED_COUNTRY_CODES = {"kor", "au", "us"}
_ALLOWED_ACCOUNT_TYPES = {"fixed", "trend", "regime"}

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


def _clean_strategy_ticker_item(account_id: str, label: str, item: Any) -> dict[str, Any]:
    """전략 종목 리스트(universe/holdings 등)의 항목 1개를 정규화한다.

    종목풀 자동 선택(ticker_types)과 달리 계좌가 이 항목을 **직접 소유**한다 — stock_meta
    재조회로 덮어쓰지 않고, 화면에서 확인(resolve)해 채운 ticker/name/ticker_type/country_code
    를 그대로 신뢰해 저장한다.
    """
    if not isinstance(item, dict):
        raise AccountSettingsStoreError(f"'{account_id}' 의 {label} 항목은 객체여야 합니다: {item}")
    ticker = str(item.get("ticker") or "").strip().upper()
    name = str(item.get("name") or "").strip()
    if not ticker or not name:
        raise AccountSettingsStoreError(f"'{account_id}' 의 {label} 항목에는 ticker/name 이 모두 필요합니다: {item}")
    row: dict[str, Any] = {"ticker": ticker, "name": name}
    ticker_type = str(item.get("ticker_type") or "").strip().lower()
    if ticker_type:
        row["ticker_type"] = ticker_type
    country_code = str(item.get("country_code") or "").strip().lower()
    if country_code:
        row["country_code"] = country_code
    return row


def _clean_strategy_ticker_list(account_id: str, label: str, raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        raise AccountSettingsStoreError(f"'{account_id}' 의 {label} 는 목록이어야 합니다.")
    seen: set[str] = set()
    cleaned: list[dict[str, Any]] = []
    for item in raw:
        row = _clean_strategy_ticker_item(account_id, label, item)
        if row["ticker"] in seen:
            continue
        seen.add(row["ticker"])
        cleaned.append(row)
    if not cleaned:
        raise AccountSettingsStoreError(f"'{account_id}' 의 {label} 는 1개 이상 필요합니다.")
    return cleaned


def _clean_fixed_holdings(account_id: str, raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        raise AccountSettingsStoreError(f"'{account_id}' 의 holdings 는 목록이어야 합니다.")
    seen: set[str] = set()
    cleaned: list[dict[str, Any]] = []
    for item in raw:
        row = _clean_strategy_ticker_item(account_id, "holdings", item)
        if row["ticker"] in seen:
            continue
        raw_weight = item.get("weight_pct")
        try:
            weight = round(float(raw_weight), 2)
        except (TypeError, ValueError) as exc:
            raise AccountSettingsStoreError(
                f"'{account_id}' 의 holdings 항목({row['ticker']})에는 숫자 weight_pct 가 필요합니다: {raw_weight}"
            ) from exc
        if not (0.0 <= weight <= 100.0):
            raise AccountSettingsStoreError(
                f"'{account_id}' 의 holdings 항목({row['ticker']}) weight_pct 는 0 ~ 100 범위여야 합니다: {weight}"
            )
        row["weight_pct"] = weight
        seen.add(row["ticker"])
        cleaned.append(row)
    if not cleaned:
        raise AccountSettingsStoreError(f"'{account_id}' 의 holdings 는 1개 이상 필요합니다.")
    return cleaned


def _clean_positive_int(account_id: str, label: str, raw: Any) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise AccountSettingsStoreError(f"'{account_id}' 의 {label} 는 정수여야 합니다: {raw}") from exc
    if value < 1:
        raise AccountSettingsStoreError(f"'{account_id}' 의 {label} 는 1 이상이어야 합니다: {value}")
    return value


def _clean_strategy(account_id: str, account_type: str, raw: Any) -> dict[str, Any]:
    """account_type 에 맞는 strategy 형태만 검증·정규화한다(타입별 필드만 저장)."""
    if not isinstance(raw, dict):
        raise AccountSettingsStoreError(f"'{account_id}' 의 strategy 는 객체여야 합니다.")

    if account_type == "fixed":
        return {"holdings": _clean_fixed_holdings(account_id, raw.get("holdings"))}

    if account_type == "trend":
        # 후보 출처 풀은 계좌-풀 연결(ticker_types)로, 선정 기준(이평선·게이팅·순위)은 그 풀 설정을
        # 그대로 쓴다. 계좌 strategy 는 보유개수(hold_count) 하나만 소유한다.
        return {
            "hold_count": _clean_positive_int(account_id, "hold_count", raw.get("hold_count")),
        }

    if account_type == "regime":
        regime_index = _clean_strategy_ticker_item(account_id, "regime_index", raw.get("regime_index"))
        return {
            "regime_index": regime_index,
            "up_universe": _clean_strategy_ticker_list(account_id, "up_universe", raw.get("up_universe")),
            "down_universe": _clean_strategy_ticker_list(account_id, "down_universe", raw.get("down_universe")),
            "long_ma_days": _clean_positive_int(account_id, "long_ma_days", raw.get("long_ma_days")),
            "short_ma_days": _clean_positive_int(account_id, "short_ma_days", raw.get("short_ma_days")),
            "hold_count": _clean_positive_int(account_id, "hold_count", raw.get("hold_count")),
        }

    raise AccountSettingsStoreError(f"'{account_id}' 의 account_type 을 알 수 없습니다: {account_type}")


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
        elif key == "benchmark":
            if not isinstance(raw, dict):
                raise AccountSettingsStoreError(f"'{account_id}' 의 benchmark 는 객체여야 합니다.")
            ticker = str(raw.get("ticker") or "").strip().upper()
            bench_name = str(raw.get("name") or "").strip()
            if not ticker or not bench_name:
                raise AccountSettingsStoreError(f"'{account_id}' 의 benchmark 에는 ticker/name 이 모두 필요합니다.")
            cleaned[key] = {"ticker": ticker, "name": bench_name}
        elif key == "account_type":
            account_type = str(raw or "").strip().lower()
            if account_type not in _ALLOWED_ACCOUNT_TYPES:
                raise AccountSettingsStoreError(
                    f"'{account_id}' 의 account_type 은 {', '.join(sorted(_ALLOWED_ACCOUNT_TYPES))} 중 하나여야 합니다: {raw}"
                )
            cleaned[key] = account_type
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
        elif key == "strategy":
            # strategy 는 account_type 형태를 따른다. 같은 저장 호출에 account_type 이 없으면
            # 기존 저장값을 기준으로 삼는다(타입 변경 없이 전략 내용만 수정하는 경우).
            account_type = str(cleaned.get("account_type") or existing_doc.get("account_type") or "").strip().lower()
            if account_type not in _ALLOWED_ACCOUNT_TYPES:
                raise AccountSettingsStoreError(
                    f"'{account_id}' 의 strategy 를 저장하려면 account_type 이 먼저 정해져야 합니다."
                )
            cleaned[key] = _clean_strategy(account_id, account_type, raw)
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
        elif key == "ma20_alarm_enabled":
            cleaned[key] = bool(raw)
        elif key == "ma20_ma_days":
            try:
                days = int(raw)
            except (TypeError, ValueError) as exc:
                raise AccountSettingsStoreError(f"'{account_id}' 의 ma20_ma_days 는 정수여야 합니다: {raw}") from exc
            if days < 2:
                raise AccountSettingsStoreError(f"'{account_id}' 의 ma20_ma_days 는 2 이상이어야 합니다: {days}")
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
# account_type 별 최소 전략 기본값(생성 직후 화면에서 상세 설정).
_DEFAULT_STRATEGY_BY_TYPE: dict[str, dict[str, Any]] = {
    "fixed": {"holdings": []},
    "trend": {"hold_count": 1},
    "regime": {},
}


def create_account(
    account_id: str,
    name: str,
    account_type: str,
    *,
    icon: str = "",
    order: int = 0,
    country_code: str = "kor",
    currency: str = "KRW",
) -> dict[str, Any]:
    """새 계좌 문서를 생성한다. account_id(원장 FK)는 사용자 입력이며 중복 불가.

    상세 전략/벤치마크/레짐지수는 생성 후 계좌 설정 화면에서 편집한다(여기선 최소 문서만 만든다).
    """
    norm_id = str(account_id or "").strip().lower()
    if not norm_id or not _ACCOUNT_ID_RE.match(norm_id):
        raise AccountSettingsStoreError("계좌 ID는 영문 소문자·숫자로 시작하고 소문자·숫자·-·_ 만 쓸 수 있습니다.")
    name_norm = str(name or "").strip()
    if not name_norm:
        raise AccountSettingsStoreError("계좌 이름을 입력하세요.")
    atype = str(account_type or "").strip().lower()
    if atype not in _ALLOWED_ACCOUNT_TYPES:
        raise AccountSettingsStoreError(f"account_type 은 {', '.join(sorted(_ALLOWED_ACCOUNT_TYPES))} 중 하나여야 합니다: {account_type}")
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
        "account_type": atype,
        "country_code": ccode,
        "currency": curr,
        "strategy": dict(_DEFAULT_STRATEGY_BY_TYPE[atype]),
        "created_at": datetime.now(timezone.utc),
        "updated_at": datetime.now(timezone.utc),
        "save_method": "계좌 추가",
    }
    db[COLLECTION].insert_one(doc)
    invalidate_account_settings_cache()
    return {"account_id": norm_id, "name": name_norm, "account_type": atype}


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
    return {"account_id": norm_id, "deleted": True}


def get_account_settings_updated_at(account_id: str) -> str | None:
    doc = _db()[COLLECTION].find_one({"_id": str(account_id or "").strip().lower()}, {"updated_at": 1, "save_method": 1})
    if not doc or doc.get("updated_at") is None:
        return None
    ua = doc["updated_at"]
    if getattr(ua, "tzinfo", None) is None:
        ua = ua.replace(tzinfo=timezone.utc)
    return ua.isoformat()
