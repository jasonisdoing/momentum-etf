"""계좌 설정의 DB 단일 소스 레이어 (MongoDB `account_settings`).

(구)accounts.json 을 대체한다. 계좌별 1개 문서:

    {_id: <account_id>, name, icon, order, country_code, currency,
     benchmark: {ticker, name}, market_regime_index?, URL?,
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

from config import CACHE_TTL_LIVE
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
    "market_regime_index",
    # 합성 슬리브별 종목풀 — 모멘텀·신고가가 서로 다른 풀을 볼 수 있다.
    # 둘은 **계좌와 같은 국가**여야 한다(거래 달력·통화가 갈리면 합성이 성립하지 않는다).
    # 합성 전략 사용 여부 — 계좌 설정에서는 이 체크만 하고, 실제 조합(전략·종목풀·배분)은
    # `/strategy-mix` 화면에서 정한다. 조합과 배분은 함께 보며 맞추는 값이라 한 화면에 둔다.
    "mix_enabled",
    # 합성 슬리브 목록 — 슬롯마다 (전략, 종목풀, 표시 이름, 배분%). 슬롯 수가 늘어도
    # 필드를 추가하지 않도록 배열 하나로 둔다. 순서가 곧 슬롯 순서(A·B·C)다.
    "mix_sleeves",
    "mix_slack_enabled",
    "mix_cash_pct",
    "broker_api",
    "URL",
    # 보유종목 알림 — 계좌는 **On/Off 만** 갖는다.
    # 이평선 일수·손절 기준(%)은 종목이 속한 종목풀 설정에서 온다 — 같은 종목을 여러 계좌가
    # 들어도 판정 기준이 갈리면 안 되기 때문이다.
    "ma_alarm_enabled",
    "stoploss_alarm_enabled",
)

# 합성 슬롯 — 최소 둘(합성이려면 섞을 것이 둘은 있어야 한다), 최대 셋.
# 슬롯을 가리키는 키는 배열 순서로 준다(첫째 "a", 둘째 "b", 셋째 "c") — 저장하지 않는다.
MIN_MIX_SLEEVES = 2
MAX_MIX_SLEEVES = 3
MIX_SLEEVE_KEYS: tuple[str, ...] = ("a", "b", "c")

# 슬리브 항목의 필드 — 전략·종목풀·표시 이름·배분(%).
MIX_SLEEVE_FIELDS: tuple[str, ...] = ("strategy", "pool", "name", "weight_pct")
# 이름 길이 상한 — 표 헤더·액션 문구에 들어가므로 짧게 제한한다.
# 비워 두면 전략 이름(「A. 모멘텀」)을 쓴다. 같은 전략을 여러 슬롯에 올릴 수 있어,
# 화면에서 무엇이 무엇인지 구분하려면 사용자가 직접 붙일 이름이 필요하다.
MIX_SLEEVE_NAME_MAX_LEN = 20

_ALLOWED_COUNTRY_CODES = {"kor", "au", "us"}
_ALLOWED_CASH_CURRENCIES = {"KRW", "USD", "AUD"}

_CACHE_TTL_SECONDS = CACHE_TTL_LIVE
_cache: tuple[float, list[dict[str, Any]]] | None = None
_cache_lock = threading.Lock()


class AccountSettingsStoreError(ValueError):
    """계좌 설정 검증/저장 오류."""


def invalidate_account_settings_cache() -> None:
    """계좌 설정 TTL 캐시를 비운다 — 저장한 프로세스가 즉시 새 값을 보게 한다.

    다른 프로세스(스케줄러·워커)는 TTL(30초) 안에 자동으로 따라온다. 읽는 쪽에 만료 없는
    캐시를 두면 이 무효화가 닿지 않아 프로세스마다 답이 갈리므로 두지 않는다
    (`settings_loader.get_account_settings` 주석 참고).
    """
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
        raise AccountSettingsStoreError("계좌 설정이 DB(account_settings)에 없습니다. 계좌 문서를 먼저 등록해주세요.")
    docs.sort(key=lambda item: (int(item.get("order") or 0), str(item["account_id"])))

    with _cache_lock:
        _cache = (now, [dict(d) for d in docs])
    return docs


def _validate_pct(account_id: str, label: str, raw: Any) -> float:
    """배분(%) 한 값 — 0~100 의 숫자. 합계 검사는 호출부가 따로 한다."""
    try:
        pct = float(raw)
    except (TypeError, ValueError) as exc:
        raise AccountSettingsStoreError(f"'{account_id}' 의 {label} 는 숫자여야 합니다: {raw}") from exc
    if not 0.0 <= pct <= 100.0:
        raise AccountSettingsStoreError(f"'{account_id}' 의 {label} 는 0~100 이어야 합니다: {pct}")
    return round(pct, 2)


def _validate_mix_pool(
    account_id: str, label: str, raw: Any, values: dict[str, Any], existing_doc: dict[str, Any]
) -> str:
    """슬리브가 볼 종목풀 — 등록된 풀이고 계좌와 국가·통화가 같아야 한다.

    국가가 다르면 거래 달력이 갈려 월초 리밸런싱 판정일이 슬리브마다 달라지고, 통화가
    다르면 원화 환산율(`_krw_rate`)과 백테스트 시작 자본이 한 값으로 정해지지 않는다 —
    어느 쪽이든 합성 곡선이 성립하지 않는다.
    """
    from utils.settings_loader import get_ticker_type_settings, list_available_ticker_types

    pool = str(raw or "").strip().lower()
    if not pool:
        raise AccountSettingsStoreError(f"'{account_id}' 의 {label} 종목풀을 고르세요.")
    allowed = list_available_ticker_types()
    if pool not in allowed:
        raise AccountSettingsStoreError(
            f"'{account_id}' 의 {label} 종목풀은 {', '.join(allowed)} 중 하나여야 합니다: {raw}"
        )

    pool_config = get_ticker_type_settings(pool) or {}
    for name, account_value, pool_value in (
        (
            "국가",
            str(values.get("country_code") or existing_doc.get("country_code") or "").strip().lower(),
            str(pool_config.get("country_code") or "").strip().lower(),
        ),
        (
            "통화",
            str(values.get("currency") or existing_doc.get("currency") or "").strip().upper(),
            str(pool_config.get("currency") or "").strip().upper(),
        ),
    ):
        if account_value and pool_value and pool_value != account_value:
            raise AccountSettingsStoreError(
                f"'{account_id}'({account_value}) 의 {label} 종목풀은 같은 {name}의 것이어야 합니다: {pool}({pool_value})"
            )
    return pool


def _validate_mix_sleeves(
    account_id: str, raw: Any, values: dict[str, Any], existing_doc: dict[str, Any]
) -> list[dict[str, Any]]:
    """합성 슬리브 목록 — [{strategy, pool, name, weight_pct}] 를 검증해 정규화한다.

    순서가 곧 슬롯 순서다(첫째 A, 둘째 B, 셋째 C). 키는 저장하지 않고 읽을 때 순서로
    붙인다(`mix_sleeves_of`) — 저장된 키와 순서가 어긋날 여지를 아예 없앤다.
    """
    from utils.mix_sleeve import STRATEGY_OPTIONS

    if not isinstance(raw, list):
        raise AccountSettingsStoreError(f"'{account_id}' 의 mix_sleeves 는 목록이어야 합니다.")
    if not MIN_MIX_SLEEVES <= len(raw) <= MAX_MIX_SLEEVES:
        raise AccountSettingsStoreError(
            f"'{account_id}' 의 합성 슬리브는 {MIN_MIX_SLEEVES}~{MAX_MIX_SLEEVES}개여야 합니다: {len(raw)}개"
        )

    sleeves: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for index, item in enumerate(raw):
        label = MIX_SLEEVE_KEYS[index].upper()
        if not isinstance(item, dict):
            raise AccountSettingsStoreError(f"'{account_id}' 의 {label} 슬리브는 객체여야 합니다.")

        strategy = str(item.get("strategy") or "").strip().lower()
        if strategy not in STRATEGY_OPTIONS:
            raise AccountSettingsStoreError(
                f"'{account_id}' 의 {label} 전략은 {', '.join(STRATEGY_OPTIONS)} 중 하나여야 합니다: {item.get('strategy')}"
            )
        pool = _validate_mix_pool(account_id, label, item.get("pool"), values, existing_doc)

        # 같은 (전략, 종목풀) 이 두 슬롯에 오면 같은 종목이 양쪽에 잡혀 계좌 보유 수량이
        # 두 번 세어진다. 전략이나 풀 중 하나만 달라도 허용한다.
        if (strategy, pool) in seen:
            raise AccountSettingsStoreError(
                f"'{account_id}' 에 같은 조합의 슬리브가 둘 있습니다 — 전략이나 종목풀 중 하나는 달라야 합니다: "
                f"{strategy}({pool})"
            )
        seen.add((strategy, pool))

        name = str(item.get("name") or "").strip()
        if len(name) > MIX_SLEEVE_NAME_MAX_LEN:
            raise AccountSettingsStoreError(
                f"'{account_id}' 의 {label} 슬리브 이름은 {MIX_SLEEVE_NAME_MAX_LEN}자 이내여야 합니다: {name}"
            )
        sleeves.append(
            {
                "strategy": strategy,
                "pool": pool,
                # 빈 이름은 None — 화면이 전략 이름을 대신 쓴다.
                "name": name or None,
                "weight_pct": _validate_pct(account_id, f"{label} 배분", item.get("weight_pct")),
            }
        )
    return sleeves


def mix_sleeves_of(account_settings: dict[str, Any]) -> list[dict[str, Any]]:
    """저장된 합성 슬리브 — [{key, strategy, pool, name, weight_pct}]. 없으면 빈 목록.

    키(a·b·c)는 여기서 순서대로 붙인다. 합성 계산·화면·알림이 모두 이 함수로 읽어야
    슬롯 순서가 한 곳에서만 정해진다.
    """
    raw = account_settings.get("mix_sleeves")
    if not isinstance(raw, list):
        return []
    sleeves: list[dict[str, Any]] = []
    for index, item in enumerate(raw[:MAX_MIX_SLEEVES]):
        if not isinstance(item, dict):
            continue
        sleeves.append(
            {
                "key": MIX_SLEEVE_KEYS[index],
                "strategy": str(item.get("strategy") or "").strip().lower(),
                "pool": str(item.get("pool") or "").strip().lower(),
                "name": str(item.get("name") or "").strip(),
                "weight_pct": float(item.get("weight_pct") or 0.0),
            }
        )
    return sleeves


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
            base_currency = str(values.get("currency") or existing_doc.get("currency") or "").strip().upper()
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
            # 호주 계좌는 `ASX:` 를 붙여 저장한다. 가격 캐시가 호주 종목을 그 형태로 보관하고,
            # 미국에도 같은 티커가 있어(예: IVV) 접두사가 없으면 구분되지 않는다.
            # 이 값이 없으면 자산 헬퍼 백테스트가 "가격 캐시 누락: IVV" 로 실패한다.
            country = str(values.get("country_code") or existing_doc.get("country_code") or "").strip().lower()
            if country == "au":
                from utils.asx_ticker import ensure_asx_prefix

                ticker = ensure_asx_prefix(ticker)
            cleaned[key] = {"ticker": ticker, "name": bench_name}
        elif key == "mix_sleeves":
            cleaned[key] = _validate_mix_sleeves(account_id, raw, values, existing_doc)
        elif key == "mix_enabled":
            # 켜면 `/strategy-mix` 목록에 오른다. 조합이 아직 없으면 그 화면에서 고르게 한다.
            cleaned[key] = bool(raw)
        elif key == "mix_slack_enabled":
            # 합성 오늘의 액션 슬랙 알람 — 새 지시·수량 증가가 생기면 발송한다.
            cleaned[key] = bool(raw)
        elif key == "mix_cash_pct":
            # 합성에서 비워 두는 현금 몫(%). 슬리브 배분과의 합이 100 인지는 아래에서 본다.
            cleaned[key] = _validate_pct(account_id, key, raw)
        elif key == "broker_api":
            # 증권사 API 연동 — {provider, account_no}. 없음이면 null.
            # provider 는 커넥터 레지스트리에 있어야 하고, 계좌번호는 화면의 '확인' 이
            # API 로 나열한 목록에서 고른 값이다(형식 검증만 하고 실조회는 하지 않는다).
            if raw in (None, "", {}):
                cleaned[key] = None
                continue
            if not isinstance(raw, dict):
                raise AccountSettingsStoreError(f"'{account_id}' 의 broker_api 는 객체여야 합니다.")
            provider = str(raw.get("provider") or "").strip().upper()
            account_no = str(raw.get("account_no") or "").strip()
            from services.broker_api_service import PROVIDERS

            allowed_providers = {p["id"] for p in PROVIDERS}
            if provider not in allowed_providers:
                raise AccountSettingsStoreError(
                    f"'{account_id}' 의 broker_api.provider 는 {', '.join(sorted(allowed_providers))} 중 하나여야 합니다: {raw}"
                )
            if not account_no:
                raise AccountSettingsStoreError(f"'{account_id}' 의 broker_api.account_no 가 비어 있습니다.")
            cleaned[key] = {"provider": provider, "account_no": account_no}
        elif key == "market_regime_index":
            # 계좌별 시장 레짐 판정 지수 — 시장추세 지수(INDICES) 중 하나(필수, {ticker, name}).
            if not isinstance(raw, dict):
                raise AccountSettingsStoreError(
                    f"'{account_id}' 의 market_regime_index 는 {{ticker, name}} 객체여야 합니다."
                )
            ticker = str(raw.get("ticker") or "").strip()
            if not ticker:
                raise AccountSettingsStoreError(
                    f"'{account_id}' 의 market_regime_index 는 필수입니다(시장 레짐 지수를 선택하세요)."
                )
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
        elif key == "stoploss_alarm_enabled":
            cleaned[key] = bool(raw)
    # 슬리브 배분과 현금은 한 묶음이다 — 하나만 바꾸면 나머지와 합이 어긋난 채로 저장된다.
    # 슬리브별 (전략, 종목풀) 중복 검사는 `_validate_mix_sleeves` 가 이미 했다.
    if ("mix_sleeves" in cleaned) != ("mix_cash_pct" in cleaned):
        raise AccountSettingsStoreError(
            f"'{account_id}' 의 합성 배분은 mix_sleeves 와 mix_cash_pct 를 함께 보내야 합니다."
        )
    if "mix_sleeves" in cleaned:
        parts = [(MIX_SLEEVE_KEYS[i].upper(), row["weight_pct"]) for i, row in enumerate(cleaned["mix_sleeves"])]
        total = sum(pct for _, pct in parts) + float(cleaned["mix_cash_pct"])
        # 0.01 은 소수 둘째 자리 반올림 오차만 허용하는 폭이다(예: 33.33+33.33+33.34).
        if abs(total - 100.0) > 0.01:
            joined = " + ".join(f"{label} {pct}%" for label, pct in parts)
            raise AccountSettingsStoreError(
                f"'{account_id}' 의 합성 배분 합계가 100%가 아닙니다: "
                f"{joined} + 현금 {cleaned['mix_cash_pct']}% = {round(total, 2)}%"
            )

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
        raise AccountSettingsStoreError(
            f"country_code 는 {', '.join(sorted(_ALLOWED_COUNTRY_CODES))} 중 하나여야 합니다: {country_code}"
        )
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
    # 딸림 데이터는 카탈로그(`utils/data_table_catalog`)가 단일 소스다 — 계좌 메모·가격 캐시
    # 컬렉션·알림 상태의 계좌별 키·원장의 빈 항목까지 한 번에 지운다. 지울 자리를 이 함수가
    # 직접 들고 있으면 컬렉션이 늘 때마다 조용히 빠진다.
    # 과거 자산 기록(`daily_snapshots` 등 집계 분류)은 카탈로그가 보존으로 못박아 뒀다 —
    # 계좌를 지웠다고 지난 수익률 그래프가 바뀌면 안 된다.
    from utils.data_table_catalog import purge_owner

    cleanup = purge_owner("account", norm_id)

    # 계좌 문서 자체는 딸림 데이터를 다 지운 뒤에 없앤다.
    db[COLLECTION].delete_one({"_id": norm_id})
    invalidate_account_settings_cache()

    return {"account_id": norm_id, "deleted": True, **cleanup}


def get_account_settings_updated_at(account_id: str) -> str | None:
    doc = _db()[COLLECTION].find_one(
        {"_id": str(account_id or "").strip().lower()}, {"updated_at": 1, "save_method": 1}
    )
    if not doc or doc.get("updated_at") is None:
        return None
    ua = doc["updated_at"]
    if getattr(ua, "tzinfo", None) is None:
        ua = ua.replace(tzinfo=timezone.utc)
    return ua.isoformat()
