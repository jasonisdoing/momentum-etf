"""종목풀 설정의 DB 단일 소스 레이어.

MongoDB `pool_settings` 컬렉션이 종목풀의 구조와 편집값을 모두 보관한다.

    구조: ticker_type, name, icon, order, country_code, currency, pool_kind
    편집: TOP_N_HOLD, SHORT_MA_DAYS, LONG_MA_DAYS,             ← 전략 공용 설정
          INTRAWEEK_EXIT, INTRAWEEK_STOP_PCT,                   ← 모멘텀 전용
          BUY_SLIPPAGE_PCT, SELL_SLIPPAGE_PCT, STOPLOSS_THRESHOLD_PCT,
          BENCHMARK, MARKET_REGIME_INDEX (선택 — 비우면 미설정)

모멘텀 전략 설정을 여기 두는 이유: 이 프로젝트에서 모멘텀은 **그 풀의 기본 판정 기준**이다.
전략 화면에서 튜닝 결과를 적용하면 순위 화면·보유종목 알림·종목풀 백테스트가 같은 값을
쓴다(예전에는 `system_config.momentum_settings` 에 따로 저장돼 8개 풀에서 값이 갈렸다).
신고가는 한 풀을 다른 설정으로 굴리는 별도 전략이라 자기 설정 문서를 그대로 쓴다.

런타임 로딩은 DB 문서가 없거나 필수 키가 누락되면 명확히 에러를 낸다.
캐시: 멀티프로세스(fastapi/scheduler/worker)에서 변경이 반영되도록 짧은 TTL(30초) 캐시를
      쓴다. 저장한 프로세스는 즉시 무효화하고, 나머지는 TTL 내 자동 반영된다.

컬렉션 문서 형태:
    {
      _id: <ticker_type>, name, icon, order, country_code, currency,
      SHORT_MA_DAYS, LONG_MA_DAYS,
      BUY_SLIPPAGE_PCT, SELL_SLIPPAGE_PCT, BENCHMARK, MARKET_REGIME_INDEX, updated_at
    }
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from config import (
    ADR_FLOOR_OPTIONS,
    POOL_KIND_OPTIONS,
    SLIPPAGE_PCT_OPTIONS,
    STOP_LOSS_PCT_OPTIONS,
    TOP_N_HOLD_OPTIONS,
)
from config import STOP_LOSS_PCT_OPTIONS as STOPLOSS_PCT_OPTIONS
from utils.logger import get_app_logger
from utils.ma_options import LONG_MA_OPTIONS, SHORT_MA_OPTIONS

logger = get_app_logger()

COLLECTION = "pool_settings"


INTERNAL_POOL_ID_PREFIX = "__"

# DB 오버라이드 대상 키 — 전부 필수이며 비어 있으면 로딩 자체가 실패한다.
# 이 셋은 **모멘텀 전략 설정이기도 하다** — 이 프로젝트에서 모멘텀은 그 풀의 기본 판정
# 기준이라, 전략 설정을 따로 두지 않고 풀 문서 하나에 모은다(순위 화면·보유종목 알림·
# 종목풀 백테스트가 같은 값을 본다). 신고가는 자기 설정 문서를 따로 쓴다.
OVERRIDABLE_KEYS: tuple[str, ...] = (
    "TOP_N_HOLD",
    "SHORT_MA_DAYS",
    "LONG_MA_DAYS",
)

# 모멘텀 전략 전용 값 — 위 셋과 함께 한 풀의 전략 설정을 이룬다. 기존 문서에 없을 수 있어
# 로딩 필수값은 아니다(미설정이면 전략 화면에서 저장해야 한다).
MOMENTUM_KEYS: tuple[str, ...] = (
    "ADR_FLOOR",  # None = 게이트 없음 (모멘텀 ADR 하한 — 시장은 MARKET_REGIME_INDEX 를 따름)
    "INTRAWEEK_EXIT",
    "INTRAWEEK_STOP_PCT",  # None = 손절 없음
    "REBALANCE_MODE",  # weekly = 매주 재선정(기존) · hold = 자격 유지
)

# 보유종목 손절 알림 기준(%). 이평선과 같은 성격의 **종목 판정 기준**이라 계좌가 아니라
# 여기에 둔다 — 같은 종목을 두 계좌가 들어도 손절선이 갈리면 안 된다.
# 기존 문서에 없을 수 있어 로딩 필수값은 아니다(미설정이면 알림이 판정 불가로 남는다).
STOPLOSS_KEYS: tuple[str, ...] = ("STOPLOSS_THRESHOLD_PCT",)


# 종목풀별 거래비용 설정. 기존 문서에 없어도 설정 화면은 열려야 하므로 로딩 필수값은 아니다.
# 단, 슬리피지를 실제로 사용하는 백테스트/계산 로직은 누락 시 명시적으로 실패한다.
SLIPPAGE_KEYS: tuple[str, ...] = (
    "BUY_SLIPPAGE_PCT",
    "SELL_SLIPPAGE_PCT",
)

# 편집 가능하지만 비워둘 수 있는 키. 필수 검사에서 제외한다.
# 빈 문자열은 '미설정'을 뜻하며, 읽는 쪽이 그 상태를 명시적으로 처리해야 한다(임의 대체 금지).
OPTIONAL_EDITABLE_KEYS: tuple[str, ...] = ("BENCHMARK", "MARKET_REGIME_INDEX")

# 종목풀 설정 화면에서 편집하는 전체 키(순서 = 화면 표시 순서).
POOL_EDITABLE_KEYS: tuple[str, ...] = (
    *OVERRIDABLE_KEYS,
    *MOMENTUM_KEYS,
    *SLIPPAGE_KEYS,
    *STOPLOSS_KEYS,
    *OPTIONAL_EDITABLE_KEYS,
)

STRUCTURAL_KEYS: tuple[str, ...] = (
    "name",
    "icon",
    "order",
    "country_code",
    "currency",
    "pool_kind",
)


_INT_KEYS = ("TOP_N_HOLD", "SHORT_MA_DAYS", "LONG_MA_DAYS")
_FLOAT_KEYS = ("BUY_SLIPPAGE_PCT", "SELL_SLIPPAGE_PCT", "STOPLOSS_THRESHOLD_PCT")
_ALLOWED_COUNTRY_CODES = {"kor", "au", "us"}
_ALLOWED_CURRENCIES = {"KRW", "AUD", "USD"}

# **종목풀 정의·오버라이드에는 TTL 캐시를 두지 않는다.** 설정을 고치면 그 즉시 화면에
# 반영돼야 하는데, TTL 이 있으면 만료 전까지 옛 값이 남아 "저장했는데 안 바뀐다"가 된다.
# pool_settings 는 풀 수십 건짜리 작은 컬렉션이라 매번 읽어도 부담이 없다.
# 설정 변경에 딸린 무거운 캐시(랭킹·전략)는 invalidate_overlay_cache() 가 계속 지운다.


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
    # 저장 시점에 종목풀 등록 표기로 맞춰 두므로(save_pool_settings) 여기서 고치지 않는다.
    # 읽으면서 조용히 보정하면 잘못 저장된 값이 드러나지 않는다.
    return str(benchmark.get("ticker") or "").strip().upper()


def get_pool_market_regime_index(settings: dict[str, Any]) -> dict[str, str] | None:
    """종목풀 설정의 시장 레짐 지수를 반환한다. 미설정이면 None."""
    index = settings.get("MARKET_REGIME_INDEX")
    if not isinstance(index, dict):
        return None
    ticker = str(index.get("ticker") or "").strip()
    name = str(index.get("name") or "").strip()
    if not ticker:
        return None
    return {"ticker": ticker, "name": name or ticker}


def invalidate_overlay_cache() -> None:
    """종목풀 설정 변경에 딸린 캐시를 비운다 (저장 직후 호출).

    설정 자체는 캐시하지 않으므로 지울 것이 없고, 설정에 의존하는 로더·랭킹 캐시만 비운다.
    """
    _invalidate_dependent_caches()


def _invalidate_dependent_caches() -> None:
    """종목풀 정의 변경에 의존하는 프로세스 내 캐시를 비운다."""
    try:
        from utils import settings_loader

        settings_loader._load_pool_configs.cache_clear()
    except Exception as exc:
        logger.warning("종목풀 로더 캐시 무효화 실패: %s", exc)
    try:
        from utils.cache_invalidation import invalidate_pool_caches

        invalidate_pool_caches()
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


def _normalize_pool_values(
    values: dict[str, Any], *, require_ticker_type: bool, check_options: bool = True
) -> dict[str, Any]:
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
    if "pool_kind" in values:
        pool_kind = str(values.get("pool_kind") or "").strip().lower()
        # 빈 값은 '미설정 유지' — 기존 문서와의 하위 호환 (토글은 항상 둘 중 하나를 보낸다).
        if pool_kind:
            if pool_kind not in POOL_KIND_OPTIONS:
                allowed = ", ".join(POOL_KIND_OPTIONS)
                raise PoolSettingsError(f"pool_kind 는 {allowed} 중 하나여야 합니다: {pool_kind}")
            cleaned["pool_kind"] = pool_kind

    editable_input = {k: values[k] for k in POOL_EDITABLE_KEYS if k in values}
    cleaned.update(_validate_values(editable_input, check_options=check_options) if editable_input else {})

    return cleaned


def _normalize_pool_doc(doc: dict[str, Any]) -> dict[str, Any]:
    pool_id = _normalize_ticker_type(doc.get("_id") or doc.get("ticker_type"))
    doc = dict(doc)
    required_keys = ("name", "icon", "order", "country_code", "currency", *OVERRIDABLE_KEYS)
    missing = [key for key in required_keys if key not in doc or doc[key] in (None, "")]
    if missing:
        raise PoolSettingsError(
            f"종목풀 '{pool_id}' 의 DB 설정에 필수 값이 없습니다: {', '.join(missing)}. "
            f"`/pools-settings` 화면에서 값을 수정하세요."
        )

    # 읽기 — 타입만 본다. 선택지 검사는 저장 경로(_validate_values(check_options=True))에서만.
    normalized = _normalize_pool_values({**doc, "ticker_type": pool_id}, require_ticker_type=True, check_options=False)
    normalized["ticker_type"] = pool_id
    if doc.get("updated_at") is not None:
        normalized["updated_at"] = doc["updated_at"]
    if doc.get("save_method") is not None:
        normalized["save_method"] = doc["save_method"]
    if doc.get("default") is not None:
        normalized["default"] = bool(doc.get("default"))
    return normalized


def load_pool_definitions() -> list[dict[str, Any]]:
    """DB에서 종목풀 정의를 읽는다. 쓰지 않을 풀은 비활성이 아니라 삭제로 관리한다."""
    docs = [
        _normalize_pool_doc(dict(doc)) for doc in _db()[COLLECTION].find({}) if not _is_internal_pool_id(doc.get("_id"))
    ]
    if not docs:
        raise PoolSettingsError(
            "종목풀 설정이 DB(pool_settings)에 없습니다. `/pools-settings` 화면에서 종목풀을 생성하세요."
        )
    docs.sort(key=lambda item: (int(item["order"]), str(item["ticker_type"])))
    return docs


def _load_overrides_from_db() -> dict[str, dict[str, Any]]:
    """pool_settings 컬렉션 전체를 {pool_id: {key: value}} 로 읽는다. 실패 시 {}."""
    try:
        result: dict[str, dict[str, Any]] = {}
        for doc in _db()[COLLECTION].find({}):
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
    """전체 오버라이드 맵을 반환한다 ({pool_id: {key: value}}). 캐시하지 않는다."""
    return _load_overrides_from_db()


def _validate_values(values: dict[str, Any], *, check_options: bool = True) -> dict[str, Any]:
    """입력값을 검증/정규화한다. 잘못된 값은 PoolSettingsError.

    ``check_options`` — 선택지(이평선·업종상한·손절·슬리피지) 포함 여부 검사. **저장할 때만** 켠다.
    DB 에서 읽을 때는 끈다: 선택지가 바뀐 뒤 옛 값이 남아 있거나(또는 서버가 옛 코드를 돌리거나)
    하면, 읽기에서 막는 순간 그 풀을 쓰지 않는 배치·화면까지 전부 죽는다. 선택지 밖 값은
    화면이 "(선택지 밖)"으로 보여주고 사용자가 고쳐 저장한다.
    """
    cleaned: dict[str, Any] = {}

    for key in _INT_KEYS:
        if key not in values:
            continue
        raw = values[key]

        try:
            num = int(raw)
        except (TypeError, ValueError) as exc:
            raise PoolSettingsError(f"{key} 은 정수여야 합니다: {raw}") from exc

        if key in ("SHORT_MA_DAYS", "LONG_MA_DAYS") and check_options:
            allowed = SHORT_MA_OPTIONS if key == "SHORT_MA_DAYS" else LONG_MA_OPTIONS
            if num not in allowed:
                options = ", ".join(str(day) for day in allowed)
                raise PoolSettingsError(f"{key} 는 다음 값 중 하나여야 합니다: {options}. 입력값: {num}")
        if key == "TOP_N_HOLD" and check_options and num not in TOP_N_HOLD_OPTIONS:
            options = ", ".join(str(value) for value in TOP_N_HOLD_OPTIONS)
            raise PoolSettingsError(f"{key} 는 다음 값 중 하나여야 합니다: {options}. 입력값: {num}")
        cleaned[key] = num

    # 모멘텀 전용 — 숫자/불리언이 섞여 있고 None 이 '없음' 을 뜻한다(임의 보정하지 않는다).
    if "ADR_FLOOR" in values:
        raw = values["ADR_FLOOR"]
        floor = None if raw in (None, "", "none") else int(raw)
        if check_options and floor not in ADR_FLOOR_OPTIONS:
            allowed = ", ".join("없음" if v is None else str(v) for v in ADR_FLOOR_OPTIONS)
            raise PoolSettingsError(f"ADR_FLOOR 는 {allowed} 중 하나여야 합니다: {raw}")
        cleaned["ADR_FLOOR"] = floor
    if "INTRAWEEK_EXIT" in values:
        cleaned["INTRAWEEK_EXIT"] = bool(values["INTRAWEEK_EXIT"])
    if "REBALANCE_MODE" in values:
        # 교체 규칙 — 값 목록은 모멘텀 서비스가 단일 소스다(여기 복사본을 두지 않는다).
        from utils.momentum_service import REBALANCE_MODE_OPTIONS, REBALANCE_MODE_WEEKLY

        raw = values["REBALANCE_MODE"]
        mode = REBALANCE_MODE_WEEKLY if raw in (None, "") else str(raw).strip().lower()
        if check_options and mode not in REBALANCE_MODE_OPTIONS:
            allowed = ", ".join(REBALANCE_MODE_OPTIONS)
            raise PoolSettingsError(f"REBALANCE_MODE 는 {allowed} 중 하나여야 합니다: {raw}")
        cleaned["REBALANCE_MODE"] = mode
    if "INTRAWEEK_STOP_PCT" in values:
        raw = values["INTRAWEEK_STOP_PCT"]
        stop = None if raw in (None, "", "none") else round(float(raw), 2)
        allowed_stops = {None, *(round(v, 2) for v in STOP_LOSS_PCT_OPTIONS)}
        if check_options and stop not in allowed_stops:
            allowed = ", ".join("없음" if v is None else f"{v:g}" for v in (None, *STOP_LOSS_PCT_OPTIONS))
            raise PoolSettingsError(f"INTRAWEEK_STOP_PCT 는 {allowed} 중 하나여야 합니다: {raw}")
        cleaned["INTRAWEEK_STOP_PCT"] = stop

    for key in _FLOAT_KEYS:
        if key not in values:
            continue
        raw = values[key]
        try:
            num = round(float(raw), 2)
        except (TypeError, ValueError) as exc:
            raise PoolSettingsError(f"{key} 은 숫자여야 합니다: {raw}") from exc
        allowed_options = STOPLOSS_PCT_OPTIONS if key in STOPLOSS_KEYS else SLIPPAGE_PCT_OPTIONS
        if check_options and num not in {round(option, 2) for option in allowed_options}:
            options = ", ".join(f"{option:g}" for option in allowed_options)
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

    if "MARKET_REGIME_INDEX" in values:
        raw = values["MARKET_REGIME_INDEX"]
        if raw in (None, ""):
            cleaned["MARKET_REGIME_INDEX"] = None
        elif not isinstance(raw, dict):
            raise PoolSettingsError("MARKET_REGIME_INDEX 는 {ticker, name} 객체여야 합니다.")
        else:
            ticker = str(raw.get("ticker") or "").strip()
            name = str(raw.get("name") or "").strip()
            if not ticker and not name:
                cleaned["MARKET_REGIME_INDEX"] = None
            elif not ticker or not name:
                raise PoolSettingsError("MARKET_REGIME_INDEX 에는 ticker/name 이 모두 필요합니다.")
            else:
                # ADR(시장 폭)이 있는 4개 시장 + **그 풀 자신**만 허용.
                # 단일 소스는 market_breadth 의 매핑과 예약 티커다.
                from utils.market_breadth_service import MARKET_BY_INDEX_TICKER, SELF_POOL_REGIME_TICKER
                from utils.market_trend_service import INDICES

                allowed = {
                    str(item["yf_ticker"]): str(item["name"])
                    for item in INDICES
                    if str(item["yf_ticker"]) in MARKET_BY_INDEX_TICKER
                }
                allowed[SELF_POOL_REGIME_TICKER] = "종목풀"
                if ticker not in allowed:
                    options = ", ".join(f"{label}({code})" for code, label in allowed.items())
                    raise PoolSettingsError(
                        f"MARKET_REGIME_INDEX 는 ADR 기준 중 하나여야 합니다: {options}. 입력값: {ticker}"
                    )
                cleaned["MARKET_REGIME_INDEX"] = {"ticker": ticker, "name": allowed[ticker]}

    if not cleaned:
        raise PoolSettingsError("저장할 값이 없습니다.")
    return cleaned


def _canonical_benchmark_for_country(
    benchmark: dict[str, Any], country_code: str, candidates: list[dict[str, Any]]
) -> dict[str, str]:
    """벤치마크를 해당 국가 종목의 표준 티커·이름으로 확정한다."""
    country = str(country_code or "").strip().lower()
    ticker = str(benchmark.get("ticker") or "").strip().upper()
    if country == "au":
        from utils.asx_ticker import ensure_asx_prefix

        ticker = ensure_asx_prefix(ticker)

    candidate_by_ticker = {
        str(item.get("ticker") or "").strip().upper(): str(item.get("name") or "").strip()
        for item in candidates
        if item.get("ticker")
    }
    name = candidate_by_ticker.get(ticker)
    if name is None:
        raise PoolSettingsError(f"벤치마크 '{ticker}' 는 국가({country})에 등록된 종목이 아닙니다.")
    return {"ticker": ticker, "name": name or ticker}


def _normalize_benchmark_for_country(cleaned: dict[str, Any], country_code: str) -> None:
    """정리된 설정의 벤치마크를 같은 국가 종목 목록으로 검증하고 제자리에서 표준화한다."""
    benchmark = cleaned.get("BENCHMARK")
    if not isinstance(benchmark, dict) or not benchmark.get("ticker"):
        return

    from utils.stock_list_io import get_etfs

    country = str(country_code or "").strip().lower()
    pool_ids = [
        str(definition["ticker_type"])
        for definition in load_pool_definitions()
        if str(definition.get("country_code") or "").strip().lower() == country
    ]
    candidates = [item for pool_id in pool_ids for item in get_etfs(pool_id)]
    cleaned["BENCHMARK"] = _canonical_benchmark_for_country(benchmark, country, candidates)


def save_pool_settings(pool_id: str, values: dict[str, Any], save_method: str = "사용자") -> dict[str, Any]:
    """편집한 값을 pool_settings 에 upsert 하고 캐시를 무효화한다.

    pool_id 는 유효한 ticker_type.
    반환: 저장된(정규화된) 값.
    """
    from utils.settings_loader import get_ticker_type_settings, list_available_ticker_types

    norm_id = str(pool_id or "").strip().lower()
    if norm_id not in list_available_ticker_types():
        raise PoolSettingsError(f"알 수 없는 종목풀입니다: {pool_id}")

    cleaned = _validate_values(values)
    country_code = str(get_ticker_type_settings(norm_id).get("country_code") or "").strip().lower()
    _normalize_benchmark_for_country(cleaned, country_code)

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
        from utils.cache_invalidation import invalidate_pool_caches

        invalidate_pool_caches()
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
    _normalize_benchmark_for_country(cleaned, str(cleaned.get("country_code") or ""))

    db = _db()
    if db[COLLECTION].find_one({"_id": pool_id}, {"_id": 1}) is not None:
        raise PoolSettingsError(f"이미 존재하는 종목풀입니다: {pool_id}")
    db[COLLECTION].insert_one(
        {
            "_id": pool_id,
            **cleaned,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "save_method": save_method,
        }
    )
    invalidate_overlay_cache()
    return {"ticker_type": pool_id, **cleaned}


def update_pool(pool_id: str, values: dict[str, Any], save_method: str = "사용자") -> dict[str, Any]:
    """기존 종목풀의 메타/설정 값을 수정한다. ticker_type 은 불변이다."""
    norm_id = _normalize_ticker_type(pool_id)
    if "ticker_type" in values and _normalize_ticker_type(values["ticker_type"]) != norm_id:
        raise PoolSettingsError("ticker_type 은 변경할 수 없습니다.")
    db = _db()
    existing = db[COLLECTION].find_one({"_id": norm_id})
    if existing is None:
        raise PoolSettingsError(f"알 수 없는 종목풀입니다: {pool_id}")
    cleaned = _normalize_pool_values({k: v for k, v in values.items() if k != "ticker_type"}, require_ticker_type=False)
    if not cleaned:
        raise PoolSettingsError("저장할 값이 없습니다.")
    country_code = str(cleaned.get("country_code") or existing.get("country_code") or "").strip().lower()
    _normalize_benchmark_for_country(cleaned, country_code)
    db[COLLECTION].update_one(
        {"_id": norm_id},
        {
            "$set": {**cleaned, "updated_at": datetime.utcnow(), "save_method": save_method},
            "$unset": {"type_source": ""},
        },
    )
    invalidate_overlay_cache()
    return cleaned


def get_pool_slippage(pool: str) -> tuple[float, float]:
    """종목풀 설정의 (매수, 매도) 편도 슬리피지(%) — 전략 백테스트가 공용으로 쓴다.

    미설정이면 임의값으로 대체하지 않고 명시적으로 실패한다(비용을 조용히 0으로
    두면 백테스트가 과대평가된다).
    """
    from utils.settings_loader import get_ticker_type_settings

    settings = get_ticker_type_settings(pool) or {}
    missing = [key for key in SLIPPAGE_KEYS if settings.get(key) in (None, "")]
    if missing:
        raise PoolSettingsError(
            f"종목풀({pool}) 설정에 {', '.join(missing)} 가 없습니다 — `/pools-settings` 에서 먼저 저장하세요."
        )
    return float(settings["BUY_SLIPPAGE_PCT"]), float(settings["SELL_SLIPPAGE_PCT"])


def get_pool_top_n_hold(pool: str) -> int:
    """종목풀의 보유 종목 수를 반환한다. 누락·선택지 밖 값은 명시적으로 실패한다."""
    from utils.settings_loader import get_ticker_type_settings

    settings = get_ticker_type_settings(pool) or {}
    raw = settings.get("TOP_N_HOLD")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise PoolSettingsError(
            f"종목풀({pool}) 설정에 TOP_N_HOLD 가 없습니다 — `/pools-settings` 에서 먼저 저장하세요."
        ) from exc
    if value not in TOP_N_HOLD_OPTIONS:
        options = ", ".join(str(item) for item in TOP_N_HOLD_OPTIONS)
        raise PoolSettingsError(f"종목풀({pool}) TOP_N_HOLD 는 {options} 중 하나여야 합니다: {value}")
    return value


def get_pool_delete_impact(pool_id: str) -> dict[str, Any]:
    """종목풀 삭제 전 영향도를 반환한다."""
    norm_id = _normalize_ticker_type(pool_id)
    db = _db()
    stock_count = db["stock_meta"].count_documents({"ticker_type": norm_id})
    exists = db[COLLECTION].find_one({"_id": norm_id}, {"_id": 1}) is not None
    return {
        "ticker_type": norm_id,
        "exists": exists,
        "stock_count": stock_count,
        "can_delete": exists,
    }


def delete_pool(pool_id: str) -> dict[str, Any]:
    """종목풀을 하드 삭제한다. 연결 종목 메타도 함께 제거한다."""
    impact = get_pool_delete_impact(pool_id)
    if not impact["exists"]:
        raise PoolSettingsError(f"알 수 없는 종목풀입니다: {pool_id}")
    norm_id = str(impact["ticker_type"])
    db = _db()

    # 딸림 데이터는 카탈로그(`utils/data_table_catalog`)가 단일 소스다 — 종목·메타 캐시·
    # 가격 캐시 컬렉션·전략 설정의 풀별 항목·이 풀을 가리키던 계좌 참조까지 한 번에 지운다.
    # 예전에는 지울 자리를 이 함수가 직접 들고 있어서, 컬렉션이 늘 때마다 조용히 빠졌다.
    from utils.data_table_catalog import purge_owner

    cleanup = purge_owner("pool", norm_id)

    # 종목풀 문서 자체는 딸림 데이터를 다 지운 뒤에 없앤다(순서가 뒤집히면 소유자를 잃은
    # 데이터가 중간 상태로 남는다).
    deleted_pool = db[COLLECTION].delete_one({"_id": norm_id}).deleted_count
    deleted_stocks = int(cleanup.get("stock_meta", 0))

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
        **cleanup,
    }
