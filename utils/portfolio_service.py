"""포트폴리오 전략 — 종목별 목표 비중을 정해 그대로 들고 가는 전략.

모멘텀·신고가와 달리 **종목을 고르지 않는다.** 순위·이평선 판정이 없으므로 교체·이탈·
손절도 없다. 사용자가 종목풀에서 종목을 골라 비중(%)을 직접 정하고, 정한 주기마다 그
비중으로 되돌린다. 종목 비중 합의 나머지는 현금이다.

설정은 MongoDB `system_config.portfolio_settings` 에 **풀별로** 저장한다
(`{settings_by_pool: {풀: {...}}}`) — 신고가와 같은 구조다. 모멘텀은 이 프로젝트의
기본 판정 기준이라 종목풀 문서에 흡수했지만, 포트폴리오는 한 풀을 다른 비중으로 굴리는
별도 전략이라 자기 문서를 쓴다.

합성(`utils/mix_sleeve`)의 세 번째 선택지가 될 것을 전제로 만든다 — 어댑터가 요구하는
`settings_map` · `validate_settings` · `load_settings` · `save_settings` 이름과 형태를
신고가와 똑같이 맞춘다.
"""

from __future__ import annotations

from typing import Any

from config import REBALANCE_BAND_PCT_OPTIONS, REBALANCE_LABELS, REBALANCE_OPTIONS
from utils.logger import get_app_logger
from utils.strategy_settings import require_start_date, validate_start_date

logger = get_app_logger()

_CONFIG_COLLECTION = "system_config"
_SETTINGS_KEY = "portfolio_settings"

# 백테스트 기간 기본값 — 화면에서 실행할 때 고르고, 저장하지 않는다(다른 전략과 동일).
DEFAULT_BACKTEST_MONTHS = 12

# 담을 수 있는 종목 수 상한 — 화면 표가 길어지는 것을 막는 안전장치일 뿐 전략 값이 아니다.
MAX_HOLDINGS = 30

# 풀을 바꾸면 그 풀의 값으로 전환되는 항목. 여기 빠진 키는 저장을 눌러도 버려진다.
PER_POOL_SETTING_KEYS = ("start_date", "weights", "cash_weight_pct", "rebalance", "band_pct")

DEFAULT_SETTINGS: dict[str, Any] = {
    # [{ticker, weight_pct}] — 순서가 화면 표 순서다(사용자가 드래그로 바꾼다).
    "weights": [],
    # 현금 비중(%) — **사용자가 직접 정한다**(`/asset-helper` 와 같은 규칙).
    # 종목합의 나머지로 자동 계산하지 않는다: 자동이면 종목 하나를 줄일 때 그만큼이
    # 조용히 현금으로 흘러가, 사용자가 어디를 조정할지 정할 기회를 잃는다.
    "cash_weight_pct": 100.0,
    "rebalance": "quarterly",
    "band_pct": 3.0,
}


def _db():
    from utils.db_manager import get_db_connection

    db = get_db_connection()
    if db is None:
        raise RuntimeError("DB 연결에 실패했습니다.")
    return db


# ── 종목풀 ─────────────────────────────────────────────────────────────────
def available_pools() -> list[str]:
    from utils.settings_loader import list_available_ticker_types

    return list_available_ticker_types()


def pool_options() -> list[dict[str, Any]]:
    """화면 셀렉트용 종목풀 목록 — 다른 전략 화면과 같은 형태(`formatPoolLabel` 이 쓴다)."""
    from utils.settings_loader import get_ticker_type_settings

    options: list[dict[str, Any]] = []
    for pool in available_pools():
        try:
            settings = get_ticker_type_settings(pool) or {}
        except Exception:
            continue
        options.append(
            {
                "ticker_type": pool,
                "name": str(settings.get("name") or pool),
                "icon": str(settings.get("icon") or ""),
                "order": settings.get("order"),
                "country_code": str(settings.get("country_code") or "").strip().lower(),
                "currency": str(settings.get("currency") or "").strip().upper(),
                "pool_kind": str(settings.get("pool_kind") or "").strip().lower(),
            }
        )
    return sorted(options, key=lambda item: (item["order"] is None, item["order"]))


def load_universe(pool: str) -> list[dict[str, str]]:
    """그 종목풀에 담긴 종목 — 비중을 매길 수 있는 후보다.

    제외 종목(`exclude_from_ranking`)도 후보에 넣는다. 이 전략은 순위를 매기지 않으므로
    '순위에서 빼는' 표시와 무관하다 — 사용자가 직접 고르는 목록이기 때문이다.
    """
    from utils.stock_list_io import get_etfs

    universe: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in get_etfs(pool) or []:
        ticker = str(item.get("ticker") or "").strip().upper()
        if not ticker or ticker in seen:
            continue
        seen.add(ticker)
        universe.append({"ticker": ticker, "name": str(item.get("name") or ticker), "pool": pool})
    return universe


def universe_metrics(pool: str) -> list[dict[str, Any]]:
    """그 풀 종목의 표시 지표 — 일간·현재가·기간수익률·MDD·소르티노.

    계산은 `/asset-helper` 와 **같은 공용 함수**를 쓴다(`utils/asset_helper_market_data`).
    화면 표가 같은 컬럼을 같은 기준으로 보여줘야 두 화면을 나란히 비교할 수 있다.
    지표를 못 구한 종목은 그 값만 None 이다 — 목록에서 빼지 않는다(비중은 정할 수 있다).
    """
    from config import METRIC_WINDOW_MONTHS
    from utils.asset_helper_market_data import (
        _build_current_price_map,
        _build_daily_change_map,
        _build_mdd_map,
        _build_return_map,
        _load_close_frame,
    )
    from utils.asset_helper_service import _compute_sortino_raw_frame
    from utils.settings_loader import get_ticker_type_settings

    universe = load_universe(pool)
    if not universe:
        return []
    country = str((get_ticker_type_settings(pool) or {}).get("country_code") or "").strip().lower()
    items = [{"ticker": row["ticker"], "ticker_type": pool, "country_code": country} for row in universe]

    close_frame, _ = _load_close_frame(items)
    price_by = _build_current_price_map(items, close_frame)
    change_by = _build_daily_change_map(items, close_frame)
    return_by = _build_return_map(close_frame)
    mdd_by = _build_mdd_map(close_frame, METRIC_WINDOW_MONTHS)
    sortino_frame = _compute_sortino_raw_frame(close_frame, METRIC_WINDOW_MONTHS)
    sortino_row = sortino_frame.iloc[-1] if not sortino_frame.empty else None

    def _sortino(ticker: str) -> float | None:
        if sortino_row is None or ticker not in sortino_row.index:
            return None
        value = sortino_row[ticker]
        return None if value is None or value != value else round(float(value), 2)

    # 종목 메모 — 계좌가 아니라 **종목**에 붙는다(순위·자산 관리 화면과 같은 값).
    from utils.stock_memo_store import get_stock_memos

    memo_by = get_stock_memos([row["ticker"] for row in universe])

    rows: list[dict[str, Any]] = []
    for row in universe:
        ticker = row["ticker"]
        returns = return_by.get(ticker) or {}
        rows.append(
            {
                **row,
                "memo": memo_by.get(ticker, ""),
                "current_price": price_by.get(ticker),
                "daily_change_pct": change_by.get(ticker),
                "return_1m_pct": returns.get("return_1m_pct"),
                "return_3m_pct": returns.get("return_3m_pct"),
                "return_6m_pct": returns.get("return_6m_pct"),
                "return_12m_pct": returns.get("return_12m_pct"),
                "mdd_pct": mdd_by.get(ticker),
                "sortino": _sortino(ticker),
            }
        )
    return rows


def benchmark_info(pool: str) -> dict[str, str]:
    """벤치마크 {ticker, name} — 종목풀 설정이 단일 소스(다른 전략과 같다)."""
    from utils.momentum_service import benchmark_info as pool_benchmark

    return pool_benchmark(pool)


# ── 설정 ───────────────────────────────────────────────────────────────────
def validate_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """설정을 검증해 정규화된 dict 를 반환한다. 실패 시 ValueError.

    비중은 **종목 합 + 현금 = 100%** 여야 한다(`/asset-helper` 와 같은 규칙). 현금은
    사용자가 직접 정하는 값이고 자동으로 흡수하지 않는다 — 합이 어긋나면 저장을 막고
    사용자가 어디를 조정할지 정한다.
    """
    if not isinstance(settings, dict):
        raise ValueError("설정 형식이 올바르지 않습니다.")

    pool = str(settings.get("pool") or "").strip().lower()
    if pool not in available_pools():
        raise ValueError(f"지원하지 않는 종목풀입니다: {settings.get('pool')}")

    rebalance = str(settings.get("rebalance") or "").strip().lower()
    if rebalance not in REBALANCE_OPTIONS:
        allowed = ", ".join(REBALANCE_LABELS[key] for key in REBALANCE_OPTIONS)
        raise ValueError(f"'rebalance' 는 {allowed} 중 하나여야 합니다 (받은 값: {settings.get('rebalance')}).")

    try:
        band_pct = round(float(settings.get("band_pct")), 2)
    except (TypeError, ValueError) as error:
        raise ValueError(f"'band_pct' 는 숫자여야 합니다: {settings.get('band_pct')}") from error
    if band_pct not in {round(option, 2) for option in REBALANCE_BAND_PCT_OPTIONS}:
        allowed = ", ".join(f"{option:g}" for option in REBALANCE_BAND_PCT_OPTIONS)
        raise ValueError(f"'band_pct' 는 {allowed} 중 하나여야 합니다 (받은 값: {band_pct}).")

    raw_weights = settings.get("weights")
    if not isinstance(raw_weights, (list, tuple)):
        raise ValueError("'weights' 는 목록이어야 합니다.")
    if len(raw_weights) > MAX_HOLDINGS:
        raise ValueError(f"종목은 최대 {MAX_HOLDINGS}개까지 담을 수 있습니다 (받은 개수: {len(raw_weights)}).")

    universe = {row["ticker"] for row in load_universe(pool)}
    weights: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in raw_weights:
        if not isinstance(item, dict):
            raise ValueError(f"종목 항목 형식이 올바르지 않습니다: {item}")
        ticker = str(item.get("ticker") or "").strip().upper()
        if not ticker:
            raise ValueError("티커가 비어 있는 항목이 있습니다.")
        if ticker in seen:
            raise ValueError(f"같은 종목이 두 번 들어 있습니다: {ticker}")
        if ticker not in universe:
            raise ValueError(f"'{pool}' 종목풀에 없는 종목입니다: {ticker}")
        seen.add(ticker)
        try:
            weight = round(float(item.get("weight_pct")), 2)
        except (TypeError, ValueError) as error:
            raise ValueError(f"'{ticker}' 의 비중은 숫자여야 합니다: {item.get('weight_pct')}") from error
        if not 0.0 <= weight <= 100.0:
            raise ValueError(f"'{ticker}' 의 비중은 0 ~ 100 이어야 합니다: {weight}")
        weights.append({"ticker": ticker, "weight_pct": weight})

    try:
        cash = round(float(settings.get("cash_weight_pct")), 2)
    except (TypeError, ValueError) as error:
        raise ValueError(f"현금 비중은 숫자여야 합니다: {settings.get('cash_weight_pct')}") from error
    if not 0.0 <= cash <= 100.0:
        raise ValueError(f"현금 비중은 0 ~ 100 이어야 합니다: {cash}")

    total = round(sum(row["weight_pct"] for row in weights) + cash, 2)
    if abs(total - 100.0) > 0.01:
        raise ValueError(f"종목 비중과 현금의 합이 100% 여야 합니다: {total}%")

    return {
        "pool": pool,
        "start_date": validate_start_date(settings.get("start_date")),
        "weights": weights,
        "cash_weight_pct": cash,
        "rebalance": rebalance,
        "band_pct": band_pct,
    }


def _load_doc() -> dict[str, Any]:
    return _db()[_CONFIG_COLLECTION].find_one({"_id": _SETTINGS_KEY}) or {}


def load_settings_map() -> dict[str, Any]:
    """풀별 저장 설정 — 화면이 풀 셀렉트를 바꿀 때 즉시 전환하는 데 쓴다."""
    return dict(_load_doc().get("settings_by_pool") or {})


def default_pool() -> str:
    """설정이 저장된 풀 중 목록에서 가장 앞선 풀 — 화면이 기억한 값이 없을 때의 기준점.

    "마지막으로 고른 풀"은 브라우저 취향이라 DB 에 두지 않는다(화면이 로컬스토리지에 기억).
    """
    saved = set(_load_doc().get("settings_by_pool") or {})
    pools = available_pools()
    for pool in pools:
        if pool in saved:
            return pool
    return pools[0] if pools else ""


def load_settings(pool: str | None = None) -> dict[str, Any]:
    """그 풀의 설정을 반환한다. 풀을 주지 않으면 `default_pool()` 을 쓴다.

    저장분이 없는 풀은 빈 비중(전액 현금)으로 시작한다 — 화면에서 종목을 담으면 된다.
    """
    doc = _load_doc()
    pools = available_pools()
    selected = str(pool or default_pool()).strip()
    if selected not in pools:
        raise ValueError(f"지원하지 않는 종목풀입니다: {pool}")
    stored = dict((doc.get("settings_by_pool") or {}).get(selected) or {})
    return validate_settings({"pool": selected, **DEFAULT_SETTINGS, **stored})


def load_settings_for_view(pool: str | None = None) -> tuple[dict[str, Any], list[str]]:
    """화면용 로드 — **종목풀에서 빠진 종목은 걷어내고** 그 내역을 함께 돌려준다.

    종목을 다른 풀로 옮기면 저장된 비중에 없는 티커가 남는다. `load_settings` 는 그걸
    에러로 막아서 화면이 열리지도 않고 고칠 수도 없었다. 화면에서는 빠진 종목을 빼고 열어
    사용자가 비중을 다시 맞출 수 있게 한다. 배치·백테스트는 그대로 `load_settings` 를 쓴다.

    반환: (설정, ["ASX:GGUS 종목풀에 없어 제외", ...])
    """
    doc = _load_doc()
    pools = available_pools()
    selected = str(pool or default_pool()).strip()
    if selected not in pools:
        raise ValueError(f"지원하지 않는 종목풀입니다: {pool}")
    stored = dict((doc.get("settings_by_pool") or {}).get(selected) or {})
    settings = {"pool": selected, **DEFAULT_SETTINGS, **stored}
    try:
        return validate_settings(settings), []
    except ValueError:
        pass

    universe = {row["ticker"] for row in load_universe(selected)}
    dropped: list[str] = []
    kept: list[dict[str, Any]] = []
    for item in settings.get("weights") or []:
        ticker = str((item or {}).get("ticker") or "").strip().upper()
        if ticker and ticker in universe:
            kept.append(item)
        elif ticker:
            dropped.append(f"{ticker} 종목풀에 없어 제외")
    return validate_settings({**settings, "weights": kept}), dropped


def save_settings(settings: dict[str, Any]) -> dict[str, Any]:
    """검증 후 그 풀의 설정으로 저장한다. 다른 풀의 저장분은 건드리지 않는다."""
    normalized = validate_settings(settings)
    require_start_date(normalized)
    pool = normalized["pool"]
    per_pool = {key: normalized[key] for key in PER_POOL_SETTING_KEYS}
    _db()[_CONFIG_COLLECTION].update_one(
        {"_id": _SETTINGS_KEY},
        {"$set": {f"settings_by_pool.{pool}": per_pool}},
        upsert=True,
    )
    # 설정을 바꿨다 되돌리면 옛 키에 그대로 걸린다 — 그 사이 달라진 종목 목록이
    # 반영되지 않은 결과가 다시 나오므로, 저장할 때마다 비우고 새로 계산하게 한다.
    from utils.cache_invalidation import invalidate_strategy_caches

    invalidate_strategy_caches()
    return normalized


__all__ = [
    "DEFAULT_BACKTEST_MONTHS",
    "DEFAULT_SETTINGS",
    "MAX_HOLDINGS",
    "PER_POOL_SETTING_KEYS",
    "REBALANCE_LABELS",
    "REBALANCE_OPTIONS",
    "available_pools",
    "benchmark_info",
    "default_pool",
    "load_settings",
    "load_settings_map",
    "load_universe",
    "universe_metrics",
    "pool_options",
    "save_settings",
    "validate_settings",
]
