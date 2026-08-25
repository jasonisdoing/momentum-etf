"""포트폴리오 설정과 목표 비중 계산 서비스."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from config import MIN_TRADING_DAYS, TRADING_DAYS_PER_MONTH
from utils.logger import get_app_logger
from utils.moving_averages import calculate_moving_average
from utils.share_allocation import ShareTarget, allocate_integer_shares

logger = get_app_logger()

# ── 분리 이동된 시장 데이터 층 re-import (내부 사용 + 하위 호환) ─────────────
from utils.asset_helper_market_data import (  # noqa: E402,F401
    IS_PRICE_PROXY,
    _build_cached_current_price_map,
    _build_cached_daily_change_map,
    _build_current_price_map,
    _build_daily_change_map,
    _build_mdd_map,
    _build_return_map,
    _cache_lookup_ticker,
    _convert_close_frame_to_krw,
    _extract_realtime_change_pct,
    _extract_realtime_price,
    _load_asset_helper_account_snapshot,
    _load_close_frame,
    _normalize_bucket_label,
    _normalize_kor_holding_ticker,
    _resolve_backtest_currency,
)

SETTINGS_ID = "default"

# 설정 스키마 — 코드 기본값(silent default) 없음. 값은 전적으로 DB에서 온다.
# 전략 필수 필드는 DB에 없으면 명시적 에러(fail loud). 사용자 선택 필드(계좌·누적수익률 기준)는
# 미설정 허용(빈값/None)이며, 그 값을 실제로 쓰는 지점에서 막는다.
SETTING_KEYS = (
    "VARIABLE_TICKERS",
    "FIXED_TICKERS",
    "MAX_TICKERS",
    "STOCK_MAX_WEIGHT",
    "ACCOUNT_ID",
)
# DB에 반드시 있어야 하는(없으면 에러) 전략 필수 필드. 코드 기본값으로 대체하지 않는다.
REQUIRED_SETTING_KEYS = (
    "VARIABLE_TICKERS",
    "FIXED_TICKERS",
    "STOCK_MAX_WEIGHT",
)
# 계좌별 편입 티커 슬롯 수의 허용 범위.
MAX_TICKERS_LIMIT = 20
DEFAULT_BACKTEST_SETTINGS: dict[str, Any] = {
    "months": 12,
    "rebalance": "none",
    "initial_amount_manwon": 10000,
}
ALLOWED_BACKTEST_REBALANCE = {"none", "weekly", "monthly", "quarterly", "yearly"}


def _db():
    from utils.db_manager import get_db_connection

    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결 실패 (자산 헬퍼 설정)")
    return db


def _load_unique_stock_meta_by_ticker(tickers: list[str]) -> dict[str, dict[str, Any]]:
    """stock_meta에 단일 활성 등록만 있는 티커의 메타를 반환한다."""
    if not tickers:
        return {}

    docs = list(
        _db().stock_meta.find(
            {
                "ticker": {"$in": tickers},
                "is_deleted": {"$ne": True},
            },
            {
                "_id": 0,
                "ticker": 1,
                "ticker_type": 1,
                "name": 1,
                "bucket": 1,
                "exclude_from_ranking": 1,
            },
        )
    )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for doc in docs:
        ticker = str(doc.get("ticker") or "").strip().upper()
        ticker_type = str(doc.get("ticker_type") or "").strip().lower()
        if ticker and ticker_type:
            grouped[ticker].append(doc)

    return {ticker: items[0] for ticker, items in grouped.items() if len(items) == 1}


def _clean_tickers(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []

    input_tickers = [
        str(item.get("ticker") or "").strip().upper()
        for item in items
        if isinstance(item, dict) and str(item.get("ticker") or "").strip()
    ]
    stock_meta_by_ticker = _load_unique_stock_meta_by_ticker(input_tickers)
    clean: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        ticker = str(item.get("ticker") or "").strip().upper()
        name = str(item.get("name") or "").strip()
        if not ticker or ticker in seen:
            continue
        # 이름 없는 행을 조용히 버리면 그 종목의 비중이 소리 없이 증발한다(silent drop 금지).
        # 이름 조회가 실패한 종목은 티커를 이름으로 대신 써서 비중을 보존한다.
        if not name:
            name = ticker
        if len(clean) >= MAX_TICKERS_LIMIT:
            raise ValueError(f"ETF는 최대 {MAX_TICKERS_LIMIT}개까지 등록할 수 있습니다.")
        seen.add(ticker)
        row: dict[str, Any] = {"ticker": ticker, "name": name}
        stock_meta = stock_meta_by_ticker.get(ticker) or {}
        ticker_type = str(stock_meta.get("ticker_type") or item.get("ticker_type") or "").strip().lower()
        country_code = str(item.get("country_code") or "").strip().lower()
        if ticker_type:
            row["ticker_type"] = ticker_type
        if country_code:
            row["country_code"] = country_code
        if isinstance(item.get("is_etf"), bool):
            row["is_etf"] = item["is_etf"]
        bucket = stock_meta.get("bucket") or item.get("bucket")
        if bucket is not None:
            row["bucket"] = int(bucket)
        if bool(stock_meta.get("exclude_from_ranking") or item.get("exclude_from_ranking")):
            row["exclude_from_ranking"] = True
        # 비중 고정 모드의 종목별 고정 비중(%). 미설정은 저장하지 않는다(임의 0 보정 금지).
        raw_weight = item.get("fixed_weight_pct")
        if raw_weight is not None and str(raw_weight).strip() != "":
            try:
                weight_val = round(float(raw_weight), 2)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"고정 비중은 숫자여야 합니다: {raw_weight}") from exc
            if not (0.0 <= weight_val <= 100.0):
                raise ValueError(f"고정 비중은 0 ~ 100 범위여야 합니다: {weight_val}")
            row["fixed_weight_pct"] = weight_val
        clean.append(row)
    return clean


def _clean_ticker_slots(items: Any) -> list[dict[str, Any]]:
    """빈 위치를 포함한 편입 ETF 슬롯을 검증·정규화한다."""
    if not isinstance(items, list):
        return []
    if len(items) > MAX_TICKERS_LIMIT:
        raise ValueError(f"ETF 슬롯은 최대 {MAX_TICKERS_LIMIT}개까지 저장할 수 있습니다.")

    clean_by_ticker = {item["ticker"]: item for item in _clean_tickers(items)}
    slots: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        ticker = str(item.get("ticker") or "").strip().upper() if isinstance(item, dict) else ""
        if not ticker or ticker in seen or ticker not in clean_by_ticker:
            slots.append({"ticker": ""})
            continue
        slots.append(dict(clean_by_ticker[ticker]))
        seen.add(ticker)
    return slots


def _clean_settings(values: dict[str, Any] | None, *, base: dict[str, Any] | None = None) -> dict[str, Any]:
    base_clean = {key: value for key, value in (base or {}).items() if value is not None}
    value_clean = {key: value for key, value in (values or {}).items() if value is not None}
    # 코드 기본값 없음 — 값은 base(DB)/values(요청)에서만 온다.
    source = {**base_clean, **value_clean}
    # 기존 문서에는 MAX_TICKERS 하나만 있다. 새 구조에서는 기존 슬롯을 모두 고정 종목으로 간주한다.
    if (
        source.get("VARIABLE_TICKERS") is None
        and source.get("FIXED_TICKERS") is None
        and source.get("MAX_TICKERS") is not None
    ):
        source["VARIABLE_TICKERS"] = 0
        source["FIXED_TICKERS"] = source.get("MAX_TICKERS")
    if source.get("STOCK_MAX_WEIGHT") is None and source.get("MAX_TICKERS") is not None:
        migrated_max_tickers = int(source.get("MAX_TICKERS"))
        if migrated_max_tickers <= 0:
            raise ValueError(f"MAX_TICKERS 은 1 이상이어야 합니다: {migrated_max_tickers}")
        # 기존 문서는 종목당 기본 슬롯(1/N)이 상한이었다. 신규 필드가 없을 때만 같은 의미로 명시 마이그레이션한다.
        source["STOCK_MAX_WEIGHT"] = max(5.0, round(100.0 / migrated_max_tickers, 1))
    # 전략 필수 필드가 하나라도 없으면 코드 기본값으로 대체하지 않고 명시적 에러(fail loud).
    missing_required = [key for key in REQUIRED_SETTING_KEYS if source.get(key) is None]
    if missing_required:
        raise ValueError(f"설정값이 없습니다(DB 미설정): {', '.join(missing_required)}. 설정 화면에서 저장해주세요.")
    cleaned: dict[str, Any] = {}

    # 계좌별 편입 티커 슬롯 수. 변동/고정은 관리용 구분이며 계산에는 합산 슬롯을 사용한다.
    try:
        variable_tickers = int(source.get("VARIABLE_TICKERS"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"VARIABLE_TICKERS 은 정수여야 합니다: {source.get('VARIABLE_TICKERS')}") from exc
    try:
        fixed_tickers = int(source.get("FIXED_TICKERS"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"FIXED_TICKERS 은 정수여야 합니다: {source.get('FIXED_TICKERS')}") from exc
    if not (0 <= variable_tickers <= MAX_TICKERS_LIMIT):
        raise ValueError(f"VARIABLE_TICKERS 은 0 ~ {MAX_TICKERS_LIMIT} 범위여야 합니다: {variable_tickers}")
    if not (0 <= fixed_tickers <= MAX_TICKERS_LIMIT):
        raise ValueError(f"FIXED_TICKERS 은 0 ~ {MAX_TICKERS_LIMIT} 범위여야 합니다: {fixed_tickers}")
    max_tickers = variable_tickers + fixed_tickers
    if not (1 <= max_tickers <= MAX_TICKERS_LIMIT):
        raise ValueError(f"편입 종목 수 합계는 1 ~ {MAX_TICKERS_LIMIT} 범위여야 합니다: {max_tickers}")
    cleaned["VARIABLE_TICKERS"] = variable_tickers
    cleaned["FIXED_TICKERS"] = fixed_tickers
    cleaned["MAX_TICKERS"] = max_tickers

    try:
        stock_max_weight = float(source.get("STOCK_MAX_WEIGHT"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"종목 최대 비중(%)은 숫자여야 합니다: {source.get('STOCK_MAX_WEIGHT')}") from exc
    if not (5.0 <= stock_max_weight <= 100.0):
        raise ValueError(f"종목 최대 비중(%)은 5 ~ 100 범위여야 합니다: {stock_max_weight}")
    cleaned["STOCK_MAX_WEIGHT"] = round(stock_max_weight, 1)

    account_id = str(source.get("ACCOUNT_ID") or "").strip()
    if account_id:
        from utils.settings_loader import list_available_accounts

        if account_id not in set(list_available_accounts()):
            raise ValueError(f"존재하지 않는 적용 계좌입니다: {account_id}")
    cleaned["ACCOUNT_ID"] = account_id

    return cleaned


def _filter_rank_excluded_tickers(
    tickers: list[dict[str, Any]], settings: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[str]]:
    """순위 고정 종목(exclude_from_ranking=true)을 계산/백테스트 유니버스에서 제외한다."""
    if not tickers:
        return [], []

    default_ticker_type = str(settings.get("POOL_TICKER_TYPE") or "").strip().lower()
    query_pairs: list[dict[str, str]] = []
    pair_by_ticker: dict[str, str] = {}
    for item in tickers:
        ticker = str(item.get("ticker") or "").strip().upper()
        ticker_type = str(item.get("ticker_type") or default_ticker_type).strip().lower()
        if ticker and ticker_type:
            query_pairs.append({"ticker_type": ticker_type, "ticker": ticker})
            pair_by_ticker[ticker] = ticker_type

    excluded_from_db: set[str] = set()
    if query_pairs:
        for doc in _db().stock_meta.find(
            {"$or": query_pairs, "is_deleted": {"$ne": True}},
            {"_id": 0, "ticker": 1, "ticker_type": 1, "exclude_from_ranking": 1},
        ):
            ticker = str(doc.get("ticker") or "").strip().upper()
            ticker_type = str(doc.get("ticker_type") or "").strip().lower()
            if ticker and ticker_type == pair_by_ticker.get(ticker) and bool(doc.get("exclude_from_ranking")):
                excluded_from_db.add(ticker)

    filtered: list[dict[str, Any]] = []
    excluded: list[str] = []
    for item in tickers:
        ticker = str(item.get("ticker") or "").strip().upper()
        if bool(item.get("exclude_from_ranking")) or ticker in excluded_from_db:
            excluded.append(ticker)
            continue
        filtered.append(item)
    return filtered, excluded


def _build_current_ma_state_maps(
    close_frame: pd.DataFrame,
    eval_date: pd.Timestamp,
    *,
    short_ma_days: int,
    long_ma_days: int,
) -> tuple[dict[str, float | None], dict[str, str | None]]:
    short_ma_cols: dict[str, pd.Series] = {}
    long_ma_cols: dict[str, pd.Series] = {}
    for ticker in close_frame.columns:
        series = close_frame[ticker].dropna()
        if series.empty:
            short_ma_cols[ticker] = pd.Series(float("nan"), index=close_frame.index, dtype="float64")
            long_ma_cols[ticker] = pd.Series(float("nan"), index=close_frame.index, dtype="float64")
            continue
        short_ma_cols[ticker] = calculate_moving_average(series, short_ma_days).reindex(close_frame.index)
        long_ma_cols[ticker] = calculate_moving_average(series, long_ma_days).reindex(close_frame.index)

    short_ma_frame = pd.DataFrame(short_ma_cols, index=close_frame.index)
    long_ma_frame = pd.DataFrame(long_ma_cols, index=close_frame.index)
    if eval_date not in long_ma_frame.index:
        return ({ticker: None for ticker in close_frame.columns}, {ticker: None for ticker in close_frame.columns})

    short_ma_row = short_ma_frame.loc[eval_date]
    long_ma_row = long_ma_frame.loc[eval_date]
    eligible_long_ma_frame = long_ma_frame.loc[long_ma_frame.index <= eval_date]
    previous_long_ma_row = (
        eligible_long_ma_frame.iloc[-2] if len(eligible_long_ma_frame.index) >= 2 else pd.Series(dtype="float64")
    )

    slope_map: dict[str, float | None] = {}
    alignment_map: dict[str, str | None] = {}
    for ticker in close_frame.columns:
        short_value = short_ma_row.get(ticker)
        long_value = long_ma_row.get(ticker)
        if pd.isna(short_value) or pd.isna(long_value):
            alignment_map[ticker] = None
        else:
            alignment_map[ticker] = "정배열" if float(short_value) >= float(long_value) else "역배열"

        previous_long_value = previous_long_ma_row.get(ticker)
        if pd.isna(long_value) or pd.isna(previous_long_value) or float(previous_long_value) == 0.0:
            slope_map[ticker] = None
        else:
            slope_map[ticker] = ((float(long_value) / float(previous_long_value)) - 1.0) * 100.0
    return slope_map, alignment_map


def _compute_ma_deviation_frame(close_frame: pd.DataFrame, ma_days: int) -> pd.DataFrame:
    """종목별 (종가/이평선-1)*100 이격률을 전체 구간에 대해 벡터화로 계산한다.

    trend 계좌(Phase 2)의 장기/단기 이격 게이트에 쓴다 — 백테스트는 리밸런싱 날짜마다
    이 프레임에서 그날의 값만 조회해 재사용한다(매번 새로 계산하지 않는다).
    """
    ma_cols: dict[str, pd.Series] = {}
    for ticker in close_frame.columns:
        series = close_frame[ticker].dropna()
        ma_cols[ticker] = (
            calculate_moving_average(series, ma_days).reindex(close_frame.index)
            if not series.empty
            else pd.Series(float("nan"), index=close_frame.index, dtype="float64")
        )
    ma_frame = pd.DataFrame(ma_cols, index=close_frame.index)
    return (close_frame / ma_frame - 1.0) * 100.0


def _clean_backtest_settings(values: Any, *, base: Any = None, require_benchmark: bool = True) -> dict[str, Any]:
    base_clean = base if isinstance(base, dict) else {}
    value_clean = values if isinstance(values, dict) else {}
    source = {**DEFAULT_BACKTEST_SETTINGS, **base_clean, **value_clean}

    benchmark_source = source.get("benchmark") if isinstance(source.get("benchmark"), dict) else {}
    ticker = str(benchmark_source.get("ticker") or "").strip().upper()
    if not ticker:
        if require_benchmark:
            raise ValueError("시장 레짐 지수가 필요합니다. 설정 화면에서 지수를 선택해주세요.")
        name = ""
    else:
        from utils.market_trend_service import INDICES

        index_meta = next((idx for idx in INDICES if idx["yf_ticker"] == ticker), None)
        if index_meta is None:
            allowed = ", ".join(idx["yf_ticker"] for idx in INDICES)
            raise ValueError(f"시장 레짐은 시장추세 지수({allowed}) 중 하나여야 합니다: {ticker}")
        name = index_meta["name"]

    try:
        months = int(source.get("months"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"백테스트 기간(개월)은 정수여야 합니다: {source.get('months')}") from exc
    # 기간 옵션은 pools-backtest 와 동일하게 가격 캐시 데이터 길이 기준으로 산정한다(재사용, 하드코딩 제거).
    from utils.pool_signal_backtest_service import get_month_options

    allowed_months = get_month_options()
    if months not in allowed_months:
        raise ValueError(f"백테스트 기간(개월)은 {', '.join(map(str, allowed_months))} 중 하나여야 합니다.")

    rebalance = str(source.get("rebalance") or "none").strip().lower()
    if rebalance not in ALLOWED_BACKTEST_REBALANCE:
        raise ValueError(f"백테스트 리밸런싱은 {', '.join(sorted(ALLOWED_BACKTEST_REBALANCE))} 중 하나여야 합니다.")

    raw_initial_amount = source.get("initial_amount_manwon")
    if isinstance(raw_initial_amount, bool):
        raise ValueError(f"백테스트 최초 금액(만원)은 정수여야 합니다: {raw_initial_amount}")
    if isinstance(raw_initial_amount, str) and not raw_initial_amount.strip().isdigit():
        raise ValueError(f"백테스트 최초 금액(만원)은 정수여야 합니다: {raw_initial_amount}")
    if isinstance(raw_initial_amount, float) and not raw_initial_amount.is_integer():
        raise ValueError(f"백테스트 최초 금액(만원)은 정수여야 합니다: {raw_initial_amount}")
    try:
        initial_amount_manwon = int(raw_initial_amount)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"백테스트 최초 금액(만원)은 정수여야 합니다: {raw_initial_amount}") from exc
    if not (1 <= initial_amount_manwon <= 1_000_000_000):
        raise ValueError(f"백테스트 최초 금액(만원)은 1 ~ 1000000000 범위여야 합니다: {initial_amount_manwon}")

    return {
        "benchmark": {"ticker": ticker, "name": name},
        "months": months,
        "rebalance": rebalance,
        "initial_amount_manwon": initial_amount_manwon,
    }


def list_asset_helper_accounts() -> list[str]:
    """자산 헬퍼를 쓸 수 있는 계좌 id 목록 (계좌 설정이 단일 소스)."""
    from utils.account_registry import load_account_configs

    return [str(config["account_id"]) for config in load_account_configs()]


def _resolve_account_id(account_id: str | None) -> str:
    """account_id 미지정 시: 계좌가 정확히 1개면 그걸 쓰고, 0개/2개 이상이면 명시적 에러(임의 선택 금지)."""
    if account_id and str(account_id).strip():
        return str(account_id).strip()
    accounts = list_asset_helper_accounts()
    if len(accounts) == 1:
        return accounts[0]
    if not accounts:
        raise ValueError("등록된 계좌가 없습니다.")
    raise ValueError(f"계좌를 지정해야 합니다(여러 계좌 존재): {', '.join(accounts)}")


def load_asset_helper_settings_for_edit(account_id: str) -> dict[str, Any]:
    """설정 편집 화면 전용 로더.

    문서가 있으면 검증된 설정을, 없으면 **초기(빈) 상태**를 반환한다.
    저장 전까지 DB 에 아무것도 쓰지 않으며, 저장 시 save 가 필수값을 검증한다(fail loud).
    """
    from utils.portfolio_io import load_portfolio_master

    resolved = str(account_id or "").strip()
    if not resolved:
        raise ValueError("계좌를 지정해야 합니다.")

    # 단일 컬렉션 원칙 — 종목·비중은 보유 목록(portfolio_master.holdings)의
    # target_ratio 필드, 계좌 단위 설정은 계좌 객체의 asset_helper 필드에서 읽는다.
    master = load_portfolio_master(resolved) or {}
    holdings = master.get("holdings") or []
    helper_doc = master.get("asset_helper") or {}

    # name 을 함께 실어 GET→PUT 왕복이 대칭이 되게 한다 — name 없이 내보내면
    # 이 응답을 그대로 저장에 되돌렸을 때 행이 걸러져 비중이 통째로 지워졌다.
    weight_tickers = [
        {
            "ticker": str(h.get("ticker") or "").strip().upper(),
            "name": str(h.get("name") or "").strip() or str(h.get("ticker") or "").strip().upper(),
            "fixed_weight_pct": float(h["target_ratio"]),
        }
        for h in holdings
        if h.get("target_ratio") is not None
    ]

    settings_base = {
        key: helper_doc[key] for key in (*SETTING_KEYS, "CASH_MAX_WEIGHT") if helper_doc.get(key) is not None
    }
    # 종목수 관련 값은 저장하지 않고 실제 보유 종목 수에서 파생한다.
    settings_base.setdefault("VARIABLE_TICKERS", 0)
    settings_base.setdefault("FIXED_TICKERS", len(weight_tickers))
    settings_base.setdefault("MAX_TICKERS", len(weight_tickers) or None)
    settings_base["ACCOUNT_ID"] = resolved
    settings = (
        _clean_settings({}, base={k: v for k, v in settings_base.items() if v is not None})
        if helper_doc
        else {key: (resolved if key == "ACCOUNT_ID" else None) for key in SETTING_KEYS}
    )
    backtest_settings = (
        _clean_backtest_settings(helper_doc.get("backtest_settings"), require_benchmark=False) if helper_doc else None
    )
    updated_at = helper_doc.get("updated_at")

    return {
        "tickers": weight_tickers,
        "cash_weight_pct": helper_doc.get("cash_weight_pct"),
        "settings": settings,
        "backtest_settings": backtest_settings,
        "updated_at": (
            (updated_at.replace(tzinfo=timezone.utc) if updated_at.tzinfo is None else updated_at).isoformat()
            if isinstance(updated_at, datetime)
            else None
        ),
    }


def save_asset_helper_settings(
    tickers: list[dict[str, Any]],
    settings: dict[str, Any] | None = None,
    backtest_settings: dict[str, Any] | None = None,
    account_id: str | None = None,
    cash_weight_pct: float | None = None,
) -> dict[str, Any]:
    from utils.portfolio_io import load_portfolio_master, update_account_asset_helper

    resolved = _resolve_account_id(account_id)
    clean_tickers = _clean_tickers(_clean_ticker_slots(tickers))
    if len(clean_tickers) < 1:
        raise ValueError("저장할 종목이 1개 이상 필요합니다.")

    # 단일 컬렉션 원칙 — 기존 설정은 portfolio_master 계좌 객체의 asset_helper 에서 읽는다.
    existing_helper = (load_portfolio_master(resolved) or {}).get("asset_helper") or {}
    stored_keys = (*SETTING_KEYS, "CASH_MAX_WEIGHT")
    settings_base = {key: existing_helper[key] for key in stored_keys if existing_helper.get(key) is not None}
    # 아래에서 VARIABLE/FIXED/MAX_TICKERS 를 실제 종목수로 덮어쓰므로, 신규 계좌라도 검증을 통과하게 미리 채운다.
    settings_base.setdefault("VARIABLE_TICKERS", 0)
    settings_base.setdefault("FIXED_TICKERS", len(clean_tickers))
    # STOCK_MAX_WEIGHT 는 변동(variable) 모드에서만 쓰이는 종목 상한(%). 고정 모드는 종목별 비중을 직접 정하므로
    # 안 쓴다 — 신규 계좌에서 미설정이면 100(=상한 없음)으로 둔다(있으면 기존 값 유지).
    settings_base.setdefault("STOCK_MAX_WEIGHT", 100.0)
    settings_base["ACCOUNT_ID"] = resolved
    clean_settings = _clean_settings(settings, base=settings_base)
    # 자산 헬퍼는 자유 테스트 포트폴리오라 계좌별 종목수 상한을 두지 않는다 —
    # 슬롯 수를 실제 종목 수에 맞춰 저장한다(전역 하드 상한 MAX_TICKERS_LIMIT 만 유지).
    clean_settings["VARIABLE_TICKERS"] = 0
    clean_settings["FIXED_TICKERS"] = len(clean_tickers)
    clean_settings["MAX_TICKERS"] = len(clean_tickers)
    clean_backtest_settings = _clean_backtest_settings(
        backtest_settings,
        base=existing_helper.get("backtest_settings") or DEFAULT_BACKTEST_SETTINGS,
        require_benchmark=False,
    )
    # 현금 목표 비중 — 저장값이 원본(로드 시 나머지로 자동 초기화하지 않기 위함).
    clean_cash_weight = existing_helper.get("cash_weight_pct")
    if cash_weight_pct is not None:
        clean_cash_weight = round(float(cash_weight_pct), 2)
        if not (0.0 <= clean_cash_weight <= 100.0):
            raise ValueError(f"현금 비중은 0 ~ 100 범위여야 합니다: {clean_cash_weight}")
    updated_at = datetime.now(timezone.utc)

    # 종목별 목표비중 — 보유 항목의 target_ratio 필드로 저장 (비중 미입력 종목은 미설정 유지).
    target_ratio_by_ticker = {
        str(item["ticker"]): float(item["fixed_weight_pct"])
        for item in clean_tickers
        if item.get("fixed_weight_pct") is not None
    }
    update_account_asset_helper(
        resolved,
        target_ratio_by_ticker=target_ratio_by_ticker,
        helper_settings={
            **clean_settings,
            "backtest_settings": clean_backtest_settings,
            "cash_weight_pct": clean_cash_weight,
            "updated_at": updated_at,
        },
    )
    return {
        "tickers": clean_tickers,
        "settings": clean_settings,
        "backtest_settings": clean_backtest_settings,
        "updated_at": updated_at.isoformat(),
    }


# IS(International Shares, 수동 고정자산)의 가격 프록시 — VGS 와 같이 움직인다고 가정한다.
def _apply_trade_plan(
    rows: list[dict[str, Any]],
    *,
    settings: dict[str, Any],
    tickers: list[dict[str, Any]],
    close_frame: pd.DataFrame,
) -> dict[str, Any]:
    account_snapshot = _load_asset_helper_account_snapshot(str(settings.get("ACCOUNT_ID") or "").strip())
    cash_row = next((row for row in rows if row.get("ticker") == "__CASH__"), None)
    if cash_row is None:
        cash_row = {
            "ticker": "__CASH__",
            "name": "현금",
            "ticker_type": "cash",
            "country_code": None,
            "target_weight_pct": 0.0,
        }
    else:
        rows.remove(cash_row)
    rows.insert(0, cash_row)

    account_amount_krw = float(account_snapshot["account_amount_krw"])
    holdings = account_snapshot["holdings"]
    account_currency = str(account_snapshot.get("currency") or "").strip().upper()
    native_mode = account_currency != "KRW"
    market_by_currency = {
        "KRW": ("kor_kr", "kor"),
        "AUD": ("aus", "au"),
        "USD": ("us", "us"),
    }
    if account_currency not in market_by_currency:
        raise ValueError(f"지원하지 않는 계좌 환종입니다: {account_currency or '-'}")
    holding_ticker_type, holding_country_code = market_by_currency[account_currency]
    current_price_map = _build_current_price_map(tickers, close_frame)
    target_asset_amount = 0.0
    target_tickers: set[str] = set()

    # 목표 주수는 종목마다 따로 내림하지 않고 **한 번에 배분**한다 (합성 전략·백테스트와 같은 함수).
    # 따로 내림하면 1주 값에 걸려 남은 몫이 아무도 쓰지 않는 현금으로 놀아, 목표 현금비중이
    # 0 이어도 채워지지 않는다. 예산은 목표 금액의 합이다.
    allocation_targets: list[ShareTarget] = []
    for row in rows:
        ticker = str(row.get("ticker") or "").strip().upper()
        price = current_price_map.get(ticker)
        weight = row.get("target_weight_pct")
        if ticker == "__CASH__" or not price or float(price) <= 0 or weight is None or account_amount_krw <= 0:
            continue
        allocation_targets.append(
            ShareTarget(key=ticker, target_amount=account_amount_krw * float(weight) / 100.0, price=float(price))
        )
    target_quantity_by_ticker = allocate_integer_shares(
        allocation_targets, budget=sum(item.target_amount for item in allocation_targets)
    )

    for row in rows:
        ticker = str(row.get("ticker") or "").strip().upper()
        if ticker and ticker != "__CASH__":
            target_tickers.add(_normalize_kor_holding_ticker(ticker))
        target_weight = row.get("target_weight_pct")
        target_amount = None
        if account_amount_krw > 0 and target_weight is not None:
            raw_target_amount = account_amount_krw * (float(target_weight) / 100.0)
            target_amount = round(raw_target_amount, 2) if native_mode else int(round(raw_target_amount))

        row["current_price"] = None
        row["target_amount_krw"] = target_amount
        row["target_value_krw"] = None
        row["target_quantity"] = None
        row["current_quantity"] = None
        row["current_amount_krw"] = None
        row["change_quantity"] = None
        row["change_amount_krw"] = None
        row["unallocated_amount_krw"] = target_amount
        row["return_pct"] = None
        row["pnl_krw"] = None
        row["bucket"] = None
        if ticker == "__CASH__":
            row["current_amount_krw"] = account_snapshot["cash_balance_krw"]
            row["bucket"] = "5. 현금"
            continue

        current_price = current_price_map.get(ticker)
        row["current_price"] = None if current_price is None else round(float(current_price), 2)
        holding = holdings.get(_normalize_kor_holding_ticker(ticker), {})
        current_quantity = int(holding.get("current_quantity") or 0)
        current_amount = float(holding.get("current_amount_krw") or 0)
        row["current_quantity"] = current_quantity
        row["current_amount_krw"] = current_amount
        row["return_pct"] = holding.get("return_pct")
        row["pnl_krw"] = holding.get("pnl_krw")
        bucket_val = holding.get("bucket")
        if not bucket_val:
            try:
                from utils.db_manager import get_db_connection

                db = get_db_connection()
                if db is not None:
                    meta_doc = db.stock_meta.find_one({"ticker_type": row.get("ticker_type"), "ticker": ticker})
                    if meta_doc:
                        b_id = meta_doc.get("bucket")
                        if b_id is not None:
                            from config import ALL_BUCKET_MAPPING

                            bucket_name = ALL_BUCKET_MAPPING.get(int(b_id))
                            if bucket_name:
                                bucket_val = f"{b_id}. {bucket_name}"
            except Exception:
                pass
        row["bucket"] = _normalize_bucket_label(bucket_val)
        if target_amount is None or current_price is None or current_price <= 0:
            continue

        quantity = target_quantity_by_ticker.get(ticker, 0)
        raw_target_buy_amount = quantity * current_price
        target_buy_amount = round(raw_target_buy_amount, 2) if native_mode else int(round(raw_target_buy_amount))
        row["target_quantity"] = quantity
        row["target_value_krw"] = target_buy_amount
        row["change_quantity"] = quantity - current_quantity
        row["change_amount_krw"] = (
            round(target_buy_amount - current_amount, 2)
            if native_mode
            else int(round(target_buy_amount - current_amount))
        )
        unallocated_amount = max(0.0, target_amount - target_buy_amount)
        row["unallocated_amount_krw"] = round(unallocated_amount, 2) if native_mode else int(unallocated_amount)
        target_asset_amount += target_buy_amount

    extra_holdings = [
        (ticker, holding)
        for ticker, holding in sorted(holdings.items())
        if ticker not in target_tickers
        and (int(holding.get("current_quantity") or 0) > 0 or float(holding.get("current_amount_krw") or 0) > 0)
    ]
    extra_ticker_items = [
        {
            "ticker": ticker,
            "ticker_type": holding_ticker_type,
            "country_code": holding_country_code,
        }
        for ticker, _holding in extra_holdings
    ]
    extra_close_frame, _extra_missing = _load_close_frame(extra_ticker_items)
    extra_return_map = _build_return_map(extra_close_frame)

    for ticker, holding in extra_holdings:
        current_quantity = int(holding.get("current_quantity") or 0)
        current_amount = float(holding.get("current_amount_krw") or 0)
        period_returns = extra_return_map.get(ticker, {})
        rows.append(
            {
                "ticker": ticker,
                "name": holding.get("name") or ticker,
                "ticker_type": holding_ticker_type,
                "country_code": holding_country_code,
                "return_1m_pct": period_returns.get("return_1m_pct"),
                "return_3m_pct": period_returns.get("return_3m_pct"),
                "return_6m_pct": period_returns.get("return_6m_pct"),
                "return_12m_pct": period_returns.get("return_12m_pct"),
                "daily_change_pct": holding.get("daily_change_pct"),
                "trend_pct": None,
                "trend_score": None,
                "sortino": None,
                "score": None,
                "target_weight_pct": 0.0,
                "rebalance_needed": None,
                "current_price": round(current_amount / current_quantity, 2) if current_quantity > 0 else None,
                "target_amount_krw": 0,
                "target_value_krw": 0,
                "target_quantity": 0,
                "current_quantity": current_quantity,
                "current_amount_krw": current_amount,
                "change_quantity": -current_quantity,
                "change_amount_krw": -current_amount,
                "unallocated_amount_krw": 0,
                "return_pct": holding.get("return_pct"),
                "pnl_krw": holding.get("pnl_krw"),
                "bucket": _normalize_bucket_label(holding.get("bucket")),
            }
        )

    remaining_cash = max(0.0, account_amount_krw - target_asset_amount) if account_amount_krw > 0 else 0.0
    remaining_cash = round(remaining_cash, 2) if native_mode else int(round(remaining_cash))
    for row in rows:
        if row.get("ticker") == "__CASH__":
            row["target_amount_krw"] = remaining_cash if account_amount_krw > 0 else row.get("target_amount_krw")
            row["target_value_krw"] = remaining_cash if account_amount_krw > 0 else row.get("target_value_krw")
            current_cash = float(row.get("current_amount_krw") or 0.0)
            cash_change = remaining_cash - current_cash
            row["change_amount_krw"] = round(cash_change, 2) if native_mode else int(round(cash_change))
            row["unallocated_amount_krw"] = (
                remaining_cash if account_amount_krw > 0 else row.get("unallocated_amount_krw")
            )

    # 변동 비중 (%) 계산
    for row in rows:
        target_weight = float(row.get("target_weight_pct") or 0.0)
        current_amount = float(row.get("current_amount_krw") or 0.0)
        current_weight = (current_amount / account_amount_krw) * 100.0 if account_amount_krw > 0 else 0.0
        row["current_weight_pct"] = round(current_weight, 2)
        row["change_weight_pct"] = round(target_weight - current_weight, 2)

    return {
        "account_id": account_snapshot["account_id"],
        "account_name": account_snapshot["account_name"],
        "currency": account_snapshot.get("currency") or "KRW",
        "account_amount_krw": account_amount_krw,
        "target_asset_amount_krw": target_asset_amount,
        "remaining_cash_krw": remaining_cash,
    }


def _normalize_target_weight_pct_rows(rows: list[dict[str, Any]]) -> None:
    weight_rows = [row for row in rows if row.get("target_weight_pct") is not None]
    if not weight_rows:
        return

    floors: list[int] = []
    remainders: list[float] = []
    raw_values: list[float] = []
    for row in weight_rows:
        raw_value = max(0.0, float(row.get("target_weight_pct") or 0.0))
        raw_tenths = raw_value * 10.0
        floor_tenths = int(np.floor(raw_tenths))
        floors.append(floor_tenths)
        remainders.append(raw_tenths - floor_tenths)
        raw_values.append(raw_value)

    tenths = floors[:]
    diff = 1000 - sum(tenths)
    if diff > 0:
        order = sorted(range(len(weight_rows)), key=lambda idx: (remainders[idx], raw_values[idx]), reverse=True)
        for idx in range(diff):
            tenths[order[idx % len(order)]] += 1
    elif diff < 0:
        order = sorted(range(len(weight_rows)), key=lambda idx: (remainders[idx], -raw_values[idx]))
        for _ in range(abs(diff)):
            for idx in order:
                if tenths[idx] > 0:
                    tenths[idx] -= 1
                    break

    for row, value in zip(weight_rows, tenths):
        row["target_weight_pct"] = round(value / 10.0, 1)


def _normalize_weight_ratio_map(weights: dict[str, float]) -> dict[str, float]:
    if not weights:
        return weights

    keys = list(weights.keys())
    floors: list[int] = []
    remainders: list[float] = []
    raw_values: list[float] = []
    for key in keys:
        raw_value = max(0.0, float(weights.get(key) or 0.0))
        raw_tenths = raw_value * 1000.0
        floor_tenths = int(np.floor(raw_tenths))
        floors.append(floor_tenths)
        remainders.append(raw_tenths - floor_tenths)
        raw_values.append(raw_value)

    tenths = floors[:]
    diff = 1000 - sum(tenths)
    if diff > 0:
        order = sorted(range(len(keys)), key=lambda idx: (remainders[idx], raw_values[idx]), reverse=True)
        for idx in range(diff):
            tenths[order[idx % len(order)]] += 1
    elif diff < 0:
        order = sorted(range(len(keys)), key=lambda idx: (remainders[idx], -raw_values[idx]))
        for _ in range(abs(diff)):
            for idx in order:
                if tenths[idx] > 0:
                    tenths[idx] -= 1
                    break

    return {key: round(value / 1000.0, 4) for key, value in zip(keys, tenths)}


def _compute_sortino_raw_frame(close_frame: pd.DataFrame, window_months: int) -> pd.DataFrame:
    if close_frame.empty:
        return pd.DataFrame(index=close_frame.index, columns=close_frame.columns, dtype=float)

    window = max(2, int(window_months) * int(TRADING_DAYS_PER_MONTH))
    # 상장 이력이 window(=지표 기간)보다 짧아도 "보유 기간만큼"(가용 데이터)으로 계산한다.
    # 최소 표본은 종목 데이터 충분 여부의 표준 기준(MIN_TRADING_DAYS)을 재사용한다.
    min_obs = min(window, max(2, int(MIN_TRADING_DAYS)))
    daily_ret = close_frame.pct_change(fill_method=None)
    mean = daily_ret.rolling(window=window, min_periods=min_obs).mean()

    def _calc_downside_std(values: np.ndarray) -> float:
        # 윈도우가 종목 첫 행까지 닿으면 pct_change 첫 NaN이 섞여 들어온다 — 제거 후 계산한다.
        valid = values[~np.isnan(values)]
        if valid.size <= 1:
            return np.nan
        downside = np.minimum(0.0, valid)
        result = np.sqrt(np.sum(downside**2) / (valid.size - 1))
        return float(result) if result > 0 else np.nan

    downside_std = daily_ret.rolling(window=window, min_periods=min_obs).apply(_calc_downside_std, raw=True)
    return (mean / downside_std.replace(0, np.nan)) * np.sqrt(252.0)


def _allocate_deviation_filtered_weights(
    ticker_deviation_pct: dict[str, float | None],
    stock_max_weight_pct: float,
) -> dict[str, float]:
    """개별 이평선 위 종목만 투자하고, 남는 비중은 최대 비중까지 재분배한다."""
    if not ticker_deviation_pct:
        raise ValueError("비중 계산 대상 종목이 없습니다.")
    stock_max_weight = float(stock_max_weight_pct) / 100.0
    if not (0.05 <= stock_max_weight <= 1.0):
        raise ValueError(f"종목 최대 비중(%)은 5 ~ 100 범위여야 합니다: {stock_max_weight_pct}")

    tickers = list(ticker_deviation_pct.keys())
    slot_weight = 1.0 / len(tickers)
    weights = {ticker: 0.0 for ticker in tickers}
    active_tickers = [
        ticker
        for ticker, deviation_pct in ticker_deviation_pct.items()
        if deviation_pct is not None and deviation_pct > 0
    ]
    if not active_tickers:
        return _normalize_weight_ratio_map({"__CASH__": 1.0, **weights})

    for ticker in active_tickers:
        weights[ticker] = min(slot_weight, stock_max_weight)

    cash_weight = max(0.0, 1.0 - sum(weights.values()))
    remaining_capacity = {ticker: max(0.0, stock_max_weight - weights[ticker]) for ticker in active_tickers}

    while cash_weight > 1e-12:
        candidates = [ticker for ticker, capacity in remaining_capacity.items() if capacity > 1e-12]
        if not candidates:
            break
        per_ticker = cash_weight / len(candidates)
        distributed = 0.0
        for ticker in candidates:
            add_weight = min(per_ticker, remaining_capacity[ticker])
            weights[ticker] += add_weight
            remaining_capacity[ticker] -= add_weight
            distributed += add_weight
        if distributed <= 1e-12:
            break
        cash_weight -= distributed

    weights["__CASH__"] = max(0.0, 1.0 - sum(weights.values()))
    return _normalize_weight_ratio_map(weights)


def calculate_asset_helper_weights_for(
    tickers: list[dict[str, Any]],
    settings: dict[str, Any],
    *,
    metric_months: int,
) -> dict[str, Any]:
    # 비중은 사용자가 지정한 고정 비중만 쓴다 — 이평선·계좌-풀 연결이 필요 없다.
    account_id = str(settings.get("ACCOUNT_ID") or "").strip()
    if not account_id:
        raise ValueError("비중 계산에는 적용 계좌가 필요합니다.")
    if len(tickers) < 3:
        raise ValueError("비중 계산에는 확인된 종목이 3개 이상 필요합니다.")

    close_frame, missing = _load_close_frame(tickers)
    if close_frame.empty:
        raise ValueError("종목의 가격 캐시가 없습니다.")

    mdd_map = _build_mdd_map(close_frame, metric_months)
    eval_date = close_frame.index.max()
    sortino_raw_frame = _compute_sortino_raw_frame(close_frame, metric_months)
    sortino_raw_row = (
        sortino_raw_frame.loc[eval_date] if eval_date in sortino_raw_frame.index else pd.Series(dtype=float)
    )
    return_map = _build_return_map(close_frame)
    daily_change_map = _build_daily_change_map(tickers, close_frame)

    # 고정 보유만 쓰므로 추세 점수·이평선 배열이 필요 없다(관련 표시 컬럼은 None으로 남긴다).
    alignment_map: dict[str, str | None] = {}

    rows: list[dict[str, Any]] = []
    ticker_meta = {item["ticker"]: item for item in tickers}
    excluded_reasons: list[str] = []
    deviation_pct_by_ticker: dict[str, float | None] = {}
    short_deviation_pct_by_ticker: dict[str, float | None] = {}
    for ticker in [item["ticker"] for item in tickers]:
        sortino_raw_value = sortino_raw_row.get(ticker)
        meta = ticker_meta[ticker]

        # 고정 보유는 가격 캐시 존재만 확인한다(이격/추세 요구 없음).
        close_count = int(close_frame[ticker].dropna().shape[0]) if ticker in close_frame.columns else 0
        if close_count <= 0:
            excluded_reasons.append(f"{ticker}: 가격 캐시 없음")
        point_deviation = None
        score = None
        deviation_pct_value = None
        deviation_pct_by_ticker[ticker] = None
        slope_value = None

        rows.append(
            {
                "ticker": ticker,
                "name": meta.get("name") or ticker,
                "ticker_type": meta.get("ticker_type"),
                "country_code": meta.get("country_code"),
                **return_map.get(
                    ticker,
                    {
                        "return_1m_pct": None,
                        "return_3m_pct": None,
                        "return_6m_pct": None,
                        "return_12m_pct": None,
                    },
                ),
                "daily_change_pct": daily_change_map.get(ticker),
                "mdd_pct": mdd_map.get(ticker),
                "deviation_pct": None if deviation_pct_value is None else round(deviation_pct_value, 2),
                "short_deviation_pct": (
                    None
                    if short_deviation_pct_by_ticker.get(ticker) is None
                    else round(short_deviation_pct_by_ticker[ticker], 2)
                ),
                "trend_pct": None if deviation_pct_value is None else round(deviation_pct_value, 2),
                "slope_pct": None if slope_value is None else round(float(slope_value), 2),
                "alignment": alignment_map.get(ticker),
                "trend_score": None if point_deviation is None else round(point_deviation, 2),
                "sortino": None if pd.isna(sortino_raw_value) else round(float(sortino_raw_value), 2),
                "score": None if score is None else round(score, 2),
                "target_weight_pct": None,
                "rebalance_needed": None,
            }
        )

    if excluded_reasons:
        raise ValueError("비중 계산에서 제외되는 종목이 있습니다. " + " / ".join(excluded_reasons))

    fixed_weights = {item["ticker"]: float(item.get("fixed_weight_pct") or 0.0) / 100.0 for item in tickers}
    sum_fixed = sum(fixed_weights.values())
    if sum_fixed > 1.0:
        for tk in fixed_weights:
            fixed_weights[tk] /= sum_fixed
        cash_weight = 0.0
    else:
        cash_weight = 1.0 - sum_fixed
    weights = {**fixed_weights, "__CASH__": cash_weight}

    for row in rows:
        weight = weights.get(row["ticker"])
        if weight is not None:
            row["target_weight_pct"] = weight * 100.0

    cash_weight = weights.get("__CASH__")
    if cash_weight is not None and cash_weight > 0:
        rows.append(
            {
                "ticker": "__CASH__",
                "name": "현금",
                "ticker_type": "cash",
                "country_code": None,
                "return_1m_pct": None,
                "return_3m_pct": None,
                "return_6m_pct": None,
                "return_12m_pct": None,
                "daily_change_pct": None,
                "deviation_pct": None,
                "trend_pct": None,
                "slope_pct": None,
                "alignment": None,
                "trend_score": None,
                "sortino": None,
                "score": None,
                "target_weight_pct": cash_weight * 100.0,
                "rebalance_needed": None,
            }
        )
    _normalize_target_weight_pct_rows(rows)

    # 1. 종목 순서는 보유 목록(portfolio_master)의 sort_order 를 그대로 따른다 (단일 소스).
    ordered_tickers = []
    try:
        from utils.portfolio_io import load_portfolio_master

        master = load_portfolio_master(str(settings.get("ACCOUNT_ID") or "").strip()) or {}
        holdings_sorted = sorted(master.get("holdings") or [], key=lambda h: int(h.get("sort_order") or 0))
        ordered_tickers = [
            str(h.get("ticker") or "").strip().upper().replace("KR:", "").replace("ASX:", "") for h in holdings_sorted
        ]
    except Exception:
        pass

    # 2. 정렬 매핑 테이블 구축 및 정렬 적용
    ticker_order_map = {ticker: idx for idx, ticker in enumerate(ordered_tickers)}

    def get_sort_key(item: dict[str, Any]) -> tuple[int, str]:
        ticker = str(item.get("ticker") or "").strip().upper()
        if ticker == "__CASH__":
            return (-1, "__CASH__")
        norm_ticker = ticker.replace("KR:", "")
        idx = ticker_order_map.get(norm_ticker)
        if idx is not None:
            return (idx, ticker)
        return (99999, ticker)

    rows.sort(key=get_sort_key)
    trade_summary = _apply_trade_plan(rows, settings=settings, tickers=tickers, close_frame=close_frame)
    return {
        "as_of_date": eval_date.strftime("%Y-%m-%d"),
        "settings": settings,
        "rows": rows,
        "missing_tickers": missing,
        "trade_summary": trade_summary,
    }


def run_asset_helper_weights(
    tickers: list[dict[str, Any]],
    settings: dict[str, Any] | None = None,
    backtest_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    clean_settings = _clean_settings(settings)
    clean_tickers = _clean_tickers(tickers)
    if len(clean_tickers) < 1:
        raise ValueError("계산할 종목이 1개 이상 필요합니다.")
    clean_backtest = _clean_backtest_settings(backtest_settings, require_benchmark=False)
    return calculate_asset_helper_weights_for(
        clean_tickers,
        clean_settings,
        metric_months=int(clean_backtest["months"]),
    )


# ── 백테스트 re-export (하위 호환) — 정의는 utils/asset_helper_backtest.py 로 이동 ──
from utils.asset_helper_backtest import run_asset_helper_backtest  # noqa: E402,F401
