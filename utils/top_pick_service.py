"""탑픽 포트폴리오 설정과 목표 비중 계산 서비스."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from config import TRADING_DAYS_PER_MONTH
from core.strategy.scoring import build_composite_rank_scores, compute_secondary_metric_points
from core.strategy.weight_allocator import calculate_ranked_score_weights_with_cash
from utils.cache_utils import get_all_ticker_type_lookup_keys, load_cached_close_series_bulk, load_cached_close_series_bulk_with_fallback
from utils.logger import get_app_logger
from utils.perf_metrics import curve_metrics, mdd_span

logger = get_app_logger()

COLLECTION = "top_pick_settings"
SETTINGS_ID = "default"
TOP_PICK_MAX_TICKERS = 10

ALLOWED_MA_TYPES = {"SMA", "EMA", "WMA", "DEMA", "TEMA", "HMA", "ALMA"}

# 탑픽 설정 스키마 — 코드 기본값(silent default) 없음. 값은 전적으로 DB에서 온다.
# 전략 필수 필드는 DB에 없으면 명시적 에러(fail loud). 사용자 선택 필드(계좌·누적수익률 기준)는
# 미설정 허용(빈값/None)이며, 그 값을 실제로 쓰는 지점에서 막는다.
SETTING_KEYS = (
    "MA_TYPE",
    "MA_MONTHS",
    "TREND_WEIGHT_RATIO",
    "SORTINO_MONTHS",
    "MIN_WEIGHT",
    "MAX_WEIGHT",
    "CASH_MAX_WEIGHT",
    "ACCOUNT_ID",
    "START_AMOUNT_MANWON",
    "START_DATE",
)
# DB에 반드시 있어야 하는(없으면 에러) 전략 필수 필드. 코드 기본값으로 대체하지 않는다.
REQUIRED_SETTING_KEYS = (
    "MA_TYPE",
    "MA_MONTHS",
    "TREND_WEIGHT_RATIO",
    "SORTINO_MONTHS",
    "MIN_WEIGHT",
    "MAX_WEIGHT",
    "CASH_MAX_WEIGHT",
)
DEFAULT_BACKTEST_SETTINGS: dict[str, Any] = {
    "months": 12,
    "rebalance": "none",
    "initial_amount_manwon": 10000,
}
ALLOWED_BACKTEST_MONTHS = {6, 12, 24}
ALLOWED_BACKTEST_REBALANCE = {"none", "weekly", "monthly", "quarterly", "yearly"}


def _db():
    from utils.db_manager import get_db_connection

    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결 실패 (top_pick_settings)")
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
        if not ticker or not name or ticker in seen:
            continue
        if len(clean) >= TOP_PICK_MAX_TICKERS:
            raise ValueError(f"탑픽 편입 ETF는 최대 {TOP_PICK_MAX_TICKERS}개까지 등록할 수 있습니다.")
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
        nickname = str(item.get("nickname") or "").strip()
        if nickname:
            row["nickname"] = nickname
        clean.append(row)
    return clean


def _clean_settings(values: dict[str, Any] | None, *, base: dict[str, Any] | None = None) -> dict[str, Any]:
    base_clean = {key: value for key, value in (base or {}).items() if value is not None}
    value_clean = {key: value for key, value in (values or {}).items() if value is not None}
    # 코드 기본값 없음 — 값은 base(DB)/values(요청)에서만 온다.
    source = {**base_clean, **value_clean}
    # 전략 필수 필드가 하나라도 없으면 코드 기본값으로 대체하지 않고 명시적 에러(fail loud).
    missing_required = [key for key in REQUIRED_SETTING_KEYS if source.get(key) is None]
    if missing_required:
        raise ValueError(
            f"탑픽 설정값이 없습니다(DB 미설정): {', '.join(missing_required)}. 설정 화면에서 저장해주세요."
        )
    ma_type = str(source.get("MA_TYPE") or "").strip().upper()
    if ma_type not in ALLOWED_MA_TYPES:
        raise ValueError(f"MA_TYPE 은 {', '.join(sorted(ALLOWED_MA_TYPES))} 중 하나여야 합니다: {ma_type}")

    cleaned: dict[str, Any] = {"MA_TYPE": ma_type}
    try:
        ma_months = int(source.get("MA_MONTHS"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"MA_MONTHS 은 정수여야 합니다: {source.get('MA_MONTHS')}") from exc
    if not (1 <= ma_months <= 24):
        raise ValueError(f"MA_MONTHS 은 1 ~ 24 범위여야 합니다: {ma_months}")
    cleaned["MA_MONTHS"] = ma_months

    try:
        trend_weight_ratio = int(source.get("TREND_WEIGHT_RATIO"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"TREND_WEIGHT_RATIO 은 정수여야 합니다: {source.get('TREND_WEIGHT_RATIO')}") from exc
    if not (0 <= trend_weight_ratio <= 100):
        raise ValueError(f"TREND_WEIGHT_RATIO 은 0 ~ 100 범위여야 합니다: {trend_weight_ratio}")
    cleaned["TREND_WEIGHT_RATIO"] = trend_weight_ratio

    try:
        sortino_months = int(source.get("SORTINO_MONTHS"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"SORTINO_MONTHS 은 정수여야 합니다: {source.get('SORTINO_MONTHS')}") from exc
    if not (1 <= sortino_months <= 6):
        raise ValueError(f"SORTINO_MONTHS 은 1 ~ 6 범위여야 합니다: {sortino_months}")
    cleaned["SORTINO_MONTHS"] = sortino_months

    float_ranges = {
        "MIN_WEIGHT": (1.0, 100.0),
        "MAX_WEIGHT": (1.0, 100.0),
        "CASH_MAX_WEIGHT": (0.0, 100.0),
    }
    for key, (low, high) in float_ranges.items():
        try:
            number = float(source.get(key))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} 은 숫자여야 합니다: {source.get(key)}") from exc
        if not (low <= number <= high):
            raise ValueError(f"{key} 은 {low} ~ {high} 범위여야 합니다: {number}")
        cleaned[key] = round(number, 1)

    if cleaned["MAX_WEIGHT"] < cleaned["MIN_WEIGHT"]:
        raise ValueError("MAX_WEIGHT 은 MIN_WEIGHT 보다 크거나 같아야 합니다.")
    account_id = str(source.get("ACCOUNT_ID") or "").strip()
    if account_id:
        from utils.settings_loader import list_available_accounts

        if account_id not in set(list_available_accounts()):
            raise ValueError(f"존재하지 않는 탑픽 적용 계좌입니다: {account_id}")
    cleaned["ACCOUNT_ID"] = account_id

    # 누적수익률 기준 시작금액(만원). 미설정(None/빈값)은 그대로 None 유지 — 임의 보정 금지.
    raw_start_amount = source.get("START_AMOUNT_MANWON")
    if raw_start_amount in (None, ""):
        cleaned["START_AMOUNT_MANWON"] = None
    else:
        try:
            start_amount_manwon = int(raw_start_amount)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"시작금액(만원)은 정수여야 합니다: {raw_start_amount}") from exc
        if not (1 <= start_amount_manwon <= 1_000_000_000):
            raise ValueError(f"시작금액(만원)은 1 ~ 1000000000 범위여야 합니다: {start_amount_manwon}")
        cleaned["START_AMOUNT_MANWON"] = start_amount_manwon

    # 누적수익률 기준 시작일자 (YYYY-MM-DD). 미설정은 None 유지.
    start_date = str(source.get("START_DATE") or "").strip()
    if not start_date:
        cleaned["START_DATE"] = None
    else:
        try:
            datetime.strptime(start_date, "%Y-%m-%d")
        except ValueError as exc:
            raise ValueError(f"시작일자는 YYYY-MM-DD 형식이어야 합니다: {start_date}") from exc
        cleaned["START_DATE"] = start_date
    return cleaned


def _clean_backtest_settings(values: Any, *, base: Any = None) -> dict[str, Any]:
    base_clean = base if isinstance(base, dict) else {}
    value_clean = values if isinstance(values, dict) else {}
    source = {**DEFAULT_BACKTEST_SETTINGS, **base_clean, **value_clean}

    benchmark_source = source.get("benchmark") if isinstance(source.get("benchmark"), dict) else {}
    ticker = str(benchmark_source.get("ticker") or "").strip().upper()
    name = str(benchmark_source.get("name") or ticker).strip()
    if not ticker:
        raise ValueError("백테스트 벤치마크 티커가 필요합니다.")

    try:
        months = int(source.get("months"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"백테스트 기간(개월)은 정수여야 합니다: {source.get('months')}") from exc
    if months not in ALLOWED_BACKTEST_MONTHS:
        raise ValueError(f"백테스트 기간(개월)은 {', '.join(map(str, sorted(ALLOWED_BACKTEST_MONTHS)))} 중 하나여야 합니다.")

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


def _serialize_doc(doc: dict[str, Any] | None) -> dict[str, Any]:
    settings = _clean_settings({}, base={key: doc[key] for key in SETTING_KEYS if doc and doc.get(key) is not None})
    updated_at = (doc or {}).get("updated_at")
    approved_at = (doc or {}).get("approved_at")
    tickers = _clean_tickers((doc or {}).get("tickers"))
    approved_weights = (doc or {}).get("approved_weights") or None
    return {
        "tickers": tickers,
        "settings": settings,
        "backtest_settings": _clean_backtest_settings((doc or {}).get("backtest_settings")),
        "approved_weights": _enrich_weight_rows_with_returns(approved_weights, tickers),
        "approved_at": (approved_at.replace(tzinfo=timezone.utc) if approved_at.tzinfo is None else approved_at).isoformat() if isinstance(approved_at, datetime) else None,
        "updated_at": (updated_at.replace(tzinfo=timezone.utc) if updated_at.tzinfo is None else updated_at).isoformat() if isinstance(updated_at, datetime) else None,
    }


def load_top_pick_settings() -> dict[str, Any]:
    doc = _db()[COLLECTION].find_one({"_id": SETTINGS_ID})
    return _serialize_doc(doc)


def save_top_pick_settings(
    tickers: list[dict[str, Any]],
    settings: dict[str, Any] | None = None,
    backtest_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    clean_tickers = _clean_tickers(tickers)
    if len(clean_tickers) < 1:
        raise ValueError("저장할 종목이 1개 이상 필요합니다.")

    current = load_top_pick_settings()
    clean_settings = _clean_settings(settings, base=current.get("settings") or {})
    clean_backtest_settings = _clean_backtest_settings(
        backtest_settings,
        base=current.get("backtest_settings") or DEFAULT_BACKTEST_SETTINGS,
    )
    updated_at = datetime.now(timezone.utc)
    _db()[COLLECTION].update_one(
        {"_id": SETTINGS_ID},
        {
            "$set": {
                "tickers": clean_tickers,
                **clean_settings,
                "backtest_settings": clean_backtest_settings,
                "updated_at": updated_at,
            },
        },
        upsert=True,
    )
    return {
        "tickers": clean_tickers,
        "settings": clean_settings,
        "backtest_settings": clean_backtest_settings,
        "approved_weights": current.get("approved_weights"),
        "approved_at": current.get("approved_at"),
        "updated_at": updated_at.isoformat(),
    }


def _load_close_frame(tickers: list[dict[str, Any]]) -> tuple[pd.DataFrame, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for item in tickers:
        ticker = str(item.get("ticker") or "").strip().upper()
        ticker_type = str(item.get("ticker_type") or "").strip().lower()
        if ticker and ticker_type:
            grouped[ticker_type].append(ticker)

    series_map: dict[str, pd.Series] = {}
    for ticker_type, group_tickers in grouped.items():
        fetched = load_cached_close_series_bulk_with_fallback(ticker_type, group_tickers)
        for ticker, series in fetched.items():
            if series is None or series.empty:
                continue
            normalized = pd.to_numeric(series, errors="coerce").dropna()
            normalized.index = pd.to_datetime(normalized.index).normalize()
            normalized = normalized[~normalized.index.duplicated(keep="last")].sort_index()
            if not normalized.empty:
                series_map[str(ticker).strip().upper()] = normalized

    unresolved = [
        str(item.get("ticker") or "").strip().upper()
        for item in tickers
        if str(item.get("ticker") or "").strip().upper() and str(item.get("ticker") or "").strip().upper() not in series_map
    ]
    for cache_key in get_all_ticker_type_lookup_keys():
        if not unresolved:
            break
        fetched = load_cached_close_series_bulk(cache_key, unresolved)
        for ticker, series in fetched.items():
            if series is None or series.empty:
                continue
            normalized = pd.to_numeric(series, errors="coerce").dropna()
            normalized.index = pd.to_datetime(normalized.index).normalize()
            normalized = normalized[~normalized.index.duplicated(keep="last")].sort_index()
            if not normalized.empty:
                series_map[str(ticker).strip().upper()] = normalized
        unresolved = [ticker for ticker in unresolved if ticker not in series_map]

    missing = [
        str(item.get("ticker") or "").strip().upper()
        for item in tickers
        if str(item.get("ticker") or "").strip().upper() not in series_map
    ]
    if not series_map:
        return pd.DataFrame(), missing

    union_index = sorted({ts for series in series_map.values() for ts in series.index})
    close_frame = pd.DataFrame(
        {ticker: series.reindex(union_index) for ticker, series in series_map.items()},
        index=pd.DatetimeIndex(union_index),
    )
    return close_frame, missing


def _calculate_period_return(series: pd.Series, months: int, eval_date: pd.Timestamp) -> float | None:
    normalized = pd.to_numeric(series, errors="coerce").dropna()
    if normalized.empty:
        return None

    latest_candidates = normalized[normalized.index <= eval_date]
    if latest_candidates.empty:
        return None

    latest_price = float(latest_candidates.iloc[-1])
    target_date = latest_candidates.index[-1] - pd.DateOffset(months=months)
    base_candidates = normalized[normalized.index <= target_date]
    if base_candidates.empty:
        return None

    base_price = float(base_candidates.iloc[-1])
    if base_price <= 0:
        return None

    return round(((latest_price / base_price) - 1.0) * 100.0, 2)


def _build_return_map(close_frame: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    if close_frame.empty:
        return {}

    eval_date = close_frame.index.max()
    result: dict[str, dict[str, float | None]] = {}
    for ticker in close_frame.columns:
        series = close_frame[ticker]
        result[str(ticker)] = {
            "return_1m_pct": _calculate_period_return(series, 1, eval_date),
            "return_3m_pct": _calculate_period_return(series, 3, eval_date),
            "return_12m_pct": _calculate_period_return(series, 12, eval_date),
        }
    return result


def _build_cached_daily_change_map(close_frame: pd.DataFrame) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for ticker in close_frame.columns:
        series = pd.to_numeric(close_frame[ticker], errors="coerce").dropna()
        if len(series) < 2:
            result[str(ticker)] = None
            continue
        current = float(series.iloc[-1])
        previous = float(series.iloc[-2])
        result[str(ticker)] = round(((current / previous) - 1.0) * 100.0, 2) if previous > 0 else None
    return result


def _extract_realtime_change_pct(snapshot: dict[str, Any]) -> float | None:
    raw_value = snapshot.get("changeRate")
    if raw_value is None:
        raw_value = snapshot.get("change_pct")
    if raw_value is None:
        return None
    try:
        return round(float(str(raw_value).replace(",", "")), 2)
    except (TypeError, ValueError):
        return None


def _extract_realtime_price(snapshot: dict[str, Any]) -> float | None:
    raw_value = snapshot.get("nowVal")
    if raw_value is None:
        raw_value = snapshot.get("price")
    if raw_value is None:
        return None
    try:
        price = float(str(raw_value).replace(",", ""))
    except (TypeError, ValueError):
        return None
    return price if price > 0 else None


def _build_cached_current_price_map(close_frame: pd.DataFrame) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for ticker in close_frame.columns:
        series = pd.to_numeric(close_frame[ticker], errors="coerce").dropna()
        if series.empty:
            result[str(ticker)] = None
            continue
        price = float(series.iloc[-1])
        result[str(ticker)] = price if price > 0 else None
    return result


def _build_current_price_map(tickers: list[dict[str, Any]], close_frame: pd.DataFrame) -> dict[str, float | None]:
    result = _build_cached_current_price_map(close_frame)
    grouped: dict[str, list[str]] = defaultdict(list)
    for item in tickers:
        ticker = str(item.get("ticker") or "").strip().upper()
        country = str(item.get("country_code") or "").strip().lower()
        if not country:
            ticker_type = str(item.get("ticker_type") or "").strip().lower()
            country = ticker_type.split("_", 1)[0] if ticker_type else ""
        if ticker and country in {"kor", "au", "us"}:
            grouped[country].append(ticker)

    if not grouped:
        return result

    from services.price_service import get_realtime_snapshot

    for country, group_tickers in grouped.items():
        try:
            snapshot_map = get_realtime_snapshot(country, group_tickers)
        except Exception as exc:
            logger.warning("탑픽 실시간 현재가 조회 실패 (%s): %s", country, exc)
            continue
        for ticker, snapshot in snapshot_map.items():
            price = _extract_realtime_price(snapshot)
            if price is not None:
                result[str(ticker).strip().upper()] = price
    return result


def _build_daily_change_map(tickers: list[dict[str, Any]], close_frame: pd.DataFrame) -> dict[str, float | None]:
    result = _build_cached_daily_change_map(close_frame)
    grouped: dict[str, list[str]] = defaultdict(list)
    for item in tickers:
        ticker = str(item.get("ticker") or "").strip().upper()
        country = str(item.get("country_code") or "").strip().lower()
        if not country:
            ticker_type = str(item.get("ticker_type") or "").strip().lower()
            country = ticker_type.split("_", 1)[0] if ticker_type else ""
        if ticker and country in {"kor", "au", "us"}:
            grouped[country].append(ticker)

    if not grouped:
        return result

    from services.price_service import get_realtime_snapshot

    for country, group_tickers in grouped.items():
        try:
            snapshot_map = get_realtime_snapshot(country, group_tickers)
        except Exception as exc:
            logger.warning("탑픽 실시간 등락률 조회 실패 (%s): %s", country, exc)
            continue
        for ticker, snapshot in snapshot_map.items():
            change_pct = _extract_realtime_change_pct(snapshot)
            if change_pct is not None:
                result[str(ticker).strip().upper()] = change_pct
    return result


def _normalize_kor_holding_ticker(value: Any) -> str:
    return str(value or "").strip().upper().replace("KR:", "")


def _load_top_pick_account_snapshot(account_id: str) -> dict[str, Any]:
    if not account_id:
        raise ValueError("탑픽 적용 계좌를 선택해주세요.")

    from utils.holdings_detail_service import load_all_holdings_detail

    payload = load_all_holdings_detail(account_id)
    summaries = [
        row
        for row in payload.get("account_summaries", [])
        if str(row.get("account_id") or "") == account_id
    ]
    if not summaries:
        raise ValueError(f"탑픽 적용 계좌 정보를 찾을 수 없습니다: {account_id}")

    summary = summaries[0]
    holdings: dict[str, dict[str, Any]] = {}
    for row in payload.get("rows", []):
        ticker = _normalize_kor_holding_ticker(row.get("ticker"))
        if not ticker or ticker == "IS":
            continue
        current = holdings.setdefault(
            ticker,
            {
                "current_quantity": 0,
                "current_amount_krw": 0,
                "name": str(row.get("name") or ticker),
                "pnl_krw": 0.0,
                "buy_amount_krw": 0.0,
                "return_pct": 0.0,
                "bucket": row.get("bucket"),
            },
        )
        current["current_quantity"] += int(float(row.get("quantity") or 0))
        current["current_amount_krw"] += int(round(float(row.get("valuation_krw") or 0)))
        current["pnl_krw"] += float(row.get("pnl_krw_num") or 0.0)
        current["buy_amount_krw"] += float(row.get("buy_amount_krw") or 0.0)
        if not current.get("name"):
            current["name"] = str(row.get("name") or ticker)

    for ticker, current in holdings.items():
        buy_amt = current["buy_amount_krw"]
        pnl = current["pnl_krw"]
        if buy_amt > 0:
            current["return_pct"] = round((pnl / buy_amt) * 100.0, 2)
        else:
            current["return_pct"] = 0.0

    return {
        "account_id": account_id,
        "account_name": summary.get("name") or account_id,
        "account_amount_krw": int(round(float(summary.get("total_assets_krw") or 0))),
        "cash_balance_krw": int(round(float(summary.get("cash_balance_krw") or 0))),
        "holdings": holdings,
    }


def _normalize_bucket_label(bucket_val: Any) -> str | None:
    if not bucket_val:
        return None
    val_str = str(bucket_val).strip()
    if not val_str:
        return None
    import re
    pure_name = re.sub(r"^(\d+\.\s*)+", "", val_str).strip()

    mapping = {
        "모멘텀": "1. 모멘텀",
        "시장지수": "2. 시장지수",
        "배당방어": "3. 배당방어",
        "비당방어": "3. 배당방어",
        "대체헷지": "4. 대체헷지",
        "현금": "5. 현금"
    }
    return mapping.get(pure_name, val_str)


def _apply_trade_plan(
    rows: list[dict[str, Any]],
    *,
    settings: dict[str, Any],
    tickers: list[dict[str, Any]],
    close_frame: pd.DataFrame,
) -> dict[str, Any]:
    account_snapshot = _load_top_pick_account_snapshot(str(settings.get("ACCOUNT_ID") or "").strip())
    account_amount_krw = int(account_snapshot["account_amount_krw"])
    holdings = account_snapshot["holdings"]
    current_price_map = _build_current_price_map(tickers, close_frame)
    target_asset_amount = 0
    target_tickers: set[str] = set()

    for row in rows:
        ticker = str(row.get("ticker") or "").strip().upper()
        if ticker and ticker != "__CASH__":
            target_tickers.add(_normalize_kor_holding_ticker(ticker))
        target_weight = row.get("target_weight_pct")
        target_amount = None
        if account_amount_krw > 0 and target_weight is not None:
            target_amount = int(round(account_amount_krw * (float(target_weight) / 100.0)))

        row["current_price"] = None
        row["target_amount_krw"] = target_amount
        row["target_quantity"] = None
        row["current_quantity"] = None
        row["current_amount_krw"] = None
        row["change_quantity"] = None
        row["unallocated_amount_krw"] = target_amount
        row["return_pct"] = None
        row["pnl_krw"] = None
        row["bucket"] = None
        if "nickname" not in row:
            row["nickname"] = None

        if ticker == "__CASH__":
            row["current_amount_krw"] = account_snapshot["cash_balance_krw"]
            row["bucket"] = "5. 현금"
            row["nickname"] = None
            continue

        current_price = current_price_map.get(ticker)
        row["current_price"] = None if current_price is None else round(float(current_price), 2)
        holding = holdings.get(_normalize_kor_holding_ticker(ticker), {})
        current_quantity = int(holding.get("current_quantity") or 0)
        current_amount = int(holding.get("current_amount_krw") or 0)
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

        quantity = int(target_amount // current_price)
        target_buy_amount = int(round(quantity * current_price))
        row["target_quantity"] = quantity
        row["change_quantity"] = quantity - current_quantity
        row["unallocated_amount_krw"] = max(0, int(target_amount - target_buy_amount))
        target_asset_amount += target_buy_amount

    for ticker, holding in sorted(holdings.items()):
        if ticker in target_tickers:
            continue
        current_quantity = int(holding.get("current_quantity") or 0)
        current_amount = int(holding.get("current_amount_krw") or 0)
        if current_quantity <= 0 and current_amount <= 0:
            continue
        rows.append(
            {
                "ticker": ticker,
                "name": holding.get("name") or ticker,
                "ticker_type": "kor_kr",
                "country_code": "kor",
                "return_1m_pct": None,
                "return_3m_pct": None,
                "return_12m_pct": None,
                "daily_change_pct": None,
                "trend_pct": None,
                "trend_score": None,
                "sortino_score": None,
                "sortino": None,
                "score": None,
                "target_weight_pct": 0.0,
                "rebalance_needed": None,
                "current_price": round(current_amount / current_quantity, 2) if current_quantity > 0 else None,
                "target_amount_krw": 0,
                "target_quantity": 0,
                "current_quantity": current_quantity,
                "current_amount_krw": current_amount,
                "change_quantity": -current_quantity,
                "unallocated_amount_krw": 0,
                "return_pct": holding.get("return_pct"),
                "pnl_krw": holding.get("pnl_krw"),
                "bucket": _normalize_bucket_label(holding.get("bucket")),
            }
        )

    remaining_cash = max(0, account_amount_krw - target_asset_amount) if account_amount_krw > 0 else 0
    for row in rows:
        if row.get("ticker") == "__CASH__":
            row["target_amount_krw"] = remaining_cash if account_amount_krw > 0 else row.get("target_amount_krw")
            row["unallocated_amount_krw"] = remaining_cash if account_amount_krw > 0 else row.get("unallocated_amount_krw")

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
        "account_amount_krw": account_amount_krw,
        "target_asset_amount_krw": target_asset_amount,
        "remaining_cash_krw": remaining_cash,
    }


def _enrich_weight_rows_with_returns(payload: Any, tickers: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        return payload

    close_frame, _missing = _load_close_frame(tickers)
    return_map = _build_return_map(close_frame)
    daily_change_map = _build_daily_change_map(tickers, close_frame)
    settings = _clean_settings(payload.get("settings") if isinstance(payload.get("settings"), dict) else None)
    eval_date = close_frame.index.max() if not close_frame.empty else None
    sortino_raw_frame = _compute_sortino_raw_frame(close_frame, int(settings["SORTINO_MONTHS"]))
    sortino_raw_row = (
        sortino_raw_frame.loc[eval_date]
        if eval_date is not None and eval_date in sortino_raw_frame.index
        else pd.Series(dtype=float)
    )
    enriched_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        ticker = str(row.get("ticker") or "").strip().upper()
        sortino_raw_value = sortino_raw_row.get(ticker)
        enriched_rows.append(
            {
                **row,
                **return_map.get(
                    ticker,
                    {
                        "return_1m_pct": None,
                        "return_3m_pct": None,
                        "return_12m_pct": None,
                    },
                ),
                "daily_change_pct": daily_change_map.get(ticker),
                "sortino": None if pd.isna(sortino_raw_value) else round(float(sortino_raw_value), 2),
            }
        )
    _normalize_target_weight_pct_rows(enriched_rows)
    trade_summary = (
        _apply_trade_plan(enriched_rows, settings=settings, tickers=tickers, close_frame=close_frame)
        if settings.get("ACCOUNT_ID")
        else {}
    )
    return {**payload, "rows": enriched_rows, "trade_summary": trade_summary}


def _normalize_target_weight_pct_rows(rows: list[dict[str, Any]]) -> None:
    weight_rows = [
        row
        for row in rows
        if row.get("target_weight_pct") is not None
    ]
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
    daily_ret = close_frame.pct_change(fill_method=None)
    mean = daily_ret.rolling(window=window, min_periods=max(2, window // 2)).mean()

    def _calc_downside_std(values: np.ndarray) -> float:
        downside = np.minimum(0.0, values)
        if len(values) <= 1:
            return np.nan
        result = np.sqrt(np.sum(downside ** 2) / (len(values) - 1))
        return float(result) if result > 0 else np.nan

    downside_std = daily_ret.rolling(window=window, min_periods=max(2, window // 2)).apply(_calc_downside_std, raw=True)
    return (mean / downside_std.replace(0, np.nan)) * np.sqrt(252.0)


def calculate_top_pick_weights_for(tickers: list[dict[str, Any]], settings: dict[str, Any]) -> dict[str, Any]:
    if len(tickers) < 3:
        raise ValueError("탑픽 비중 계산에는 확인된 종목이 3개 이상 필요합니다.")

    min_weight = float(settings["MIN_WEIGHT"]) / 100.0
    max_weight = float(settings["MAX_WEIGHT"]) / 100.0
    cash_max_weight = float(settings["CASH_MAX_WEIGHT"]) / 100.0
    forced_cash_weight: float | None = None
    regime_forced_cash_zero = False  # 상승장 자동 현금 제어로 현금 최대가 0%로 강제됐는지
    if min_weight * len(tickers) > 1.0:
        raise ValueError("최소 비중과 종목 수가 맞지 않습니다. 최소 비중을 낮추거나 종목 수를 줄이세요.")
    if (max_weight * len(tickers)) + cash_max_weight < 1.0:
        raise ValueError("최대 비중과 현금 최대 비중이 맞지 않습니다. 최대 비중 또는 현금 최대 비중을 높이세요.")

    # 벤치마크 실시간 레짐 판단에 따른 현금 조절
    try:
        db_settings = load_top_pick_settings()
        benchmark = db_settings.get("backtest_settings", {}).get("benchmark", {})
        bench_ticker = benchmark.get("ticker")
        if bench_ticker:
            from utils.cache_utils import load_cached_frames_bulk_from_all_ticker_types
            bench_frames = load_cached_frames_bulk_from_all_ticker_types([bench_ticker])
            bench_df = bench_frames.get(bench_ticker)
            if bench_df is not None and not bench_df.empty:
                regimes = _calculate_benchmark_regimes(bench_df)
                if not regimes.empty:
                    latest_date = regimes.index.max()
                    current_regime = regimes.loc[latest_date]
                    if current_regime == "accel_up":
                        cash_max_weight = 0.0
                        regime_forced_cash_zero = True
                    elif current_regime == "accel_down":
                        forced_cash_weight = cash_max_weight
    except Exception:
        pass

    close_frame, missing = _load_close_frame(tickers)
    if close_frame.empty:
        raise ValueError("탑픽 종목의 가격 캐시가 없습니다.")

    ma_rules = [
        {
            "order": 1,
            "ma_type": settings["MA_TYPE"],
            "ma_months": settings["MA_MONTHS"],
            "score_column": "추세",
        }
    ]
    composite_frame, trend_by_order, _ = build_composite_rank_scores(close_frame, ma_rules)
    sortino_months = int(settings["SORTINO_MONTHS"])
    trend_share = float(settings["TREND_WEIGHT_RATIO"]) / 100.0
    sortino_share = 1.0 - trend_share
    sortino_frame = compute_secondary_metric_points(close_frame, "SORTINO", window_months=sortino_months)
    sortino_raw_frame = _compute_sortino_raw_frame(close_frame, sortino_months)

    eval_date = close_frame.index.max()
    composite_row = composite_frame.loc[eval_date] if eval_date in composite_frame.index else pd.Series(dtype=float)
    trend_frame = trend_by_order[1]
    trend_row = trend_frame.loc[eval_date] if eval_date in trend_frame.index else pd.Series(dtype=float)
    sortino_row = sortino_frame.loc[eval_date] if eval_date in sortino_frame.index else pd.Series(dtype=float)
    sortino_raw_row = sortino_raw_frame.loc[eval_date] if eval_date in sortino_raw_frame.index else pd.Series(dtype=float)
    return_map = _build_return_map(close_frame)
    daily_change_map = _build_daily_change_map(tickers, close_frame)

    raw_scores: dict[str, float] = {}
    defensive_tickers: set[str] = set()
    rows: list[dict[str, Any]] = []
    ticker_meta = {item["ticker"]: item for item in tickers}
    for ticker in [item["ticker"] for item in tickers]:
        trend_value = composite_row.get(ticker)
        sortino_value = sortino_row.get(ticker)
        sortino_raw_value = sortino_raw_row.get(ticker)
        point_trend = None if pd.isna(trend_value) else float(trend_value)
        point_sortino = 0.0 if pd.isna(sortino_value) else float(sortino_value)
        score = None if point_trend is None else (trend_share * point_trend) + (sortino_share * point_sortino)
        if score is not None:
            raw_scores[ticker] = score

        trend_pct = trend_row.get(ticker)
        trend_pct_value = None if pd.isna(trend_pct) else float(trend_pct)
        if trend_pct_value is not None and trend_pct_value <= 0:
            defensive_tickers.add(ticker)
        meta = ticker_meta[ticker]
        rows.append(
            {
                "ticker": ticker,
                "name": meta.get("name") or ticker,
                "nickname": meta.get("nickname"),
                "ticker_type": meta.get("ticker_type"),
                "country_code": meta.get("country_code"),
                **return_map.get(
                    ticker,
                    {
                        "return_1m_pct": None,
                        "return_3m_pct": None,
                        "return_12m_pct": None,
                    },
                ),
                "daily_change_pct": daily_change_map.get(ticker),
                "trend_pct": None if trend_pct_value is None else round(trend_pct_value, 2),
                "trend_score": None if point_trend is None else round(point_trend, 2),
                "sortino_score": round(point_sortino, 2),
                "sortino": None if pd.isna(sortino_raw_value) else round(float(sortino_raw_value), 2),
                "score": None if score is None else round(score, 2),
                "target_weight_pct": None,
                "rebalance_needed": None,
            }
        )

    if len(raw_scores) < 3:
        raise ValueError(f"비중 계산 가능한 종목이 3개 미만입니다. 가격 캐시 누락: {', '.join(missing) or '-'}")

    try:
        weights = calculate_ranked_score_weights_with_cash(
            raw_scores,
            defensive_tickers=defensive_tickers,
            min_weight=min_weight,
            max_weight=max_weight,
            cash_max_weight=cash_max_weight,
            forced_cash_weight=forced_cash_weight,
        )
    except ValueError as exc:
        # 상승장 자동 현금 제어(현금 0%)로 인한 실패면, UI 설정값(현금 최대 %)과 달라 혼란스러우므로 사유를 명시한다.
        if regime_forced_cash_zero and "채울 수 없습니다" in str(exc):
            n_scored = len(raw_scores)
            need_pct = 100.0 / n_scored if n_scored else 100.0
            raise ValueError(
                f"현재 벤치마크가 상승장이라 '시장 연동형 현금 제어'로 현금 비중이 0%로 적용됩니다"
                f"(설정한 현금 최대 {float(settings['CASH_MAX_WEIGHT']):.0f}%는 상승장에서 무시). "
                f"이 상태에서는 최대 비중 {max_weight * 100:.1f}% × {n_scored}종목 = {max_weight * n_scored * 100:.0f}%로 "
                f"100%를 채울 수 없습니다. 최대 비중을 {need_pct:.1f}% 이상으로 높이거나 종목 수를 늘리세요."
            ) from exc
        raise
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
                "return_12m_pct": None,
                "daily_change_pct": None,
                "trend_pct": None,
                "trend_score": None,
                "sortino_score": None,
                "sortino": None,
                "score": None,
                "target_weight_pct": cash_weight * 100.0,
                "rebalance_needed": None,
            }
        )
    _normalize_target_weight_pct_rows(rows)

    # 1. top-pick-settings 에 저장된 tickers 순서 로드
    ordered_tickers = []
    try:
        doc = _db()[COLLECTION].find_one({"_id": SETTINGS_ID})
        if doc:
            tickers_list = doc.get("tickers") or []
            ordered_tickers = [
                str(t.get("ticker") or "").strip().upper().replace("KR:", "")
                for t in tickers_list
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


def calculate_top_pick_weights() -> dict[str, Any]:
    payload = load_top_pick_settings()
    result = calculate_top_pick_weights_for(payload["tickers"], payload["settings"])
    result["updated_at"] = payload.get("updated_at")
    return result


def run_top_pick_weights(tickers: list[dict[str, Any]], settings: dict[str, Any] | None = None) -> dict[str, Any]:
    clean_tickers = _clean_tickers(tickers)
    if len(clean_tickers) < 1:
        raise ValueError("계산할 종목이 1개 이상 필요합니다.")
    clean_settings = _clean_settings(settings)
    return calculate_top_pick_weights_for(clean_tickers, clean_settings)


def approve_top_pick_weights(tickers: list[dict[str, Any]], settings: dict[str, Any] | None = None) -> dict[str, Any]:
    result = run_top_pick_weights(tickers, settings)
    approved_at = datetime.now(timezone.utc)
    clean_tickers = _clean_tickers(tickers)
    clean_settings = _clean_settings(settings)
    _db()[COLLECTION].update_one(
        {"_id": SETTINGS_ID},
        {
            "$set": {
                "tickers": clean_tickers,
                **clean_settings,
                "approved_weights": result,
                "approved_at": approved_at,
                "updated_at": approved_at,
            }
        },
        upsert=True,
    )
    return {
        **result,
        "approved_at": approved_at.isoformat(),
        "tickers": clean_tickers,
    }


def _select_rebalance_dates(index: pd.DatetimeIndex, rebalance: str) -> list[pd.Timestamp]:
    if index.empty:
        return []
    if rebalance == "none":
        return [index[0]]

    freq_map = {"weekly": "W", "monthly": "M", "quarterly": "Q", "yearly": "Y"}
    periods = index.to_period(freq_map[rebalance])
    selected: dict[Any, pd.Timestamp] = {}
    for date, period in zip(index, periods):
        selected[period] = date
    dates = sorted(set(selected.values()))
    if index[0] not in dates:
        dates.insert(0, index[0])
    return dates


def _select_friday_history_dates(index: pd.DatetimeIndex) -> set[pd.Timestamp]:
    if index.empty:
        return set()

    max_date = index[-1].normalize()
    periods = index.to_period("W-FRI")
    selected: dict[Any, pd.Timestamp] = {}
    for date, period in zip(index, periods):
        period_end = period.end_time.normalize()
        if period_end <= max_date:
            selected[period] = date
    res = set(selected.values())
    res.add(index[-1])
    return res


def _build_top_pick_weight_engine(
    close_frame: pd.DataFrame,
    tickers: list[dict[str, Any]],
    settings: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    ma_rules = [
        {
            "order": 1,
            "ma_type": settings["MA_TYPE"],
            "ma_months": settings["MA_MONTHS"],
            "score_column": "추세",
        }
    ]
    composite_frame, trend_by_order, _ = build_composite_rank_scores(close_frame, ma_rules)
    return composite_frame, trend_by_order


def _calculate_top_pick_weights_on_date(
    eval_date: pd.Timestamp,
    tickers: list[dict[str, Any]],
    settings: dict[str, Any],
    composite_frame: pd.DataFrame,
    trend_frame: pd.DataFrame,
    sortino_frame: pd.DataFrame,
    forced_cash_weight: float | None = None,
) -> dict[str, float] | None:
    if eval_date not in composite_frame.index:
        return None

    composite_row = composite_frame.loc[eval_date]
    trend_row = trend_frame.loc[eval_date] if eval_date in trend_frame.index else pd.Series(dtype=float)
    sortino_row = sortino_frame.loc[eval_date] if eval_date in sortino_frame.index else pd.Series(dtype=float)
    trend_share = float(settings["TREND_WEIGHT_RATIO"]) / 100.0
    sortino_share = 1.0 - trend_share
    raw_scores: dict[str, float] = {}
    defensive_tickers: set[str] = set()

    for item in tickers:
        ticker = str(item.get("ticker") or "").strip().upper()
        trend_value = composite_row.get(ticker)
        if pd.isna(trend_value):
            continue
        point_trend = float(trend_value)
        sortino_value = sortino_row.get(ticker)
        point_sortino = 0.0 if pd.isna(sortino_value) else float(sortino_value)
        raw_scores[ticker] = (trend_share * point_trend) + (sortino_share * point_sortino)

        trend_pct = trend_row.get(ticker)
        if not pd.isna(trend_pct) and float(trend_pct) <= 0:
            defensive_tickers.add(ticker)

    if len(raw_scores) < 3:
        return None

    weights = calculate_ranked_score_weights_with_cash(
        raw_scores,
        defensive_tickers=defensive_tickers,
        min_weight=float(settings["MIN_WEIGHT"]) / 100.0,
        max_weight=float(settings["MAX_WEIGHT"]) / 100.0,
        cash_max_weight=float(settings["CASH_MAX_WEIGHT"]) / 100.0,
        forced_cash_weight=forced_cash_weight,
    )
    return _normalize_weight_ratio_map(weights)


def run_top_pick_backtest(
    tickers: list[dict[str, Any]],
    settings: dict[str, Any] | None = None,
    backtest_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    clean_tickers = _clean_tickers(tickers)
    if len(clean_tickers) < 3:
        raise ValueError("탑픽 백테스트에는 확인된 종목이 3개 이상 필요합니다.")

    clean_settings = _clean_settings(settings)
    clean_backtest = _clean_backtest_settings(backtest_settings)
    benchmark = clean_backtest["benchmark"]
    months = int(clean_backtest["months"])
    rebalance = str(clean_backtest["rebalance"])
    initial_capital_krw = float(int(clean_backtest["initial_amount_manwon"]) * 10_000)

    ticker_order = [item["ticker"] for item in clean_tickers]
    combined_tickers = clean_tickers + [benchmark]
    close_frame, missing = _load_close_frame(combined_tickers)
    if close_frame.empty:
        raise ValueError("탑픽 백테스트 가격 캐시가 없습니다.")

    missing_required = [ticker for ticker in ticker_order + [benchmark["ticker"]] if ticker not in close_frame.columns]
    if missing_required:
        raise ValueError(f"탑픽 백테스트 가격 캐시 누락: {', '.join(missing_required)}")

    today = pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None).normalize()
    start_target = (today - pd.DateOffset(months=months)).normalize()
    candidate_close = close_frame[ticker_order].sort_index()
    composite_frame, trend_by_order = _build_top_pick_weight_engine(candidate_close, clean_tickers, clean_settings)
    trend_frame = trend_by_order[1]
    sortino_frame = compute_secondary_metric_points(
        candidate_close,
        "SORTINO",
        window_months=int(clean_settings["SORTINO_MONTHS"]),
    )

    simulation_columns = list(dict.fromkeys(ticker_order + [benchmark["ticker"]]))
    simulation_frame = close_frame[simulation_columns].sort_index()
    simulation_frame = simulation_frame[simulation_frame.index >= start_target].dropna(how="all").ffill()
    simulation_frame = simulation_frame.dropna(subset=[benchmark["ticker"]])
    if simulation_frame.empty or len(simulation_frame) < 2:
        raise ValueError("탑픽 백테스트 기간의 가격 데이터가 부족합니다.")

    # 벤치마크 지수의 과거 레짐을 미리 일별 시계열로 구함
    try:
        from utils.cache_utils import load_cached_frames_bulk_from_all_ticker_types
        bench_frames = load_cached_frames_bulk_from_all_ticker_types([benchmark["ticker"]])
        bench_df = bench_frames.get(benchmark["ticker"])
        if bench_df is not None and not bench_df.empty:
            benchmark_regimes = _calculate_benchmark_regimes(bench_df)
        else:
            benchmark_regimes = pd.Series()
    except Exception:
        benchmark_regimes = pd.Series()

    requested_rebalance_dates = _select_rebalance_dates(simulation_frame.index, rebalance)
    weights_by_date: dict[pd.Timestamp, dict[str, float]] = {}
    for date in requested_rebalance_dates:
        # 그날의 벤치마크 시장 레짐에 따라 동적으로 현금 비중 제어
        adjusted_settings = clean_settings.copy()
        forced_cash_weight = None
        regime = benchmark_regimes.get(date)
        if regime == "accel_up":
            # 시장 상승 시 현금 최대 비중을 0%로 강제 (100% 주식 가동)
            adjusted_settings["CASH_MAX_WEIGHT"] = 0
        elif regime == "accel_down":
            # 시장 하락 시 설정된 현금 최대 비중을 우선 확보한다.
            forced_cash_weight = float(clean_settings["CASH_MAX_WEIGHT"]) / 100.0

        weights = _calculate_top_pick_weights_on_date(
            date,
            clean_tickers,
            adjusted_settings,
            composite_frame,
            trend_frame,
            sortino_frame,
            forced_cash_weight=forced_cash_weight,
        )
        if weights is not None:
            weights_by_date[date] = weights

    if not weights_by_date:
        raise ValueError("백테스트 기간에 계산 가능한 탑픽 비중이 없습니다.")

    start_date = min(weights_by_date)
    sim = simulation_frame[simulation_frame.index >= start_date].copy()
    if len(sim) < 2:
        raise ValueError("탑픽 백테스트 시뮬레이션 기간이 부족합니다.")

    current_weights = weights_by_date[start_date]
    cash_value = initial_capital_krw * float(current_weights.get("__CASH__", 0.0))
    asset_values = {
        ticker: initial_capital_krw * float(current_weights.get(ticker, 0.0))
        for ticker in ticker_order
    }
    profit_by_ticker = {ticker: 0.0 for ticker in ticker_order}
    curve_values: list[float] = [initial_capital_krw]
    weight_history: list[dict[str, Any]] = []
    friday_history_dates = _select_friday_history_dates(sim.index)

    # 종목별/현금의 일별 실제 비중(%) 이력을 추적합니다.
    weights_history_by_ticker: dict[str, list[float]] = {
        ticker: [] for ticker in ticker_order
    }
    weights_history_by_ticker["__CASH__"] = []

    def record_current_weights():
        total_val = sum(asset_values.values()) + cash_value
        if total_val > 0:
            for t in ticker_order:
                weights_history_by_ticker[t].append((asset_values[t] / total_val) * 100.0)
            weights_history_by_ticker["__CASH__"].append((cash_value / total_val) * 100.0)

    # 1. start_date 시점의 비중 기록
    record_current_weights()

    def append_weight_history(date: pd.Timestamp, values: dict[str, float], cash: float) -> None:
        row: dict[str, Any] = {"date": date.date().isoformat()}
        for item in clean_tickers:
            ticker = item["ticker"]
            row[ticker] = round(float(values.get(ticker, 0.0)), 0)
        row["__CASH__"] = round(float(cash), 0)
        weight_history.append(row)

    if start_date in friday_history_dates:
        append_weight_history(start_date, asset_values, cash_value)
    sim_index = list(sim.index)
    for idx in range(1, len(sim_index)):
        prev_date = sim_index[idx - 1]
        date = sim_index[idx]
        previous_prices = sim.loc[prev_date, ticker_order]
        current_prices = sim.loc[date, ticker_order]
        for ticker in ticker_order:
            previous_price = float(previous_prices[ticker])
            current_price = float(current_prices[ticker])
            if previous_price > 0 and current_price > 0:
                previous_value = float(asset_values.get(ticker, 0.0))
                current_value = previous_value * (current_price / previous_price)
                profit_by_ticker[ticker] += current_value - previous_value
                asset_values[ticker] = current_value

        if date in weights_by_date and date != start_date:
            current_weights = weights_by_date[date]
            total_value = sum(asset_values.values()) + cash_value
            asset_values = {
                ticker: total_value * float(current_weights.get(ticker, 0.0))
                for ticker in ticker_order
            }
            cash_value = total_value * float(current_weights.get("__CASH__", 0.0))

        curve_values.append(sum(asset_values.values()) + cash_value)
        # 매일 최종 자산 상태 기준 비중 기록
        record_current_weights()
        if date in friday_history_dates:
            append_weight_history(date, asset_values, cash_value)

    # 비중 이력 리스트의 최소/최대비중을 각 종목별로 집계합니다.
    min_weights = {}
    max_weights = {}
    for t in weights_history_by_ticker:
        lst = weights_history_by_ticker[t]
        if lst:
            min_weights[t] = min(lst)
            max_weights[t] = max(lst)
        else:
            min_weights[t] = 0.0
            max_weights[t] = 0.0

    curve = np.asarray(curve_values, dtype=np.float64)
    summary = curve_metrics(initial_capital_krw, curve)

    bench_series = pd.to_numeric(sim[benchmark["ticker"]], errors="coerce").dropna()
    bench_start = float(bench_series.iloc[0])
    bench_curve = (bench_series / bench_start * initial_capital_krw).to_numpy(dtype=np.float64)
    benchmark_summary = curve_metrics(initial_capital_krw, bench_curve)

    positions: list[dict[str, Any]] = []
    for item in clean_tickers:
        ticker = item["ticker"]
        series = pd.to_numeric(sim[ticker], errors="coerce").dropna()
        if len(series) < 2:
            continue
        start_price = float(series.iloc[0])
        end_price = float(series.iloc[-1])
        metrics = curve_metrics(start_price, series.to_numpy(dtype=np.float64))
        mdd_peak, mdd_trough, _ = mdd_span(series.to_numpy(dtype=np.float64))
        positions.append(
            {
                "ticker": ticker,
                "name": item.get("name") or ticker,
                "buy_date": series.index[0].date().isoformat(),
                "late_entry": series.index[0] > start_date,
                "shares": 0,
                "buy_price": round(start_price, 4),
                "last_price": round(end_price, 4),
                "return_pct": round((end_price / start_price - 1.0) * 100.0, 2) if start_price > 0 else 0.0,
                "mdd_pct": round(metrics["mdd_pct"], 2),
                "mdd_start": series.index[mdd_peak].date().isoformat(),
                "mdd_end": series.index[mdd_trough].date().isoformat(),
                "sortino": round(metrics["sortino"], 2),
                "profit": round(float(profit_by_ticker.get(ticker, 0.0)), 0),
                "value": round(float(asset_values.get(ticker, 0.0)), 0),
                "min_weight": round(min_weights.get(ticker, 0.0), 1),
                "max_weight": round(max_weights.get(ticker, 0.0), 1),
            }
        )

    return {
        "months": months,
        "rebalance": rebalance,
        "buy_date": start_date.date().isoformat(),
        "end_date": sim.index[-1].date().isoformat(),
        "has_late_entry": any(position["late_entry"] for position in positions),
        "initial_capital": round(initial_capital_krw, 0),
        "final_value": round(float(curve[-1]), 0),
        "summary": {key: round(value, 2) for key, value in summary.items()},
        "benchmark": {
            **benchmark,
            "summary": {key: round(value, 2) for key, value in benchmark_summary.items()},
        },
        "positions": positions,
        "cash_min_weight": round(min_weights.get("__CASH__", 0.0), 1),
        "cash_max_weight": round(max_weights.get("__CASH__", 0.0), 1),
        "chart": {
            "dates": [date.date().isoformat() for date in sim.index],
            "portfolio_value": [round(float(value), 0) for value in curve],
            "benchmark_value": [round(float(value), 0) for value in bench_curve],
            "portfolio_pct": [round((value / initial_capital_krw - 1.0) * 100.0, 3) for value in curve],
            "benchmark_pct": [round((value / initial_capital_krw - 1.0) * 100.0, 3) for value in bench_curve],
        },
        "weight_history": weight_history,
        "weight_items": [
            {"key": item["ticker"], "label": item.get("name") or item["ticker"]}
            for item in clean_tickers
        ]
        + [{"key": "__CASH__", "label": "현금"}],
        "missing_tickers": missing,
    }


def _calculate_benchmark_regimes(bench_df: pd.DataFrame) -> pd.Series:
    """벤치마크 OHLC DataFrame을 입력받아 날짜별 레짐(accel_up, accel_down, neutral) 시리즈를 계산해 반환한다."""
    from utils.market_trend_service import (
        _calculate_supertrend,
        _regime_step,
    )
    from config import (
        MARKET_TREND_REGIME_BUFFER_PCT_DEFAULT,
        MARKET_TREND_REGIME_MA_TYPE,
        MARKET_TREND_REGIME_SHORT_MA_DAYS,
        MARKET_TREND_SUPERTREND_MULTIPLIER_DEFAULT,
        MARKET_TREND_SUPERTREND_PERIOD,
    )
    from utils.moving_averages import calculate_moving_average

    df = bench_df.dropna(subset=["Close"]).sort_index()
    if df.empty:
        return pd.Series(dtype=object)

    close_series = df["Close"]
    
    # 1. 20일 이동평균선(MA) 계산
    ma_days = MARKET_TREND_REGIME_SHORT_MA_DAYS
    try:
        ma_series = calculate_moving_average(close_series, ma_days, MARKET_TREND_REGIME_MA_TYPE).fillna(close_series)
    except Exception:
        ma_series = close_series
    
    # 2. 휩소 방지용 부드러운 종가 (20일 MA)
    close_smooth = ma_series
    
    # 3. 슈퍼트렌드 계산 (벤치마크는 시장추세 지수 목록에 없으므로 공통 기본 곱수를 쓴다)
    supertrend_dir_series = None
    try:
        st_df = _calculate_supertrend(
            df,
            period=MARKET_TREND_SUPERTREND_PERIOD,
            multiplier=MARKET_TREND_SUPERTREND_MULTIPLIER_DEFAULT,
        )
        if st_df is not None and "direction" in st_df.columns:
            supertrend_dir_series = st_df["direction"]
    except Exception:
        pass

    # 4. 일별 레짐 계산 (ST 방향 주도 + MA±버퍼 보조)
    regimes = {}

    st_dir_map = {}
    if supertrend_dir_series is not None:
        st_dir_map = supertrend_dir_series.dropna().to_dict()

    for idx in range(len(df)):
        date_value = df.index[idx]
        st_dir = st_dir_map.get(date_value)
        if st_dir is not None:
            st_dir = int(st_dir)

        regimes[date_value] = _regime_step(
            float(close_series.iloc[idx]),
            float(close_smooth.iloc[idx]),
            st_dir,
            MARKET_TREND_REGIME_BUFFER_PCT_DEFAULT,
        )

    return pd.Series(regimes)
