"""탑픽 포트폴리오 설정과 목표 비중 계산 서비스."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from config import MIN_TRADING_DAYS, TRADING_DAYS_PER_MONTH
from core.strategy.scoring import build_composite_rank_scores
from utils.cache_utils import (
    get_all_ticker_type_lookup_keys,
    load_cached_close_series_bulk,
    load_cached_close_series_bulk_with_fallback,
)
from utils.logger import get_app_logger
from utils.moving_averages import calculate_moving_average
from utils.perf_metrics import curve_metrics, mdd_span

logger = get_app_logger()

COLLECTION = "top_pick_settings"
SETTINGS_ID = "default"

# 탑픽 설정 스키마 — 코드 기본값(silent default) 없음. 값은 전적으로 DB에서 온다.
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
MAX_TICKERS_LIMIT = 10
DEFAULT_BACKTEST_SETTINGS: dict[str, Any] = {
    "months": 12,
    "rebalance": "none",
    "initial_amount_manwon": 10000,
}
ALLOWED_BACKTEST_MONTHS = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 24, 36}
ALLOWED_BACKTEST_REBALANCE = {"none", "weekly", "monthly", "quarterly", "yearly"}
ALLOWED_WEIGHT_MODES = {"variable", "fixed"}


def _db():
    from utils.db_manager import get_db_connection

    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결 실패 (top_pick_settings)")
    return db


def _clean_weight_mode(value: Any) -> str:
    mode = str(value or "").strip().lower()
    if mode not in ALLOWED_WEIGHT_MODES:
        raise ValueError("비중 방식은 variable 또는 fixed여야 합니다.")
    return mode


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
        if not ticker or not name or ticker in seen:
            continue
        if len(clean) >= MAX_TICKERS_LIMIT:
            raise ValueError(f"탑픽 편입 ETF는 최대 {MAX_TICKERS_LIMIT}개까지 등록할 수 있습니다.")
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
        raise ValueError(f"탑픽 편입 ETF 슬롯은 최대 {MAX_TICKERS_LIMIT}개까지 저장할 수 있습니다.")

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
        raise ValueError(
            f"탑픽 설정값이 없습니다(DB 미설정): {', '.join(missing_required)}. 설정 화면에서 저장해주세요."
        )
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
            raise ValueError(f"존재하지 않는 탑픽 적용 계좌입니다: {account_id}")
    cleaned["ACCOUNT_ID"] = account_id

    return cleaned


def _with_account_top_pick_basis(settings: dict[str, Any], *, weight_mode: str | None = None) -> dict[str, Any]:
    """설정 응답에 계좌별 이평선 기준을 결합한다.

    weight_mode="fixed"(고정 보유)는 이평선·종목풀 연결이 필요 없으므로 그 필드는 비운 채 반환한다
    (계좌-풀 연결을 요구하지 않는다). trend 계좌는 연결 풀 이평선 + 보유개수를 붙인다.
    """
    account_id = str(settings.get("ACCOUNT_ID") or "").strip()
    if weight_mode == "fixed":
        ma_context = {"POOL_TICKER_TYPE": None, "POOL_NAME": None, "SHORT_MA_DAYS": None, "LONG_MA_DAYS": None}
    else:
        ma_context = _load_account_trend_context(account_id) or _load_account_pool_ma_context(account_id)
    return {**settings, **ma_context}


def _load_account_pool_ma_context(account_id: str) -> dict[str, Any]:
    """계좌에 연결된 단일 종목풀의 이평선 설정을 탑픽 계산 기준으로 반환한다."""
    normalized_account_id = str(account_id or "").strip()
    if not normalized_account_id:
        return {"POOL_TICKER_TYPE": None, "POOL_NAME": None, "SHORT_MA_DAYS": None, "LONG_MA_DAYS": None}

    from utils.settings_loader import get_account_settings, get_ticker_type_settings

    account_settings = get_account_settings(normalized_account_id)
    ticker_types = account_settings.get("ticker_types")
    if not isinstance(ticker_types, list) or len(ticker_types) != 1 or not str(ticker_types[0] or "").strip():
        raise ValueError(f"계좌 '{normalized_account_id}'에 연결된 종목풀이 1개여야 합니다.")

    ticker_type = str(ticker_types[0]).strip().lower()
    pool_settings = get_ticker_type_settings(ticker_type)
    try:
        short_ma_days = int(pool_settings["SHORT_MA_DAYS"])
        long_ma_days = int(pool_settings["LONG_MA_DAYS"])
    except KeyError as exc:
        raise ValueError(f"종목풀 '{ticker_type}'에 이평선 설정(SHORT_MA_DAYS/LONG_MA_DAYS)이 없습니다.") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"종목풀 '{ticker_type}'의 이평선 설정은 정수여야 합니다.") from exc
    if short_ma_days <= 0 or long_ma_days <= 0:
        raise ValueError(f"종목풀 '{ticker_type}'의 이평선 설정은 1일 이상이어야 합니다.")

    return {
        "POOL_TICKER_TYPE": ticker_type,
        "POOL_NAME": str(pool_settings.get("name") or ticker_type),
        "SHORT_MA_DAYS": short_ma_days,
        "LONG_MA_DAYS": long_ma_days,
    }


def _load_account_trend_context(account_id: str) -> dict[str, Any] | None:
    """trend 계좌면 연결 종목풀의 선정 기준(이평선/풀명) + 계좌 보유개수(hold_count)를 반환한다.

    후보 출처·선정 기준(이평선·게이팅·순위)은 계좌에 연결된 종목풀(ticker_types)을 그대로 쓰고,
    계좌 strategy 는 보유개수(hold_count) 하나만 소유한다. account_type 이 trend 가 아니거나
    hold_count 가 없으면 None을 반환해 호출부가 기존 풀 기반 방식으로 동작하게 한다.
    """
    normalized_account_id = str(account_id or "").strip()
    if not normalized_account_id:
        return None

    from utils.settings_loader import get_account_settings

    account_settings = get_account_settings(normalized_account_id)
    if account_settings.get("account_type") != "trend":
        return None
    strategy = account_settings.get("strategy")
    if not isinstance(strategy, dict):
        return None
    try:
        hold_count = int(strategy["hold_count"])
    except (KeyError, TypeError, ValueError):
        return None
    # 이평선·풀명은 연결된 종목풀에서 그대로 가져온다(계좌가 따로 소유하지 않는다).
    pool_context = _load_account_pool_ma_context(normalized_account_id)
    return {**pool_context, "HOLD_COUNT": hold_count}


def _load_pool_universe_tickers(pool_ticker_type: str) -> list[dict[str, Any]]:
    """trend 후보군 = 연결 종목풀의 종목 전체. get_etfs 로 풀 종목을 읽어 계산 유니버스로 만든다."""
    pool = str(pool_ticker_type or "").strip().lower()
    if not pool:
        raise ValueError("추세 계좌의 후보 종목풀(ticker_types)이 지정되지 않았습니다.")

    from utils.settings_loader import get_ticker_type_settings
    from utils.stock_list_io import get_etfs

    country_code = str(get_ticker_type_settings(pool).get("country_code") or "").strip().lower()
    universe: list[dict[str, Any]] = []
    for item in get_etfs(pool):
        ticker = str(item.get("ticker") or "").strip().upper()
        if not ticker:
            continue
        universe.append(
            {
                "ticker": ticker,
                "name": str(item.get("name") or ticker),
                "ticker_type": pool,
                "country_code": country_code or None,
            }
        )
    if not universe:
        raise ValueError(f"종목풀 '{pool}'에 종목이 없습니다.")
    return universe


def _build_top_pick_ma_rule(settings: dict[str, Any]) -> dict[str, Any]:
    long_ma_days = settings.get("LONG_MA_DAYS")
    if long_ma_days is None:
        account_id = str(settings.get("ACCOUNT_ID") or "").strip()
        long_ma_days = _load_account_pool_ma_context(account_id)["LONG_MA_DAYS"]
    return {"order": 1, "long_ma_days": int(long_ma_days), "score_column": "이격"}


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


def _allocate_trend_hold_weights(
    long_deviation_pct: dict[str, float | None],
    short_deviation_pct: dict[str, float | None],
    hold_count: int,
) -> dict[str, float]:
    """이격(장기) 상위 hold_count개를 균등 슬롯(1/hold_count)으로 보유한다 (trend 계좌, Phase 2).

    보유 조건: 장기이격>0 이고 단기이격>=0 인 종목만 후보. 장기 이평선은 종목 선택, 단기
    이평선은 손절/익절 담당(docs/account_types_plan.md 선정 규칙과 동일). 후보가 hold_count
    보다 적으면 남는 슬롯은 현금으로 남긴다(억지로 채우지 않는다) — 종목풀 백테스트의
    top_n 방식과 동일한 관례.
    """
    if hold_count < 1:
        raise ValueError(f"보유종목수(hold_count)는 1 이상이어야 합니다: {hold_count}")

    eligible = [
        ticker
        for ticker, long_dev in long_deviation_pct.items()
        if long_dev is not None
        and long_dev > 0
        and short_deviation_pct.get(ticker) is not None
        and float(short_deviation_pct[ticker]) >= 0
    ]
    eligible.sort(key=lambda t: long_deviation_pct[t], reverse=True)
    selected = eligible[:hold_count]

    slot_weight = 1.0 / hold_count
    weights = {ticker: 0.0 for ticker in long_deviation_pct}
    for ticker in selected:
        weights[ticker] = slot_weight
    weights["__CASH__"] = max(0.0, 1.0 - slot_weight * len(selected))
    return weights


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
    if months not in ALLOWED_BACKTEST_MONTHS:
        raise ValueError(
            f"백테스트 기간(개월)은 {', '.join(map(str, sorted(ALLOWED_BACKTEST_MONTHS)))} 중 하나여야 합니다."
        )

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
    stored_keys = (*SETTING_KEYS, "CASH_MAX_WEIGHT")
    settings = _clean_settings({}, base={key: doc[key] for key in stored_keys if doc and doc.get(key) is not None})
    doc_weight_mode = _clean_weight_mode((doc or {}).get("weight_mode"))
    settings = _with_account_top_pick_basis(settings, weight_mode=doc_weight_mode)
    backtest_settings = _clean_backtest_settings((doc or {}).get("backtest_settings"), require_benchmark=False)
    updated_at = (doc or {}).get("updated_at")
    approved_at = (doc or {}).get("approved_at")
    tickers = _clean_ticker_slots((doc or {}).get("tickers"))
    approved_weights = (doc or {}).get("approved_weights") or None
    return {
        "tickers": tickers,
        "weight_mode": doc_weight_mode,
        "settings": settings,
        "backtest_settings": backtest_settings,
        "approved_weights": _enrich_weight_rows_with_returns(
            approved_weights,
            _clean_tickers(tickers),
            settings,
            metric_months=int(backtest_settings["months"]),
            weight_mode=doc_weight_mode,
        ),
        "approved_at": (
            approved_at.replace(tzinfo=timezone.utc) if approved_at.tzinfo is None else approved_at
        ).isoformat()
        if isinstance(approved_at, datetime)
        else None,
        "updated_at": (updated_at.replace(tzinfo=timezone.utc) if updated_at.tzinfo is None else updated_at).isoformat()
        if isinstance(updated_at, datetime)
        else None,
    }


def list_top_pick_accounts() -> list[str]:
    """탑픽 설정 문서가 존재하는 계좌 id 목록 (문서 _id = account_id)."""
    return [str(doc["_id"]) for doc in _db()[COLLECTION].find({}, {"_id": 1})]


def _resolve_account_id(account_id: str | None) -> str:
    """account_id 미지정 시: 계좌가 정확히 1개면 그걸 쓰고, 0개/2개 이상이면 명시적 에러(임의 선택 금지)."""
    if account_id and str(account_id).strip():
        return str(account_id).strip()
    accounts = list_top_pick_accounts()
    if len(accounts) == 1:
        return accounts[0]
    if not accounts:
        raise ValueError("등록된 탑픽 계좌가 없습니다.")
    raise ValueError(f"탑픽 계좌를 지정해야 합니다(여러 계좌 존재): {', '.join(accounts)}")


def load_top_pick_settings_for_edit(account_id: str) -> dict[str, Any]:
    """설정 편집 화면 전용 로더.

    문서가 있으면 검증된 설정을, 없으면 **초기(빈) 상태**를 반환한다.
    저장 전까지 DB 에 아무것도 쓰지 않으며, 저장 시 save 가 필수값을 검증한다(fail loud).
    """
    resolved = str(account_id or "").strip()
    if not resolved:
        raise ValueError("계좌를 지정해야 합니다.")
    doc = _db()[COLLECTION].find_one({"_id": resolved})
    if doc is not None:
        return _serialize_doc(doc)
    # 저장된 문서 없음 → 모든 값 미설정인 초기 상태(계좌 id 만 채움).
    # 초기 weight_mode 는 계좌 타입(account_settings.account_type)을 따른다 — fixed 계좌는
    # 종목풀 연결이 없어도(이번 개편으로 더 이상 요구하지 않음) 화면이 열려야 한다.
    from utils.settings_loader import get_account_settings

    initial_weight_mode = "fixed" if get_account_settings(resolved).get("account_type") == "fixed" else "variable"
    return {
        "tickers": [],
        "weight_mode": initial_weight_mode,
        "settings": _with_account_top_pick_basis(
            {key: (resolved if key == "ACCOUNT_ID" else None) for key in SETTING_KEYS},
            weight_mode=initial_weight_mode,
        ),
        "backtest_settings": None,
        "approved_weights": None,
        "approved_at": None,
        "updated_at": None,
    }


def save_top_pick_settings(
    tickers: list[dict[str, Any]],
    weight_mode: str,
    settings: dict[str, Any] | None = None,
    backtest_settings: dict[str, Any] | None = None,
    account_id: str | None = None,
) -> dict[str, Any]:
    resolved = _resolve_account_id(account_id)
    clean_weight_mode = _clean_weight_mode(weight_mode)
    ticker_slots = _clean_ticker_slots(tickers)
    clean_tickers = _clean_tickers(ticker_slots)
    if len(clean_tickers) < 1:
        raise ValueError("저장할 종목이 1개 이상 필요합니다.")

    # 신규 계좌면 기존 문서가 없다(base 비움). 요청에 필수 설정이 다 있어야 저장된다.
    existing_doc = _db()[COLLECTION].find_one({"_id": resolved}) or {}
    stored_keys = (*SETTING_KEYS, "CASH_MAX_WEIGHT")
    settings_base = {key: existing_doc[key] for key in stored_keys if existing_doc.get(key) is not None}
    clean_settings = _clean_settings(settings, base=settings_base)
    # 자산 헬퍼는 자유 테스트 포트폴리오라 계좌별 종목수 상한을 두지 않는다 —
    # 슬롯 수를 실제 종목 수에 맞춰 저장한다(전역 하드 상한 MAX_TICKERS_LIMIT 만 유지).
    clean_settings["VARIABLE_TICKERS"] = 0
    clean_settings["FIXED_TICKERS"] = len(clean_tickers)
    clean_settings["MAX_TICKERS"] = len(clean_tickers)
    clean_backtest_settings = _clean_backtest_settings(
        backtest_settings,
        base=existing_doc.get("backtest_settings") or DEFAULT_BACKTEST_SETTINGS,
        require_benchmark=False,
    )
    updated_at = datetime.now(timezone.utc)
    _db()[COLLECTION].update_one(
        {"_id": resolved},
        {
            "$set": {
                "tickers": ticker_slots,
                "weight_mode": clean_weight_mode,
                **clean_settings,
                "backtest_settings": clean_backtest_settings,
                "updated_at": updated_at,
            },
            "$unset": {"CASH_MAX_WEIGHT": "", "START_AMOUNT_MANWON": "", "START_DATE": "", "MA_MONTHS": ""},
        },
        upsert=True,
    )
    return {
        "tickers": ticker_slots,
        "weight_mode": clean_weight_mode,
        "settings": _with_account_top_pick_basis(clean_settings, weight_mode=clean_weight_mode),
        "backtest_settings": clean_backtest_settings,
        "approved_weights": existing_doc.get("approved_weights"),
        "approved_at": existing_doc.get("approved_at"),
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
        if str(item.get("ticker") or "").strip().upper()
        and str(item.get("ticker") or "").strip().upper() not in series_map
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
            "return_6m_pct": _calculate_period_return(series, 6, eval_date),
            "return_12m_pct": _calculate_period_return(series, 12, eval_date),
        }
    return result


def _build_mdd_map(close_frame: pd.DataFrame, window_months: int) -> dict[str, float | None]:
    """Sortino 설정과 동일한 N개월 거래일 구간의 종목별 MDD를 반환한다."""
    if close_frame.empty:
        return {}

    window = max(2, int(window_months) * int(TRADING_DAYS_PER_MONTH))
    result: dict[str, float | None] = {}
    for ticker in close_frame.columns:
        series = pd.to_numeric(close_frame[ticker], errors="coerce").dropna().iloc[-window:]
        if len(series) < 2:
            result[str(ticker)] = None
            continue
        running_max = series.cummax()
        drawdowns = (series / running_max) - 1.0
        result[str(ticker)] = round(float(drawdowns.min()) * 100.0, 2)
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
    # 보유내역 티커 표시 접두사(KR:, ASX:)를 제거해 탑픽 티커와 매칭 가능한 형태로 정규화한다.
    return str(value or "").strip().upper().replace("KR:", "").replace("ASX:", "")


def _load_top_pick_account_snapshot(account_id: str) -> dict[str, Any]:
    if not account_id:
        raise ValueError("탑픽 적용 계좌를 선택해주세요.")

    from utils.holdings_detail_service import load_all_holdings_detail

    payload = load_all_holdings_detail(account_id)
    summaries = [row for row in payload.get("account_summaries", []) if str(row.get("account_id") or "") == account_id]
    if not summaries:
        raise ValueError(f"탑픽 적용 계좌 정보를 찾을 수 없습니다: {account_id}")

    summary = summaries[0]
    currency = str(summary.get("currency") or "KRW").strip().upper()
    # KRW 계좌는 KRW 금액 그대로, 그 외(AUD 등)는 계좌 환종 네이티브 금액으로 통일한다.
    # (가격 캐시의 현재가가 네이티브라 금액·수량 계산이 같은 통화로 맞아떨어져야 한다)
    native_mode = currency != "KRW"

    account_rows = [row for row in payload.get("rows", []) if str(row.get("account_id") or "") == account_id]
    holdings: dict[str, dict[str, Any]] = {}
    total_native_value = 0.0
    for row in account_rows:
        ticker = _normalize_kor_holding_ticker(row.get("ticker"))
        if not ticker:
            continue
        # IS는 수동으로 관리하는 고정 자산이므로 탑픽 운용 자산과 성과 계산에서 제외한다.
        if ticker == "IS":
            continue
        quantity = int(float(row.get("quantity") or 0))
        valuation_krw = float(row.get("valuation_krw") or 0)
        if native_mode:
            price_native = float(row.get("current_price_num") or 0)
            value = quantity * price_native
            # KRW 환산에 쓰인 환율을 역산해 손익/매입금액도 같은 통화로 복원한다.
            row_rate = (valuation_krw / value) if value > 0 else 0.0
            pnl = float(row.get("pnl_krw_num") or 0.0) / row_rate if row_rate > 0 else 0.0
            buy_amount = float(row.get("buy_amount_krw") or 0.0) / row_rate if row_rate > 0 else 0.0
            total_native_value += value
        else:
            value = valuation_krw
            pnl = float(row.get("pnl_krw_num") or 0.0)
            buy_amount = float(row.get("buy_amount_krw") or 0.0)
        current = holdings.setdefault(
            ticker,
            {
                "current_quantity": 0,
                "current_amount_krw": 0,
                "name": str(row.get("name") or ticker),
                "pnl_krw": 0.0,
                "buy_amount_krw": 0.0,
                "return_pct": 0.0,
                "daily_change_pct": row.get("daily_change_pct"),
                "bucket": row.get("bucket"),
            },
        )
        current["current_quantity"] += quantity
        current["current_amount_krw"] += round(value, 2) if native_mode else int(round(value))
        current["pnl_krw"] += pnl
        current["buy_amount_krw"] += buy_amount
        if not current.get("name"):
            current["name"] = str(row.get("name") or ticker)
        if current.get("daily_change_pct") is None and row.get("daily_change_pct") is not None:
            current["daily_change_pct"] = row.get("daily_change_pct")

    for ticker, current in holdings.items():
        buy_amt = current["buy_amount_krw"]
        pnl = current["pnl_krw"]
        if buy_amt > 0:
            current["return_pct"] = round((pnl / buy_amt) * 100.0, 2)
        else:
            current["return_pct"] = 0.0

    if native_mode:
        cash_native = summary.get("cash_balance_native")
        if cash_native is None and float(summary.get("cash_balance_krw") or 0) > 0:
            raise ValueError(f"계좌 '{account_id}' 의 현금 원화({currency}) 잔액 정보가 없습니다.")
        cash_balance = float(cash_native or 0.0)
        account_amount = total_native_value + cash_balance
    else:
        cash_balance = float(summary.get("cash_balance_krw") or 0)
        account_amount = float(summary.get("total_assets_krw") or 0)

    return {
        "account_id": account_id,
        "account_name": summary.get("name") or account_id,
        "currency": currency,
        "account_amount_krw": round(account_amount, 2) if native_mode else int(round(account_amount)),
        "cash_balance_krw": round(cash_balance, 2) if native_mode else int(round(cash_balance)),
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
        "현금": "5. 현금",
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
        raise ValueError(f"탑픽에서 지원하지 않는 계좌 환종입니다: {account_currency or '-'}")
    holding_ticker_type, holding_country_code = market_by_currency[account_currency]
    current_price_map = _build_current_price_map(tickers, close_frame)
    target_asset_amount = 0.0
    target_tickers: set[str] = set()

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

        quantity = int(target_amount // current_price)
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


def _enrich_weight_rows_with_returns(
    payload: Any,
    tickers: list[dict[str, Any]],
    current_settings: dict[str, Any],
    *,
    metric_months: int,
    weight_mode: str | None = None,
) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        return payload

    close_frame, _missing = _load_close_frame(tickers)
    return_map = _build_return_map(close_frame)
    daily_change_map = _build_daily_change_map(tickers, close_frame)
    # 승인 당시 박제된 settings 스냅샷 — 옛 스냅샷엔 MAX_TICKERS 같은 신규 필드가 없다.
    # 계산에 필요한 신규 필드는 현재 설정으로 보완하고, 나머지는 스냅샷 값을 쓴다.
    snap_settings = payload.get("settings") if isinstance(payload.get("settings"), dict) else {}
    settings = _clean_settings({**current_settings, **snap_settings})
    is_fixed = weight_mode == "fixed"
    if is_fixed:
        settings = {**settings, "POOL_TICKER_TYPE": None, "POOL_NAME": None, "SHORT_MA_DAYS": None, "LONG_MA_DAYS": None}
    else:
        settings = {**settings, **_load_account_pool_ma_context(str(settings.get("ACCOUNT_ID") or "").strip())}
    eval_date = close_frame.index.max() if not close_frame.empty else None
    slope_map, alignment_map = (
        _build_current_ma_state_maps(
            close_frame,
            eval_date,
            short_ma_days=int(settings["SHORT_MA_DAYS"]),
            long_ma_days=int(settings["LONG_MA_DAYS"]),
        )
        if eval_date is not None and not is_fixed
        else ({}, {})
    )
    sortino_raw_frame = _compute_sortino_raw_frame(close_frame, metric_months)
    mdd_map = _build_mdd_map(close_frame, metric_months)
    sortino_raw_row = (
        sortino_raw_frame.loc[eval_date]
        if eval_date is not None and eval_date in sortino_raw_frame.index
        else pd.Series(dtype=float)
    )
    ticker_to_bucket = {str(t.get("ticker")).upper(): t.get("bucket") for t in tickers}

    enriched_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        ticker = str(row.get("ticker") or "").strip().upper()
        sortino_raw_value = sortino_raw_row.get(ticker)
        bucket = ticker_to_bucket.get(ticker)

        enriched_row = {
            **row,
            "bucket": int(bucket) if bucket is not None else None,
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
            "deviation_pct": row.get("deviation_pct", row.get("trend_pct")),
            "slope_pct": None if slope_map.get(ticker) is None else round(float(slope_map[ticker]), 2),
            "alignment": alignment_map.get(ticker),
            "sortino": None if pd.isna(sortino_raw_value) else round(float(sortino_raw_value), 2),
        }
        enriched_row.pop("nickname", None)
        enriched_rows.append(enriched_row)
    _normalize_target_weight_pct_rows(enriched_rows)
    trade_summary = (
        _apply_trade_plan(enriched_rows, settings=settings, tickers=tickers, close_frame=close_frame)
        if settings.get("ACCOUNT_ID")
        else {}
    )
    return {**payload, "rows": enriched_rows, "trade_summary": trade_summary}


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


def calculate_top_pick_weights_for(
    tickers: list[dict[str, Any]],
    settings: dict[str, Any],
    weight_mode: str = "variable",
    *,
    metric_months: int,
) -> dict[str, Any]:
    # weight_mode="fixed"(고정 보유)는 추세/이평선이 필요 없다 — 계좌-풀 연결도 요구하지 않는다.
    # 그 외(variable/trend)는 계좌에 연결된 풀의 이평선 설정을 쓴다.
    # trend 계좌는 후보군 = 연결 풀의 종목 전체(get_etfs)로 대체한다.
    account_id = str(settings.get("ACCOUNT_ID") or "").strip()
    if not account_id:
        raise ValueError("탑픽 비중 계산에는 적용 계좌가 필요합니다.")

    is_fixed = weight_mode == "fixed"
    if is_fixed:
        excluded_fixed_tickers: list[str] = []
    else:
        ma_context = _load_account_trend_context(account_id) or _load_account_pool_ma_context(account_id)
        settings = {**settings, **ma_context}
        if settings.get("HOLD_COUNT") is not None:
            tickers = _load_pool_universe_tickers(settings["POOL_TICKER_TYPE"])
        tickers, excluded_fixed_tickers = _filter_rank_excluded_tickers(tickers, settings)
    if len(tickers) < 3:
        suffix = f" 고정 종목 제외: {', '.join(excluded_fixed_tickers)}" if excluded_fixed_tickers else ""
        raise ValueError(f"탑픽 비중 계산에는 고정 종목 제외 후 확인된 종목이 3개 이상 필요합니다.{suffix}")

    close_frame, missing = _load_close_frame(tickers)
    if close_frame.empty:
        raise ValueError("탑픽 종목의 가격 캐시가 없습니다.")

    mdd_map = _build_mdd_map(close_frame, metric_months)
    eval_date = close_frame.index.max()
    sortino_raw_frame = _compute_sortino_raw_frame(close_frame, metric_months)
    sortino_raw_row = (
        sortino_raw_frame.loc[eval_date] if eval_date in sortino_raw_frame.index else pd.Series(dtype=float)
    )
    return_map = _build_return_map(close_frame)
    daily_change_map = _build_daily_change_map(tickers, close_frame)

    if is_fixed:
        # 고정 보유는 추세 점수·이평선 배열이 필요 없다(관련 표시 컬럼은 전부 None으로 남긴다).
        composite_row: pd.Series = pd.Series(dtype=float)
        trend_row: pd.Series = pd.Series(dtype=float)
        slope_map: dict[str, float] = {}
        alignment_map: dict[str, str | None] = {}
    else:
        ma_rules = [_build_top_pick_ma_rule(settings)]
        composite_frame, trend_by_order, _ = build_composite_rank_scores(close_frame, ma_rules)
        composite_row = composite_frame.loc[eval_date] if eval_date in composite_frame.index else pd.Series(dtype=float)
        trend_frame = trend_by_order[1]
        trend_row = trend_frame.loc[eval_date] if eval_date in trend_frame.index else pd.Series(dtype=float)
        slope_map, alignment_map = _build_current_ma_state_maps(
            close_frame,
            eval_date,
            short_ma_days=int(settings["SHORT_MA_DAYS"]),
            long_ma_days=int(settings["LONG_MA_DAYS"]),
        )

    # trend 계좌(Phase 2, HOLD_COUNT 있음)만 단기이격 게이트를 계산한다 — 그 외는 불필요한 연산.
    hold_count = settings.get("HOLD_COUNT")
    if hold_count is not None:
        short_dev_frame = _compute_ma_deviation_frame(close_frame, int(settings["SHORT_MA_DAYS"]))
        short_dev_row = short_dev_frame.loc[eval_date] if eval_date in short_dev_frame.index else pd.Series(dtype=float)
    else:
        short_dev_row = pd.Series(dtype=float)

    rows: list[dict[str, Any]] = []
    ticker_meta = {item["ticker"]: item for item in tickers}
    excluded_reasons: list[str] = []
    deviation_pct_by_ticker: dict[str, float | None] = {}
    short_deviation_pct_by_ticker: dict[str, float | None] = {}
    for ticker in [item["ticker"] for item in tickers]:
        sortino_raw_value = sortino_raw_row.get(ticker)
        meta = ticker_meta[ticker]

        if is_fixed:
            # 고정 보유는 가격 캐시 존재만 확인한다(이격/추세 요구 없음).
            close_count = int(close_frame[ticker].dropna().shape[0]) if ticker in close_frame.columns else 0
            if close_count <= 0:
                excluded_reasons.append(f"{ticker}: 가격 캐시 없음")
            point_deviation = None
            score = None
            deviation_pct_value = None
            slope_value = None
        else:
            deviation_score_value = composite_row.get(ticker)
            point_deviation = None if pd.isna(deviation_score_value) else float(deviation_score_value)
            score = point_deviation
            if score is None:
                close_count = int(close_frame[ticker].dropna().shape[0]) if ticker in close_frame.columns else 0
                if close_count <= 0:
                    excluded_reasons.append(f"{ticker}: 가격 캐시 없음")
                elif close_count < int(MIN_TRADING_DAYS):
                    excluded_reasons.append(f"{ticker}: 가격 데이터 {close_count}개로 부족(최소 {MIN_TRADING_DAYS}개)")
                else:
                    excluded_reasons.append(f"{ticker}: 이격 계산 불가")

            deviation_pct = trend_row.get(ticker)
            deviation_pct_value = None if pd.isna(deviation_pct) else float(deviation_pct)
            if score is not None and deviation_pct_value is None:
                excluded_reasons.append(f"{ticker}: 이격 계산 불가")
            deviation_pct_by_ticker[ticker] = deviation_pct_value
            if hold_count is not None:
                short_dev_value = short_dev_row.get(ticker)
                short_deviation_pct_by_ticker[ticker] = None if pd.isna(short_dev_value) else float(short_dev_value)
            slope_value = slope_map.get(ticker)

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

    if weight_mode == "fixed":
        fixed_weights = {}
        for item in tickers:
            tk = item["ticker"]
            fixed_weights[tk] = float(item.get("fixed_weight_pct") or 0.0) / 100.0

        sum_fixed = sum(fixed_weights.values())
        if sum_fixed > 1.0:
            for tk in fixed_weights:
                fixed_weights[tk] /= sum_fixed
            cash_weight = 0.0
        else:
            cash_weight = 1.0 - sum_fixed
        weights = {**fixed_weights, "__CASH__": cash_weight}
    elif hold_count is not None:
        weights = _allocate_trend_hold_weights(deviation_pct_by_ticker, short_deviation_pct_by_ticker, int(hold_count))
    else:
        weights = _allocate_deviation_filtered_weights(deviation_pct_by_ticker, float(settings["STOCK_MAX_WEIGHT"]))

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

    # 1. top-pick-settings 에 저장된 tickers 순서 로드
    ordered_tickers = []
    try:
        doc = _db()[COLLECTION].find_one({"_id": str(settings.get("ACCOUNT_ID") or "").strip()})
        if doc:
            tickers_list = doc.get("tickers") or []
            ordered_tickers = [str(t.get("ticker") or "").strip().upper().replace("KR:", "") for t in tickers_list]
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
        "excluded_fixed_tickers": excluded_fixed_tickers,
        "trade_summary": trade_summary,
    }


def run_top_pick_weights(
    tickers: list[dict[str, Any]],
    settings: dict[str, Any] | None = None,
    backtest_settings: dict[str, Any] | None = None,
    weight_mode: str = "variable",
) -> dict[str, Any]:
    clean_settings = _clean_settings(settings)
    account_id = str(clean_settings.get("ACCOUNT_ID") or "").strip()
    # trend 계좌는 후보군 = 연결 풀 종목 전체(전달받은 tickers 는 무시).
    trend_context = None if weight_mode == "fixed" else _load_account_trend_context(account_id)
    if trend_context is not None:
        # 풀 유니버스는 이미 정제된 형태이며 20개 상한(선택 종목용)을 적용하지 않는다.
        clean_tickers = _load_pool_universe_tickers(trend_context["POOL_TICKER_TYPE"])
    else:
        clean_tickers = _clean_tickers(tickers)
    if len(clean_tickers) < 1:
        raise ValueError("계산할 종목이 1개 이상 필요합니다.")
    response_settings = _with_account_top_pick_basis(clean_settings, weight_mode=weight_mode)
    clean_backtest = _clean_backtest_settings(backtest_settings, require_benchmark=False)
    return calculate_top_pick_weights_for(
        clean_tickers,
        response_settings,
        weight_mode=weight_mode,
        metric_months=int(clean_backtest["months"]),
    )


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
    ma_rules = [_build_top_pick_ma_rule(settings)]
    composite_frame, trend_by_order, _ = build_composite_rank_scores(close_frame, ma_rules)
    return composite_frame, trend_by_order


def _calculate_top_pick_weights_on_date(
    eval_date: pd.Timestamp,
    tickers: list[dict[str, Any]],
    settings: dict[str, Any],
    composite_frame: pd.DataFrame,
    trend_frame: pd.DataFrame,
) -> dict[str, float] | None:
    eligible_dates = composite_frame.index[composite_frame.index <= eval_date]
    if eligible_dates.empty:
        return None

    score_date = eligible_dates.max()
    composite_row = composite_frame.loc[score_date]
    trend_row = trend_frame.loc[score_date] if score_date in trend_frame.index else pd.Series(dtype=float)
    deviation_pct_by_ticker: dict[str, float | None] = {}
    for item in tickers:
        ticker = str(item.get("ticker") or "").strip().upper()
        score_value = composite_row.get(ticker)
        deviation_pct = trend_row.get(ticker)
        if pd.isna(score_value) or pd.isna(deviation_pct):
            deviation_pct_by_ticker[ticker] = None
        elif float(deviation_pct) > 0:
            deviation_pct_by_ticker[ticker] = float(deviation_pct)
        else:
            deviation_pct_by_ticker[ticker] = float(deviation_pct)
    return _allocate_deviation_filtered_weights(deviation_pct_by_ticker, float(settings["STOCK_MAX_WEIGHT"]))


def _calculate_trend_weights_on_date(
    eval_date: pd.Timestamp,
    tickers: list[dict[str, Any]],
    long_dev_frame: pd.DataFrame,
    short_dev_frame: pd.DataFrame,
    hold_count: int,
) -> dict[str, float] | None:
    """trend 계좌(Phase 2)의 특정 리밸런싱 날짜 비중 — 이격 상위 hold_count개, 균등 슬롯."""
    eligible_dates = long_dev_frame.index[long_dev_frame.index <= eval_date]
    if eligible_dates.empty:
        return None
    score_date = eligible_dates.max()
    long_row = long_dev_frame.loc[score_date]
    short_row = short_dev_frame.loc[score_date] if score_date in short_dev_frame.index else pd.Series(dtype=float)

    long_dev: dict[str, float | None] = {}
    short_dev: dict[str, float | None] = {}
    for item in tickers:
        ticker = str(item.get("ticker") or "").strip().upper()
        long_value = long_row.get(ticker)
        short_value = short_row.get(ticker)
        long_dev[ticker] = None if pd.isna(long_value) else float(long_value)
        short_dev[ticker] = None if pd.isna(short_value) else float(short_value)
    return _allocate_trend_hold_weights(long_dev, short_dev, hold_count)


def _resolve_slippage_by_ticker(clean_tickers: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    """종목별 (매수, 매도) 슬리피지 비율 — 종목풀 DB 설정을 그대로 사용한다.

    ticker_type 미상이거나 종목풀 슬리피지 설정이 없으면 임의 기본값 없이 명시적 에러.
    """
    from utils.settings_loader import get_ticker_type_settings

    rates: dict[str, tuple[float, float]] = {}
    problems: list[str] = []
    settings_cache: dict[str, dict[str, Any]] = {}
    for item in clean_tickers:
        ticker = str(item["ticker"])
        ticker_type = str(item.get("ticker_type") or "").strip().lower()
        if ticker_type:
            if ticker_type not in settings_cache:
                settings_cache[ticker_type] = get_ticker_type_settings(ticker_type)
        config = settings_cache.get(ticker_type, {})
        if (
            not ticker_type
            or config.get("BUY_SLIPPAGE_PCT") in (None, "")
            or config.get("SELL_SLIPPAGE_PCT") in (None, "")
        ):
            problems.append(f"{ticker}({ticker_type or '종목풀 미상'})")
            continue
        rates[ticker] = (float(config["BUY_SLIPPAGE_PCT"]) / 100.0, float(config["SELL_SLIPPAGE_PCT"]) / 100.0)
    if problems:
        raise ValueError(
            f"종목풀 슬리피지 설정이 없는 종목이 있습니다: {', '.join(problems)}. "
            "/pools-settings 에서 해당 종목풀의 매수/매도 슬리피지를 저장하세요."
        )
    return rates


def run_top_pick_backtest(
    tickers: list[dict[str, Any]],
    settings: dict[str, Any] | None = None,
    backtest_settings: dict[str, Any] | None = None,
    weight_mode: str = "variable",
) -> dict[str, Any]:
    # weight_mode="fixed"(고정 보유)는 이평선·종목풀 연결이 필요 없다. 그 외는 동일 슬롯 + 개별 이평선 필터.
    # 백테스트는 전달받은 "현재 종목"만 검증한다 — trend 라도 풀 전체를 재선정하지 않는다
    # (풀 백테스트는 종목풀 화면에 별도로 있음).
    clean_tickers = _clean_tickers(tickers)
    clean_settings = _with_account_top_pick_basis(_clean_settings(settings), weight_mode=weight_mode)
    clean_tickers, excluded_fixed_tickers = _filter_rank_excluded_tickers(clean_tickers, clean_settings)
    if len(clean_tickers) < 3:
        suffix = f" 고정 종목 제외: {', '.join(excluded_fixed_tickers)}" if excluded_fixed_tickers else ""
        raise ValueError(f"탑픽 백테스트에는 고정 종목 제외 후 확인된 종목이 3개 이상 필요합니다.{suffix}")
    clean_backtest = _clean_backtest_settings(backtest_settings, require_benchmark=False)
    # 결과 비교용 벤치마크 — 계좌 설정(/account-settings)의 벤치마크를 쓴다(계좌별로 다름).
    account_id = str(clean_settings.get("ACCOUNT_ID") or "").strip()
    if not account_id:
        raise ValueError("탑픽 백테스트에는 적용 계좌가 필요합니다.")
    from utils.settings_loader import get_account_settings

    _account_bench = get_account_settings(account_id).get("benchmark")
    if not isinstance(_account_bench, dict) or not str(_account_bench.get("ticker") or "").strip():
        raise ValueError(f"계좌 '{account_id}' 설정에 벤치마크가 없습니다. 계좌 설정에서 등록해주세요.")
    benchmark = {
        "ticker": str(_account_bench["ticker"]).strip().upper(),
        "name": str(_account_bench.get("name") or _account_bench["ticker"]).strip(),
    }
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

    simulation_columns = list(dict.fromkeys(ticker_order + [benchmark["ticker"]]))
    simulation_frame = close_frame[simulation_columns].sort_index()
    simulation_frame = simulation_frame[simulation_frame.index >= start_target].dropna(how="all").ffill()
    simulation_frame = simulation_frame.dropna(subset=[benchmark["ticker"]])
    if simulation_frame.empty or len(simulation_frame) < 2:
        raise ValueError("탑픽 백테스트 기간의 가격 데이터가 부족합니다.")

    requested_rebalance_dates = _select_rebalance_dates(simulation_frame.index, rebalance)
    weights_by_date: dict[pd.Timestamp, dict[str, float]] = {}
    hold_count = clean_settings.get("HOLD_COUNT")

    if weight_mode == "fixed":
        fixed_weights = {item["ticker"]: float(item.get("fixed_weight_pct") or 0.0) / 100.0 for item in clean_tickers}
        sum_fixed = sum(fixed_weights.values())
        if sum_fixed > 1.0:
            for tk in fixed_weights:
                fixed_weights[tk] /= sum_fixed
            cash_weight = 0.0
        else:
            cash_weight = 1.0 - sum_fixed
        fixed_weights["__CASH__"] = cash_weight

        weights_by_date = {date: fixed_weights for date in requested_rebalance_dates}
    elif hold_count is not None:
        # trend 계좌(Phase 2): 계좌 소유 장기/단기 이평선으로 이격 프레임을 미리 계산해두고
        # 리밸런싱 날짜마다 그 값만 조회한다(기존 종목풀 기반 엔진과 별개 경로).
        long_dev_frame = _compute_ma_deviation_frame(candidate_close, int(clean_settings["LONG_MA_DAYS"]))
        short_dev_frame = _compute_ma_deviation_frame(candidate_close, int(clean_settings["SHORT_MA_DAYS"]))
        for date in requested_rebalance_dates:
            weights = _calculate_trend_weights_on_date(date, clean_tickers, long_dev_frame, short_dev_frame, int(hold_count))
            if weights is not None:
                weights_by_date[date] = weights
    else:
        composite_frame, trend_by_order = _build_top_pick_weight_engine(candidate_close, clean_tickers, clean_settings)
        trend_frame = trend_by_order[1]

        for date in requested_rebalance_dates:
            weights = _calculate_top_pick_weights_on_date(
                date,
                clean_tickers,
                clean_settings,
                composite_frame,
                trend_frame,
            )
            if weights is not None:
                weights_by_date[date] = weights

    if not weights_by_date:
        raise ValueError("백테스트 기간에 계산 가능한 탑픽 비중이 없습니다.")

    start_date = min(weights_by_date)
    sim = simulation_frame[simulation_frame.index >= start_date].copy()
    if len(sim) < 2:
        raise ValueError("탑픽 백테스트 시뮬레이션 기간이 부족합니다.")

    # 슬리피지(종목풀별 DB 설정) — 매수/매도 금액에 비용으로 차감한다.
    slippage_by_ticker = _resolve_slippage_by_ticker(clean_tickers)
    total_slippage_cost = 0.0

    def allocate_with_slippage(
        total_value: float, weights: dict[str, float], current_values: dict[str, float]
    ) -> tuple[dict[str, float], float, float, dict[str, float]]:
        """총자산을 목표비중으로 재배치하며 종목별 매수/매도 금액 × 슬리피지율을 비용으로 차감한다."""
        cost = 0.0
        costs_by_ticker: dict[str, float] = {}
        for ticker in ticker_order:
            target = total_value * float(weights.get(ticker, 0.0))
            delta = target - float(current_values.get(ticker, 0.0))
            buy_ratio, sell_ratio = slippage_by_ticker[ticker]
            ticker_cost = 0.0
            if delta > 0:
                ticker_cost = delta * buy_ratio
            elif delta < 0:
                ticker_cost = (-delta) * sell_ratio
            costs_by_ticker[ticker] = ticker_cost
            cost += ticker_cost
        net_total = total_value - cost
        values = {ticker: net_total * float(weights.get(ticker, 0.0)) for ticker in ticker_order}
        cash = net_total * float(weights.get("__CASH__", 0.0))
        return values, cash, cost, costs_by_ticker

    current_weights = weights_by_date[start_date]
    profit_by_ticker = {ticker: 0.0 for ticker in ticker_order}
    # 최초 편입 — 전액 현금에서 매수하므로 매수 슬리피지가 발생한다.
    asset_values, cash_value, initial_cost, initial_costs_by_ticker = allocate_with_slippage(
        initial_capital_krw, current_weights, {ticker: 0.0 for ticker in ticker_order}
    )
    total_slippage_cost += initial_cost
    for ticker, ticker_cost in initial_costs_by_ticker.items():
        profit_by_ticker[ticker] -= ticker_cost
    curve_values: list[float] = [initial_capital_krw]
    weight_history: list[dict[str, Any]] = []
    friday_history_dates = _select_friday_history_dates(sim.index)

    # 종목별/현금의 일별 실제 비중(%) 이력을 추적합니다.
    weights_history_by_ticker: dict[str, list[float]] = {ticker: [] for ticker in ticker_order}
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
            asset_values, cash_value, rebalance_cost, rebalance_costs_by_ticker = allocate_with_slippage(
                total_value, current_weights, asset_values
            )
            total_slippage_cost += rebalance_cost
            for ticker, ticker_cost in rebalance_costs_by_ticker.items():
                profit_by_ticker[ticker] -= ticker_cost

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
                "bucket": item.get("bucket"),
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
        "slippage": {
            "total_cost": round(total_slippage_cost, 0),
            "total_cost_pct": round(total_slippage_cost / initial_capital_krw * 100.0, 2),
        },
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
            {
                "key": item["ticker"],
                "label": item.get("name") or item["ticker"],
                "bucket": item.get("bucket"),
            }
            for item in clean_tickers
        ]
        + [{"key": "__CASH__", "label": "현금"}],
        "missing_tickers": missing,
        "excluded_fixed_tickers": excluded_fixed_tickers,
    }
