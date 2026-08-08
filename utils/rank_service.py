from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from threading import Lock
from time import monotonic
from typing import Any

import pandas as pd

from services.stock_cache_service import get_stock_cache_meta_map
from utils.data_loader import get_trading_days
from utils.rankings import (
    MONTHLY_RETURN_LABEL_COUNT,
    build_effective_ma_rules,
    build_ticker_type_rankings,
    get_recent_monthly_return_labels,
)
from utils.stock_list_io import get_etfs
from utils.ticker_registry import load_ticker_type_configs, pick_default_ticker_type

_RANK_DATA_CACHE_TTL_SECONDS = 300.0
_RankCacheKey = tuple[str, str, tuple[tuple[int, int], ...]]
_RANK_DATA_CACHE: dict[_RankCacheKey, tuple[float, dict[str, Any]]] = {}
_RANK_DATA_CACHE_LOCK = Lock()
_RANK_DATA_INFLIGHT_LOCKS: dict[_RankCacheKey, Lock] = {}


def _build_rank_cache_key(
    ticker_type: str,
    as_of_date: pd.Timestamp | None,
    ma_rules: list[dict[str, Any]],
) -> _RankCacheKey:
    as_of_date_key = as_of_date.date().isoformat() if as_of_date is not None else ""
    ma_rule_key = tuple(
        (
            int(rule.get("short_ma_days") or 0),
            int(rule.get("long_ma_days") or 0),
            int(rule.get("slope_days") or 0),
        )
        for rule in ma_rules
    )
    return (ticker_type, as_of_date_key, ma_rule_key)


def invalidate_rank_data_cache(ticker_type: str | None = None) -> None:
    """랭킹 응답 메모리 캐시를 무효화한다."""

    with _RANK_DATA_CACHE_LOCK:
        if ticker_type is None:
            _RANK_DATA_CACHE.clear()
            return

        target = str(ticker_type or "").strip().lower()
        if not target:
            return

        for cache_key in list(_RANK_DATA_CACHE):
            if cache_key[0] == target:
                _RANK_DATA_CACHE.pop(cache_key, None)


def _get_rank_data_cache(cache_key: _RankCacheKey) -> dict[str, Any] | None:
    with _RANK_DATA_CACHE_LOCK:
        cached = _RANK_DATA_CACHE.get(cache_key)
        if cached is None:
            return None

        cached_at, payload = cached
        if monotonic() - cached_at > _RANK_DATA_CACHE_TTL_SECONDS:
            _RANK_DATA_CACHE.pop(cache_key, None)
            return None

        return deepcopy(payload)


def _set_rank_data_cache(cache_key: _RankCacheKey, payload: dict[str, Any]) -> None:
    with _RANK_DATA_CACHE_LOCK:
        _RANK_DATA_CACHE[cache_key] = (monotonic(), deepcopy(payload))


def _get_rank_data_inflight_lock(cache_key: _RankCacheKey) -> Lock:
    with _RANK_DATA_CACHE_LOCK:
        lock = _RANK_DATA_INFLIGHT_LOCKS.get(cache_key)
        if lock is None:
            lock = Lock()
            _RANK_DATA_INFLIGHT_LOCKS[cache_key] = lock
        return lock


def _serialize_datetime(value: Any) -> str | None:
    if value is None:
        return None

    if isinstance(value, str):
        return value

    if isinstance(value, datetime):
        return value.isoformat()

    try:
        timestamp = pd.Timestamp(value)
    except Exception:
        return str(value)

    if pd.isna(timestamp):
        return None
    return timestamp.isoformat()


def _serialize_value(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, str):
        return value

    if isinstance(value, bool):
        return value

    if isinstance(value, int):
        return value

    if isinstance(value, float):
        if pd.isna(value):
            return None
        return value

    if isinstance(value, datetime):
        return value.isoformat()

    if isinstance(value, list):
        return [_serialize_value(item) for item in value]

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    if isinstance(value, pd.Timestamp):
        return _serialize_datetime(value)

    return value


def _serialize_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in df.to_dict(orient="records"):
        row = {str(key): _serialize_value(value) for key, value in record.items()}
        rows.append(row)
    return rows


def _format_listed_date(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    if len(normalized) == 10 and normalized[4] == "-" and normalized[7] == "-":
        return normalized
    if len(normalized) == 8 and normalized.isdigit():
        return f"{normalized[:4]}-{normalized[4:6]}-{normalized[6:8]}"
    return normalized


def _apply_sector_labels(dataframe: pd.DataFrame) -> pd.DataFrame:
    """stock_meta 의 섹터·업종을 한글 표기(utils.sector_labels)로 붙인다.

    배치 B 가 yfinance 로 수집한 값이며, 없는 종목(ETF·코스닥 소형주 등)은 빈 값이다.
    화면은 값이 있는 풀에서만 컬럼을 노출한다.
    """
    if dataframe.empty or "티커" not in dataframe.columns:
        return dataframe

    from utils.db_manager import get_db_connection
    from utils.sector_labels import industry_ko, sector_ko

    db = get_db_connection()
    if db is None:
        dataframe["섹터"] = ""
        dataframe["업종"] = ""
        return dataframe

    tickers = sorted({str(v or "").strip().upper() for v in dataframe["티커"] if str(v or "").strip()})
    sector_by: dict[str, tuple[str, str]] = {}
    cursor = db.stock_meta.find(
        {"ticker": {"$in": tickers}, "sector": {"$nin": [None, ""]}},
        {"_id": 0, "ticker": 1, "sector": 1, "industry": 1},
    )
    for doc in cursor:
        ticker = str(doc.get("ticker") or "").strip().upper()
        if ticker and ticker not in sector_by:
            sector_by[ticker] = (
                sector_ko(doc.get("sector")),
                industry_ko(doc.get("industry")),
            )

    upper = dataframe["티커"].astype(str).str.strip().str.upper()
    dataframe["섹터"] = upper.map(lambda t: sector_by.get(t, ("", ""))[0])
    dataframe["업종"] = upper.map(lambda t: sector_by.get(t, ("", ""))[1])
    return dataframe


def _apply_rank_info_cache(dataframe: pd.DataFrame, ticker_type: str) -> pd.DataFrame:
    if dataframe.empty:
        return dataframe

    from utils.stock_list_io import get_all_etfs_including_deleted
    meta_docs = get_all_etfs_including_deleted(ticker_type)
    meta_stats_map = {}
    for d in meta_docs:
        t = str(d.get("ticker") or "").strip().upper()
        if t and "backtest_stats" in d:
            meta_stats_map[t] = d["backtest_stats"]

    tickers = [
        str(record.get("티커") or "").strip().upper()
        for record in dataframe.to_dict(orient="records")
        if str(record.get("티커") or "").strip()
    ]
    cache_map = get_stock_cache_meta_map(ticker_type, tickers)
    if not cache_map:
        enriched = dataframe.copy()
        enriched["배당률"] = None
        enriched["보수"] = None
        enriched["순자산총액"] = None
        enriched["상장일"] = None
        enriched["backtest_stats"] = enriched["티커"].map(lambda t: meta_stats_map.get(str(t).strip().upper()))
        enriched.attrs.update(dict(dataframe.attrs))
        return enriched

    rows: list[dict[str, Any]] = []
    for row in dataframe.to_dict(orient="records"):
        ticker = str(row.get("티커") or "").strip().upper()
        doc = cache_map.get(ticker, {})
        meta_cache = doc.get("meta_cache") if isinstance(doc, dict) else {}
        meta_cache = meta_cache if isinstance(meta_cache, dict) else {}
        row["배당률"] = meta_cache.get("dividend_yield_ttm")
        row["보수"] = meta_cache.get("expense_ratio")
        row["순자산총액"] = meta_cache.get("total_net_assets")
        row["상장일"] = _format_listed_date(meta_cache.get("listed_date") or row.get("상장일"))
        row["backtest_stats"] = meta_stats_map.get(ticker)

        rows.append(row)

    enriched = pd.DataFrame(rows)
    enriched.attrs.update(dict(dataframe.attrs))
    return enriched


def _build_configs_payload() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    configs = load_ticker_type_configs()
    if not configs:
        raise ValueError("사용 가능한 종목풀이 없습니다.")

    default_config = pick_default_ticker_type(configs)
    payload = [
        {
            "ticker_type": str(cfg["ticker_type"]),
            "order": int(cfg["order"]),
            "name": str(cfg["name"]),
            "icon": str(cfg.get("icon") or ""),
            "country_code": str(cfg.get("country_code") or ""),
            "top_n_hold": int(cfg["settings"].get("TOP_N_HOLD", 0)),
            "currency": str(cfg["settings"].get("currency") or ""),
        }
        for cfg in configs
    ]
    return payload, default_config


def _build_missing_ticker_labels(ticker_type: str, missing_tickers: list[str]) -> list[str]:
    if not missing_tickers:
        return []

    ticker_name_map = {
        str(item.get("ticker") or "").strip().upper(): str(item.get("name") or "").strip()
        for item in get_etfs(ticker_type)
        if str(item.get("ticker") or "").strip()
    }

    labels: list[str] = []
    for ticker in missing_tickers:
        normalized_ticker = str(ticker or "").strip().upper()
        name = ticker_name_map.get(normalized_ticker, "")
        if name:
            labels.append(f"{name}({normalized_ticker})")
        else:
            labels.append(normalized_ticker)
    return labels


def _build_missing_ticker_rows(
    selected_ticker_type: str,
    missing_tickers: list[str],
    is_all: bool,
) -> list[dict[str, Any]]:
    """캐시 누락 ETF를 그리드에서 선택·삭제할 수 있도록 placeholder 행으로 만든다.

    추세/순위 등 지표는 None(그리드에 "-")으로 두고, 식별·삭제에 필요한
    티커/종목명/source_ticker_type 만 채운다. (캐시 누락으로 순위 계산이 막혀도
    사용자가 신규 ETF를 제거할 수 있게 한다 — 경고 배너는 그대로 표시됨.)
    """
    if not missing_tickers:
        return []

    name_maps: dict[str, dict[str, str]] = {}

    def _name_of(src: str, ticker: str) -> str:
        cached = name_maps.get(src)
        if cached is None:
            cached = {
                str(item.get("ticker") or "").strip().upper(): str(item.get("name") or "").strip()
                for item in get_etfs(src)
                if str(item.get("ticker") or "").strip()
            }
            name_maps[src] = cached
        return cached.get(ticker, "")

    rows: list[dict[str, Any]] = []
    for entry in missing_tickers:
        text = str(entry or "").strip()
        if is_all and ":" in text:
            source, _, ticker = text.partition(":")
            source = source.strip().lower() or selected_ticker_type
            ticker = ticker.strip().upper()
        else:
            source = selected_ticker_type
            ticker = text.upper()
        if not ticker:
            continue
        rows.append(
            {
                "티커": ticker,
                "종목명": _name_of(source, ticker),
                "source_ticker_type": source,
                "순번": "-",
                "순위": None,
                "이전순위": None,
                "1주순위": None,
                "버킷": "",
                "bucket": None,
                "상장일": "-",
                "추세": None,
                "보유": "",
                "보유대상": False,
                "현재가": None,
                "exclude_from_ranking": False,
                "cache_missing": True,
            }
        )
    return rows


def _rows_with_missing_placeholders(
    dataframe: pd.DataFrame,
    selected_ticker_type: str,
    is_all: bool,
) -> list[dict[str, Any]]:
    """직렬화된 행 + 캐시 누락 ETF placeholder 행. 누락 종목을 그리드에서 제거할 수 있게 한다."""
    rows = _serialize_rows(dataframe)
    missing = [str(item) for item in (dataframe.attrs.get("missing_tickers") or [])]
    if not missing:
        return rows
    existing = {str(row.get("티커") or "").strip().upper() for row in rows}
    placeholders = [
        row
        for row in _build_missing_ticker_rows(selected_ticker_type, missing, is_all)
        if str(row.get("티커") or "").strip().upper() not in existing
    ]
    return rows + placeholders


def _get_nth_previous_trading_day(
    country_code: str,
    reference_date: pd.Timestamp | None,
    offset: int,
) -> pd.Timestamp | None:
    if offset < 1:
        raise ValueError("이전 거래일 offset은 1 이상이어야 합니다.")
    if reference_date is None:
        return None

    try:
        reference = pd.Timestamp(reference_date)
    except Exception:
        return None
    if pd.isna(reference):
        return None
    reference = reference.normalize()
    search_start = reference - pd.DateOffset(days=max(15, offset * 4 + 10))
    trading_days = get_trading_days(
        search_start.strftime("%Y-%m-%d"),
        reference.strftime("%Y-%m-%d"),
        country_code,
    )
    previous_days = sorted(
        {pd.Timestamp(day).normalize() for day in trading_days if pd.Timestamp(day).normalize() < reference}
    )
    if len(previous_days) < offset:
        return None
    return previous_days[-offset]


def _get_previous_trading_day(country_code: str, reference_date: pd.Timestamp | None) -> pd.Timestamp | None:
    return _get_nth_previous_trading_day(country_code, reference_date, 1)


def _build_rank_map(dataframe: pd.DataFrame) -> dict[str, int]:
    rank_map: dict[str, int] = {}
    if dataframe.empty:
        return rank_map

    for index, row in enumerate(dataframe.to_dict(orient="records"), start=1):
        ticker = str(row.get("티커") or "").strip().upper()
        if ticker:
            rank_map[ticker] = index
    return rank_map


def _normalize_trend_value(value: Any) -> float | None:
    if value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(score):
        return None
    return score


def _build_score_ranked_rows(dataframe: pd.DataFrame) -> list[dict[str, Any]]:
    rows_with_index: list[dict[str, Any]] = []
    for index, row in enumerate(dataframe.to_dict(orient="records")):
        trend = _normalize_trend_value(row.get("추세"))
        rows_with_index.append(
            {
                **row,
                "추세": trend,
                "__base_index": index,
            }
        )

    rows_with_index.sort(
        key=lambda row: (
            1 if row.get("추세") is None else 0,
            -(float(row["추세"]) if row.get("추세") is not None else 0.0),
            int(row["__base_index"]),
        )
    )

    bm_score = None
    for row in rows_with_index:
        if row.get("is_benchmark") and row.get("추세") is not None:
            bm_score = float(row["추세"])
            break

    ranked_rows: list[dict[str, Any]] = []
    current_rank = 1
    for row in rows_with_index:
        normalized = dict(row)
        normalized.pop("__base_index", None)
        is_excl = bool(normalized.get("exclude_from_ranking"))
        is_bm = bool(normalized.get("is_benchmark"))
        score = normalized.get("추세")
        is_non_pos = score is not None and float(score) <= 0.0

        is_below_bm = False
        if bm_score is not None and score is not None:
            is_below_bm = float(score) < bm_score

        normalized["is_below_benchmark"] = is_below_bm

        normalized["순위"] = current_rank
        current_rank += 1
        ranked_rows.append(normalized)
    return ranked_rows


def _build_rank_map_from_rows(rows: list[dict[str, Any]]) -> dict[str, int]:
    rank_map: dict[str, int] = {}
    for row in rows:
        ticker = _build_rank_row_key(row)
        rank = row.get("순위")
        if ticker and isinstance(rank, int):
            rank_map[ticker] = rank
    return rank_map


def _build_rank_row_key(row: dict[str, Any] | pd.Series) -> str:
    ticker = str(row.get("티커") or "").strip().upper()
    source_ticker_type = str(row.get("source_ticker_type") or "").strip().lower()
    if source_ticker_type:
        return f"{source_ticker_type}:{ticker}"
    return ticker


def _strip_pool_order_prefix(name: str) -> str:
    text = str(name or "").strip()
    if ". " in text:
        prefix, _, suffix = text.partition(". ")
        if prefix.isdigit() and suffix.strip():
            return suffix.strip()
    return text


def load_rank_toolbar_data(ticker_type: str | None = None) -> dict[str, Any]:
    configs_payload, default_config = _build_configs_payload()
    target = str(ticker_type or "").strip().lower()
    available_ids = [str(cfg["ticker_type"]).lower() for cfg in configs_payload]

    if target and target in available_ids:
        selected_ticker_type = target
    else:
        selected_ticker_type = str(default_config["ticker_type"]).strip().lower()

    selected_config = next((cfg for cfg in configs_payload if str(cfg["ticker_type"]).lower() == selected_ticker_type), None)
    if selected_config is None:
        raise ValueError("선택된 종목풀 설정을 찾을 수 없습니다.")

    ma_rules = build_effective_ma_rules(selected_ticker_type, None)

    return {
        "ticker_types": configs_payload,
        "ticker_type": selected_ticker_type,
        "ma_rules": ma_rules,
    }


def _compute_rank_data_payload(
    *,
    configs_payload: list[dict[str, Any]],
    selected_ticker_type: str,
    country_code: str,
    ma_rules: list[dict[str, Any]],
    selected_as_of_date: pd.Timestamp | None,
) -> dict[str, Any]:
    dataframe = build_ticker_type_rankings(
        selected_ticker_type,
        ma_rules=ma_rules,
        as_of_date=selected_as_of_date,
    )
    effective_as_of_date = selected_as_of_date
    raw_as_of_date = dataframe.attrs.get("as_of_date")
    if raw_as_of_date is not None:
        try:
            parsed_as_of_date = pd.Timestamp(raw_as_of_date)
        except Exception:
            parsed_as_of_date = None
        if parsed_as_of_date is not None and not pd.isna(parsed_as_of_date):
            effective_as_of_date = parsed_as_of_date.normalize()
    current_rows = _build_score_ranked_rows(dataframe)
    current_rank_map = _build_rank_map_from_rows(current_rows)
    previous_rank_map: dict[str, int] = {}
    weekly_rank_map: dict[str, int] = {}
    raw_latest_trading_day = dataframe.attrs.get("latest_trading_day")
    previous_trading_day = _get_previous_trading_day(country_code, raw_latest_trading_day)
    weekly_rank_trading_day = _get_nth_previous_trading_day(country_code, raw_latest_trading_day, 5)
    if previous_trading_day is not None:
        previous_dataframe = build_ticker_type_rankings(
            selected_ticker_type,
            ma_rules=ma_rules,
            as_of_date=previous_trading_day,
        )
        previous_rows = _build_score_ranked_rows(previous_dataframe)
        previous_rank_map = _build_rank_map_from_rows(previous_rows)
    if weekly_rank_trading_day is not None:
        weekly_dataframe = build_ticker_type_rankings(
            selected_ticker_type,
            ma_rules=ma_rules,
            as_of_date=weekly_rank_trading_day,
        )
        weekly_rows = _build_score_ranked_rows(weekly_dataframe)
        weekly_rank_map = _build_rank_map_from_rows(weekly_rows)

    if current_rows:
        dataframe_attrs = dict(dataframe.attrs)
        enriched_rows: list[dict[str, Any]] = []
        for row in current_rows:
            row_key = _build_rank_row_key(row)
            row["순위"] = current_rank_map.get(row_key)
            row["이전순위"] = previous_rank_map.get(row_key)
            row["1주순위"] = weekly_rank_map.get(row_key)
            enriched_rows.append(row)
        dataframe = pd.DataFrame(enriched_rows)
        dataframe.attrs.update(dataframe_attrs)

    dataframe = _apply_rank_info_cache(dataframe, selected_ticker_type)
    dataframe = _apply_sector_labels(dataframe)

    return {
        "ticker_types": configs_payload,
        "ticker_type": selected_ticker_type,
        "ma_rules": ma_rules,
        "as_of_date": _serialize_datetime(effective_as_of_date),
        "monthly_return_labels": get_recent_monthly_return_labels(
            MONTHLY_RETURN_LABEL_COUNT,
            reference_date=effective_as_of_date,
        ),
        "rows": _rows_with_missing_placeholders(dataframe, selected_ticker_type, False),
        "cache_blocked": bool(dataframe.attrs.get("cache_blocked", False)),
        "latest_trading_day": _serialize_datetime(dataframe.attrs.get("latest_trading_day")),
        "cache_updated_at": _serialize_datetime(dataframe.attrs.get("cache_updated_at")),
        "ranking_computed_at": _serialize_datetime(dataframe.attrs.get("ranking_computed_at")),
        "realtime_fetched_at": _serialize_datetime(dataframe.attrs.get("realtime_fetched_at")),
        "previous_trading_day": _serialize_datetime(previous_trading_day),
        "weekly_rank_trading_day": _serialize_datetime(weekly_rank_trading_day),
        "missing_tickers": [str(item) for item in (dataframe.attrs.get("missing_tickers") or [])],
        "missing_ticker_labels": _build_missing_ticker_labels(
            selected_ticker_type,
            [str(item) for item in (dataframe.attrs.get("missing_tickers") or [])],
        ),
        "stale_tickers": [str(item) for item in (dataframe.attrs.get("stale_tickers") or [])],
    }


def load_rank_data(
    *,
    ticker_type: str | None = None,
    ma_rule_override: dict[str, Any] | None = None,
    as_of_date: str | None = None,
) -> dict[str, Any]:
    configs_payload, default_config = _build_configs_payload()

    # 요청받은 ticker_type이 현재 유효한 목록 내에 있는지 검사 (없으면 기본값 사용)
    target = str(ticker_type or "").strip().lower()
    available_ids = [str(cfg["ticker_type"]).lower() for cfg in configs_payload]

    if target and target in available_ids:
        selected_ticker_type = target
    else:
        selected_ticker_type = str(default_config["ticker_type"]).strip().lower()
    selected_config = next((cfg for cfg in configs_payload if str(cfg["ticker_type"]).lower() == selected_ticker_type), None)
    country_code = str(selected_config.get("country_code") or "") if selected_config else ""

    ma_rules = build_effective_ma_rules(selected_ticker_type, ma_rule_override)
    selected_as_of_date: pd.Timestamp | None = None
    if as_of_date:
        try:
            selected_as_of_date = pd.to_datetime(as_of_date).normalize()
        except Exception as exc:
            raise ValueError(f"기준일 형식이 올바르지 않습니다: {as_of_date}") from exc
        today_korea = pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None).normalize()
        if selected_as_of_date > today_korea:
            raise ValueError("기준일은 오늘 이후로 선택할 수 없습니다.")

    if selected_config is None:
        raise ValueError("선택된 종목풀 설정을 찾을 수 없습니다.")

    cache_key = _build_rank_cache_key(
        selected_ticker_type, selected_as_of_date, ma_rules
    )
    cached_payload = _get_rank_data_cache(cache_key)
    if cached_payload is not None:
        return cached_payload

    inflight_lock = _get_rank_data_inflight_lock(cache_key)
    with inflight_lock:
        cached_payload = _get_rank_data_cache(cache_key)
        if cached_payload is not None:
            return cached_payload

        payload = _compute_rank_data_payload(
            configs_payload=configs_payload,
            selected_ticker_type=selected_ticker_type,
            country_code=country_code,
            ma_rules=ma_rules,
            selected_as_of_date=selected_as_of_date,
        )
        _set_rank_data_cache(cache_key, payload)
        return payload
