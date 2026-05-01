from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from threading import Lock
from time import monotonic
from typing import Any

import pandas as pd

from services.stock_cache_service import get_stock_cache_meta_map
from utils.rankings import (
    ALLOWED_MA_TYPES,
    MONTHLY_RETURN_LABEL_COUNT,
    build_effective_ma_rules,
    build_ticker_type_rankings,
    get_rank_months_max,
    get_recent_monthly_return_labels,
)
from utils.data_loader import get_trading_days
from utils.stock_list_io import get_etfs
from utils.ticker_registry import load_ticker_type_configs, pick_default_ticker_type
from config import NAVER_ETF_CATEGORY_CONFIG

_RANK_DATA_CACHE_TTL_SECONDS = 300.0
_RankCacheKey = tuple[str, str, tuple[tuple[str, int], ...], int]
_RANK_DATA_CACHE: dict[_RankCacheKey, tuple[float, dict[str, Any]]] = {}
_RANK_DATA_CACHE_LOCK = Lock()
_RANK_DATA_INFLIGHT_LOCKS: dict[_RankCacheKey, Lock] = {}


def _build_rank_cache_key(
    ticker_type: str,
    as_of_date: pd.Timestamp | None,
    ma_rules: list[dict[str, Any]],
    held_bonus_score: int,
) -> _RankCacheKey:
    as_of_date_key = as_of_date.date().isoformat() if as_of_date is not None else ""
    ma_rule_key = tuple((str(rule.get("ma_type") or ""), int(rule.get("ma_months") or 0)) for rule in ma_rules)
    return ticker_type, as_of_date_key, ma_rule_key, int(held_bonus_score)


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


def _apply_rank_info_cache(dataframe: pd.DataFrame, ticker_type: str) -> pd.DataFrame:
    if dataframe.empty:
        return dataframe

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

        # Naver 상세 분류 정보(cat_*) 추가
        for cat in NAVER_ETF_CATEGORY_CONFIG:
            cat_code = cat["code"]
            cat_name = cat["name"]
            row[cat_name] = meta_cache.get(f"cat_{cat_code}")

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
            "holding_bonus_score": int(cfg["settings"].get("HOLDING_BONUS_SCORE", 0)),
            "top_n_hold": int(cfg["settings"].get("TOP_N_HOLD", 0)),
            "rsi_limit": (
                float(cfg["settings"]["RSI_LIMIT"])
                if "RSI_LIMIT" in cfg["settings"] and cfg["settings"]["RSI_LIMIT"] is not None
                else None
            ),
            "type_source": str(cfg["settings"].get("type_source") or ""),
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


def _get_previous_trading_day(country_code: str, reference_date: pd.Timestamp | None) -> pd.Timestamp | None:
    if reference_date is None:
        return None

    try:
        reference = pd.Timestamp(reference_date)
    except Exception:
        return None
    if pd.isna(reference):
        return None
    reference = reference.normalize()
    search_start = reference - pd.DateOffset(days=15)
    trading_days = get_trading_days(
        search_start.strftime("%Y-%m-%d"),
        reference.strftime("%Y-%m-%d"),
        country_code,
    )
    previous_days = [pd.Timestamp(day).normalize() for day in trading_days if pd.Timestamp(day).normalize() < reference]
    if not previous_days:
        return None
    return max(previous_days)


def _build_rank_map(dataframe: pd.DataFrame) -> dict[str, int]:
    rank_map: dict[str, int] = {}
    if dataframe.empty:
        return rank_map

    for index, row in enumerate(dataframe.to_dict(orient="records"), start=1):
        ticker = str(row.get("티커") or "").strip().upper()
        if ticker:
            rank_map[ticker] = index
    return rank_map


def _normalize_score_value(value: Any) -> float | None:
    if value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if pd.isna(score):
        return None
    return score


def _build_bonus_adjusted_rows(
    dataframe: pd.DataFrame,
    held_bonus_score: int,
) -> list[dict[str, Any]]:
    rows_with_index: list[dict[str, Any]] = []
    for index, row in enumerate(dataframe.to_dict(orient="records")):
        score = _normalize_score_value(row.get("점수"))
        is_held = str(row.get("보유") or "").strip() != ""
        adjusted_score = score
        if score is not None and is_held:
            adjusted_score = round(score + held_bonus_score, 1)
        rows_with_index.append(
            {
                **row,
                "점수": adjusted_score,
                "__base_index": index,
            }
        )

    rows_with_index.sort(
        key=lambda row: (
            1 if row.get("exclude_from_ranking") else 0,
            1 if row.get("점수") is None else 0,
            -(float(row["점수"]) if row.get("점수") is not None else 0.0),
            int(row["__base_index"]),
        )
    )

    ranked_rows: list[dict[str, Any]] = []
    current_rank = 1
    for row in rows_with_index:
        normalized = dict(row)
        normalized.pop("__base_index", None)
        if normalized.get("exclude_from_ranking"):
            normalized["순위"] = None
        else:
            normalized["순위"] = current_rank
            current_rank += 1
        ranked_rows.append(normalized)
    return ranked_rows


def _build_rank_map_from_rows(rows: list[dict[str, Any]]) -> dict[str, int]:
    rank_map: dict[str, int] = {}
    for row in rows:
        ticker = str(row.get("티커") or "").strip().upper()
        rank = row.get("순위")
        if ticker and isinstance(rank, int):
            rank_map[ticker] = rank
    return rank_map


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

    return {
        "ticker_types": configs_payload,
        "ticker_type": selected_ticker_type,
        "ma_rules": build_effective_ma_rules(selected_ticker_type, None),
        "ma_type_options": ALLOWED_MA_TYPES,
        "ma_months_max": get_rank_months_max(),
        "held_bonus_score": int(selected_config["holding_bonus_score"]),
    }


def _compute_rank_data_payload(
    *,
    configs_payload: list[dict[str, Any]],
    selected_ticker_type: str,
    country_code: str,
    ma_rules: list[dict[str, Any]],
    selected_as_of_date: pd.Timestamp | None,
    bonus_score: int,
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
    current_rows = _build_bonus_adjusted_rows(dataframe, bonus_score)
    current_rank_map = _build_rank_map_from_rows(current_rows)
    previous_rank_map: dict[str, int] = {}
    raw_latest_trading_day = dataframe.attrs.get("latest_trading_day")
    previous_trading_day = _get_previous_trading_day(country_code, raw_latest_trading_day)
    if previous_trading_day is not None:
        previous_dataframe = build_ticker_type_rankings(
            selected_ticker_type,
            ma_rules=ma_rules,
            as_of_date=previous_trading_day,
        )
        previous_rows = _build_bonus_adjusted_rows(previous_dataframe, bonus_score)
        previous_rank_map = _build_rank_map_from_rows(previous_rows)

    if current_rows:
        dataframe_attrs = dict(dataframe.attrs)
        enriched_rows: list[dict[str, Any]] = []
        for row in current_rows:
            ticker = str(row.get("티커") or "").strip().upper()
            row["순위"] = current_rank_map.get(ticker)
            row["이전순위"] = previous_rank_map.get(ticker)
            enriched_rows.append(row)
        dataframe = pd.DataFrame(enriched_rows)
        dataframe.attrs.update(dataframe_attrs)

    dataframe = _apply_rank_info_cache(dataframe, selected_ticker_type)

    return {
        "ticker_types": configs_payload,
        "ticker_type": selected_ticker_type,
        "ma_rules": ma_rules,
        "ma_type_options": ALLOWED_MA_TYPES,
        "ma_months_max": get_rank_months_max(),
        "as_of_date": _serialize_datetime(effective_as_of_date),
        "monthly_return_labels": get_recent_monthly_return_labels(
            MONTHLY_RETURN_LABEL_COUNT,
            reference_date=effective_as_of_date,
        ),
        "rows": _serialize_rows(dataframe),
        "held_bonus_score": bonus_score,
        "cache_blocked": bool(dataframe.attrs.get("cache_blocked", False)),
        "latest_trading_day": _serialize_datetime(dataframe.attrs.get("latest_trading_day")),
        "cache_updated_at": _serialize_datetime(dataframe.attrs.get("cache_updated_at")),
        "ranking_computed_at": _serialize_datetime(dataframe.attrs.get("ranking_computed_at")),
        "realtime_fetched_at": _serialize_datetime(dataframe.attrs.get("realtime_fetched_at")),
        "previous_trading_day": _serialize_datetime(previous_trading_day),
        "missing_tickers": [str(item) for item in (dataframe.attrs.get("missing_tickers") or [])],
        "missing_ticker_labels": _build_missing_ticker_labels(
            selected_ticker_type,
            [str(item) for item in (dataframe.attrs.get("missing_tickers") or [])],
        ),
        "stale_tickers": [str(item) for item in (dataframe.attrs.get("stale_tickers") or [])],
        "naver_category_config": [c for c in NAVER_ETF_CATEGORY_CONFIG if c.get("show")],
    }


def load_rank_data(
    *,
    ticker_type: str | None = None,
    ma_rule_override: dict[str, Any] | None = None,
    as_of_date: str | None = None,
    held_bonus_score: int | None,
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

    if held_bonus_score is None:
        if selected_config is None:
            raise ValueError("선택된 종목풀 설정을 찾을 수 없습니다.")
        bonus_score = int(selected_config["holding_bonus_score"])
    else:
        bonus_score = int(held_bonus_score)

    cache_key = _build_rank_cache_key(selected_ticker_type, selected_as_of_date, ma_rules, bonus_score)
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
            bonus_score=bonus_score,
        )
        _set_rank_data_cache(cache_key, payload)
        return payload
