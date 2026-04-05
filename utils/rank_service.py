from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from utils.rankings import (
    ALLOWED_MA_TYPES,
    build_effective_ma_rules,
    build_ticker_type_rankings,
    get_rank_months_max,
    get_recent_monthly_return_labels,
)
from utils.stock_list_io import get_etfs
from utils.ticker_registry import load_ticker_type_configs, pick_default_ticker_type


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


def _build_configs_payload() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    configs = load_ticker_type_configs()
    if not configs:
        raise ValueError("사용 가능한 종목 타입이 없습니다.")

    default_config = pick_default_ticker_type(configs)
    payload = [
        {
            "ticker_type": str(cfg["ticker_type"]),
            "order": int(cfg["order"]),
            "name": str(cfg["name"]),
            "icon": str(cfg.get("icon") or ""),
            "country_code": str(cfg.get("country_code") or ""),
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


def load_rank_data(
    *,
    ticker_type: str | None = None,
    ma_rule_overrides: list[dict[str, Any]] | None = None,
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

    ma_rules = build_effective_ma_rules(selected_ticker_type, ma_rule_overrides)
    selected_as_of_date: pd.Timestamp | None = None
    if as_of_date:
        try:
            selected_as_of_date = pd.to_datetime(as_of_date).normalize()
        except Exception as exc:
            raise ValueError(f"기준일 형식이 올바르지 않습니다: {as_of_date}") from exc
        today_korea = pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None).normalize()
        if selected_as_of_date > today_korea:
            raise ValueError("기준일은 오늘 이후로 선택할 수 없습니다.")

    dataframe = build_ticker_type_rankings(
        selected_ticker_type,
        ma_rules=ma_rules,
        as_of_date=selected_as_of_date,
    )
    effective_as_of_date = (
        pd.Timestamp(dataframe.attrs.get("as_of_date")).normalize()
        if dataframe.attrs.get("as_of_date") is not None
        else selected_as_of_date
    )

    return {
        "ticker_types": configs_payload,
        "ticker_type": selected_ticker_type,
        "ma_rules": ma_rules,
        "ma_type_options": ALLOWED_MA_TYPES,
        "ma_months_max": get_rank_months_max(),
        "as_of_date": _serialize_datetime(effective_as_of_date),
        "monthly_return_labels": get_recent_monthly_return_labels(reference_date=effective_as_of_date),
        "rows": _serialize_rows(dataframe),
        "cache_blocked": bool(dataframe.attrs.get("cache_blocked", False)),
        "latest_trading_day": _serialize_datetime(dataframe.attrs.get("latest_trading_day")),
        "cache_updated_at": _serialize_datetime(dataframe.attrs.get("cache_updated_at")),
        "ranking_computed_at": _serialize_datetime(dataframe.attrs.get("ranking_computed_at")),
        "realtime_fetched_at": _serialize_datetime(dataframe.attrs.get("realtime_fetched_at")),
        "missing_tickers": [str(item) for item in (dataframe.attrs.get("missing_tickers") or [])],
        "missing_ticker_labels": _build_missing_ticker_labels(
            selected_ticker_type,
            [str(item) for item in (dataframe.attrs.get("missing_tickers") or [])],
        ),
        "stale_tickers": [str(item) for item in (dataframe.attrs.get("stale_tickers") or [])],
    }
