from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from utils.account_registry import load_account_configs, pick_default_account
from utils.rankings import ALLOWED_MA_TYPES, build_account_rankings, get_account_rank_defaults, get_rank_months_max
from utils.stock_list_io import get_etfs


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


def _build_accounts_payload() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    accounts = load_account_configs()
    if not accounts:
        raise ValueError("사용 가능한 계정이 없습니다.")

    default_account = pick_default_account(accounts)
    payload = [
        {
            "account_id": str(account["account_id"]),
            "order": int(account["order"]),
            "name": str(account["name"]),
            "icon": str(account.get("icon") or ""),
            "country_code": str(account.get("country_code") or ""),
        }
        for account in accounts
    ]
    return payload, default_account


def _build_missing_ticker_labels(account_id: str, missing_tickers: list[str]) -> list[str]:
    if not missing_tickers:
        return []

    ticker_name_map = {
        str(item.get("ticker") or "").strip().upper(): str(item.get("name") or "").strip()
        for item in get_etfs(account_id)
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
    account_id: str | None = None,
    ma_type: str | None = None,
    ma_months: int | None = None,
) -> dict[str, Any]:
    accounts_payload, default_account = _build_accounts_payload()
    selected_account_id = str(account_id or default_account["account_id"]).strip().lower()

    default_ma_type, default_ma_months = get_account_rank_defaults(selected_account_id)
    selected_ma_type = str(ma_type or default_ma_type).strip().upper()
    selected_ma_months = int(ma_months or default_ma_months)

    dataframe = build_account_rankings(
        selected_account_id,
        ma_type=selected_ma_type,
        ma_months=selected_ma_months,
    )

    return {
        "accounts": accounts_payload,
        "account_id": selected_account_id,
        "ma_type": selected_ma_type,
        "ma_months": selected_ma_months,
        "ma_type_options": ALLOWED_MA_TYPES,
        "ma_months_max": get_rank_months_max(),
        "rows": _serialize_rows(dataframe),
        "cache_blocked": bool(dataframe.attrs.get("cache_blocked", False)),
        "latest_trading_day": _serialize_datetime(dataframe.attrs.get("latest_trading_day")),
        "cache_updated_at": _serialize_datetime(dataframe.attrs.get("cache_updated_at")),
        "ranking_computed_at": _serialize_datetime(dataframe.attrs.get("ranking_computed_at")),
        "realtime_fetched_at": _serialize_datetime(dataframe.attrs.get("realtime_fetched_at")),
        "missing_tickers": [str(item) for item in (dataframe.attrs.get("missing_tickers") or [])],
        "missing_ticker_labels": _build_missing_ticker_labels(
            selected_account_id,
            [str(item) for item in (dataframe.attrs.get("missing_tickers") or [])],
        ),
        "stale_tickers": [str(item) for item in (dataframe.attrs.get("stale_tickers") or [])],
    }
