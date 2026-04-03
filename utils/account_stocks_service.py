from __future__ import annotations

from typing import Any

from services.price_service import get_exchange_rates
from services.price_service import get_realtime_snapshot
from utils.account_registry import load_account_configs, pick_default_account
from utils.account_stocks_io import get_account_targets
from utils.cache_utils import load_cached_close_series_bulk_with_fallback
from utils.db_manager import get_db_connection
from utils.holdings_detail_service import load_all_holdings_detail
from utils.normalization import normalize_nullable_number, normalize_text
from utils.rankings import build_recent_monthly_return_metrics, get_recent_monthly_return_labels

BUCKETS: dict[int, str] = {
    1: "1. 모멘텀",
    2: "2. 시장지수",
    3: "3. 배당방어",
    4: "4. 대체헷지",
}


def _build_accounts_payload(account_id: str | None) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    accounts = load_account_configs()
    if not accounts:
        raise ValueError("사용 가능한 계정이 없습니다.")

    default_account = pick_default_account(accounts)
    
    payload = [
        {
            "account_id": str(acc["account_id"]),
            "order": int(acc["order"]),
            "name": str(acc["name"]),
            "icon": str(acc.get("icon") or ""),
            "country_code": str(acc.get("country_code") or ""),
            "currency": str(acc.get("settings", {}).get("currency") or "KRW").strip().upper(),
            "ticker_codes": acc.get("settings", {}).get("ticker_codes", []),
        }
        for acc in accounts
    ]
    
    selected_account_id = str(account_id or default_account["account_id"]).strip().lower()
    selected_account = next((a for a in payload if a["account_id"] == selected_account_id), payload[0])
    
    return payload, selected_account, selected_account_id


def _compute_account_total_assets_native(account_id: str, account_currency: str) -> float:
    holdings_detail = load_all_holdings_detail(account_id=account_id)
    rows = holdings_detail.get("rows") or []
    cash_info = holdings_detail.get("cash") or {}
    total_valuation_krw = sum(float(row.get("valuation_krw") or 0.0) for row in rows)

    currency = str(account_currency or "KRW").strip().upper() or "KRW"
    if currency == "KRW":
        return total_valuation_krw + float(cash_info.get("cash_balance_krw") or 0.0)

    if currency == "AUD":
        rates = get_exchange_rates()
        aud_rate = float(((rates or {}).get("AUD") or {}).get("rate") or 0.0)
        if aud_rate <= 0:
            raise RuntimeError("AUD 환율을 가져오지 못했습니다.")
        cash_native = normalize_nullable_number(cash_info.get("cash_balance_native"))
        if cash_native is None:
            cash_native = float(cash_info.get("cash_balance_krw") or 0.0) / aud_rate
        return (total_valuation_krw / aud_rate) + float(cash_native or 0.0)

    return total_valuation_krw


def load_account_stocks_data(account_id: str | None) -> dict[str, Any]:
    accounts, selected_account, selected_account_id = _build_accounts_payload(account_id)
    
    ticker_codes = selected_account.get("ticker_codes", [])
    country_code = selected_account.get("country_code", "kor")
    account_currency = str(selected_account.get("currency") or "KRW").strip().upper() or "KRW"
    monthly_return_labels = get_recent_monthly_return_labels()
    account_total_assets = _compute_account_total_assets_native(selected_account_id, account_currency)
    
    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결에 실패했습니다.")

    saved_targets = get_account_targets(selected_account_id)
    target_dict = {str(t.get("ticker")).upper(): float(t.get("ratio") or 0.0) for t in saved_targets}

    docs = list(
        db.stock_meta.find(
            {
                "ticker_type": {"$in": ticker_codes},
                "is_deleted": {"$ne": True},
            },
            {
                "ticker_type": 1,
                "ticker": 1,
                "name": 1,
                "bucket": 1,
                "added_date": 1,
                "listing_date": 1,
                "1_week_avg_volume": 1,
            },
        )
    )

    tickers = [doc.get("ticker", "") for doc in docs if doc.get("ticker")]
    realtime_snapshot = {}
    try:
        realtime_snapshot = get_realtime_snapshot(country_code, tickers)
    except Exception:
        pass

    close_series_by_ticker: dict[str, Any] = {}
    ticker_groups: dict[str, list[str]] = {}
    for doc in docs:
        ticker_type = normalize_text(doc.get("ticker_type"), "").lower()
        ticker = normalize_text(doc.get("ticker"), "")
        if not ticker_type or not ticker:
            continue
        ticker_groups.setdefault(ticker_type, []).append(ticker)

    for ticker_type, grouped_tickers in ticker_groups.items():
        close_series_by_ticker.update(load_cached_close_series_bulk_with_fallback(ticker_type, grouped_tickers))

    available_tickers = []

    for doc in docs:
        ticker = normalize_text(doc.get("ticker"), "")
        name = normalize_text(doc.get("name"), "")
        if not ticker:
            continue

        monthly_returns = build_recent_monthly_return_metrics(
            close_series_by_ticker.get(ticker),
            labels=monthly_return_labels,
        )
        item = {
            "ticker": ticker,
            "name": name,
            "bucket_id": int(doc.get("bucket") or 1),
            "bucket_name": BUCKETS.get(int(doc.get("bucket") or 1), BUCKETS[1]),
            "added_date": normalize_text(doc.get("added_date"), "-"),
            "listing_date": normalize_text(doc.get("listing_date"), "-"),
            "week_volume": normalize_nullable_number(doc.get("1_week_avg_volume")),
            "return_1d": normalize_nullable_number(
                realtime_snapshot.get(ticker, {}).get("changeRate")
            ),
        }
        for label in monthly_return_labels:
            item[label] = normalize_nullable_number(monthly_returns.get(label))
        available_tickers.append(item)

    rows = []
    for item in available_tickers:
        ticker = item["ticker"]
        if ticker in target_dict:
            row_item = dict(item)
            row_item["ratio"] = target_dict[ticker]
            row_item["target_amount"] = round(account_total_assets * (float(row_item["ratio"]) / 100.0), 2)
            rows.append(row_item)

    rows = sorted(
        rows,
        key=lambda row: (
            -(row.get("ratio") or 0.0),
            -(row.get("return_1d") if row.get("return_1d") is not None else float("-inf")),
        ),
    )

    return {
        "accounts": accounts,
        "account_id": selected_account_id,
        "account_currency": account_currency,
        "account_total_assets": account_total_assets,
        "monthly_return_labels": monthly_return_labels,
        "available_tickers": available_tickers,
        "rows": rows,
    }
