from __future__ import annotations

import datetime
import re
from typing import Any

from utils.account_registry import load_account_configs
from utils.db_manager import get_db_connection

BUCKET_NAME_TO_ID = {
    "1. 모멘텀": 1,
    "2. 시장지수": 2,
    "3. 배당방어": 3,
    "4. 대체헷지": 4,
}


def _require_db():
    db = get_db_connection()
    if db is None:
        raise RuntimeError("DB 연결 실패")
    return db


def _normalize_ticker(value: str) -> str:
    text = str(value or "").strip()
    if ":" not in text:
        return text.upper()
    return text.split(":")[-1].strip().upper()


def _normalize_numeric_text(value: str) -> float:
    normalized = re.sub(r"[^\d.-]", "", str(value or "").replace(",", ""))
    try:
        return float(normalized)
    except ValueError:
        return 0.0


def _get_last_business_day_text() -> str:
    current = datetime.datetime.now(datetime.timezone.utc)
    while current.weekday() >= 5:
        current -= datetime.timedelta(days=1)
    return current.astimezone(datetime.timezone(datetime.timedelta(hours=9))).strftime("%Y-%m-%d")


def _build_account_name_map(accounts: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for account in accounts:
        plain_name = str(account["name"]).strip()
        ordered_name = f"{int(account['order'])}. {plain_name}"
        mapping[plain_name] = account
        mapping[ordered_name] = account
    return mapping


def _build_account_order_map(accounts: list[dict[str, Any]]) -> dict[str, int]:
    return {str(account["account_id"]): int(account["order"]) for account in accounts}


def _parse_tsv_rows(raw_text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for line in str(raw_text or "").splitlines():
        trimmed = line.rstrip()
        if trimmed.strip():
            rows.append(trimmed.split("\t"))
    return rows


def parse_bulk_import_text(raw_text: str) -> dict[str, Any]:
    text = str(raw_text or "")
    if not text.strip():
        raise ValueError("붙여넣은 데이터가 비어 있습니다.")

    accounts = load_account_configs()
    account_name_map = _build_account_name_map(accounts)
    account_order_map = _build_account_order_map(accounts)
    parsed_lines = _parse_tsv_rows(text)
    if not parsed_lines:
        raise ValueError("붙여넣은 데이터가 비어 있습니다.")

    errors: list[str] = []
    rows: list[dict[str, Any]] = []

    for index, columns in enumerate(parsed_lines, start=1):
        if len(columns) < 7:
            errors.append(f"{index}행: TSV 7컬럼이 필요합니다.")
            continue

        account_name_raw, currency_raw, bucket_text_raw, ticker_raw, stock_name_raw, quantity_raw, price_raw = columns[
            :7
        ]
        account_name = str(account_name_raw or "").strip()
        currency = str(currency_raw or "").strip().upper()
        bucket_text = str(bucket_text_raw or "").strip()
        ticker = _normalize_ticker(ticker_raw)
        stock_name = str(stock_name_raw or "").strip()
        quantity = _normalize_numeric_text(quantity_raw)
        average_buy_price = _normalize_numeric_text(price_raw)

        account = account_name_map.get(account_name)
        if account is None:
            errors.append(f"{index}행: 계좌 '{account_name}'을(를) 찾을 수 없습니다.")
            continue

        bucket = BUCKET_NAME_TO_ID.get(bucket_text)
        if bucket is None:
            errors.append(f"{index}행: 버킷 '{bucket_text}'을(를) 찾을 수 없습니다.")
            continue

        if not ticker:
            errors.append(f"{index}행: 티커가 비어 있습니다.")
            continue

        rows.append(
            {
                "account_name": account_name,
                "account_id": str(account["account_id"]),
                "currency": currency,
                "bucket_text": bucket_text,
                "bucket": bucket,
                "ticker": ticker,
                "name": stock_name,
                "quantity": quantity,
                "average_buy_price": average_buy_price,
            }
        )

    if errors:
        raise ValueError("\n".join(errors))

    rows.sort(
        key=lambda row: (
            account_order_map.get(str(row["account_id"]), 999),
            str(row["ticker"]),
        )
    )
    return {
        "rows": rows,
        "row_count": len(rows),
        "account_count": len({str(row["account_id"]) for row in rows}),
    }


def _load_reference_name_map(account_id: str, tickers: list[str]) -> dict[str, str]:
    db = _require_db()
    by_account: dict[str, str] = {}

    docs = list(
        db.stock_meta.find(
            {"account_id": account_id, "ticker": {"$in": tickers}, "is_deleted": {"$ne": True}},
            {"ticker": 1, "name": 1},
        )
    )
    for doc in docs:
        ticker = str(doc.get("ticker") or "").strip().upper()
        name = str(doc.get("name") or "").strip()
        if ticker and name:
            by_account[ticker] = name

    missing = [ticker for ticker in tickers if ticker not in by_account]
    if not missing:
        return by_account

    fallback_docs = list(
        db.stock_meta.find({"ticker": {"$in": missing}, "is_deleted": {"$ne": True}}, {"ticker": 1, "name": 1})
    )
    for doc in fallback_docs:
        ticker = str(doc.get("ticker") or "").strip().upper()
        name = str(doc.get("name") or "").strip()
        if ticker and name and ticker not in by_account:
            by_account[ticker] = name

    return by_account


def save_bulk_import_rows(rows: list[dict[str, Any]]) -> dict[str, int]:
    if not rows:
        raise ValueError("반영할 데이터가 없습니다.")

    db = _require_db()
    doc = db.portfolio_master.find_one({"master_id": "GLOBAL"}) or {"master_id": "GLOBAL", "accounts": []}
    accounts = list(doc.get("accounts") or [])

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["account_id"]), []).append(row)

    now = datetime.datetime.now()
    default_first_buy_date = _get_last_business_day_text()
    updated_accounts = 0

    for account_id, account_rows in grouped.items():
        existing_account = next(
            (account for account in accounts if str(account.get("account_id") or "") == account_id), None
        )
        existing_holdings = list((existing_account or {}).get("holdings") or [])
        existing_date_map: dict[str, str] = {}
        existing_name_map: dict[str, str] = {}

        for holding in existing_holdings:
            ticker = str(holding.get("ticker") or "").strip().upper()
            if not ticker:
                continue
            first_buy_date = str(holding.get("first_buy_date") or "").strip()
            name = str(holding.get("name") or "").strip()
            if first_buy_date:
                existing_date_map[ticker] = first_buy_date
            if name:
                existing_name_map[ticker] = name

        tickers = [str(row["ticker"]) for row in account_rows]
        reference_name_map = _load_reference_name_map(account_id, tickers)
        next_holdings = [
            {
                "ticker": str(row["ticker"]),
                "name": existing_name_map.get(str(row["ticker"]))
                or reference_name_map.get(str(row["ticker"]))
                or str(row.get("name") or row["ticker"]),
                "quantity": int(float(row.get("quantity") or 0)),
                "average_buy_price": float(row.get("average_buy_price") or 0),
                "currency": str(row.get("currency") or ""),
                "bucket": int(row.get("bucket") or 0),
                "first_buy_date": existing_date_map.get(str(row["ticker"])) or default_first_buy_date,
            }
            for row in account_rows
        ]

        if existing_account is not None:
            existing_account["holdings"] = next_holdings
            existing_account["updated_at"] = now
        else:
            accounts.append(
                {
                    "account_id": account_id,
                    "total_principal": 0,
                    "cash_balance": 0,
                    "holdings": next_holdings,
                    "updated_at": now,
                }
            )

        updated_accounts += 1

    db.portfolio_master.update_one(
        {"master_id": "GLOBAL"},
        {"$set": {"master_id": "GLOBAL", "accounts": accounts}},
        upsert=True,
    )
    return {"updated_accounts": updated_accounts}
