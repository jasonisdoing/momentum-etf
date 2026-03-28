from __future__ import annotations

import datetime
from typing import Any

from utils.account_registry import load_account_configs
from utils.db_manager import get_db_connection


def _require_db():
    db = get_db_connection()
    if db is None:
        raise RuntimeError("DB 연결 실패")
    return db


def _normalize_number(value: Any) -> float:
    return float(value or 0)


def _normalize_nullable_number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _to_updated_at_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime.datetime):
        return value.isoformat()
    return str(value)


def _normalize_currency(value: Any, fallback: str) -> str:
    text = str(value or "").strip().upper()
    return text or fallback


def load_cash_accounts() -> dict[str, list[dict[str, Any]]]:
    db = _require_db()
    doc = db.portfolio_master.find_one({"master_id": "GLOBAL"}) or {}
    account_docs = {
        str(account.get("account_id") or ""): account
        for account in (doc.get("accounts") or [])
        if isinstance(account, dict)
    }

    rows: list[dict[str, Any]] = []
    for account in load_account_configs():
        account_id = str(account["account_id"])
        settings = account.get("settings") or {}
        currency = str(settings.get("currency") or "KRW").strip().upper() or "KRW"
        account_doc = account_docs.get(account_id, {})
        cash_currency = _normalize_currency(account_doc.get("cash_currency"), currency)

        rows.append(
            {
                "account_id": account_id,
                "order": int(account["order"]),
                "name": str(account["name"]),
                "icon": str(account.get("icon") or ""),
                "country_code": str(account.get("country_code") or ""),
                "currency": currency,
                "total_principal": _normalize_number(account_doc.get("total_principal")),
                "cash_balance_krw": _normalize_number(account_doc.get("cash_balance")),
                "cash_balance_native": _normalize_nullable_number(account_doc.get("cash_balance_native")),
                "cash_currency": cash_currency,
                "intl_shares_value": (
                    _normalize_nullable_number(account_doc.get("intl_shares_value"))
                    if account_id == "aus_account"
                    else None
                ),
                "intl_shares_change": (
                    _normalize_nullable_number(account_doc.get("intl_shares_change"))
                    if account_id == "aus_account"
                    else None
                ),
                "updated_at": _to_updated_at_text(account_doc.get("updated_at")),
            }
        )

    return {"accounts": rows}


def save_cash_accounts(updates: list[dict[str, Any]]) -> dict[str, str]:
    if not updates:
        raise ValueError("저장할 계좌 데이터가 없습니다.")

    db = _require_db()
    collection = db.portfolio_master
    doc = collection.find_one({"master_id": "GLOBAL"}) or {"master_id": "GLOBAL", "accounts": []}
    accounts = list(doc.get("accounts") or [])
    now = datetime.datetime.now()

    for update in updates:
        account_id = str(update.get("account_id") or "").strip()
        if not account_id:
            raise ValueError("account_id가 필요합니다.")

        row = {
            "account_id": account_id,
            "total_principal": float(update.get("total_principal") or 0),
            "cash_balance": float(update.get("cash_balance_krw") or 0),
            "cash_balance_native": _normalize_nullable_number(update.get("cash_balance_native")),
            "cash_currency": str(update.get("cash_currency") or "").strip().upper(),
            "intl_shares_value": _normalize_nullable_number(update.get("intl_shares_value")),
            "intl_shares_change": _normalize_nullable_number(update.get("intl_shares_change")),
            "updated_at": now,
        }

        index = next((i for i, item in enumerate(accounts) if str(item.get("account_id") or "") == account_id), -1)
        if index >= 0:
            current = accounts[index]
            accounts[index] = {
                **current,
                **row,
                "holdings": current.get("holdings") if isinstance(current.get("holdings"), list) else [],
            }
        else:
            row["holdings"] = []
            accounts.append(row)

    collection.update_one({"master_id": "GLOBAL"}, {"$set": {"accounts": accounts}}, upsert=True)
    return {"message": "자산 관리 저장 완료"}
