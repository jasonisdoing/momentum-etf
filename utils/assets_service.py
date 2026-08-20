from __future__ import annotations

import datetime
from typing import Any

from utils.account_registry import load_account_configs
from utils.cash_model import resolve_cash_currencies, resolve_cash_native_map
from utils.db_manager import get_db_connection
from utils.normalization import normalize_nullable_number, normalize_number, to_iso_string


def _require_db():
    db = get_db_connection()
    if db is None:
        raise RuntimeError("DB 연결 실패")
    return db


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
        cash_currencies = resolve_cash_currencies(settings)
        cash_map = resolve_cash_native_map(account_doc, currency)

        rows.append(
            {
                "account_id": account_id,
                "order": int(account["order"]),
                "name": str(account["name"]),
                "icon": str(account.get("icon") or ""),
                "country_code": str(account.get("country_code") or ""),
                "currency": currency,
                "total_principal": normalize_number(account_doc.get("total_principal")),
                "cash_balance_krw": normalize_number(account_doc.get("cash_balance")),
                "cash_balance_native": normalize_nullable_number(account_doc.get("cash_balance_native")),
                "cash_currency": cash_currency,
                "cash_currencies": cash_currencies,
                "cash": cash_map,
                "cash_target_ratio": normalize_number(account_doc.get("cash_target_ratio")),
                "intl_shares_value": (
                    normalize_nullable_number(account_doc.get("intl_shares_value"))
                    if account_id == "aus_account"
                    else None
                ),
                "intl_shares_change": (
                    normalize_nullable_number(account_doc.get("intl_shares_change"))
                    if account_id == "aus_account"
                    else None
                ),
                "updated_at": to_iso_string(account_doc.get("updated_at")),
                "updated_by": str(account_doc.get("updated_by") or ""),
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
            "cash_balance_native": normalize_nullable_number(update.get("cash_balance_native")),
            "cash_currency": str(update.get("cash_currency") or "").strip().upper(),
            "cash_target_ratio": float(update.get("cash_target_ratio") or 0),
            "intl_shares_value": normalize_nullable_number(update.get("intl_shares_value")),
            "intl_shares_change": normalize_nullable_number(update.get("intl_shares_change")),
            "updated_at": now,
            "updated_by": "user",
        }

        # 통화별 native 현금 맵 동기화 (신 형식 우선, 없으면 레거시 필드에서 합성).
        # 단, 현금 관련 키가 아예 없는 요청(예: Intl Value 만 저장)에서는 현금을 건드리지
        # 않는다. 레거시 필드로 재합성하면 신 형식 `cash` 맵이 0 으로 덮여 잔액이 사라진다.
        cash_input = update.get("cash")
        has_cash_intent = isinstance(cash_input, dict) or any(
            key in update for key in ("cash_balance_krw", "cash_balance_native")
        )
        if not has_cash_intent:
            for key in ("cash", "cash_balance", "cash_balance_native"):
                row.pop(key, None)
        elif isinstance(cash_input, dict) and cash_input:
            cash_map: dict[str, float] = {}
            for key, value in cash_input.items():
                code = str(key or "").strip().upper()
                if code:
                    try:
                        cash_map[code] = float(value or 0)
                    except (TypeError, ValueError):
                        cash_map[code] = 0.0
            row["cash"] = cash_map
            # 레거시 cash_balance = 통화별 native 를 원화로 환산한 합계(대시보드·자산헬퍼 호환).
            from services.price_service import get_exchange_rates
            from utils.cash_model import cash_total_krw

            row["cash_balance"] = round(cash_total_krw(cash_map, get_exchange_rates()), 2)
        else:
            row["cash"] = resolve_cash_native_map(row, row["cash_currency"])

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
    from utils.snapshot_service import update_today_snapshot_all_accounts

    update_today_snapshot_all_accounts()
    return {"message": "자산 관리 저장 완료"}
