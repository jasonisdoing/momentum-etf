from __future__ import annotations

import datetime
from typing import Any
from zoneinfo import ZoneInfo

from utils.account_registry import load_account_configs
from utils.cash_model import resolve_cash_currencies, resolve_cash_native_map
from utils.db_manager import get_db_connection
from utils.normalization import normalize_nullable_number, normalize_number, to_iso_string

KST = ZoneInfo("Asia/Seoul")


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


def save_cash_accounts(updates: list[dict[str, Any]]) -> dict[str, Any]:
    """현금·원금을 저장하고, 저장된 계좌의 현금 상태를 함께 돌려준다.

    반환의 ``accounts`` 는 화면이 **전체 리로드 없이** 현금 칸과 파생값(총자산·현금비중)을
    바로 갱신하는 데 쓴다. 통화별 native 맵을 원화로 합치려면 환율이 필요해 화면이
    스스로 계산할 수 없다 — 그래서 서버가 계산한 값을 그대로 내려준다.
    """
    if not updates:
        raise ValueError("저장할 계좌 데이터가 없습니다.")

    db = _require_db()
    collection = db.portfolio_master
    doc = collection.find_one({"master_id": "GLOBAL"}) or {"master_id": "GLOBAL", "accounts": []}
    accounts = list(doc.get("accounts") or [])
    # tz 를 붙여 저장한다 — naive 로 넣으면 Mongo 가 그 값을 UTC 로 보관하고, 읽는 쪽
    # (to_iso_string)도 UTC 로 해석해 화면에 9시간 뒤(미래)로 찍힌다.
    now = datetime.datetime.now(KST)

    saved: list[dict[str, Any]] = []
    for update in updates:
        account_id = str(update.get("account_id") or "").strip()
        if not account_id:
            raise ValueError("account_id가 필요합니다.")

        row = {
            "account_id": account_id,
            "total_principal": float(update.get("total_principal") or 0),
            "cash_currency": str(update.get("cash_currency") or "").strip().upper(),
            "cash_target_ratio": float(update.get("cash_target_ratio") or 0),
            "intl_shares_value": normalize_nullable_number(update.get("intl_shares_value")),
            "intl_shares_change": normalize_nullable_number(update.get("intl_shares_change")),
            "updated_at": now,
            "updated_by": "user",
        }

        # 현금의 진실은 통화별 `cash` 맵 하나다. 원화 합(cash_balance)·계좌 통화 잔액
        # (cash_balance_native)은 맵에서 **파생**해 캐시로만 갱신한다(레거시 읽기 호환).
        # 레거시 단일 금액으로 현금을 바꾸는 저장은 폐기(2026-09) — 원금만 저장하는 요청이
        # 현금 필드를 같이 보내면 백엔드가 맵을 재합성해, USD 잔액이 통째로 KRW 로 바뀌는
        # 사고가 났다. 현금 키가 아예 없는 요청(원금·비율·Intl 저장)은 현금을 건드리지 않는다.
        cash_input = update.get("cash")
        if isinstance(cash_input, dict) and cash_input:
            cash_map: dict[str, float] = {}
            for key, value in cash_input.items():
                code = str(key or "").strip().upper()
                if code:
                    try:
                        cash_map[code] = float(value or 0)
                    except (TypeError, ValueError):
                        cash_map[code] = 0.0
            row["cash"] = cash_map
            from services.price_service import get_exchange_rates
            from utils.cash_model import cash_total_krw

            row["cash_balance"] = round(cash_total_krw(cash_map, get_exchange_rates()), 2)
            row["cash_balance_native"] = cash_map.get(row["cash_currency"]) if row["cash_currency"] else None
        elif any(key in update for key in ("cash_balance_krw", "cash_balance_native")):
            raise ValueError(
                "현금 금액은 통화별 금액(cash 맵)으로만 저장합니다 — 단일 환산 금액 저장은 통화별 잔액을 덮어써 폐기했습니다."
            )

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

        merged = accounts[index] if index >= 0 else row
        saved.append(
            {
                "account_id": account_id,
                "cash": merged.get("cash") or {},
                "cash_balance_krw": normalize_number(merged.get("cash_balance")),
                "cash_balance_native": normalize_nullable_number(merged.get("cash_balance_native")),
                "cash_target_ratio": normalize_number(merged.get("cash_target_ratio")),
                "total_principal": normalize_number(merged.get("total_principal")),
                "updated_at": to_iso_string(now),
                "updated_by": "user",
            }
        )

    collection.update_one({"master_id": "GLOBAL"}, {"$set": {"accounts": accounts}}, upsert=True)
    from utils.snapshot_service import refresh_today_snapshot_async

    refresh_today_snapshot_async()
    return {"message": "자산 관리 저장 완료", "accounts": saved}
