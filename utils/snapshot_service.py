from __future__ import annotations

from typing import Any

from utils.account_registry import load_account_configs
from utils.db_manager import get_db_connection


def _normalize_number(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def load_snapshot_list() -> list[dict[str, Any]]:
    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결에 실패했습니다.")

    docs = list(db.daily_snapshots.find().sort("snapshot_date", -1))
    configs = load_account_configs()
    account_map = {config["account_id"]: {"name": config["name"], "order": config["order"]} for config in configs}

    snapshots: list[dict[str, Any]] = []
    for doc in docs:
        accounts = sorted(
            [
                {
                    "account_id": str(account.get("account_id") or ""),
                    "account_name": account_map.get(str(account.get("account_id") or ""), {}).get(
                        "name",
                        str(account.get("account_id") or ""),
                    ),
                    "order": int(
                        account_map.get(str(account.get("account_id") or ""), {}).get("order", 999),
                    ),
                    "total_assets": _normalize_number(account.get("total_assets")),
                    "total_principal": _normalize_number(account.get("total_principal")),
                    "cash_balance": _normalize_number(account.get("cash_balance")),
                    "valuation_krw": _normalize_number(account.get("valuation_krw")),
                }
                for account in (doc.get("accounts") or [])
                if isinstance(account, dict)
            ],
            key=lambda item: item["order"],
        )

        snapshots.append(
            {
                "id": str(doc.get("_id")),
                "snapshot_date": str(doc.get("snapshot_date") or ""),
                "total_assets": _normalize_number(doc.get("total_assets")),
                "total_principal": _normalize_number(doc.get("total_principal")),
                "cash_balance": _normalize_number(doc.get("cash_balance")),
                "valuation_krw": _normalize_number(doc.get("valuation_krw")),
                "account_count": len(accounts),
                "accounts": accounts,
            }
        )

    return snapshots
