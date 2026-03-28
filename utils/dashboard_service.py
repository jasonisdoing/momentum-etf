from __future__ import annotations

from typing import Any

from utils.account_registry import load_account_configs
from utils.db_manager import get_db_connection
from utils.normalization import normalize_number, to_iso_string

INITIAL_TOTAL_PRINCIPAL_DATE = "2024-01-31"
INITIAL_TOTAL_PRINCIPAL_VALUE = 56_000_000


def _calculate_weekly_docs(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    docs_by_date: dict[str, dict[str, Any]] = {}
    for doc in docs:
        week_date = str(doc.get("week_date") or "")
        docs_by_date[week_date] = {
            **doc,
            "total_expense": normalize_number(doc.get("withdrawal_personal"))
            + normalize_number(doc.get("withdrawal_mom"))
            + normalize_number(doc.get("nh_principal_interest")),
            "total_stocks": normalize_number(doc.get("profit_count")) + normalize_number(doc.get("loss_count")),
        }

    running_total = float(INITIAL_TOTAL_PRINCIPAL_VALUE)
    running_expense = 0.0
    previous_cumulative_profit = 0.0

    for week_date in sorted(docs_by_date.keys()):
        doc = docs_by_date[week_date]

        if week_date <= INITIAL_TOTAL_PRINCIPAL_DATE:
            doc["total_principal"] = float(INITIAL_TOTAL_PRINCIPAL_VALUE)
        else:
            running_total += normalize_number(doc.get("deposit_withdrawal"))
            doc["total_principal"] = running_total

        running_expense += normalize_number(doc.get("total_expense"))
        cumulative_profit = (
            normalize_number(doc.get("total_assets")) - normalize_number(doc.get("total_principal")) - running_expense
        )
        weekly_profit = cumulative_profit - previous_cumulative_profit
        total_principal = normalize_number(doc.get("total_principal"))

        if total_principal > 0:
            weekly_return_pct = (weekly_profit / total_principal) * 100
            cumulative_return_pct = (cumulative_profit / total_principal) * 100
        else:
            weekly_return_pct = 0.0
            cumulative_return_pct = 0.0

        doc["cumulative_profit"] = cumulative_profit
        doc["weekly_profit"] = weekly_profit
        doc["weekly_return_pct"] = weekly_return_pct
        doc["cumulative_return_pct"] = cumulative_return_pct

        previous_cumulative_profit = cumulative_profit

    return [
        docs_by_date[str(doc.get("week_date") or "")]
        for doc in sorted(docs, key=lambda item: str(item.get("week_date") or ""), reverse=True)
    ]


def load_dashboard_data() -> dict[str, Any]:
    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결에 실패했습니다.")

    configs = load_account_configs()
    portfolio_doc = db.portfolio_master.find_one({"master_id": "GLOBAL"}) or {}
    snapshot_docs = list(db.daily_snapshots.find().sort("snapshot_date", -1).limit(2))
    weekly_docs = list(db.weekly_fund_data.find().sort("week_date", -1))

    latest_snapshot = snapshot_docs[0] if snapshot_docs else None
    previous_snapshot = snapshot_docs[1] if len(snapshot_docs) > 1 else None
    latest_weekly = _calculate_weekly_docs(weekly_docs)[0] if weekly_docs else None

    portfolio_accounts = {
        str(account.get("account_id") or ""): account
        for account in (portfolio_doc.get("accounts") or [])
        if isinstance(account, dict)
    }
    snapshot_accounts = {
        str(account.get("account_id") or ""): account
        for account in ((latest_snapshot or {}).get("accounts") or [])
        if isinstance(account, dict)
    }

    accounts: list[dict[str, Any]] = []
    for config in configs:
        portfolio_account = portfolio_accounts.get(config["account_id"], {})
        snapshot_account = snapshot_accounts.get(config["account_id"], {})
        total_principal = normalize_number(
            portfolio_account.get("total_principal", snapshot_account.get("total_principal"))
        )
        cash_balance = normalize_number(portfolio_account.get("cash_balance", snapshot_account.get("cash_balance")))
        valuation_krw = normalize_number(snapshot_account.get("valuation_krw"))
        total_assets = valuation_krw + cash_balance
        net_profit = total_assets - total_principal
        net_profit_pct = (net_profit / total_principal) * 100 if total_principal > 0 else 0.0
        cash_ratio = (cash_balance / total_assets) * 100 if total_assets > 0 else 0.0

        accounts.append(
            {
                "account_id": config["account_id"],
                "account_name": config["name"],
                "order": config["order"],
                "total_assets": total_assets,
                "total_principal": total_principal,
                "valuation_krw": valuation_krw,
                "cash_balance": cash_balance,
                "cash_ratio": cash_ratio,
                "net_profit": net_profit,
                "net_profit_pct": net_profit_pct,
            }
        )

    total_assets = sum(account["total_assets"] for account in accounts)
    total_principal = sum(account["total_principal"] for account in accounts)
    total_cash = sum(account["cash_balance"] for account in accounts)
    valuation_amount = sum(account["valuation_krw"] for account in accounts)
    previous_total_assets = normalize_number((previous_snapshot or {}).get("total_assets"))
    daily_profit = total_assets - previous_total_assets if previous_snapshot else 0.0
    daily_return_pct = (daily_profit / previous_total_assets) * 100 if previous_total_assets > 0 else 0.0

    metrics = [
        {"label": "총 자산", "value": total_assets, "kind": "money"},
        {"label": "투자 원금", "value": total_principal, "kind": "money"},
        {"label": "현금 잔고", "value": total_cash, "kind": "money"},
        {"label": "금일 손익", "value": daily_profit, "kind": "money"},
        {"label": "금주 손익", "value": normalize_number((latest_weekly or {}).get("weekly_profit")), "kind": "money"},
        {
            "label": "누적 손익",
            "value": normalize_number((latest_weekly or {}).get("cumulative_profit")),
            "kind": "money",
        },
    ]

    buckets = [
        {"label": "1. 모멘텀", "weight_pct": normalize_number((latest_weekly or {}).get("bucket_pct_momentum"))},
        {"label": "2. 혁신기술", "weight_pct": normalize_number((latest_weekly or {}).get("bucket_pct_innovation"))},
        {"label": "3. 시장지수", "weight_pct": normalize_number((latest_weekly or {}).get("bucket_pct_market"))},
        {"label": "4. 배당방어", "weight_pct": normalize_number((latest_weekly or {}).get("bucket_pct_dividend"))},
        {"label": "5. 대체헷지", "weight_pct": normalize_number((latest_weekly or {}).get("bucket_pct_alternative"))},
        {"label": "6. 현금", "weight_pct": normalize_number((latest_weekly or {}).get("bucket_pct_cash"))},
    ]

    stats = [
        {
            "label": "매입 금액",
            "value": normalize_number((latest_weekly or {}).get("purchase_amount")),
            "kind": "money",
        },
        {
            "label": "평가 금액",
            "value": valuation_amount or normalize_number((latest_weekly or {}).get("valuation_amount")),
            "kind": "money",
        },
        {
            "label": "현금 비중",
            "value": (total_cash / total_assets) * 100 if total_assets > 0 else 0.0,
            "kind": "percent",
        },
        {"label": "일간 수익률", "value": daily_return_pct, "kind": "percent"},
        {
            "label": "주 수익률",
            "value": normalize_number((latest_weekly or {}).get("weekly_return_pct")),
            "kind": "percent",
        },
        {
            "label": "누적 수익률",
            "value": normalize_number((latest_weekly or {}).get("cumulative_return_pct")),
            "kind": "percent",
        },
        {
            "label": "수익 종목 수",
            "value": normalize_number((latest_weekly or {}).get("profit_count")),
            "kind": "count",
        },
        {"label": "손실 종목 수", "value": normalize_number((latest_weekly or {}).get("loss_count")), "kind": "count"},
    ]

    updated_at_candidates = [
        to_iso_string(f"{latest_snapshot.get('snapshot_date')}T00:00:00+09:00")
        if latest_snapshot and latest_snapshot.get("snapshot_date")
        else None,
        to_iso_string((latest_weekly or {}).get("updated_at")),
        *[to_iso_string(account.get("updated_at")) for account in portfolio_accounts.values()],
    ]
    updated_at_values = sorted([value for value in updated_at_candidates if value])

    return {
        "metrics": metrics,
        "accounts": accounts,
        "buckets": buckets,
        "stats": stats,
        "latest_snapshot_date": latest_snapshot.get("snapshot_date") if latest_snapshot else None,
        "weekly_date": (latest_weekly or {}).get("week_date"),
        "updated_at": updated_at_values[-1] if updated_at_values else None,
    }
