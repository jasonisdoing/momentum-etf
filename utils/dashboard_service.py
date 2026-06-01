from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from utils.account_registry import load_account_configs
from utils.daily_fund_service import calculate_period_return_pct, load_daily_docs_for_aggregation
from utils.db_manager import get_db_connection
from utils.logger import get_app_logger
from utils.monthly_service import _load_monthly_docs as _load_monthly_docs_with_running
from utils.normalization import normalize_number, to_iso_string
from utils.portfolio_io import load_real_holdings_table
from utils.yearly_service import _load_yearly_docs as _load_yearly_docs_with_running

logger = get_app_logger()


BUCKET_NAMES = ["1. 모멘텀", "2. 시장지수", "3. 배당방어", "4. 대체헷지"]


def _get_week_start_date_kst() -> str:
    now = datetime.now().astimezone()
    week_start = now.date() - timedelta(days=now.weekday())
    return week_start.isoformat()


def _find_latest_snapshot_before_week_start(db: Any) -> dict[str, Any] | None:
    week_start = _get_week_start_date_kst()
    return db.daily_snapshots.find_one(
        {"snapshot_date": {"$lt": week_start}},
        sort=[("snapshot_date", -1)],
    )


def _compute_account_buckets(
    account_id: str, cash_balance: float, df_live: pd.DataFrame | None = None
) -> list[dict[str, Any]]:
    """계좌 하나의 버킷별 비중을 계산한다."""
    bucket_totals: dict[str, float] = {name: 0.0 for name in BUCKET_NAMES}

    try:
        df = df_live if df_live is not None else load_real_holdings_table(account_id)
        if df is not None and not df.empty:
            for bucket_name in BUCKET_NAMES:
                bucket_totals[bucket_name] = float(df.loc[df["버킷"] == bucket_name, "평가금액(KRW)"].sum())
    except Exception as exc:
        logger.warning("계좌 %s 버킷 계산 실패: %s", account_id, exc)

    total = sum(bucket_totals.values()) + cash_balance
    if total <= 0:
        return [{"label": name, "weight_pct": 0.0} for name in BUCKET_NAMES] + [{"label": "5. 현금", "weight_pct": 0.0}]

    result = [{"label": name, "weight_pct": round((val / total) * 100, 2)} for name, val in bucket_totals.items()]
    result.append({"label": "5. 현금", "weight_pct": round((cash_balance / total) * 100, 2)})
    return result


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

    # 수익률 계산 규칙: weekly_return_pct = 입출금 제거 1주 수익률, cumulative_return_pct = ROI.
    # 상세는 docs/developer_guide.md (자산 수익률 계산 정책) 참고.
    # 시드 row(2023년 마지막 거래일) 의 deposit_withdrawal 에 초기 입금이 들어있고
    # 이후 row 들의 입출금이 누적되어 total_principal 이 계산된다.
    running_total = 0.0
    running_expense = 0.0
    previous_cumulative_profit = 0.0
    previous_total_assets = 0.0

    for week_date in sorted(docs_by_date.keys()):
        doc = docs_by_date[week_date]
        running_total += normalize_number(doc.get("deposit_withdrawal"))
        doc["total_principal"] = running_total

        running_expense += normalize_number(doc.get("total_expense"))
        cumulative_profit = (
            normalize_number(doc.get("total_assets")) - normalize_number(doc.get("total_principal")) - running_expense
        )
        weekly_profit = cumulative_profit - previous_cumulative_profit
        total_principal = normalize_number(doc.get("total_principal"))

        weekly_return_pct = calculate_period_return_pct(weekly_profit, previous_total_assets)
        cumulative_return_pct = (cumulative_profit / total_principal) * 100 if total_principal > 0 else 0.0

        doc["cumulative_profit"] = cumulative_profit
        doc["weekly_profit"] = weekly_profit
        doc["weekly_return_pct"] = weekly_return_pct
        doc["cumulative_return_pct"] = cumulative_return_pct

        previous_cumulative_profit = cumulative_profit
        previous_total_assets = normalize_number(doc.get("total_assets"))

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
    weekly_base_snapshot = _find_latest_snapshot_before_week_start(db)
    weekly_docs = [
        doc for doc in db.weekly_fund_data.find().sort("week_date", -1) if normalize_number(doc.get("total_assets")) > 0
    ]

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
    weekly_base_snapshot_accounts = {
        str(account.get("account_id") or ""): account
        for account in ((weekly_base_snapshot or {}).get("accounts") or [])
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
        # 실시간 평가액 직접 계산 (데이터 정합성 확보)
        df_live = load_real_holdings_table(config["account_id"])
        valuation_krw = float(df_live["평가금액(KRW)"].sum()) if df_live is not None else 0.0
        previous_snapshot_account = (
            next(
                (
                    account
                    for account in ((previous_snapshot or {}).get("accounts") or [])
                    if isinstance(account, dict) and str(account.get("account_id") or "") == config["account_id"]
                ),
                {},
            )
            if previous_snapshot
            else {}
        )
        weekly_base_snapshot_account = weekly_base_snapshot_accounts.get(config["account_id"], {})

        total_assets = valuation_krw + cash_balance
        net_profit = total_assets - total_principal
        net_profit_pct = (net_profit / total_principal) * 100 if total_principal > 0 else 0.0
        cash_ratio = (cash_balance / total_assets) * 100 if total_assets > 0 else 0.0
        previous_total_assets = normalize_number(previous_snapshot_account.get("total_assets"))
        weekly_base_total_assets = normalize_number(weekly_base_snapshot_account.get("total_assets"))
        # daily_profit / weekly_profit 은 입출금 영향을 제거한 시장 변동분만 계산한다.
        previous_account_principal = normalize_number(previous_snapshot_account.get("total_principal"))
        weekly_base_account_principal = normalize_number(weekly_base_snapshot_account.get("total_principal"))
        daily_deposit = (total_principal - previous_account_principal) if previous_snapshot_account else 0.0
        weekly_deposit = (total_principal - weekly_base_account_principal) if weekly_base_snapshot_account else 0.0
        daily_profit = (total_assets - previous_total_assets - daily_deposit) if previous_snapshot_account else 0.0
        weekly_profit = (
            (total_assets - weekly_base_total_assets - weekly_deposit) if weekly_base_snapshot_account else 0.0
        )

        daily_return_pct_acc = calculate_period_return_pct(daily_profit, previous_total_assets)
        weekly_return_pct_acc = calculate_period_return_pct(weekly_profit, weekly_base_total_assets)
        accounts.append(
            {
                "account_id": config["account_id"],
                "account_name": config["name"],
                "account_url": str(config.get("settings", {}).get("URL") or "").strip() or None,
                "order": config["order"],
                "total_assets": total_assets,
                "total_principal": total_principal,
                "valuation_krw": valuation_krw,
                "cash_balance": cash_balance,
                "cash_ratio": cash_ratio,
                "net_profit": net_profit,
                "net_profit_pct": net_profit_pct,
                "daily_profit": daily_profit,
                "daily_return_pct": daily_return_pct_acc,
                "weekly_profit": weekly_profit,
                "weekly_return_pct": weekly_return_pct_acc,
                "_df_live": df_live,  # 버킷 계산을 위해 임시 보관
            }
        )

    total_assets = sum(account["total_assets"] for account in accounts)
    total_principal = sum(account["total_principal"] for account in accounts)
    total_cash = sum(account["cash_balance"] for account in accounts)
    # /assets 와 /daily, /weekly 의 일/주 수익률을 일치시키기 위해
    # daily_fund_data 와 weekly_fund_data 에서 마지막 row 를 직접 사용한다.
    # (자산 수익률 계산 정책: docs/developer_guide.md 참고)
    try:
        latest_daily_doc = next(iter(load_daily_docs_for_aggregation()), None)
    except Exception as exc:
        logger.warning("daily_fund_data 조회 실패: %s", exc)
        latest_daily_doc = None
    daily_profit = normalize_number((latest_daily_doc or {}).get("daily_profit"))
    daily_return_pct = normalize_number((latest_daily_doc or {}).get("daily_return_pct"))
    weekly_profit = normalize_number((latest_weekly or {}).get("weekly_profit"))
    weekly_return_pct = normalize_number((latest_weekly or {}).get("weekly_return_pct"))

    # 월별/년별 최신 doc 의 금월/금년 손익 (캐시 누락 시 0 으로 폴백)
    try:
        latest_monthly_doc = next(iter(_load_monthly_docs_with_running()), None)
    except Exception as exc:
        logger.warning("monthly_fund_data 조회 실패: %s", exc)
        latest_monthly_doc = None
    monthly_profit = normalize_number((latest_monthly_doc or {}).get("monthly_profit"))
    monthly_return_pct = normalize_number((latest_monthly_doc or {}).get("monthly_return_pct"))

    try:
        latest_yearly_doc = next(iter(_load_yearly_docs_with_running()), None)
    except Exception as exc:
        logger.warning("yearly_fund_data 조회 실패: %s", exc)
        latest_yearly_doc = None
    yearly_profit = normalize_number((latest_yearly_doc or {}).get("yearly_profit"))
    yearly_return_pct = normalize_number((latest_yearly_doc or {}).get("yearly_return_pct"))

    period_profits = {
        "daily": {"profit": daily_profit, "return_pct": daily_return_pct},
        "weekly": {"profit": weekly_profit, "return_pct": weekly_return_pct},
        "monthly": {"profit": monthly_profit, "return_pct": monthly_return_pct},
        "yearly": {"profit": yearly_profit, "return_pct": yearly_return_pct},
    }

    metrics_row1 = [
        {"label": "총 자산", "value": total_assets, "kind": "money"},
        {"label": "투자 원금", "value": total_principal, "kind": "money"},
        {
            "label": "금일 손익",
            "value": daily_profit,
            "kind": "money",
            "sub_value": daily_return_pct,
            "sub_kind": "percent",
        },
        {
            "label": "금주 손익",
            "value": weekly_profit,
            "kind": "money",
            "sub_value": weekly_return_pct,
            "sub_kind": "percent",
        },
    ]

    profit_count = normalize_number((latest_weekly or {}).get("profit_count"))
    loss_count = normalize_number((latest_weekly or {}).get("loss_count"))

    # 평가 손익 (인출분 합산): 총 자산 − 투자 원금 (인출액도 수익에 포함한 운용 성과 총합)
    gross_cumulative_profit = total_assets - total_principal
    gross_cumulative_return_pct = (
        round((gross_cumulative_profit / total_principal) * 100, 2) if total_principal > 0 else 0.0
    )

    metrics_row2 = [
        {
            "label": "누적 손익",
            "value": normalize_number((latest_weekly or {}).get("cumulative_profit")),
            "kind": "money",
            "sub_value": normalize_number((latest_weekly or {}).get("cumulative_return_pct")),
            "sub_kind": "percent",
        },
        {
            "label": "평가 손익 (인출분 합산)",
            "value": gross_cumulative_profit,
            "kind": "money",
            "sub_value": gross_cumulative_return_pct,
            "sub_kind": "percent",
        },
        {"label": "현금 잔고", "value": total_cash, "kind": "money"},
        {
            "label": "현금 비중",
            "value": (total_cash / total_assets) * 100 if total_assets > 0 else 0.0,
            "kind": "percent",
        },
        {
            "label": "수익/손실 종목 수",
            "value": profit_count,
            "kind": "count",
            "sub_value": loss_count,
            "sub_kind": "count",
        },
    ]

    momentum_weight_pct = normalize_number((latest_weekly or {}).get("bucket_pct_momentum"))
    buckets = [
        {"label": "1. 모멘텀", "weight_pct": momentum_weight_pct},
        {"label": "2. 시장지수", "weight_pct": normalize_number((latest_weekly or {}).get("bucket_pct_market"))},
        {"label": "3. 배당방어", "weight_pct": normalize_number((latest_weekly or {}).get("bucket_pct_dividend"))},
        {"label": "4. 대체헷지", "weight_pct": normalize_number((latest_weekly or {}).get("bucket_pct_alternative"))},
        {"label": "5. 현금", "weight_pct": normalize_number((latest_weekly or {}).get("bucket_pct_cash"))},
    ]

    updated_at_candidates = [
        to_iso_string(f"{latest_snapshot.get('snapshot_date')}T00:00:00+09:00")
        if latest_snapshot and latest_snapshot.get("snapshot_date")
        else None,
        to_iso_string((latest_weekly or {}).get("updated_at")),
        *[to_iso_string(account.get("updated_at")) for account in portfolio_accounts.values()],
    ]
    updated_at_values = sorted([value for value in updated_at_candidates if value])

    # 스파크라인용 주별 히스토리 (최근 52주, 오래된 순)
    calculated_weekly = _calculate_weekly_docs(weekly_docs) if weekly_docs else []
    sparkline_source = list(reversed(calculated_weekly))[-52:]
    dates = [str(d.get("week_date") or "") for d in sparkline_source]

    def _spark(values: list[float]) -> list[dict[str, object]]:
        return [{"date": dt, "value": v} for dt, v in zip(dates, values)]

    sparklines: dict[str, list[dict[str, object]]] = {
        "총 자산": _spark([normalize_number(d.get("total_assets")) for d in sparkline_source]),
        "투자 원금": _spark([normalize_number(d.get("total_principal")) for d in sparkline_source]),
        "누적 손익": _spark([normalize_number(d.get("cumulative_profit")) for d in sparkline_source]),
        "평가 손익 (인출분 합산)": _spark(
            [
                normalize_number(d.get("total_assets")) - normalize_number(d.get("total_principal"))
                for d in sparkline_source
            ]
        ),
        "현금 잔고": _spark(
            [
                normalize_number(d.get("total_principal")) - normalize_number(d.get("purchase_amount"))
                for d in sparkline_source
            ]
        ),
    }

    # 계좌별 버킷 비중 계산
    account_buckets: dict[str, list[dict[str, Any]]] = {}
    for account in accounts:
        aid = account["account_id"]
        # 임시 보관한 실시간 데이터를 넘겨주어 중복 조회를 방지한다.
        df_live = account.pop("_df_live", None)
        account_buckets[aid] = _compute_account_buckets(aid, account["cash_balance"], df_live=df_live)

    try:
        from utils.system_service import is_deploying as _is_deploying

        deploying = bool(_is_deploying())
    except Exception:
        deploying = False

    return {
        "metrics_row1": metrics_row1,
        "metrics_row2": metrics_row2,
        "period_profits": period_profits,
        "is_deploying": deploying,
        "accounts": accounts,
        "totals": {
            "total_assets": total_assets,
            "total_principal": total_principal,
            "daily_profit": daily_profit,
            "daily_return_pct": daily_return_pct,
            "weekly_profit": weekly_profit,
            "weekly_return_pct": weekly_return_pct,
        },
        "buckets": buckets,
        "account_buckets": account_buckets,
        "sparklines": sparklines,
        "latest_snapshot_date": latest_snapshot.get("snapshot_date") if latest_snapshot else None,
        "weekly_date": (latest_weekly or {}).get("week_date"),
        "updated_at": updated_at_values[-1] if updated_at_values else None,
    }
