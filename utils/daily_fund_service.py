from __future__ import annotations

import datetime
from typing import Any
from zoneinfo import ZoneInfo

from services.price_service import get_exchange_rate_series
from utils.account_registry import load_account_configs
from utils.db_manager import get_db_connection
from utils.normalization import to_iso_string
from utils.portfolio_io import load_portfolio_master, load_real_holdings_table

DAILY_COLLECTION = "daily_fund_data"
WEEKLY_COLLECTION = "weekly_fund_data"
KST = ZoneInfo("Asia/Seoul")

READ_ONLY_FIELDS = {
    "total_expense",
    "total_principal",
    "total_assets",
    "purchase_amount",
    "valuation_amount",
    "profit_loss",
    "daily_profit",
    "cumulative_profit",
    "daily_return_pct",
    "cumulative_return_pct",
    "exchange_rate",
    "bucket_pct_momentum",
    "bucket_pct_market",
    "bucket_pct_dividend",
    "bucket_pct_alternative",
    "bucket_pct_cash",
    "total_stocks",
    "profit_count",
    "loss_count",
}
CORE_VIEW_HIDDEN_KEYS = [
    "withdrawal_personal",
    "withdrawal_mom",
    "nh_principal_interest",
    "deposit_withdrawal",
]

FIELD_DEFS = [
    {"key": "withdrawal_personal", "label": "개인 인출", "type": "int"},
    {"key": "withdrawal_mom", "label": "엄마", "type": "int"},
    {"key": "nh_principal_interest", "label": "농협원리금", "type": "int"},
    {"key": "total_expense", "label": "지출 합계", "type": "int"},
    {"key": "deposit_withdrawal", "label": "입출금", "type": "int"},
    {"key": "total_principal", "label": "총 원금", "type": "int"},
    {"key": "total_assets", "label": "총 자산", "type": "int"},
    {"key": "purchase_amount", "label": "매입 금액", "type": "int"},
    {"key": "valuation_amount", "label": "평가 금액", "type": "int"},
    {"key": "profit_loss", "label": "평가 손익", "type": "int"},
    {"key": "daily_profit", "label": "금일 손익", "type": "int"},
    {"key": "cumulative_profit", "label": "누적 손익", "type": "int"},
    {"key": "daily_return_pct", "label": "일수익률 (%)", "type": "float"},
    {"key": "cumulative_return_pct", "label": "누적 수익률 (%)", "type": "float"},
    {"key": "memo", "label": "비고", "type": "text"},
    {"key": "exchange_rate", "label": "환율", "type": "float"},
    {"key": "bucket_pct_momentum", "label": "1. 모멘텀 (%)", "type": "float"},
    {"key": "bucket_pct_market", "label": "2. 시장지수 (%)", "type": "float"},
    {"key": "bucket_pct_dividend", "label": "3. 배당방어 (%)", "type": "float"},
    {"key": "bucket_pct_alternative", "label": "4. 대체헷지 (%)", "type": "float"},
    {"key": "bucket_pct_cash", "label": "5. 현금 (%)", "type": "float"},
    {"key": "total_stocks", "label": "총 종목 수", "type": "int"},
    {"key": "profit_count", "label": "수익 종목 수", "type": "int"},
    {"key": "loss_count", "label": "손실 종목 수", "type": "int"},
]


def _require_db() -> Any:
    db = get_db_connection()
    if db is None:
        raise RuntimeError("DB 연결 실패")
    return db


def _get_now_kst() -> datetime.datetime:
    return datetime.datetime.now(KST)


def _get_active_daily_date() -> str:
    """오늘 이하의 마지막 한국/호주 합집합 거래일을 반환한다.

    자산 스냅샷(`daily_snapshots`)과 같은 기준을 써야 해서 공용 함수를 부른다.
    """
    from utils.data_loader import resolve_active_trading_date

    return resolve_active_trading_date()


def _to_int(value: object) -> int:
    return int(float(value or 0))


def _to_float(value: object) -> float:
    return float(value or 0.0)


def calculate_period_return_pct(period_profit: float, previous_total_assets: float) -> float:
    """입출금 제거 손익을 직전 기간 총자산 대비 수익률로 계산한다."""
    if previous_total_assets <= 0:
        return 0.0
    return (period_profit / previous_total_assets) * 100


def _format_date_display(date_str: str) -> str:
    weekday_kr = ["월", "화", "수", "목", "금", "토", "일"]
    dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    return f"{dt.year}. {dt.month}. {dt.day} ({weekday_kr[dt.weekday()]})"


def _normalize_bucket_percentages(source: dict[str, Any]) -> dict[str, float]:
    return {
        "bucket_pct_momentum": _to_float(source.get("bucket_pct_momentum", 0.0)),
        "bucket_pct_market": _to_float(source.get("bucket_pct_market", 0.0)),
        "bucket_pct_dividend": _to_float(source.get("bucket_pct_dividend", 0.0)),
        "bucket_pct_alternative": _to_float(source.get("bucket_pct_alternative", 0.0)),
        "bucket_pct_cash": _to_float(source.get("bucket_pct_cash", 0.0)),
    }


def _calculate_total_expense(source: dict[str, Any]) -> int:
    return (
        _to_int(source.get("withdrawal_personal", 0))
        + _to_int(source.get("withdrawal_mom", 0))
        + _to_int(source.get("nh_principal_interest", 0))
    )


def _calculate_profit_loss(source: dict[str, Any]) -> int:
    valuation_amount = _to_int(source.get("valuation_amount", 0))
    purchase_amount = _to_int(source.get("purchase_amount", 0))
    return valuation_amount - purchase_amount


def _calculate_total_stocks(source: dict[str, Any]) -> int:
    return _to_int(source.get("profit_count", 0)) + _to_int(source.get("loss_count", 0))


def _apply_derived_fields(source: dict[str, Any]) -> dict[str, Any]:
    updated = dict(source)
    updated["total_expense"] = _calculate_total_expense(updated)
    updated["profit_loss"] = _calculate_profit_loss(updated)
    updated["total_stocks"] = _calculate_total_stocks(updated)
    return updated


def _apply_running_total_principal(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """수익률 계산 규칙 (전체 페이지 통일):

    - daily_return_pct (입출금 제거 1일 수익률):
        daily_profit / previous_total_assets × 100
        분자는 입출금 제거된 순수 시장변동, 분모는 직전 기간 총자산.
    - cumulative_return_pct (ROI):
        cumulative_profit / total_principal × 100
        내가 투입한 누적 원금 대비 총 수익률.

    자세한 정책은 docs/developer_guide.md (자산 수익률 계산 정책) 참고.
    """
    docs_by_date = {str(doc["date"]): _apply_derived_fields(doc) for doc in docs}
    # 시드 row(2023-12-28) 의 deposit_withdrawal 에 초기 입금이 들어있고
    # 이후 row 들의 입출금이 누적되어 total_principal 이 계산된다.
    running_total = 0
    running_total_expense = 0
    previous_cumulative_profit = 0
    previous_total_assets = 0

    for date_str in sorted(docs_by_date):
        doc = docs_by_date[date_str]
        running_total += _to_int(doc.get("deposit_withdrawal", 0))
        doc["total_principal"] = running_total

        running_total_expense += _to_int(doc.get("total_expense", 0))
        doc["cumulative_profit"] = (
            _to_int(doc.get("total_assets", 0)) - _to_int(doc.get("total_principal", 0)) - running_total_expense
        )
        doc["daily_profit"] = _to_int(doc.get("cumulative_profit", 0)) - previous_cumulative_profit
        total_principal = _to_int(doc.get("total_principal", 0))
        doc["daily_return_pct"] = round(
            calculate_period_return_pct(_to_int(doc.get("daily_profit", 0)), previous_total_assets),
            2,
        )
        if total_principal == 0:
            doc["cumulative_return_pct"] = 0.0
        else:
            doc["cumulative_return_pct"] = round((_to_int(doc.get("cumulative_profit", 0)) / total_principal) * 100, 2)
        previous_cumulative_profit = _to_int(doc.get("cumulative_profit", 0))
        previous_total_assets = _to_int(doc.get("total_assets", 0))

    return [docs_by_date[str(doc["date"])] for doc in sorted(docs, key=lambda item: item["date"], reverse=True)]


def _doc_to_api_row(doc: dict[str, Any]) -> dict[str, Any]:
    computed_doc = _apply_derived_fields(doc)
    exchange_rate = _to_float(computed_doc.get("exchange_rate", 0.0))
    bucket_percentages = _normalize_bucket_percentages(computed_doc)

    return {
        "date": str(computed_doc["date"]),
        "date_display": _format_date_display(str(computed_doc["date"])),
        "withdrawal_personal": _to_int(computed_doc.get("withdrawal_personal", 0)),
        "withdrawal_mom": _to_int(computed_doc.get("withdrawal_mom", 0)),
        "nh_principal_interest": _to_int(computed_doc.get("nh_principal_interest", 0)),
        "total_expense": _to_int(computed_doc.get("total_expense", 0)),
        "deposit_withdrawal": _to_int(computed_doc.get("deposit_withdrawal", 0)),
        "total_principal": _to_int(computed_doc.get("total_principal", 0)),
        "total_assets": _to_int(computed_doc.get("total_assets", 0)),
        "purchase_amount": _to_int(computed_doc.get("purchase_amount", 0)),
        "valuation_amount": _to_int(computed_doc.get("valuation_amount", 0)),
        "profit_loss": _to_int(computed_doc.get("profit_loss", 0)),
        "daily_profit": _to_int(computed_doc.get("daily_profit", 0)),
        "cumulative_profit": _to_int(computed_doc.get("cumulative_profit", 0)),
        "daily_return_pct": round(float(computed_doc.get("daily_return_pct", 0.0) or 0.0), 2),
        "cumulative_return_pct": round(float(computed_doc.get("cumulative_return_pct", 0.0) or 0.0), 2),
        "memo": str(computed_doc.get("memo", "") or ""),
        "exchange_rate": round(exchange_rate, 2),
        "exchange_rate_change_pct": 0.0,
        **{key: round(value, 2) for key, value in bucket_percentages.items()},
        "total_stocks": _to_int(computed_doc.get("total_stocks", 0)),
        "profit_count": _to_int(computed_doc.get("profit_count", 0)),
        "loss_count": _to_int(computed_doc.get("loss_count", 0)),
        "updated_at": to_iso_string(computed_doc.get("updated_at")),
    }


def _build_api_rows(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [_doc_to_api_row(doc) for doc in docs]
    for idx, row in enumerate(rows):
        current_rate = float(row.get("exchange_rate", 0.0) or 0.0)
        older_rate = float(rows[idx + 1].get("exchange_rate", 0.0) or 0.0) if idx + 1 < len(rows) else 0.0
        if older_rate > 0:
            row["exchange_rate_change_pct"] = round(((current_rate / older_rate) - 1.0) * 100, 2)
        else:
            row["exchange_rate_change_pct"] = 0.0
    return rows


def _get_live_exchange_rate() -> float:
    now = datetime.datetime.now(KST).replace(tzinfo=None)
    start = now - datetime.timedelta(days=5)
    series = get_exchange_rate_series(start, now)
    if series.empty:
        return 0.0
    return float(series.iloc[-1])


# 총자산 급변 허용 한도(%) — 직전 기록 대비 이 비율을 넘는 변동(입출금 제외)은
# 캐시/조회 사고로 간주하고 일별 집계 기록을 거부한다.
# 2026-07-24 사고(해외 평가액 누락)가 -16.6% 였으므로 그보다 낮게 잡는다.
# (현금 비중이 높아 정상 일일 변동은 1% 미만 — 10% 면 여유가 충분하다)
DAILY_TOTAL_JUMP_GUARD_PCT = 10.0


def _collect_live_daily_summary() -> dict[str, Any]:
    total_assets = 0.0
    total_purchase = 0.0
    total_valuation = 0.0
    total_cash = 0.0
    total_profit_count = 0
    total_loss_count = 0
    live_exchange_rate = _get_live_exchange_rate()
    bucket_totals = {
        "1. 모멘텀": 0.0,
        "2. 시장지수": 0.0,
        "3. 배당방어": 0.0,
        "4. 대체헷지": 0.0,
    }
    all_missing_tickers: list[str] = []
    all_zero_valuation_tickers: list[str] = []

    for account in load_account_configs():
        if not account.get("settings", {}).get("show_hold", True):
            continue

        account_id = account["account_id"]
        master_data = load_portfolio_master(account_id)
        cash_balance = float(master_data.get("cash_balance", 0.0) if master_data else 0.0)
        total_cash += cash_balance

        holdings_df = load_real_holdings_table(account_id)
        if holdings_df is None or holdings_df.empty:
            # 원장에는 보유가 있는데 테이블이 비면 조회 실패다 — 평가액 0 으로 기록하면
            # 총자산이 내려앉으므로 집계를 중단한다 (2026-07-24 캐시 사고 재발 방지).
            master_holding_count = len((master_data or {}).get("holdings") or [])
            if master_holding_count > 0:
                raise RuntimeError(
                    f"'{account_id}' 계좌 보유 {master_holding_count}종목의 조회가 실패해 일별 집계를 중단합니다."
                )
            account_purchase = 0.0
            account_valuation = 0.0
        else:
            missing = holdings_df.attrs.get("missing_price_tickers") or []
            if missing:
                all_missing_tickers.extend(missing)
            # 수량이 있는데 평가금액이 0 이하인 종목 = 가격을 못 얻은 종목.
            # (캐시 문서는 있으나 값이 비정상인 경우 — missing_price_tickers 가드가 못 잡는 구멍)
            zero_rows = holdings_df[
                (holdings_df["수량"].fillna(0) > 0) & (holdings_df["평가금액(KRW)"].fillna(0) <= 0)
            ]
            if not zero_rows.empty:
                all_zero_valuation_tickers.extend(str(t) for t in zero_rows["티커"].tolist())

            account_purchase = float(holdings_df["매입금액(KRW)"].sum())
            account_valuation = float(holdings_df["평가금액(KRW)"].sum())
            total_profit_count += int((holdings_df["평가손익(KRW)"] >= 0).sum())
            total_loss_count += int((holdings_df["평가손익(KRW)"] < 0).sum())
            for bucket_name in bucket_totals:
                bucket_totals[bucket_name] += float(
                    holdings_df.loc[holdings_df["버킷"] == bucket_name, "평가금액(KRW)"].sum()
                )

        total_assets += account_valuation + cash_balance
        total_purchase += account_purchase
        total_valuation += account_valuation

    if all_missing_tickers:
        joined = ", ".join(all_missing_tickers)
        raise RuntimeError(
            f"가격 캐시가 없는 종목이 있어 일별 집계를 중단합니다: {joined}. "
            "종목 관리에서 해당 종목의 메타/캐시 새로고침을 실행하세요."
        )
    if all_zero_valuation_tickers:
        joined = ", ".join(all_zero_valuation_tickers)
        raise RuntimeError(
            f"평가금액을 계산하지 못한 종목이 있어 일별 집계를 중단합니다: {joined}. "
            "가격 캐시 상태를 확인한 뒤 다시 실행하세요."
        )

    if total_assets > 0:
        bucket_raw = {
            "bucket_pct_momentum": (bucket_totals["1. 모멘텀"] / total_assets) * 100,
            "bucket_pct_market": (bucket_totals["2. 시장지수"] / total_assets) * 100,
            "bucket_pct_dividend": (bucket_totals["3. 배당방어"] / total_assets) * 100,
            "bucket_pct_alternative": (bucket_totals["4. 대체헷지"] / total_assets) * 100,
            "bucket_pct_cash": (total_cash / total_assets) * 100,
        }
        bucket_rounded = {key: round(value, 2) for key, value in bucket_raw.items()}
        diff = round(round(sum(bucket_raw.values()), 2) - sum(bucket_rounded.values()), 2)
        if diff != 0:
            largest = max(bucket_rounded, key=lambda key: bucket_rounded[key])
            bucket_rounded[largest] = round(bucket_rounded[largest] + diff, 2)
    else:
        bucket_rounded = {
            "bucket_pct_momentum": 0.0,
            "bucket_pct_market": 0.0,
            "bucket_pct_dividend": 0.0,
            "bucket_pct_alternative": 0.0,
            "bucket_pct_cash": 0.0,
        }

    return {
        "total_assets": int(round(total_assets)),
        "purchase_amount": int(round(total_purchase)),
        "valuation_amount": int(round(total_valuation)),
        "exchange_rate": round(float(live_exchange_rate), 2),
        "profit_count": total_profit_count,
        "loss_count": total_loss_count,
        **bucket_rounded,
    }


def _load_daily_docs() -> list[dict[str, Any]]:
    db = _require_db()
    docs = list(db[DAILY_COLLECTION].find().sort("date", -1))
    if not docs:
        raise RuntimeError(
            "daily_fund_data 데이터가 없습니다. 먼저 ./.venv/bin/python scripts/seed_daily_fund_data.py 를 실행하세요."
        )
    return _apply_running_total_principal(docs)


def load_daily_docs_for_aggregation() -> list[dict[str, Any]]:
    return _load_daily_docs()


def _convert_weekly_doc_to_daily_seed(doc: dict[str, Any]) -> dict[str, Any]:
    normalized = _apply_derived_fields(doc)
    return {
        "date": str(normalized["week_date"]),
        "withdrawal_personal": _to_int(normalized.get("withdrawal_personal", 0)),
        "withdrawal_mom": _to_int(normalized.get("withdrawal_mom", 0)),
        "nh_principal_interest": _to_int(normalized.get("nh_principal_interest", 0)),
        "deposit_withdrawal": _to_int(normalized.get("deposit_withdrawal", 0)),
        "total_assets": _to_int(normalized.get("total_assets", 0)),
        "purchase_amount": _to_int(normalized.get("purchase_amount", 0)),
        "valuation_amount": _to_int(normalized.get("valuation_amount", 0)),
        "memo": str(normalized.get("memo", "") or ""),
        "exchange_rate": round(_to_float(normalized.get("exchange_rate", 0.0)), 2),
        **{key: round(value, 2) for key, value in _normalize_bucket_percentages(normalized).items()},
        "profit_count": _to_int(normalized.get("profit_count", 0)),
        "loss_count": _to_int(normalized.get("loss_count", 0)),
        "seed_source": WEEKLY_COLLECTION,
        "created_at": normalized.get("created_at") or _get_now_kst(),
        "updated_at": normalized.get("updated_at") or _get_now_kst(),
    }


def load_daily_table_data() -> dict[str, Any]:
    daily_docs = _load_daily_docs()
    latest_date = str(daily_docs[0]["date"]) if daily_docs else ""
    return {
        "latest_date": latest_date,
        "rows": _build_api_rows(daily_docs),
        "editable_fields": FIELD_DEFS,
        "read_only_keys": list(READ_ONLY_FIELDS),
        "core_hidden_keys": list(CORE_VIEW_HIDDEN_KEYS),
    }


def update_daily_row(date: str, payload: dict[str, Any]) -> dict[str, str]:
    target_date = str(date or "").strip()
    if not target_date:
        raise RuntimeError("수정할 일자를 찾을 수 없습니다.")

    update_doc: dict[str, Any] = {}
    for field in FIELD_DEFS:
        key = field["key"]
        if key in READ_ONLY_FIELDS or key not in payload:
            continue

        raw_value = payload[key]
        if field["type"] == "text":
            update_doc[key] = str(raw_value or "").strip()
            continue

        try:
            numeric_value = float(raw_value or 0)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"{field['label']} 값이 올바르지 않습니다.") from exc

        if field["type"] == "int":
            update_doc[key] = int(numeric_value)
        else:
            update_doc[key] = round(numeric_value, 4)

    update_doc["updated_at"] = _get_now_kst()

    db = _require_db()
    result = db[DAILY_COLLECTION].update_one(
        {"date": target_date},
        {"$set": update_doc},
    )
    if result.matched_count == 0:
        raise RuntimeError("수정할 일별 데이터를 찾지 못했습니다.")

    return {"date": target_date}


def seed_daily_data_from_weekly() -> dict[str, int]:
    db = _require_db()
    weekly_docs = list(db[WEEKLY_COLLECTION].find().sort("week_date", 1))
    if not weekly_docs:
        raise RuntimeError("weekly_fund_data 데이터가 없습니다.")
    today_str = _get_now_kst().date().isoformat()

    seen_dates: set[str] = set()
    for doc in weekly_docs:
        date_str = str(doc.get("week_date", "")).strip()
        if not date_str:
            raise RuntimeError("weekly_fund_data에 week_date가 비어 있는 문서가 있습니다.")
        if date_str in seen_dates:
            raise RuntimeError(f"weekly_fund_data에 중복 week_date가 있습니다: {date_str}")
        seen_dates.add(date_str)

    seeded = 0
    skipped = 0
    for doc in _apply_running_total_principal(
        [
            {"date": str(item["week_date"]), **{k: v for k, v in item.items() if k != "week_date"}}
            for item in weekly_docs
        ]
    ):
        if str(doc["date"]) > today_str:
            skipped += 1
            continue
        daily_doc = _convert_weekly_doc_to_daily_seed(
            {"week_date": doc["date"], **{k: v for k, v in doc.items() if k != "date"}}
        )
        exists = db[DAILY_COLLECTION].count_documents({"date": daily_doc["date"]}, limit=1)
        if exists:
            skipped += 1
            continue
        db[DAILY_COLLECTION].insert_one(daily_doc)
        seeded += 1

    return {"seeded": seeded, "skipped": skipped, "total_weekly_rows": len(weekly_docs)}


def remove_future_daily_rows() -> dict[str, int]:
    db = _require_db()
    active_date = _get_active_daily_date()
    deleted = db[DAILY_COLLECTION].delete_many({"date": {"$gt": active_date}}).deleted_count
    return {"deleted": int(deleted)}


def _check_total_jump_guard(
    new_total: float, baseline_total: float | None, deposit_withdrawal: float
) -> None:
    """총자산 급변 서킷브레이커 — 직전 기록 대비 허용 범위를 벗어나면 기록을 거부한다.

    2026-07-24 캐시 사고처럼 조회 실패로 평가액이 통째로 빠진 값이 통계에 들어가는 것을
    마지막 단계에서 막는다. 입출금(deposit_withdrawal) 반영분은 정상 변동으로 제외한다.
    진짜 폭락/급등이면 로그를 확인하고 수동으로 기록하면 된다.
    """
    if baseline_total is None or baseline_total <= 0:
        return
    jump_pct = abs(new_total - baseline_total - deposit_withdrawal) / baseline_total * 100.0
    if jump_pct > DAILY_TOTAL_JUMP_GUARD_PCT:
        raise RuntimeError(
            f"총자산이 직전 기록 대비 {jump_pct:.1f}% 변동해 일별 집계 기록을 거부합니다 "
            f"(허용 {DAILY_TOTAL_JUMP_GUARD_PCT:.0f}%, 직전 {baseline_total:,.0f} → 신규 {new_total:,.0f}). "
            "가격 캐시 상태를 확인하세요. 실제 급변이면 수동으로 기록하세요."
        )


def aggregate_today_daily_data() -> dict[str, str]:
    db = _require_db()
    target_date = _get_active_daily_date()
    update_doc = _collect_live_daily_summary()
    update_doc["updated_at"] = _get_now_kst()

    # 서킷브레이커 기준: 대상일 직전의 마지막 기록(과거 문서). 첫 기록이면 통과.
    previous_doc = db[DAILY_COLLECTION].find_one(
        {"date": {"$lt": target_date}}, sort=[("date", -1)]
    )
    existing_doc = db[DAILY_COLLECTION].find_one({"date": target_date}) or {}
    _check_total_jump_guard(
        float(update_doc.get("total_assets") or 0),
        float(previous_doc["total_assets"]) if previous_doc and previous_doc.get("total_assets") else None,
        float(existing_doc.get("deposit_withdrawal") or 0),
    )

    db[DAILY_COLLECTION].update_one(
        {"date": target_date},
        {
            "$set": update_doc,
            "$setOnInsert": {
                "date": target_date,
                "withdrawal_personal": 0,
                "withdrawal_mom": 0,
                "nh_principal_interest": 0,
                "deposit_withdrawal": 0,
                "memo": "",
                "created_at": _get_now_kst(),
            },
        },
        upsert=True,
    )
    return {"date": target_date}
