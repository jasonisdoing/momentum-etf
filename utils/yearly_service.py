from __future__ import annotations

import datetime
from collections import defaultdict
from typing import Any
from zoneinfo import ZoneInfo

from utils.daily_fund_service import load_daily_docs_for_aggregation
from utils.data_loader import get_trading_days
from utils.db_manager import get_db_connection
from utils.normalization import to_iso_string

YEARLY_COLLECTION = "yearly_fund_data"
KST = ZoneInfo("Asia/Seoul")
INITIAL_TOTAL_PRINCIPAL_DATE = "2024-01-31"
INITIAL_TOTAL_PRINCIPAL_VALUE = 56_000_000
READ_ONLY_FIELDS = {
    "withdrawal_personal",
    "withdrawal_mom",
    "nh_principal_interest",
    "total_expense",
    "deposit_withdrawal",
    "total_principal",
    "total_assets",
    "purchase_amount",
    "valuation_amount",
    "profit_loss",
    "cumulative_profit",
    "yearly_profit",
    "yearly_return_pct",
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
    {"key": "cumulative_profit", "label": "누적 손익", "type": "int"},
    {"key": "yearly_profit", "label": "금년 손익", "type": "int"},
    {"key": "yearly_return_pct", "label": "연수익률 (%)", "type": "float"},
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
EDITABLE_FIELD_KEYS = {"memo"}


def _require_db() -> Any:
    db = get_db_connection()
    if db is None:
        raise RuntimeError("DB 연결 실패")
    return db


def _get_now_kst() -> datetime.datetime:
    return datetime.datetime.now(KST)


def _to_int(value: object) -> int:
    return int(float(value or 0))


def _to_float(value: object) -> float:
    return float(value or 0.0)


def _format_year_date_display(date_str: str) -> str:
    weekday_kr = ["월", "화", "수", "목", "금", "토", "일"]
    dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    return f"{dt.year}. {dt.month}. {dt.day} ({weekday_kr[dt.weekday()]})"


def _get_year_range(year: int) -> tuple[datetime.date, datetime.date]:
    """해당 연도의 첫날과 마지막 날을 반환한다."""
    return datetime.date(year, 1, 1), datetime.date(year, 12, 31)


def _get_last_trading_day_of_year(year: int) -> str:
    """해당 연도의 한국 시장 마지막 거래일 날짜를 반환한다.

    아직 진행 중인 연도(연말 거래일 이전)의 경우 오늘 이전 가장 최근 거래일을 사용한다.
    """
    first_day, last_day = _get_year_range(year)
    today = datetime.date.today()
    upper = min(last_day, today)
    try:
        days = get_trading_days(str(first_day), str(upper), "kor")
        if days:
            return str(days[-1].date())
    except Exception:
        pass
    raise RuntimeError(f"{year} 한국 시장 거래일을 조회하지 못했습니다.")


def _normalize_bucket_percentages(source: dict[str, Any]) -> dict[str, float]:
    momentum = _to_float(source.get("bucket_pct_momentum", 0.0))
    innovation = _to_float(source.get("bucket_pct_innovation", 0.0))
    market = _to_float(source.get("bucket_pct_market", 0.0))
    dividend = _to_float(source.get("bucket_pct_dividend", 0.0))
    alternative = _to_float(source.get("bucket_pct_alternative", 0.0))
    cash = _to_float(source.get("bucket_pct_cash", 0.0))
    return {
        "bucket_pct_momentum": momentum + innovation,
        "bucket_pct_market": market,
        "bucket_pct_dividend": dividend,
        "bucket_pct_alternative": alternative,
        "bucket_pct_cash": cash,
    }


def _calculate_total_expense(source: dict[str, Any]) -> int:
    return (
        _to_int(source.get("withdrawal_personal", 0))
        + _to_int(source.get("withdrawal_mom", 0))
        + _to_int(source.get("nh_principal_interest", 0))
    )


def _calculate_profit_loss(source: dict[str, Any]) -> int:
    return _to_int(source.get("valuation_amount", 0)) - _to_int(source.get("purchase_amount", 0))


def _calculate_total_stocks(source: dict[str, Any]) -> int:
    return _to_int(source.get("profit_count", 0)) + _to_int(source.get("loss_count", 0))


def _apply_derived_fields(source: dict[str, Any]) -> dict[str, Any]:
    updated = dict(source)
    updated["total_expense"] = _calculate_total_expense(updated)
    updated["profit_loss"] = _calculate_profit_loss(updated)
    updated["total_stocks"] = _calculate_total_stocks(updated)
    return updated


def _apply_running_total_principal(docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """수익률 계산 규칙: yearly_return_pct = TWR(1년), cumulative_return_pct = ROI.
    상세는 docs/developer_guide.md (자산 수익률 계산 정책) 참고."""
    docs_by_date = {str(doc["year_date"]): _apply_derived_fields(doc) for doc in docs}
    running_total = INITIAL_TOTAL_PRINCIPAL_VALUE
    running_total_expense = 0
    previous_cumulative_profit = 0
    previous_total_assets = 0

    for year_date in sorted(docs_by_date):
        doc = docs_by_date[year_date]
        if year_date <= INITIAL_TOTAL_PRINCIPAL_DATE:
            doc["total_principal"] = INITIAL_TOTAL_PRINCIPAL_VALUE
        else:
            running_total += _to_int(doc.get("deposit_withdrawal", 0))
            doc["total_principal"] = running_total

        running_total_expense += _to_int(doc.get("total_expense", 0))
        doc["cumulative_profit"] = (
            _to_int(doc.get("total_assets", 0)) - _to_int(doc.get("total_principal", 0)) - running_total_expense
        )
        doc["yearly_profit"] = _to_int(doc.get("cumulative_profit", 0)) - previous_cumulative_profit
        total_principal = _to_int(doc.get("total_principal", 0))
        deposit_withdrawal = _to_int(doc.get("deposit_withdrawal", 0))
        twr_base = previous_total_assets + deposit_withdrawal
        if twr_base > 0:
            doc["yearly_return_pct"] = round((_to_int(doc.get("yearly_profit", 0)) / twr_base) * 100, 2)
        else:
            doc["yearly_return_pct"] = 0.0
        if total_principal == 0:
            doc["cumulative_return_pct"] = 0.0
        else:
            doc["cumulative_return_pct"] = round((_to_int(doc.get("cumulative_profit", 0)) / total_principal) * 100, 2)
        previous_cumulative_profit = _to_int(doc.get("cumulative_profit", 0))
        previous_total_assets = _to_int(doc.get("total_assets", 0))

    return [
        docs_by_date[str(doc["year_date"])] for doc in sorted(docs, key=lambda item: item["year_date"], reverse=True)
    ]


def _doc_to_api_row(doc: dict[str, Any]) -> dict[str, Any]:
    computed_doc = _apply_derived_fields(doc)
    exchange_rate = _to_float(computed_doc.get("exchange_rate", 0.0))
    bucket_percentages = _normalize_bucket_percentages(computed_doc)
    return {
        "year_date": str(computed_doc["year_date"]),
        "year_date_display": _format_year_date_display(str(computed_doc["year_date"])),
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
        "cumulative_profit": _to_int(computed_doc.get("cumulative_profit", 0)),
        "yearly_profit": _to_int(computed_doc.get("yearly_profit", 0)),
        "yearly_return_pct": round(float(computed_doc.get("yearly_return_pct", 0.0) or 0.0), 2),
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


def _load_yearly_docs() -> list[dict[str, Any]]:
    db = _require_db()
    docs = list(db[YEARLY_COLLECTION].find().sort("year_date", -1))
    if not docs:
        raise RuntimeError("yearly_fund_data 데이터가 없습니다. 먼저 일별/년별 집계를 실행하세요.")
    return _apply_running_total_principal(docs)


def _get_year_group_key(date_str: str) -> int:
    dt = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
    return dt.year


def _build_yearly_doc_from_group(year: int, docs: list[dict[str, Any]], memo: str) -> dict[str, Any]:
    sorted_docs = sorted(docs, key=lambda item: str(item["date"]))
    last_doc = sorted_docs[-1]
    bucket_percentages = _normalize_bucket_percentages(last_doc)
    now = _get_now_kst()
    # 종료일을 '데이터가 있는 마지막 날'이 아니라 '해당 연도의 마지막 영업일'로 고정
    year_date = _get_last_trading_day_of_year(year)

    return {
        "year_date": year_date,
        "withdrawal_personal": sum(_to_int(doc.get("withdrawal_personal", 0)) for doc in sorted_docs),
        "withdrawal_mom": sum(_to_int(doc.get("withdrawal_mom", 0)) for doc in sorted_docs),
        "nh_principal_interest": sum(_to_int(doc.get("nh_principal_interest", 0)) for doc in sorted_docs),
        "deposit_withdrawal": sum(_to_int(doc.get("deposit_withdrawal", 0)) for doc in sorted_docs),
        "total_assets": _to_int(last_doc.get("total_assets", 0)),
        "purchase_amount": _to_int(last_doc.get("purchase_amount", 0)),
        "valuation_amount": _to_int(last_doc.get("valuation_amount", 0)),
        "memo": memo,
        "exchange_rate": round(_to_float(last_doc.get("exchange_rate", 0.0)), 2),
        **{key: round(value, 2) for key, value in bucket_percentages.items()},
        "profit_count": _to_int(last_doc.get("profit_count", 0)),
        "loss_count": _to_int(last_doc.get("loss_count", 0)),
        "created_at": last_doc.get("created_at") or now,
        "updated_at": now,
    }


def _build_yearly_docs_from_daily() -> list[dict[str, Any]]:
    daily_docs = load_daily_docs_for_aggregation()
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for doc in daily_docs:
        grouped[_get_year_group_key(str(doc["date"]))].append(doc)

    db = _require_db()
    # 메모를 year 키로 로드 (영업일 변경 대비)
    existing_memo_by_year: dict[int, str] = {}
    for doc in db[YEARLY_COLLECTION].find({}, {"year_date": 1, "memo": 1}):
        year_date = str(doc["year_date"])
        year_key = _get_year_group_key(year_date)
        existing_memo_by_year[year_key] = str(doc.get("memo", "") or "")

    yearly_docs: list[dict[str, Any]] = []
    for year, group_docs in grouped.items():
        memo = existing_memo_by_year.get(year, "")
        yearly_docs.append(_build_yearly_doc_from_group(year, group_docs, memo))

    return sorted(yearly_docs, key=lambda item: str(item["year_date"]))


def load_yearly_table_data() -> dict[str, Any]:
    yearly_docs = _load_yearly_docs()
    active_year_date = str(yearly_docs[0]["year_date"]) if yearly_docs else ""
    return {
        "active_year_date": active_year_date,
        "rows": _build_api_rows(yearly_docs),
        "editable_fields": [field for field in FIELD_DEFS if field["key"] in EDITABLE_FIELD_KEYS],
        "read_only_keys": list(READ_ONLY_FIELDS),
        "core_hidden_keys": list(CORE_VIEW_HIDDEN_KEYS),
    }


def aggregate_active_year_data() -> dict[str, str]:
    db = _require_db()
    yearly_docs = _build_yearly_docs_from_daily()
    if not yearly_docs:
        raise RuntimeError("년별 집계에 사용할 daily_fund_data 데이터가 없습니다.")

    valid_year_dates = {str(doc["year_date"]) for doc in yearly_docs}
    db[YEARLY_COLLECTION].delete_many({"year_date": {"$nin": list(valid_year_dates)}})
    for doc in yearly_docs:
        db[YEARLY_COLLECTION].update_one(
            {"year_date": doc["year_date"]},
            {"$set": doc},
            upsert=True,
        )
    return {"year_date": str(yearly_docs[-1]["year_date"])}


def update_yearly_row(year_date: str, payload: dict[str, Any]) -> dict[str, str]:
    target_year_date = str(year_date or "").strip()
    if not target_year_date:
        raise RuntimeError("수정할 연도를 찾을 수 없습니다.")

    invalid_keys = [key for key in payload if key not in {"year_date", "memo"}]
    if invalid_keys:
        raise RuntimeError("년별에서는 비고만 수정할 수 있습니다.")

    db = _require_db()
    result = db[YEARLY_COLLECTION].update_one(
        {"year_date": target_year_date},
        {"$set": {"memo": str(payload.get("memo", "") or "").strip(), "updated_at": _get_now_kst()}},
    )
    if result.matched_count == 0:
        raise RuntimeError("수정할 년별 데이터를 찾지 못했습니다.")
    return {"year_date": target_year_date}
