import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from utils.data_loader import get_exchange_rate_series
from utils.db_manager import get_db_connection
from utils.report import format_kr_money
from utils.ui import create_loading_status

# MongoDB 컬렉션 이름
WEEKLY_COLLECTION = "weekly_fund_data"
READ_ONLY_FIELDS = {
    "total_expense",
    "total_principal",
    "total_assets",
    "purchase_amount",
    "valuation_amount",
    "profit_loss",
    "cumulative_profit",
    "weekly_profit",
    "weekly_return_pct",
    "cumulative_return_pct",
    "exchange_rate",
    "total_stocks",
}
YELLOW_COLUMNS = ["개인 인출", "엄마", "농협원리금", "입출금"]
WHITE_COLUMNS = ["지출 합계", "총 원금"]
HOME_SUMMARY_COLUMN_STYLES = {
    "총 자산": "#93c47d",
    "매입 금액": "#76a5af",
    "평가 금액": "#6fa8dc",
}
CORE_VIEW_HIDDEN_COLUMNS = ["개인 인출", "엄마", "농협원리금", "입출금"]
KST = ZoneInfo("Asia/Seoul")
INITIAL_TOTAL_PRINCIPAL_DATE = "2024-01-31"
INITIAL_TOTAL_PRINCIPAL_VALUE = 56_000_000
MONEY_COLUMNS = {
    "개인 인출",
    "엄마",
    "농협원리금",
    "지출 합계",
    "입출금",
    "총 원금",
    "총 자산",
    "매입 금액",
    "평가 금액",
    "평가 손익",
    "누적 손익",
    "금주 손익",
}
PROFIT_DISPLAY_COLUMNS = ["평가 손익", "누적 손익", "금주 손익", "주수익률", "누적 수익률", "환율(변동)"]

# 계산 컬럼 메모:
# - total_expense(지출 합계) = withdrawal_personal(개인 인출) + withdrawal_mom(엄마) + nh_principal_interest(농협원리금)
# - total_principal(총 원금) = 2024-01-31의 시작 원금 56,000,000원을 기준으로, 이후 행은 전 행 총 원금 + deposit_withdrawal(입출금)
# - profit_loss(평가 손익) = valuation_amount(평가 금액) - purchase_amount(매입 금액)
# - cumulative_profit(누적 손익) = total_assets(총 자산) - total_principal(총 원금) - 현재 행까지 누적 total_expense(지출 합계)
# - weekly_profit(금주 손익) = cumulative_profit(이번주 누적 손익) - 이전 행 cumulative_profit(지난주 누적 손익)
# - weekly_return_pct(주 수익률) = weekly_profit(금주 손익) / total_principal(총 원금) * 100
# - cumulative_return_pct(누적 수익률) = cumulative_profit(누적 손익) / total_principal(총 원금) * 100
# - total_stocks(총 종목 수) = profit_count(수익 종목 수) + loss_count(손실 종목 수)
# - total_assets/purchase_amount/valuation_amount는 사용자가 직접 입력하지 않고, 데이터 집계 버튼으로 daily_snapshots 기준 값을 반영한다.
# 계산 컬럼은 사용자가 직접 수정하지 않고, 화면 표시 시점에 항상 다시 계산한다.
# exchange_rate(환율)만 예외적으로 집계/백필 결과를 DB에 저장한다.


def _get_now_kst() -> datetime.datetime:
    """한국 시간 기준 현재 시각을 반환한다."""
    return datetime.datetime.now(KST)


def _get_active_week_date() -> str:
    """활성 주차 기준일을 YYYY-MM-DD 형식으로 반환한다."""
    now = _get_now_kst()
    this_week_monday = now.date() - datetime.timedelta(days=now.weekday())
    this_week_friday = this_week_monday + datetime.timedelta(days=4)

    # 다음 주 월요일 09:00 전까지는 지난 금요일 행을 유지한다.
    if now.weekday() == 0 and now.time() < datetime.time(hour=9):
        active_friday = this_week_friday - datetime.timedelta(days=7)
    else:
        active_friday = this_week_friday

    return active_friday.strftime("%Y-%m-%d")


def _format_week_date_display(date_str: str) -> str:
    """YYYY-MM-DD → 'YYYY. M. D (요일)' 형식으로 변환."""
    weekday_kr = ["월", "화", "수", "목", "금", "토", "일"]
    dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    wd = weekday_kr[dt.weekday()]
    return f"{dt.year}. {dt.month}. {dt.day} ({wd})"


def _new_empty_doc(week_date: str) -> dict:
    """빈 주별 데이터 문서를 생성한다."""
    return {
        "week_date": week_date,
        "withdrawal_personal": 0,
        "withdrawal_mom": 0,
        "nh_principal_interest": 0,
        "deposit_withdrawal": 0,
        "total_assets": 0,
        "purchase_amount": 0,
        "valuation_amount": 0,
        "memo": "",
        "exchange_rate": 0.0,
        "bucket_pct_momentum": 0.0,
        "bucket_pct_innovation": 0.0,
        "bucket_pct_market": 0.0,
        "bucket_pct_dividend": 0.0,
        "bucket_pct_alternative": 0.0,
        "bucket_pct_cash": 0.0,
        "profit_count": 0,
        "loss_count": 0,
        "created_at": _get_now_kst(),
        "updated_at": _get_now_kst(),
    }


def _calculate_total_expense(source: dict) -> int:
    """지출 합계를 계산한다."""
    withdrawal_personal = int(source.get("withdrawal_personal", 0) or 0)
    withdrawal_mom = int(source.get("withdrawal_mom", 0) or 0)
    nh_principal_interest = int(source.get("nh_principal_interest", 0) or 0)
    return withdrawal_personal + withdrawal_mom + nh_principal_interest


def _calculate_profit_loss(source: dict) -> int:
    """평가 손익을 계산한다."""
    valuation_amount = int(source.get("valuation_amount", 0) or 0)
    purchase_amount = int(source.get("purchase_amount", 0) or 0)
    return valuation_amount - purchase_amount


def _calculate_total_stocks(source: dict) -> int:
    """총 종목 수를 계산한다."""
    profit_count = int(source.get("profit_count", 0) or 0)
    loss_count = int(source.get("loss_count", 0) or 0)
    return profit_count + loss_count


def _to_int(value: object) -> int:
    """숫자/빈값을 정수로 정규화한다."""
    return int(value or 0)


def _format_display_money(value: object) -> str:
    """원 단위 값을 화면에서만 만원 반올림 한글 형식으로 표시한다."""
    rounded_to_manwon = round(_to_int(value) / 10000) * 10000
    return format_kr_money(rounded_to_manwon).replace("만 ", "만")


@st.cache_data(ttl=1800, show_spinner=False)
def _get_live_exchange_rate() -> float:
    """현재 시점의 USD/KRW 환율을 반환한다."""
    now = pd.Timestamp(_get_now_kst()).tz_localize(None)
    # 주별 화면의 환율은 보조 지표이므로, 단기 누락이 있어도 부분 캐시를 우선 사용합니다.
    series = get_exchange_rate_series(now - pd.Timedelta(days=5), now, allow_partial=True)
    if series.empty:
        return 0.0
    return float(series.iloc[-1])


def _ensure_historical_exchange_rates() -> None:
    """과거 주차의 누락 환율을 일괄 조회해 저장한다."""
    db = get_db_connection()
    if db is None:
        return

    active_week_date = _get_active_week_date()
    target_docs = list(
        db[WEEKLY_COLLECTION]
        .find(
            {
                "week_date": {"$ne": active_week_date},
                "$or": [{"exchange_rate": {"$exists": False}}, {"exchange_rate": 0}, {"exchange_rate": 0.0}],
            },
            {"week_date": 1},
        )
        .sort("week_date", 1)
    )
    if not target_docs:
        return

    start_date = target_docs[0]["week_date"]
    end_date = target_docs[-1]["week_date"]
    rate_series = get_exchange_rate_series(
        pd.Timestamp(start_date) - pd.Timedelta(days=7),
        pd.Timestamp(end_date),
        allow_partial=True,
    )
    if rate_series.empty:
        return

    week_dates = [pd.Timestamp(doc["week_date"]) for doc in target_docs]
    aligned_rates = rate_series.reindex(pd.DatetimeIndex(week_dates), method="ffill")
    now = _get_now_kst()

    for doc, rate in zip(target_docs, aligned_rates.tolist(), strict=True):
        if pd.isna(rate):
            continue
        db[WEEKLY_COLLECTION].update_one(
            {"week_date": doc["week_date"]},
            {
                "$set": {
                    "exchange_rate": float(rate),
                    "updated_at": now,
                }
            },
        )


def _get_profit_text_color(value: object) -> str:
    """손익/수익률 값의 부호에 따라 글자색을 반환한다."""
    if isinstance(value, str):
        cleaned = value.replace(",", "").strip()
        sign = -1 if cleaned.startswith("-") else 1
        normalized = cleaned.lstrip("+-")
        if "%" in normalized:
            numeric_value = float(normalized.replace("%", "").strip() or 0)
        elif "억" in normalized or "만" in normalized or "원" in normalized:
            eok_part = 0
            man_part = 0
            if "억" in normalized:
                eok_text, remainder = normalized.split("억", 1)
                eok_part = int((eok_text or "0").strip() or 0) * 100_000_000
                normalized = remainder
            if "만" in normalized:
                man_text, remainder = normalized.split("만", 1)
                man_part = int((man_text or "0").strip() or 0) * 10_000
                normalized = remainder
            won_text = normalized.replace("원", "").strip()
            won_part = int(won_text or 0)
            numeric_value = float(sign * (eok_part + man_part + won_part))
        else:
            numeric_value = float(normalized or 0) * sign
    else:
        numeric_value = float(value or 0)
    if numeric_value > 0:
        return "#e06666"
    if numeric_value < 0:
        return "#3d85c6"
    return "#000000"


def _get_effective_exchange_rate(doc: dict) -> float:
    """표시에 사용할 환율 값을 반환한다."""
    active_week_date = _get_active_week_date()
    exchange_rate = float(doc.get("exchange_rate", 0.0) or 0.0)
    if str(doc.get("week_date", "")) == active_week_date:
        live_exchange_rate = _get_live_exchange_rate()
        if live_exchange_rate > 0:
            exchange_rate = live_exchange_rate
    return exchange_rate


def _apply_derived_fields(source: dict) -> dict:
    """계산 컬럼 값을 반영한 사본을 반환한다."""
    updated = dict(source)
    # 지출 합계는 입력값이 아니라 개인 인출 + 엄마 + 농협원리금으로 고정 계산한다.
    updated["total_expense"] = _calculate_total_expense(updated)
    # 평가 손익은 입력값이 아니라 평가 금액 - 매입 금액으로 고정 계산한다.
    updated["profit_loss"] = _calculate_profit_loss(updated)
    # 총 종목 수는 입력값이 아니라 수익 종목 수 + 손실 종목 수로 고정 계산한다.
    updated["total_stocks"] = _calculate_total_stocks(updated)
    return updated


def _apply_running_total_principal(docs: list[dict]) -> list[dict]:
    """총 원금, 손익, 수익률을 날짜 오름차순 기준으로 계산해 원래 순서로 반환한다."""
    docs_by_date = {doc["week_date"]: _apply_derived_fields(doc) for doc in docs}
    running_total = INITIAL_TOTAL_PRINCIPAL_VALUE
    running_total_expense = 0
    previous_cumulative_profit = 0

    for week_date in sorted(docs_by_date):
        doc = docs_by_date[week_date]
        if week_date <= INITIAL_TOTAL_PRINCIPAL_DATE:
            doc["total_principal"] = INITIAL_TOTAL_PRINCIPAL_VALUE
        else:
            # 시작 기준일 이후 행은 직전 총 원금 + 현재 행 입출금으로 누적 계산한다.
            running_total += _to_int(doc.get("deposit_withdrawal", 0))
            doc["total_principal"] = running_total

        running_total_expense += _to_int(doc.get("total_expense", 0))
        doc["cumulative_profit"] = (
            _to_int(doc.get("total_assets", 0)) - _to_int(doc.get("total_principal", 0)) - running_total_expense
        )
        doc["weekly_profit"] = _to_int(doc.get("cumulative_profit", 0)) - previous_cumulative_profit
        total_principal = _to_int(doc.get("total_principal", 0))
        if total_principal == 0:
            doc["weekly_return_pct"] = 0.0
            doc["cumulative_return_pct"] = 0.0
        else:
            doc["weekly_return_pct"] = (_to_int(doc.get("weekly_profit", 0)) / total_principal) * 100
            doc["cumulative_return_pct"] = (_to_int(doc.get("cumulative_profit", 0)) / total_principal) * 100
        previous_cumulative_profit = _to_int(doc.get("cumulative_profit", 0))

    return [docs_by_date[doc["week_date"]] for doc in docs]


def _ensure_active_week_row() -> None:
    """활성 주차 데이터가 없으면 빈 행을 생성한다."""
    db = get_db_connection()
    if db is None:
        return

    active_week_date = _get_active_week_date()
    existing = db[WEEKLY_COLLECTION].find_one({"week_date": active_week_date})
    if existing:
        return

    db[WEEKLY_COLLECTION].insert_one(_new_empty_doc(active_week_date))


# --- 필드 정의: (DB key, 표시명, 타입) ---
FIELD_DEFS: list[tuple[str, str, str]] = [
    ("withdrawal_personal", "개인 인출", "int"),
    ("withdrawal_mom", "엄마", "int"),
    ("nh_principal_interest", "농협원리금", "int"),
    ("total_expense", "지출 합계", "int"),
    ("deposit_withdrawal", "입출금", "int"),
    ("total_principal", "총 원금", "int"),
    ("total_assets", "총 자산", "int"),
    ("purchase_amount", "매입 금액", "int"),
    ("valuation_amount", "평가 금액", "int"),
    ("profit_loss", "평가 손익", "int"),
    ("cumulative_profit", "누적 손익", "int"),
    ("weekly_profit", "금주 손익", "int"),
    ("weekly_return_pct", "주수익률 (%)", "float"),
    ("cumulative_return_pct", "누적 수익률 (%)", "float"),
    ("memo", "비고", "text"),
    ("exchange_rate", "환율", "float"),
    ("bucket_pct_momentum", "1. 모멘텀 (%)", "float"),
    ("bucket_pct_innovation", "2. 혁신기술 (%)", "float"),
    ("bucket_pct_market", "3. 시장지수 (%)", "float"),
    ("bucket_pct_dividend", "4. 배당방어 (%)", "float"),
    ("bucket_pct_alternative", "5. 대체헷지 (%)", "float"),
    ("bucket_pct_cash", "6. 현금 (%)", "float"),
    ("total_stocks", "총 종목 수", "int"),
    ("profit_count", "수익 종목 수", "int"),
    ("loss_count", "손실 종목 수", "int"),
]


def _doc_to_display_row(doc: dict) -> dict:
    """MongoDB 문서를 표시용 딕셔너리로 변환."""
    computed_doc = _apply_derived_fields(doc)
    exchange_rate = _get_effective_exchange_rate(computed_doc)

    return {
        "날짜": _format_week_date_display(computed_doc["week_date"]),
        "개인 인출": _format_display_money(computed_doc.get("withdrawal_personal", 0)),
        "엄마": _format_display_money(computed_doc.get("withdrawal_mom", 0)),
        "농협원리금": _format_display_money(computed_doc.get("nh_principal_interest", 0)),
        "지출 합계": _format_display_money(computed_doc.get("total_expense", 0)),
        "입출금": _format_display_money(computed_doc.get("deposit_withdrawal", 0)),
        "총 원금": _format_display_money(computed_doc.get("total_principal", 0)),
        "총 자산": _format_display_money(computed_doc.get("total_assets", 0)),
        "매입 금액": _format_display_money(computed_doc.get("purchase_amount", 0)),
        "평가 금액": _format_display_money(computed_doc.get("valuation_amount", 0)),
        "평가 손익": _format_display_money(computed_doc.get("profit_loss", 0)),
        "누적 손익": _format_display_money(computed_doc.get("cumulative_profit", 0)),
        "금주 손익": _format_display_money(computed_doc.get("weekly_profit", 0)),
        "주수익률": computed_doc.get("weekly_return_pct", 0.0),
        "누적 수익률": computed_doc.get("cumulative_return_pct", 0.0),
        "비고": computed_doc.get("memo", ""),
        "환율(변동)": 0.0,
        "환율": exchange_rate,
        "1. 모멘텀": computed_doc.get("bucket_pct_momentum", 0.0),
        "2. 혁신기술": computed_doc.get("bucket_pct_innovation", 0.0),
        "3. 시장지수": computed_doc.get("bucket_pct_market", 0.0),
        "4. 배당방어": computed_doc.get("bucket_pct_dividend", 0.0),
        "5. 대체헷지": computed_doc.get("bucket_pct_alternative", 0.0),
        "6. 현금": computed_doc.get("bucket_pct_cash", 0.0),
        "총 종목 수": computed_doc.get("total_stocks", 0),
        "수익 종목 수": computed_doc.get("profit_count", 0),
        "손실 종목 수": computed_doc.get("loss_count", 0),
    }


def _build_display_rows(docs: list[dict]) -> list[dict]:
    """표시용 행 리스트를 생성한다."""
    rows = [_doc_to_display_row(doc) for doc in docs]
    for idx, row in enumerate(rows):
        current_rate = float(row.get("환율", 0.0) or 0.0)
        older_rate = float(rows[idx + 1].get("환율", 0.0) or 0.0) if idx + 1 < len(rows) else 0.0
        if older_rate > 0:
            row["환율(변동)"] = ((current_rate / older_rate) - 1.0) * 100
        else:
            row["환율(변동)"] = 0.0
    return rows


def _load_weekly_docs() -> list[dict]:
    """MongoDB에서 주별 데이터 원본 문서 리스트 반환 (최신순)."""
    db = get_db_connection()
    if db is None:
        return []
    docs = list(db[WEEKLY_COLLECTION].find().sort("week_date", -1))
    return _apply_running_total_principal(docs)


def sync_active_week_summary() -> dict:
    """활성 주차 행을 최신 실시간 합계로 집계한 뒤 계산 컬럼이 반영된 문서를 반환한다."""
    _ensure_active_week_row()
    _ensure_historical_exchange_rates()
    active_week_date = _aggregate_live_summary_into_active_week()

    for doc in _load_weekly_docs():
        if str(doc.get("week_date", "")) == active_week_date:
            return doc

    raise RuntimeError(f"활성 주차({active_week_date}) 데이터를 찾지 못했습니다.")


def _aggregate_live_summary_into_active_week() -> str:
    """홈 화면과 같은 실시간 합계를 활성 주차 1행에 반영한다."""
    from utils.account_registry import load_account_configs
    from utils.portfolio_io import load_portfolio_master, load_real_holdings_table

    db = get_db_connection()
    if db is None:
        raise RuntimeError("DB 연결 실패")

    active_week_date = _get_active_week_date()
    total_assets = 0.0
    total_purchase = 0.0
    total_valuation = 0.0
    total_cash = 0.0
    total_profit_count = 0
    total_loss_count = 0
    live_exchange_rate = _get_live_exchange_rate()
    bucket_totals = {
        "1. 모멘텀": 0.0,
        "2. 혁신기술": 0.0,
        "3. 시장지수": 0.0,
        "4. 배당방어": 0.0,
        "5. 대체헷지": 0.0,
    }

    for account in load_account_configs():
        if not account.get("settings", {}).get("show_hold", True):
            continue

        account_id = account["account_id"]
        master_data = load_portfolio_master(account_id)
        cash_balance = float(master_data.get("cash_balance", 0.0) if master_data else 0.0)
        total_cash += cash_balance

        holdings_df = load_real_holdings_table(account_id)
        if holdings_df is None or holdings_df.empty:
            account_purchase = 0.0
            account_valuation = 0.0
        else:
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

    if total_assets > 0:
        bucket_pct_momentum = (bucket_totals["1. 모멘텀"] / total_assets) * 100
        bucket_pct_innovation = (bucket_totals["2. 혁신기술"] / total_assets) * 100
        bucket_pct_market = (bucket_totals["3. 시장지수"] / total_assets) * 100
        bucket_pct_dividend = (bucket_totals["4. 배당방어"] / total_assets) * 100
        bucket_pct_alternative = (bucket_totals["5. 대체헷지"] / total_assets) * 100
        bucket_pct_cash = (total_cash / total_assets) * 100
    else:
        bucket_pct_momentum = 0.0
        bucket_pct_innovation = 0.0
        bucket_pct_market = 0.0
        bucket_pct_dividend = 0.0
        bucket_pct_alternative = 0.0
        bucket_pct_cash = 0.0

    db[WEEKLY_COLLECTION].update_one(
        {"week_date": active_week_date},
        {
            "$set": {
                "total_assets": int(round(total_assets)),
                "purchase_amount": int(round(total_purchase)),
                "valuation_amount": int(round(total_valuation)),
                "exchange_rate": float(live_exchange_rate),
                "bucket_pct_momentum": float(bucket_pct_momentum),
                "bucket_pct_innovation": float(bucket_pct_innovation),
                "bucket_pct_market": float(bucket_pct_market),
                "bucket_pct_dividend": float(bucket_pct_dividend),
                "bucket_pct_alternative": float(bucket_pct_alternative),
                "bucket_pct_cash": float(bucket_pct_cash),
                "profit_count": total_profit_count,
                "loss_count": total_loss_count,
                "updated_at": _get_now_kst(),
            }
        },
        upsert=False,
    )
    return active_week_date


def _style_weekly_table(df: pd.DataFrame):
    """주별 데이터 테이블 컬럼 색상을 적용한다."""
    styler = df.style
    for column in YELLOW_COLUMNS:
        if column in df.columns:
            styler = styler.set_properties(subset=[column], **{"background-color": "#fff2cc"})
    for column in WHITE_COLUMNS:
        if column in df.columns:
            styler = styler.set_properties(subset=[column], **{"background-color": "#ffffff"})
    for column, color in HOME_SUMMARY_COLUMN_STYLES.items():
        if column in df.columns:
            styler = styler.set_properties(subset=[column], **{"background-color": color, "font-weight": "bold"})
    for column in MONEY_COLUMNS:
        if column in df.columns:
            styler = styler.set_properties(subset=[column], **{"text-align": "right"})
    if "수익 종목 수" in df.columns:
        styler = styler.map(
            lambda _: "color: #e06666; font-weight: bold;",
            subset=["수익 종목 수"],
        )
    if "손실 종목 수" in df.columns:
        styler = styler.map(
            lambda _: "color: #3d85c6; font-weight: bold;",
            subset=["손실 종목 수"],
        )
    for column in PROFIT_DISPLAY_COLUMNS:
        if column in df.columns:
            styler = styler.map(
                lambda value: f"color: {_get_profit_text_color(value)}; font-weight: bold;",
                subset=[column],
            )
    if "환율(변동)" in df.columns and "환율" in df.columns:

        def _style_exchange_rate_row(row: pd.Series) -> list[str]:
            styles = [""] * len(row)
            change_value = row.get("환율(변동)", 0.0)
            color = _get_profit_text_color(change_value)
            for idx, column in enumerate(row.index):
                if column in {"환율(변동)", "환율"}:
                    styles[idx] = f"color: {color}; font-weight: bold;"
            return styles

        styler = styler.apply(_style_exchange_rate_row, axis=1)
    return styler


@st.dialog("✏️ 주별 데이터 수정", width="large")
def _edit_weekly_modal(doc: dict):
    """주별 데이터 수정 모달. 좌우 2컬럼 레이아웃."""
    week_date = doc["week_date"]
    computed_doc = _apply_derived_fields(doc)
    st.markdown(f"### 📅 {_format_week_date_display(week_date)}")
    st.divider()

    # 버튼 스타일 (기존 모달 참고)
    st.markdown(
        """<style>
        .st-key-btn_weekly_save button { background-color: #2e7d32 !important; color: white !important; }
        .st-key-btn_weekly_cancel button { background-color: #757575 !important; color: white !important; }
        div[data-testid="stDialog"] div[data-testid="stNumberInput"] button {
            display: none !important;
        }
        </style>""",
        unsafe_allow_html=True,
    )

    # 좌우 2컬럼으로 필드 배치
    editable_fields = [field for field in FIELD_DEFS if field[0] not in READ_ONLY_FIELDS]
    mid = (len(editable_fields) + 1) // 2
    left_fields = editable_fields[:mid]
    right_fields = editable_fields[mid:]

    values: dict[str, object] = {}

    col_left, col_right = st.columns(2)

    with col_left:
        for db_key, label, field_type in left_fields:
            current_val = computed_doc.get(db_key, 0)
            if field_type == "int":
                values[db_key] = st.number_input(
                    label,
                    value=int(current_val or 0),
                    step=1,
                    key=f"wk_{db_key}",
                    disabled=db_key in READ_ONLY_FIELDS,
                )
            elif field_type == "float":
                values[db_key] = st.number_input(
                    label,
                    value=float(current_val or 0.0),
                    step=0.01,
                    format="%.2f",
                    key=f"wk_{db_key}",
                    disabled=db_key in READ_ONLY_FIELDS,
                )
            elif field_type == "text":
                values[db_key] = st.text_input(
                    label,
                    value=str(current_val or ""),
                    key=f"wk_{db_key}",
                    disabled=db_key in READ_ONLY_FIELDS,
                )

    with col_right:
        for db_key, label, field_type in right_fields:
            current_val = computed_doc.get(db_key, 0)
            if field_type == "int":
                values[db_key] = st.number_input(
                    label,
                    value=int(current_val or 0),
                    step=1,
                    key=f"wk_{db_key}",
                    disabled=db_key in READ_ONLY_FIELDS,
                )
            elif field_type == "float":
                values[db_key] = st.number_input(
                    label,
                    value=float(current_val or 0.0),
                    step=0.01,
                    format="%.2f",
                    key=f"wk_{db_key}",
                    disabled=db_key in READ_ONLY_FIELDS,
                )
            elif field_type == "text":
                values[db_key] = st.text_input(
                    label,
                    value=str(current_val or ""),
                    key=f"wk_{db_key}",
                    disabled=db_key in READ_ONLY_FIELDS,
                )

    st.divider()
    b_cancel, b_save = st.columns(2)
    if b_cancel.button("❌ 취소", width="stretch", key="btn_weekly_cancel"):
        st.rerun()

    if b_save.button("💾 저장", width="stretch", key="btn_weekly_save"):
        db = get_db_connection()
        if db is None:
            st.error("DB 연결 실패")
            return
        values["updated_at"] = _get_now_kst()
        db[WEEKLY_COLLECTION].update_one(
            {"week_date": week_date},
            {"$set": values},
        )
        st.success("✅ 저장 완료!")
        st.rerun()


def render_weekly_data_page():
    """주별 자금 관리 데이터 페이지."""
    loading = create_loading_status()
    try:
        loading.update("주별 데이터 화면 준비")

        # 활성 주차 데이터 자동 생성
        _ensure_active_week_row()
        _ensure_historical_exchange_rates()

        _render_weekly_table()
    finally:
        loading.clear()


def _render_weekly_table():
    """주별 데이터 테이블 + 행 선택 → 수정 모달."""
    st.subheader("📊 주별")
    aggregate_button_label = "🟢 이번주 데이터 집계"

    st.markdown(
        """<style>
        .st-key-btn_weekly_aggregate button {
            background-color: #2e7d32 !important;
            color: white !important;
            border: 1px solid #2e7d32 !important;
        }
        .st-key-btn_weekly_aggregate button:hover {
            background-color: #256628 !important;
            color: white !important;
            border: 1px solid #256628 !important;
        }
        </style>""",
        unsafe_allow_html=True,
    )

    if st.button(aggregate_button_label, key="btn_weekly_aggregate", width="content"):
        try:
            aggregated_week_date = _aggregate_live_summary_into_active_week()
            st.success(f"{_format_week_date_display(aggregated_week_date)} 행 데이터를 집계했습니다.")
            st.rerun()
        except RuntimeError as exc:
            st.error(str(exc))

    st.caption("홈 화면과 같은 실시간 합계를 활성 주차 1행에 반영합니다.")

    if "weekly_table_view_mode" not in st.session_state:
        st.session_state["weekly_table_view_mode"] = "핵심만 보기"

    selected_view_mode = st.segmented_control(
        "보기 모드",
        options=["핵심만 보기", "전체 보기"],
        default=st.session_state["weekly_table_view_mode"],
        key="weekly_table_view_mode_selector",
        label_visibility="collapsed",
    )
    if selected_view_mode:
        st.session_state["weekly_table_view_mode"] = selected_view_mode
    else:
        selected_view_mode = st.session_state["weekly_table_view_mode"]

    docs = _load_weekly_docs()
    if not docs:
        st.info("저장된 주별 데이터가 없습니다.")
        return

    # 표시용 DataFrame 생성
    rows = _build_display_rows(docs)
    df = pd.DataFrame(rows)
    if selected_view_mode == "핵심만 보기":
        visible_columns = [column for column in df.columns if column not in CORE_VIEW_HIDDEN_COLUMNS]
        df = df[visible_columns]

    # 테이블 표시 (행 선택 활성화)
    selection = st.dataframe(
        _style_weekly_table(df),
        width="stretch",
        height=720,
        hide_index=True,
        on_select="rerun",
        selection_mode="single-row",
        key="weekly_data_table",
        column_config={
            "날짜": st.column_config.TextColumn(width=100),
            "개인 인출": st.column_config.TextColumn(width=90),
            "엄마": st.column_config.TextColumn(width=90),
            "농협원리금": st.column_config.TextColumn(width=90),
            "지출 합계": st.column_config.TextColumn(width=90),
            "입출금": st.column_config.TextColumn(width=90),
            "총 원금": st.column_config.TextColumn(width=90),
            "총 자산": st.column_config.TextColumn(width=90),
            "매입 금액": st.column_config.TextColumn(width=90),
            "평가 금액": st.column_config.TextColumn(width=90),
            "평가 손익": st.column_config.TextColumn(width=90),
            "누적 손익": st.column_config.TextColumn(width=90),
            "금주 손익": st.column_config.TextColumn(width=90),
            "주수익률": st.column_config.NumberColumn(format="%.2f%%", width=80),
            "누적 수익률": st.column_config.NumberColumn(format="%.2f%%", width=90),
            "비고": st.column_config.TextColumn(width=220),
            "환율(변동)": st.column_config.NumberColumn(format="%.1f%%", width=80),
            "환율": st.column_config.NumberColumn(format="%.2f", width=70),
            "1. 모멘텀": st.column_config.NumberColumn(format="%.2f%%", width=80),
            "2. 혁신기술": st.column_config.NumberColumn(format="%.2f%%", width=80),
            "3. 시장지수": st.column_config.NumberColumn(format="%.2f%%", width=80),
            "4. 배당방어": st.column_config.NumberColumn(format="%.2f%%", width=80),
            "5. 대체헷지": st.column_config.NumberColumn(format="%.2f%%", width=80),
            "6. 현금": st.column_config.NumberColumn(format="%.2f%%", width=80),
            "총 종목 수": st.column_config.NumberColumn(format="%d", width=80),
            "수익 종목 수": st.column_config.NumberColumn(format="%d", width=90),
            "손실 종목 수": st.column_config.NumberColumn(format="%d", width=90),
        },
    )

    # 행 선택 시 수정 모달 열기
    if selection and selection.selection and selection.selection.rows:
        selected_idx = selection.selection.rows[0]
        if selected_idx < len(docs):
            _edit_weekly_modal(docs[selected_idx])


def build_weekly_data_page(page_cls, *, title: str = "주별", url_path: str = "transactions_weekly"):
    """주별 페이지 빌드 헬퍼."""
    return page_cls(
        render_weekly_data_page,
        title=title,
        icon="📅",
        url_path=url_path,
    )
