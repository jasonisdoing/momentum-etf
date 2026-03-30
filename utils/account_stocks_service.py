from __future__ import annotations

from typing import Any

from utils.account_registry import load_account_configs, pick_default_account
from utils.account_stocks_io import get_account_targets
from utils.db_manager import get_db_connection
from utils.normalization import normalize_nullable_number, normalize_text
from utils.stock_list_io import get_etfs
from services.price_service import get_realtime_snapshot


BUCKETS: dict[int, str] = {
    1: "1. 모멘텀",
    2: "2. 시장지수",
    3: "3. 배당방어",
    4: "4. 대체헷지",
}


def _build_accounts_payload(account_id: str | None) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    accounts = load_account_configs()
    if not accounts:
        raise ValueError("사용 가능한 계정이 없습니다.")

    default_account = pick_default_account(accounts)
    
    payload = [
        {
            "account_id": str(acc["account_id"]),
            "order": int(acc["order"]),
            "name": str(acc["name"]),
            "icon": str(acc.get("icon") or ""),
            "country_code": str(acc.get("country_code") or ""),
            "ticker_codes": acc.get("settings", {}).get("ticker_codes", []),
        }
        for acc in accounts
    ]
    
    selected_account_id = str(account_id or default_account["account_id"]).strip().lower()
    selected_account = next((a for a in payload if a["account_id"] == selected_account_id), payload[0])
    
    return payload, selected_account, selected_account_id


def load_account_stocks_data(account_id: str | None) -> dict[str, Any]:
    accounts, selected_account, selected_account_id = _build_accounts_payload(account_id)
    
    # 해당 계좌에 설정된 ticker_codes 목록 (예: ["kor_kr", "kor_us"])
    ticker_codes = selected_account.get("ticker_codes", [])
    country_code = selected_account.get("country_code", "kor")
    
    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결에 실패했습니다.")

    # 저장된 목표 비중들을 가져온다
    saved_targets = get_account_targets(selected_account_id)
    target_dict = {str(t.get("ticker")).upper(): float(t.get("ratio") or 0.0) for t in saved_targets}

    # 허용된 ticker_codes들에 매핑된 모든 활성 종목(stock_meta)들을 가져온다. (종목 추가 모달을 위해 전체 풀이 필요)
    docs = list(
        db.stock_meta.find(
            {
                "ticker_type": {"$in": ticker_codes},
                "is_deleted": {"$ne": True},
            },
            {
                "ticker": 1,
                "name": 1,
                "bucket": 1,
                "added_date": 1,
                "listing_date": 1,
                "1_week_avg_volume": 1,
                "1_week_earn_rate": 1,
                "2_week_earn_rate": 1,
                "1_month_earn_rate": 1,
                "3_month_earn_rate": 1,
                "6_month_earn_rate": 1,
                "12_month_earn_rate": 1,
            },
        )
    )

    # 실시간 스냅샷 통합
    tickers = [doc.get("ticker", "") for doc in docs if doc.get("ticker")]
    realtime_snapshot = {}
    try:
        realtime_snapshot = get_realtime_snapshot(country_code, tickers)
    except Exception:
        pass

    available_tickers = []
    
    # 전체 메타 순회하면서 available_tickers 구성
    for doc in docs:
        ticker = normalize_text(doc.get("ticker"), "")
        name = normalize_text(doc.get("name"), "")
        if not ticker:
            continue
            
        available_tickers.append({
            "ticker": ticker,
            "name": name,
            "bucket_id": int(doc.get("bucket") or 1),
            "bucket_name": BUCKETS.get(int(doc.get("bucket") or 1), BUCKETS[1]),
            "added_date": normalize_text(doc.get("added_date"), "-"),
            "listing_date": normalize_text(doc.get("listing_date"), "-"),
            "week_volume": normalize_nullable_number(doc.get("1_week_avg_volume")),
            "return_1d": normalize_nullable_number(
                realtime_snapshot.get(ticker, {}).get("changeRate")
            ),
            "return_1w": normalize_nullable_number(doc.get("1_week_earn_rate")),
            "return_2w": normalize_nullable_number(doc.get("2_week_earn_rate")),
            "return_1m": normalize_nullable_number(doc.get("1_month_earn_rate")),
            "return_3m": normalize_nullable_number(doc.get("3_month_earn_rate")),
            "return_6m": normalize_nullable_number(doc.get("6_month_earn_rate")),
            "return_12m": normalize_nullable_number(doc.get("12_month_earn_rate")),
        })
        
    # 현재 targets에 저장된 항목만 필터링하여 rows 생성
    # DB에 저장되어있지 않은(예: account_targets에 없는) stock_meta는 리스트에 표시하지 않음
    rows = []
    for item in available_tickers:
        ticker = item["ticker"]
        if ticker in target_dict:
            # 딕셔너리 복사
            row_item = dict(item)
            row_item["ratio"] = target_dict[ticker]
            rows.append(row_item)

    # 비중이 큰 순서대로, 비중이 같으면 1주 수익률이 좋은 순서대로 정렬
    rows = sorted(
        rows,
        key=lambda row: (
            -(row.get("ratio") or 0.0),
            -(row.get("return_1w") if row.get("return_1w") is not None else float("-inf")),
        ),
    )

    return {
        "accounts": accounts,
        "account_id": selected_account_id,
        "available_tickers": available_tickers,
        "rows": rows,
    }
