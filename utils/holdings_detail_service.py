"""계좌 상세 — 전체 계좌의 보유 종목을 한 테이블로 반환한다."""

from __future__ import annotations

from typing import Any

from config import BUCKET_MAPPING
from services.price_service import get_exchange_rates
from utils.account_registry import load_account_configs
from utils.cash_service import load_cash_accounts
from utils.logger import get_app_logger
from utils.portfolio_io import load_portfolio_master, load_real_holdings_table, save_portfolio_master

logger = get_app_logger()


def load_all_holdings_detail(account_id: str | None = None) -> dict[str, Any]:
    """모든 계좌 또는 특정 계좌의 보유 종목을 반환한다."""
    all_accounts = load_account_configs()
    rates = get_exchange_rates()

    target_id = str(account_id or "").strip()
    if target_id.upper() == "TOTAL":
        target_id = ""
    if not target_id and all_accounts:
        target_id = str(all_accounts[0]["account_id"])

    all_rows: list[dict[str, Any]] = []

    for account in all_accounts:
        curr_account_id = str(account["account_id"])

        # 필터링: account_id가 있으면 해당 계좌만, 없으면 전체
        if target_id and curr_account_id != target_id:
            continue

        account_name = str(account.get("name") or curr_account_id)

        try:
            df = load_real_holdings_table(
                curr_account_id,
                preloaded_exchange_rates=rates,
            )
        except Exception as exc:
            logger.warning("holdings 로드 실패 (%s): %s", curr_account_id, exc)
            continue

        if df is None or df.empty:
            continue

        settings = account.get("settings") or {}
        country_code = str(settings.get("country_code") or "").strip().lower()
        currency = str(settings.get("currency") or "KRW").strip().upper()

        for _, row in df.iterrows():
            ticker_raw = str(row.get("티커") or "").strip()
            row_currency = str(row.get("환종") or currency).strip().upper()

            # 종목코드 포맷: 호주는 ASX:TICKER
            if country_code == "au" and ticker_raw != "IS":
                display_ticker = f"ASX:{ticker_raw}"
            else:
                display_ticker = ticker_raw

            bucket_id = int(row.get("bucket_id") or 0)
            bucket_name = BUCKET_MAPPING.get(bucket_id, f"{bucket_id}. Bucket")

            avg_price = float(row.get("평균 매입가") or 0)
            current_price = float(row.get("현재가") or 0)
            quantity = int(row.get("수량") or 0)
            buy_amount = int(row.get("매입금액(KRW)") or 0)
            val_amount = int(row.get("평가금액(KRW)") or 0)
            pnl = int(row.get("평가손익(KRW)") or 0)
            ret_pct = float(row.get("수익률(%)") or 0)

            # 현지 통화 가격 포맷
            price_prefix = ""
            if row_currency == "AUD":
                price_prefix = "A$"
            elif row_currency == "USD":
                price_prefix = "$"

            all_rows.append(
                {
                    "account_name": account_name,
                    "currency": row_currency,
                    "bucket": bucket_name,
                    "bucket_id": bucket_id,
                    "ticker": display_ticker,
                    "name": str(row.get("종목명") or ""),
                    "quantity": quantity,
                    "average_buy_price": f"{price_prefix}{avg_price:,.4f}" if price_prefix else f"{avg_price:,.0f}원",
                    "current_price": f"{price_prefix}{current_price:,.2f}"
                    if price_prefix
                    else f"{current_price:,.0f}원",
                    "pnl_krw": pnl,
                    "return_pct": round(ret_pct, 2),
                    "buy_amount_krw": buy_amount,
                    "valuation_krw": val_amount,
                }
            )

    # 현재 계좌의 cash 정보
    cash_data = load_cash_accounts()
    cash_accounts = cash_data.get("accounts", [])
    cash_info = next((c for c in cash_accounts if c["account_id"] == target_id), None)

    return {
        "accounts": [
            *[
                {
                    "account_id": a["account_id"],
                    "name": a["name"],
                    "icon": a["icon"],
                }
                for a in all_accounts
            ]
        ],
        "account_id": target_id,
        "cash": cash_info,
        "rows": all_rows
    }


def delete_holding(account_id: str, ticker: str) -> dict[str, str]:
    """계좌에서 특정 종목을 삭제한다."""
    account_id = str(account_id or "").strip()
    ticker = str(ticker or "").strip()
    if not account_id or not ticker:
        raise RuntimeError("계좌 ID와 종목코드가 필요합니다.")

    # ASX: 접두어 제거
    raw_ticker = ticker.replace("ASX:", "")

    master = load_portfolio_master(account_id)
    if not master:
        raise RuntimeError("계좌 데이터를 찾을 수 없습니다.")

    holdings = master.get("holdings", [])
    new_holdings = [h for h in holdings if str(h.get("ticker", "")).strip() != raw_ticker]

    if len(new_holdings) == len(holdings):
        raise RuntimeError(f"종목 {ticker}을 찾을 수 없습니다.")

    save_portfolio_master(account_id, new_holdings)
    return {"deleted": ticker}


def update_holding(account_id: str, ticker: str, quantity: int | None = None, average_buy_price: float | None = None) -> dict[str, str]:
    """계좌의 특정 종목 수량/매입단가를 수정한다."""
    account_id = str(account_id or "").strip()
    ticker = str(ticker or "").strip()
    if not account_id or not ticker:
        raise RuntimeError("계좌 ID와 종목코드가 필요합니다.")

    raw_ticker = ticker.replace("ASX:", "")

    master = load_portfolio_master(account_id)
    if not master:
        raise RuntimeError("계좌 데이터를 찾을 수 없습니다.")

    holdings = master.get("holdings", [])
    found = False
    for h in holdings:
        if str(h.get("ticker", "")).strip() == raw_ticker:
            if quantity is not None:
                h["quantity"] = int(quantity)
            if average_buy_price is not None:
                h["average_buy_price"] = float(average_buy_price)
            found = True
            break

    if not found:
        raise RuntimeError(f"종목 {ticker}을 찾을 수 없습니다.")

    save_portfolio_master(account_id, holdings)
    return {"updated": ticker}


def add_holding(account_id: str, ticker: str, quantity: int, average_buy_price: float) -> dict[str, Any]:
    """계좌에 새로운 종목을 추가한다."""
    account_id = str(account_id or "").strip()
    ticker = str(ticker or "").strip()

    if not account_id or not ticker:
        raise RuntimeError("계좌 ID와 종목코드가 필요합니다.")

    raw_ticker = ticker.replace("ASX:", "")

    master = load_portfolio_master(account_id)
    if not master:
        raise RuntimeError("계좌 데이터를 찾을 수 없습니다.")

    holdings = master.get("holdings", [])

    # 중복 확인
    for h in holdings:
        if str(h.get("ticker", "")).strip() == raw_ticker:
            raise RuntimeError(f"종목 {ticker}은 이미 등록되어 있습니다.")

    # 새로운 종목 추가
    new_holding = {
        "ticker": raw_ticker,
        "quantity": int(quantity),
        "average_buy_price": float(average_buy_price),
    }

    holdings.append(new_holding)
    save_portfolio_master(account_id, holdings)

    return {"added": ticker}


def validate_ticker_for_account(account_id: str, ticker: str) -> dict[str, Any]:
    """계좌에 추가할 수 있는 유효한 티커인지 검증한다."""
    from pymongo import MongoClient
    from config import DB_NAME

    account_id = str(account_id or "").strip()
    ticker = str(ticker or "").strip().upper()

    if not account_id or not ticker:
        raise RuntimeError("계좌 ID와 종목코드가 필요합니다.")

    # ASX: 접두어 제거
    raw_ticker = ticker.replace("ASX:", "").strip().upper()
    if not raw_ticker:
        raise RuntimeError("유효한 티커를 입력하세요.")

    # 계좌 설정 조회
    all_accounts = load_account_configs()
    account = next((a for a in all_accounts if str(a.get("account_id")) == account_id), None)
    if not account:
        raise RuntimeError("계좌를 찾을 수 없습니다.")

    settings = account.get("settings") or {}
    country_code = str(settings.get("country_code") or "").strip().lower()

    # stock_meta에서 조회
    try:
        client = MongoClient("mongodb://localhost:27017")
        db = client[DB_NAME]
        stock_meta = db["stock_meta"]

        # ticker_type 결정 (임시: country_code 기반)
        if country_code == "kor":
            ticker_types = ["kr-fund", "kr-stock"]
        elif country_code == "us":
            ticker_types = ["us-etf", "us-stock"]
        elif country_code == "au":
            ticker_types = ["au-etf", "au-stock"]
        else:
            ticker_types = ["kr-fund", "kr-stock"]  # 기본값

        # 등록된 종목 찾기
        doc = None
        for tt in ticker_types:
            doc = stock_meta.find_one({
                "ticker_type": tt,
                "ticker": raw_ticker,
                "is_deleted": {"$ne": True},
            })
            if doc:
                break

        if not doc:
            raise RuntimeError(f"등록되지 않은 종목입니다: {ticker}")

        return {
            "ticker": raw_ticker,
            "name": str(doc.get("name") or ""),
            "bucket_id": int(doc.get("bucket") or 1),
        }

    except Exception as e:
        if "등록되지 않은 종목" in str(e):
            raise
        raise RuntimeError(f"티커 검증 중 오류: {e}")
    finally:
        if 'client' in locals():
            client.close()
