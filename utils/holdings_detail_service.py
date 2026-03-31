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
