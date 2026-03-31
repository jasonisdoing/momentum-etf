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

    # target_id가 비어있으면 모든 계좌를 순회하며 데이터를 수집함
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
            # NaN/None 방어 로직 추가
            def safe_int(val):
                import pandas as pd
                if pd.isna(val) or val is None:
                    return 0
                try:
                    return int(float(val))
                except (ValueError, TypeError):
                    return 0

            quantity = safe_int(row.get("수량"))
            buy_amount = safe_int(row.get("매입금액(KRW)"))
            val_amount = safe_int(row.get("평가금액(KRW)"))
            pnl = safe_int(row.get("평가손익(KRW)"))
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
                    "current_price_num": current_price,
                    "days_held": str(row.get("보유일", "-")),
                    "pnl_krw": pnl,
                    "pnl_krw_num": pnl,
                    "return_pct": round(ret_pct, 2),
                    "daily_change_pct": float(row.get("일간(%)") or 0) if row.get("일간(%)") is not None else None,
                    "buy_amount_krw": buy_amount,
                    "valuation_krw": val_amount,
                    "memo": str(row.get("memo") or "").strip(),
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


def update_holding(account_id: str, ticker: str, quantity: int | None = None, average_buy_price: float | None = None, memo: str | None = None) -> dict[str, str]:
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
            if memo is not None:
                h["memo"] = str(memo).strip()
            found = True
            break

    if not found:
        raise RuntimeError(f"종목 {ticker}을 찾을 수 없습니다.")

    save_portfolio_master(account_id, holdings)
    return {"updated": ticker}


def add_holding(account_id: str, ticker: str, quantity: int, average_buy_price: float, memo: str | None = None) -> dict[str, Any]:
    """계좌에 새로운 종목을 추가한다."""
    account_id = str(account_id or "").strip()
    ticker = str(ticker or "").strip()

    if not account_id or not ticker:
        raise RuntimeError("계좌 ID와 종목코드가 필요합니다.")

    # 1. 티커 검증 및 상세 메타데이터 가져오기 (종목명, 버킷 등)
    res = validate_ticker_for_account(account_id, ticker)
    raw_ticker = res["ticker"]

    # 2. 계좌 설정에서 통화 정보 가져오기
    from utils.settings_loader import get_account_settings
    try:
        settings = get_account_settings(account_id)
        currency = settings.get("currency", "KRW")
    except Exception:
        currency = "KRW"

    master = load_portfolio_master(account_id)
    if not master:
        raise RuntimeError("계좌 데이터를 찾을 수 없습니다.")

    holdings = master.get("holdings", [])

    # 중복 확인
    for h in holdings:
        if str(h.get("ticker", "")).strip() == raw_ticker:
            raise RuntimeError(f"종목 {ticker}은 이미 등록되어 있습니다.")

    # 3. 정석적인 구조로 새로운 종목 구성
    from datetime import datetime
    new_holding = {
        "ticker": raw_ticker,
        "name": res["name"],
        "quantity": int(quantity),
        "average_buy_price": float(average_buy_price),
        "currency": currency,
        "bucket": res.get("bucket_id", 1),
        "first_buy_date": datetime.now().strftime("%Y-%m-%d"),
        "memo": str(memo or "").strip(),
    }

    holdings.append(new_holding)
    save_portfolio_master(account_id, holdings)

    return {"added": ticker, "name": res["name"]}


def validate_ticker_for_account(account_id: str, ticker: str) -> dict[str, Any]:
    """계좌에 추가할 수 있는 유효한 티커인지 검증한다."""
    account_id = str(account_id or "").strip()
    ticker = str(ticker or "").strip().upper()

    if not account_id or not ticker:
        raise RuntimeError("계좌 ID와 종목코드가 필요합니다.")

    # ASX: 접두어 제거
    raw_ticker = ticker.replace("ASX:", "").strip().upper()
    if not raw_ticker:
        raise RuntimeError("유효한 티커를 입력하세요.")

    from utils.settings_loader import get_account_settings
    from utils.stocks_service import validate_stock_candidate

    # 1. 계좌 설정 로드 (zaccounts/ 하위의 실제 설정 파일 읽기)
    try:
        settings = get_account_settings(account_id)
        # account_settings["settings"]가 아닌 top-level에 있는 경우가 많음
        inner_settings = settings.get("settings") or settings
    except Exception as e:
        raise RuntimeError(f"계좌 설정을 찾을 수 없습니다: {account_id} ({e})")

    # 2. 계좌의 ticker_types 목록 추출 (실제 필드명인 'ticker_codes' 사용)
    ticker_types = settings.get("ticker_codes") or []
    
    if isinstance(ticker_types, str):
        ticker_types = [ticker_types]

    if not ticker_types:
        available_keys = list(settings.keys())
        raise RuntimeError(f"계좌 설정({account_id})에서 'ticker_codes'를 찾을 수 없습니다. (Keys: {available_keys})")

    # 3. 기존 "종목 추가" 모달과 동일한 검증 엔진 사용
    last_error = None
    validated_res = None
    
    for tt in ticker_types:
        try:
            # StocksManager가 사용하는 동일한 함수 호출
            validated_res = validate_stock_candidate(tt, ticker)
            break # 성공하면 루프 중단
        except Exception as e:
            last_error = str(e)
            continue

    if not validated_res:
        raise RuntimeError(last_error or f"등록되지 않은 종목입니다: {ticker}")

    return {
        "ticker": validated_res["ticker"],
        "name": validated_res["name"],
        "bucket_id": validated_res.get("bucket_id") or 1,
        "status": "success"
    }
