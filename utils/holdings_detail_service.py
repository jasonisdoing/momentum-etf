"""계좌 상세 — 전체 계좌의 보유 종목을 한 테이블로 반환한다."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from config import BUCKET_MAPPING
from services.price_service import get_exchange_rates
from utils.account_registry import load_account_configs
from utils.assets_service import load_cash_accounts
from utils.asx_ticker import ensure_asx_prefix, strip_asx_prefix
from utils.cash_model import cash_total_krw
from utils.logger import get_app_logger
from utils.portfolio_io import load_portfolio_master, load_real_holdings_table, save_portfolio_master

logger = get_app_logger()


def _normalize_target_ticker(ticker: str) -> str:
    """비교 전용 키 — 접두사 유무와 무관하게 같은 종목이 같은 값이 되도록 벗긴다.

    저장·표시용이 아니다. 저장/표시에는 `ensure_asx_prefix` 로 접두사를 붙인 값을 쓴다.
    """
    return strip_asx_prefix(ticker)


def _assign_sort_order(holdings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """보유 종목 배열에 순서를 다시 부여한다."""
    normalized_holdings: list[dict[str, Any]] = []
    for index, holding in enumerate(holdings):
        next_holding = dict(holding)
        next_holding["sort_order"] = index
        normalized_holdings.append(next_holding)
    return normalized_holdings


def _compute_account_total_assets_native(
    rows: list[dict[str, Any]],
    cash_info: dict[str, Any] | None,
    account_currency: str,
    rates: dict[str, Any],
) -> float:
    total_valuation_krw = sum(float(row.get("valuation_krw") or 0.0) for row in rows)
    currency = str(account_currency or "KRW").strip().upper() or "KRW"
    cash_info = cash_info or {}

    if currency == "KRW":
        return total_valuation_krw + float(cash_info.get("cash_balance_krw") or 0.0)

    if currency == "AUD":
        aud_rate = float(((rates or {}).get("AUD") or {}).get("rate") or 0.0)
        if aud_rate <= 0:
            raise RuntimeError("AUD 환율을 가져오지 못했습니다.")
        cash_native = cash_info.get("cash_balance_native")
        if cash_native in (None, ""):
            cash_native = float(cash_info.get("cash_balance_krw") or 0.0) / aud_rate
        return (total_valuation_krw / aud_rate) + float(cash_native or 0.0)

    return total_valuation_krw


def _compute_target_quantity(target_amount: float, current_price: float, currency: str) -> float | int | None:
    # NaN 은 `<= 0` 비교를 통과하므로(NaN 비교는 항상 False) 유한성 검사를 먼저 한다.
    # 가격·목표금액을 알 수 없으면 목표수량도 계산 불가(None → 화면 '-') 로 명시한다.
    if not math.isfinite(current_price) or current_price <= 0 or not math.isfinite(target_amount):
        return None
    quantity = target_amount / current_price
    currency_code = str(currency or "KRW").strip().upper()
    if currency_code == "AUD":
        return round(quantity, 4)
    return max(math.floor(quantity), 0)


def _apply_target_metrics(
    rows: list[dict[str, Any]],
    account_id: str,
    cash_info: dict[str, Any] | None,
    account_currency: str,
    rates: dict[str, Any],
) -> list[dict[str, Any]]:
    if not account_id:
        return rows

    if not rows:
        return rows

    # 목표비중은 보유 항목(portfolio_master.holdings)의 target_ratio 필드가 단일 소스다.
    master = load_portfolio_master(account_id) or {}
    target_map = {
        _normalize_target_ticker(str(h.get("ticker") or "")): float(h["target_ratio"])
        for h in master.get("holdings") or []
        if h.get("target_ratio") is not None
    }
    account_total_assets = _compute_account_total_assets_native(rows, cash_info, account_currency, rates)
    # 목표수량은 통화 혼합 계좌(KRW 계좌의 USD 종목 등)를 위해 전부 KRW 로 환산해 계산한다.
    account_total_krw = sum(float(row.get("valuation_krw") or 0.0) for row in rows) + float(
        (cash_info or {}).get("cash_balance_krw") or 0.0
    )

    def _row_fx_rate_krw(row_currency: str) -> float | None:
        currency_code = str(row_currency or "KRW").strip().upper() or "KRW"
        if currency_code == "KRW":
            return 1.0
        rate = float(((rates or {}).get(currency_code) or {}).get("rate") or 0.0)
        return rate if rate > 0 else None

    enriched_rows: list[dict[str, Any]] = []
    for row in rows:
        target_ratio = target_map.get(_normalize_target_ticker(str(row.get("ticker") or "")))
        next_row = dict(row)
        next_row["target_ratio"] = target_ratio
        if target_ratio is None:
            next_row["target_amount"] = None
            next_row["target_quantity"] = None
        else:
            target_amount = round(account_total_assets * (target_ratio / 100.0), 2)
            next_row["target_amount"] = target_amount
            # 목표수량 = KRW 목표금액 ÷ (현재가 × 종목 통화 환율). 환율이 없으면 계산 불가(None).
            fx_rate = _row_fx_rate_krw(str(next_row.get("currency") or ""))
            price_krw = float(next_row.get("current_price_num") or 0.0) * fx_rate if fx_rate else 0.0
            next_row["target_quantity"] = _compute_target_quantity(
                account_total_krw * (target_ratio / 100.0),
                price_krw,
                str(next_row.get("currency") or account_currency),
            )
        enriched_rows.append(next_row)
    return enriched_rows


def _set_holding_target_ratio(holding: dict[str, Any], target_ratio: float | None) -> None:
    """보유 항목의 목표비중 필드를 갱신한다 — 0 이하/None 은 미설정(필드 제거)으로 처리."""
    if target_ratio is not None and float(target_ratio) > 0:
        holding["target_ratio"] = float(target_ratio)
    else:
        holding.pop("target_ratio", None)


def load_all_holdings_detail(account_id: str | None = None) -> dict[str, Any]:
    """모든 계좌 또는 특정 계좌의 보유 종목을 반환한다."""
    all_accounts = load_account_configs()
    rates = get_exchange_rates()
    cash_data = load_cash_accounts()
    cash_accounts = cash_data.get("accounts", [])
    cash_map = {str(account.get("account_id") or ""): account for account in cash_accounts if isinstance(account, dict)}

    target_id = str(account_id or "").strip()
    if target_id.upper() == "TOTAL":
        target_id = ""

    # target_id가 비어있으면 모든 계좌를 순회하며 데이터를 수집함
    all_rows: list[dict[str, Any]] = []
    account_summaries: list[dict[str, Any]] = []
    selected_cash_info: dict[str, Any] | None = None

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

        # 보유 종목이 0개인 계좌도 합계/현금/원금은 표시해야 한다.
        # df 가 None 이거나 비어있으면 빈 DataFrame 으로 처리해서 이어 진행한다.
        if df is None or df.empty:
            import pandas as pd  # 지연 import (위 모듈 import 와 일관성)

            df = pd.DataFrame()

        settings = account.get("settings") or {}
        country_code = str(settings.get("country_code") or "").strip().lower()
        currency = str(settings.get("currency") or "KRW").strip().upper()
        cash_info = cash_map.get(curr_account_id)
        if curr_account_id == target_id:
            selected_cash_info = cash_info

        account_rows: list[dict[str, Any]] = []

        for _, row in df.iterrows():
            ticker_raw = str(row.get("티커") or "").strip()
            row_currency = str(row.get("환종") or currency).strip().upper()

            # 종목코드 포맷: 호주는 ASX:TICKER (이미 붙어 있으면 그대로 — 중복 방지)
            if country_code == "au" and ticker_raw != "IS":
                display_ticker = ensure_asx_prefix(ticker_raw)
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

            # IS(가상 VGS 수량) 처럼 소수 수량이 있는 행은 소수점을 보존한다.
            try:
                quantity = float(row.get("수량"))
            except (TypeError, ValueError):
                quantity = 0.0
            if not math.isfinite(quantity):
                quantity = 0.0
            quantity = int(quantity) if quantity.is_integer() else round(quantity, 4)
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

            # 종목 통화 → 원화 환율 배수(프론트 라이브 비중 계산용). KRW=1.0.
            if row_currency == "USD":
                fx_rate_krw = float(((rates or {}).get("USD") or {}).get("rate") or 0.0)
            elif row_currency == "AUD":
                fx_rate_krw = float(((rates or {}).get("AUD") or {}).get("rate") or 0.0)
            else:
                fx_rate_krw = 1.0

            account_rows.append(
                {
                    "account_id": curr_account_id,
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
                    "fx_rate_krw": fx_rate_krw,
                    "pnl_krw": pnl,
                    "pnl_krw_num": pnl,
                    "return_pct": round(ret_pct, 2),
                    "weight_pct": float(row.get("weight_pct") or 0),
                    "daily_change_pct": float(row.get("일간(%)") or 0) if row.get("일간(%)") is not None else None,
                    "buy_amount_krw": buy_amount,
                    "valuation_krw": val_amount,
                    "memo": str(row.get("memo") or "").strip(),
                    "sort_order": safe_int(row.get("sort_order")),
                    "ticker_type": str(row.get("ticker_type") or "").strip(),
                    "country_code": str(row.get("country_code") or "").strip(),
                    "is_etf": bool(row.get("is_etf")),
                }
            )

        account_rows = _apply_target_metrics(
            account_rows,
            account_id=curr_account_id,
            cash_info=cash_info,
            account_currency=currency,
            rates=rates,
        )
        all_rows.extend(account_rows)

        valuation_krw = sum(float(row.get("valuation_krw") or 0.0) for row in account_rows)
        # 현금: 통화별 native 맵을 원화로 환산·합산한다(다통화 계좌). 맵이 없으면 레거시 단일 현금.
        cash_native_map = (cash_info or {}).get("cash") or {}
        if cash_native_map:
            cash_balance_krw = round(cash_total_krw(cash_native_map, rates or {}), 2)
        else:
            cash_balance_krw = float((cash_info or {}).get("cash_balance_krw") or 0.0)
            # 호주 계좌: cash_balance_krw가 0이고 cash_balance_native가 있으면 환율 적용하여 KRW 환산
            if cash_balance_krw == 0.0 and currency == "AUD":
                cash_native = float((cash_info or {}).get("cash_balance_native") or 0.0)
                if cash_native > 0:
                    aud_rate = float(((rates or {}).get("AUD") or {}).get("rate") or 0.0)
                    if aud_rate > 0:
                        cash_balance_krw = round(cash_native * aud_rate, 2)
        cash_target_ratio = float((cash_info or {}).get("cash_target_ratio") or 0.0)
        # 현금 컬럼 표시용: 원화 합계를 계좌의 주 통화로 환산한 값(호주=AUD, 국내=KRW).
        if currency == "KRW":
            base_rate = 1.0
        else:
            base_rate = float(((rates or {}).get(currency) or {}).get("rate") or 0.0)
        cash_display_native = round(cash_balance_krw / base_rate, 2) if base_rate > 0 else cash_balance_krw
        target_ratio_total = sum(float(row.get("target_ratio") or 0.0) for row in account_rows) + cash_target_ratio
        account_summaries.append(
            {
                "account_id": curr_account_id,
                "order": int(account["order"]),
                "name": account_name,
                "account_url": str(settings.get("URL") or "").strip() or None,
                "icon": str(account.get("icon") or ""),
                "currency": currency,
                "total_principal": float((cash_info or {}).get("total_principal") or 0.0),
                "cash_balance_krw": cash_balance_krw,
                "cash_balance_native": (cash_info or {}).get("cash_balance_native"),
                "cash_currency": str((cash_info or {}).get("cash_currency") or currency).strip().upper(),
                "cash": cash_native_map,
                "cash_currencies": (cash_info or {}).get("cash_currencies") or [],
                "cash_display_native": cash_display_native,
                "cash_display_currency": currency,
                "cash_target_ratio": cash_target_ratio,
                # 자산 헬퍼에서 저장한 현금 목표 비중(%) — /assets 목표비중 칸의 유일한 소스.
                # 미저장이면 None 그대로 내려 화면이 '-' 로 표시한다(파생·기본값 금지).
                "helper_cash_weight_pct": (
                    (load_portfolio_master(curr_account_id) or {}).get("asset_helper") or {}
                ).get("cash_weight_pct"),
                "intl_shares_value": (cash_info or {}).get("intl_shares_value"),
                "intl_shares_change": (cash_info or {}).get("intl_shares_change"),
                "updated_at": (cash_info or {}).get("updated_at"),
                "updated_by": (cash_info or {}).get("updated_by"),
                "valuation_krw": valuation_krw,
                "total_assets_krw": valuation_krw + cash_balance_krw,
                "holdings_count": len([r for r in account_rows if str(r.get("ticker") or "") != "IS"]),
                "target_ratio_total": target_ratio_total,
            }
        )

    if target_id:
        account_summaries = [row for row in account_summaries if str(row.get("account_id") or "") == target_id]

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
        "cash": selected_cash_info,
        "rows": all_rows,
        "account_summaries": sorted(account_summaries, key=lambda row: int(row.get("order") or 0)),
    }


def delete_holding(account_id: str, ticker: str) -> dict[str, str]:
    """계좌에서 특정 종목을 삭제한다."""
    account_id = str(account_id or "").strip()
    ticker = str(ticker or "").strip()
    if not account_id or not ticker:
        raise RuntimeError("계좌 ID와 종목코드가 필요합니다.")

    # 저장된 티커와 요청 티커 양쪽을 같은 규칙으로 정규화해 비교한다
    # (ASX: 접두사 유무가 달라도 같은 종목으로 매칭되도록).
    target_ticker = _normalize_target_ticker(ticker)

    master = load_portfolio_master(account_id)
    if not master:
        raise RuntimeError("계좌 데이터를 찾을 수 없습니다.")

    holdings = master.get("holdings", [])
    new_holdings = [h for h in holdings if _normalize_target_ticker(str(h.get("ticker", ""))) != target_ticker]

    if len(new_holdings) == len(holdings):
        raise RuntimeError(f"종목 {ticker}을 찾을 수 없습니다.")

    save_portfolio_master(account_id, _assign_sort_order(new_holdings))

    # 변경 사항을 스냅샷에 즉시 동기화
    try:
        from utils.snapshot_service import update_today_snapshot_all_accounts

        update_today_snapshot_all_accounts()
    except Exception as e:
        from utils.logger import get_app_logger

        get_app_logger().warning(f"Failed to update snapshot after deletion: {e}")

    return {"deleted": ticker}


def update_holding(
    account_id: str,
    ticker: str,
    quantity: int | None = None,
    average_buy_price: float | None = None,
    memo: str | None = None,
    target_ratio: float | None = None,
) -> dict[str, str]:
    """계좌의 특정 종목 수량/매입단가를 수정한다."""
    account_id = str(account_id or "").strip()
    ticker = str(ticker or "").strip()
    if not account_id or not ticker:
        raise RuntimeError("계좌 ID와 종목코드가 필요합니다.")

    # 저장된 티커와 요청 티커를 같은 규칙으로 정규화해 비교한다(ASX: 접두사 유무 무관).
    target_ticker = _normalize_target_ticker(ticker)

    master = load_portfolio_master(account_id)
    if not master:
        raise RuntimeError("계좌 데이터를 찾을 수 없습니다.")

    holdings = master.get("holdings", [])
    found = False
    for h in holdings:
        if _normalize_target_ticker(str(h.get("ticker", ""))) == target_ticker:
            if quantity is not None:
                h["quantity"] = int(quantity)
            if average_buy_price is not None:
                h["average_buy_price"] = float(average_buy_price)
            if memo is not None:
                h["memo"] = str(memo).strip()
            if target_ratio is not None:
                _set_holding_target_ratio(h, float(target_ratio))
            found = True
            break

    if not found:
        raise RuntimeError(f"종목 {ticker}을 찾을 수 없습니다.")

    save_portfolio_master(account_id, _assign_sort_order(holdings))

    # 변경 사항을 스냅샷에 즉시 동기화
    try:
        from utils.snapshot_service import update_today_snapshot_all_accounts

        update_today_snapshot_all_accounts()
    except Exception as e:
        from utils.logger import get_app_logger

        get_app_logger().warning(f"Failed to update snapshot after update: {e}")

    return {"updated": ticker}


def add_holding(
    account_id: str,
    ticker: str,
    quantity: int,
    average_buy_price: float,
    memo: str | None = None,
    target_ratio: float | None = None,
) -> dict[str, Any]:
    """계좌에 새로운 종목을 추가한다."""
    account_id = str(account_id or "").strip()
    ticker = str(ticker or "").strip()

    if not account_id or not ticker:
        raise RuntimeError("계좌 ID와 종목코드가 필요합니다.")

    # 1. 티커 검증 및 상세 메타데이터 가져오기 (종목명, 버킷 등)
    res = validate_ticker_for_account(account_id, ticker)
    raw_ticker = res["ticker"]

    # 2. 통화는 계좌가 아니라 "종목 실제 통화" 로 각인한다(다통화 계좌 지원).
    #    미국 티커는 USD, 호주는 AUD, 국내는 KRW. 계좌 통화에 종속되지 않는다.
    from utils.cash_model import currency_for_country

    currency = currency_for_country(res.get("country_code"))

    # 원장(portfolio_master) 항목이 아직 없으면(신규 계좌·자산 미설정) 빈 목록에서 시작한다.
    # save_portfolio_master 가 저장 시 계좌 항목을 새로 만들어준다.
    master = load_portfolio_master(account_id)
    holdings = master.get("holdings", []) if master else []

    # 중복 확인 — ASX: 접두사 유무가 달라도 같은 종목으로 판정한다.
    for h in holdings:
        if _normalize_target_ticker(str(h.get("ticker", ""))) == _normalize_target_ticker(raw_ticker):
            raise RuntimeError(f"종목 {ticker}은 이미 등록되어 있습니다.")

    # 3. 정석적인 구조로 새로운 종목 구성
    next_sort_order = max((int(h.get("sort_order") or 0) for h in holdings), default=-1) + 1
    new_holding = {
        "ticker": raw_ticker,
        "name": res["name"],
        "quantity": int(quantity),
        "average_buy_price": float(average_buy_price),
        "currency": currency,
        "first_buy_date": datetime.now().strftime("%Y-%m-%d"),
        "last_buy_date": datetime.now().strftime("%Y-%m-%d"),
        "memo": str(memo or "").strip(),
        "sort_order": next_sort_order,
    }

    if target_ratio is not None:
        _set_holding_target_ratio(new_holding, float(target_ratio))
    holdings.append(new_holding)
    save_portfolio_master(account_id, _assign_sort_order(holdings))

    # 변경 사항을 스냅샷에 즉시 동기화
    try:
        from utils.snapshot_service import update_today_snapshot_all_accounts

        update_today_snapshot_all_accounts()
    except Exception as e:
        from utils.logger import get_app_logger

        get_app_logger().warning(f"Failed to update snapshot after addition: {e}")

    return {"added": ticker, "name": res["name"]}


def reorder_holdings(account_id: str, ordered_tickers: list[str]) -> dict[str, Any]:
    """계좌 보유 종목의 사용자 지정 순서를 저장한다."""
    normalized_account_id = str(account_id or "").strip()
    if not normalized_account_id:
        raise RuntimeError("계좌 ID가 필요합니다.")

    normalized_tickers = [_normalize_target_ticker(ticker) for ticker in ordered_tickers]
    normalized_tickers = [ticker for ticker in normalized_tickers if ticker]
    if not normalized_tickers:
        raise RuntimeError("정렬할 종목코드가 필요합니다.")

    # IS(가짜 보유 행)는 실보유가 아니므로 목록에서 분리하고, 위치만 계좌 필드로 저장한다.
    intl_shares_sort_order = None
    if "IS" in normalized_tickers:
        intl_shares_sort_order = normalized_tickers.index("IS")
        normalized_tickers = [ticker for ticker in normalized_tickers if ticker != "IS"]
    if not normalized_tickers:
        raise RuntimeError("정렬할 종목코드가 필요합니다.")

    master = load_portfolio_master(normalized_account_id)
    if not master:
        raise RuntimeError("계좌 데이터를 찾을 수 없습니다.")

    holdings = [dict(holding) for holding in (master.get("holdings") or [])]
    if not holdings:
        raise RuntimeError("정렬할 종목이 없습니다.")

    holding_map = {_normalize_target_ticker(str(holding.get("ticker") or "")): holding for holding in holdings}
    missing_tickers = [ticker for ticker in normalized_tickers if ticker not in holding_map]
    if missing_tickers:
        joined = ", ".join(missing_tickers)
        raise RuntimeError(f"순서를 저장할 종목을 찾을 수 없습니다: {joined}")

    ordered_holdings: list[dict[str, Any]] = []
    seen: set[str] = set()
    for ticker in normalized_tickers:
        if ticker in seen:
            continue
        ordered_holdings.append(holding_map[ticker])
        seen.add(ticker)

    for holding in holdings:
        ticker = _normalize_target_ticker(str(holding.get("ticker") or ""))
        if ticker not in seen:
            ordered_holdings.append(holding)

    save_portfolio_master(
        normalized_account_id,
        _assign_sort_order(ordered_holdings),
        intl_shares_sort_order=intl_shares_sort_order,
    )

    try:
        from utils.snapshot_service import update_today_snapshot_all_accounts

        update_today_snapshot_all_accounts()
    except Exception as e:
        from utils.logger import get_app_logger

        get_app_logger().warning(f"Failed to update snapshot after reorder: {e}")

    return {"reordered": len(ordered_holdings)}


def validate_ticker_for_account(account_id: str, ticker: str) -> dict[str, Any]:
    """계좌에 추가할 수 있는 유효한 티커인지 검증한다."""
    account_id = str(account_id or "").strip()
    ticker = str(ticker or "").strip().upper()

    if not account_id or not ticker:
        raise RuntimeError("계좌 ID와 종목코드가 필요합니다.")

    # 시장 접두어(ASX:/US:/KOR:)로 시장을 명시할 수 있다(같은 티커가 여러 시장에 있을 때 구분용).
    forced_country: str | None = None
    if ticker.startswith("ASX:"):
        forced_country, raw_ticker = "au", ticker[len("ASX:") :].strip().upper()
    elif ticker.startswith("US:"):
        forced_country, raw_ticker = "us", ticker[len("US:") :].strip().upper()
    elif ticker.startswith("KOR:"):
        forced_country, raw_ticker = "kor", ticker[len("KOR:") :].strip().upper()
    else:
        raw_ticker = ticker.strip().upper()
    if not raw_ticker:
        raise RuntimeError("유효한 티커를 입력하세요.")

    from utils.settings_loader import get_account_settings
    from utils.stocks_service import validate_stock_candidate

    # 1. 계좌 설정 로드 (DB account_settings 읽기)
    try:
        settings = get_account_settings(account_id)
        # account_settings["settings"]가 아닌 top-level에 있는 경우가 많음
        inner_settings = settings.get("settings") or settings
    except Exception as e:
        raise RuntimeError(f"계좌 설정을 찾을 수 없습니다: {account_id} ({e})")

    # 2. 전체 종목풀을 대상으로 종목 추가 가능 여부를 검사한다.
    from utils.settings_loader import list_available_ticker_types

    ticker_types = list_available_ticker_types()
    if not ticker_types:
        raise RuntimeError("사용 가능한 종목풀이 없습니다.")

    # 3. 종목풀(stock_meta)에 이미 등록된 종목만 계좌에 담을 수 있다.
    #    미등록 종목(status="new": fetch 는 되지만 stock_meta 부재)은 여기서 막고,
    #    최초 등록 창구인 '종목 순위(pools-rank)' 로 안내한다. (pools-rank 는 이 함수를 거치지 않는다.)
    last_error = None
    saw_unregistered = False
    # 시장(country_code)별 최초 active 후보. 같은 시장의 여러 종목풀 중복은 하나로 취급.
    candidates_by_country: dict[str, dict[str, Any]] = {}

    for tt in ticker_types:
        try:
            # StocksManager가 사용하는 동일한 함수 호출
            candidate = validate_stock_candidate(tt, raw_ticker)
        except Exception as e:
            last_error = str(e)
            continue
        if candidate.get("status") == "active":
            cc = str(candidate.get("country_code") or "").strip().lower()
            candidates_by_country.setdefault(cc, candidate)
        else:
            saw_unregistered = True

    if not candidates_by_country:
        if saw_unregistered:
            raise RuntimeError(
                f"종목풀에 등록되지 않은 종목입니다: {raw_ticker}. "
                "'종목 순위' 화면에서 먼저 종목을 추가한 뒤 계좌에 담아주세요."
            )
        raise RuntimeError(last_error or f"등록되지 않은 종목입니다: {raw_ticker}")

    # 시장 결정: 접두어 지정 > 단일 시장 > 계좌 현금 통화로 후보 필터(A).
    if forced_country is not None:
        if forced_country not in candidates_by_country:
            raise RuntimeError(f"'{raw_ticker}' 는 지정한 시장({forced_country})에 등록돼 있지 않습니다.")
        validated_res = candidates_by_country[forced_country]
    elif len(candidates_by_country) == 1:
        validated_res = next(iter(candidates_by_country.values()))
    else:
        # 같은 티커가 여러 시장에 존재 — 계좌가 보유하는 현금 통화로 후보를 좁힌다.
        from utils.cash_model import currency_for_country, resolve_cash_currencies

        account_currencies = set(resolve_cash_currencies(inner_settings))
        matched = {
            cc: cand for cc, cand in candidates_by_country.items() if currency_for_country(cc) in account_currencies
        }
        if len(matched) == 1:
            validated_res = next(iter(matched.values()))
        else:
            markets = " / ".join(sorted(f"{currency_for_country(cc)}({cc})" for cc in candidates_by_country))
            raise RuntimeError(
                f"'{raw_ticker}' 는 여러 시장에 등록돼 있습니다: {markets}. "
                f"'US:{raw_ticker}' 또는 'ASX:{raw_ticker}' 처럼 시장을 지정해 주세요."
            )

    return {
        "ticker": validated_res["ticker"],
        "name": validated_res["name"],
        "bucket_id": validated_res.get("bucket_id") or 1,
        "country_code": str(validated_res.get("country_code") or "").strip().lower(),
        "status": "success",
    }
