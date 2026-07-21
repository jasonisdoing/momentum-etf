from __future__ import annotations

from typing import Any

from services.price_service import get_exchange_rates
from utils.account_registry import load_account_configs
from utils.db_manager import get_db_connection
from utils.normalization import normalize_number


def load_snapshot_list() -> list[dict[str, Any]]:
    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결에 실패했습니다.")

    docs = list(db.daily_snapshots.find().sort("snapshot_date", -1))
    configs = load_account_configs()
    account_map = {config["account_id"]: {"name": config["name"], "order": config["order"]} for config in configs}

    snapshots: list[dict[str, Any]] = []
    for doc in docs:
        accounts = sorted(
            [
                {
                    "account_id": str(account.get("account_id") or ""),
                    "account_name": account_map.get(str(account.get("account_id") or ""), {}).get(
                        "name",
                        str(account.get("account_id") or ""),
                    ),
                    "order": int(
                        account_map.get(str(account.get("account_id") or ""), {}).get("order", 999),
                    ),
                    "total_assets": normalize_number(account.get("total_assets")),
                    "total_principal": normalize_number(account.get("total_principal")),
                    "cash_balance": normalize_number(account.get("cash_balance")),
                    "cash_balance_native": normalize_number(account.get("cash_balance_native")),
                    "cash_currency": str(account.get("cash_currency") or "").strip().upper(),
                    "valuation_krw": normalize_number(account.get("valuation_krw")),
                }
                for account in (doc.get("accounts") or [])
                if isinstance(account, dict)
            ],
            key=lambda item: item["order"],
        )

        snapshots.append(
            {
                "id": str(doc.get("_id")),
                "snapshot_date": str(doc.get("snapshot_date") or ""),
                "total_assets": normalize_number(doc.get("total_assets")),
                "total_principal": normalize_number(doc.get("total_principal")),
                "cash_balance": normalize_number(doc.get("cash_balance")),
                "valuation_krw": normalize_number(doc.get("valuation_krw")),
                "account_count": len(accounts),
                "accounts": accounts,
            }
        )

    return snapshots


def update_today_snapshot_all_accounts() -> dict[str, Any]:
    """
    모든 계좌 정보를 실시간으로 계산하여 오늘 자 통합 스냅샷을 갱신한다.
    (자산 관리 페이지에서 데이터 변경 시 호출된다.)
    """
    from utils.portfolio_io import load_portfolio_master, load_real_holdings_table, save_daily_snapshot

    configs = load_account_configs()
    exchange_rates = get_exchange_rates()
    account_summaries = []
    global_principal = 0.0
    global_cash = 0.0
    global_valuation = 0.0
    global_purchase = 0.0

    for config in configs:
        aid = config["account_id"]
        # 원금 및 현금 로드
        m_data = load_portfolio_master(aid)
        acc_principal = normalize_number(m_data.get("total_principal")) if m_data else 0.0
        acc_cash = normalize_number(m_data.get("cash_balance")) if m_data else 0.0
        cash_balance_native = normalize_number(m_data.get("cash_balance_native")) if m_data else 0.0
        cash_currency = str(m_data.get("cash_currency") or "").strip().upper() if m_data else ""

        if m_data:
            if cash_currency and cash_currency != "KRW" and acc_cash <= 0:
                native_cash = cash_balance_native
                if native_cash > 0:
                    rate_info = (exchange_rates or {}).get(cash_currency)
                    rate = normalize_number((rate_info or {}).get("rate"))
                    if rate <= 0:
                        raise RuntimeError(f"{cash_currency} 환율을 가져오지 못했습니다.")
                    acc_cash = native_cash * rate

        global_principal += acc_principal
        global_cash += acc_cash

        # 실시간 보유 현황 및 평가액 로드
        try:
            df = load_real_holdings_table(aid)
            if df is not None and not df.empty:
                acc_valuation = float(df["평가금액(KRW)"].sum())
                acc_purchase = float(df["매입금액(KRW)"].sum())
                
                # 보유 종목 상세 (티커만; 보유일 컬럼은 더 이상 사용하지 않음)
                holding_details = []
                for _, row in df.iterrows():
                    ticker = str(row.get("ticker", row.get("티커", ""))).strip().upper()
                    if ticker and ticker != "NAN":
                        holding_details.append({"ticker": ticker})
            else:
                acc_valuation = 0.0
                acc_purchase = 0.0
                holding_details = []
        except Exception as e:
            from utils.logger import get_app_logger
            get_app_logger().error(f"Failed to calculate valuation for account {aid}: {e}")
            acc_valuation = 0.0
            acc_purchase = 0.0
            holding_details = []

        acc_total_assets = acc_valuation + acc_cash
        global_valuation += acc_valuation
        global_purchase += acc_purchase

        # intl_shares_value 로드 (호주 계좌)
        intl_shares_value = None
        if aid == "aus_account" and m_data:
            intl_shares_value = normalize_number(m_data.get("intl_shares_value"))

        # 계좌별 개별 스냅샷 저장 데이터 구성
        account_summaries.append({
            "account_id": aid,
            "total_assets": acc_total_assets,
            "total_principal": acc_principal,
            "cash_balance": acc_cash,
            "cash_balance_native": cash_balance_native,
            "cash_currency": cash_currency,
            "valuation_krw": acc_valuation,
            "purchase_amount": acc_purchase,
            "holding_details": holding_details,
            "intl_shares_value": intl_shares_value,
        })

        # DB에 개별 계좌 데이터로 먼저 저장 (save_daily_snapshot 호출)
        save_daily_snapshot(
            aid,
            acc_total_assets,
            acc_principal,
            acc_cash,
            acc_valuation,
            purchase_amount=acc_purchase,
            holding_details=holding_details,
            cash_balance_native=cash_balance_native if cash_currency and cash_currency != "KRW" else None,
            cash_currency=cash_currency if cash_currency and cash_currency != "KRW" else None,
            intl_shares_value=intl_shares_value,
        )

    # 전체 통합 데이터 저장
    total_assets = global_valuation + global_cash
    save_daily_snapshot(
        "TOTAL",
        total_assets,
        global_principal,
        global_cash,
        global_valuation,
        purchase_amount=global_purchase
    )

    return {
        "total_assets": total_assets,
        "account_count": len(account_summaries)
    }


# 계좌별 최근 거래일 기간 수익률 창.
# 1주/2주/3주/1달/3달/6달/1년 = 5/10/15/21/63/126/252 거래일.
ACCOUNT_RETURN_TRADING_DAYS = [5, 10, 15, 21, 63, 126, 252]


def compute_account_period_returns() -> dict[str, dict[str, Any]]:
    """계좌별 최근 거래일 기간 수익률(입출금 제거)을 반환한다.

    손익 = 총자산(total_assets, 보유평가 + 현금) − 원금(total_principal)로 입출금 영향을 제거하고,
    기간 수익률 = (손익_현재 − 손익_N거래일전) / N거래일전 총자산 × 100 으로 계산한다
    (일별 펀드 화면과 동일한 calculate_period_return_pct 공식 재사용).
    ※ valuation_krw(보유 평가금액)는 현금을 빼먹어 총자산과 불일치하므로 쓰지 않는다.
    스냅샷 이력이 부족한 기간은 None.
    반환: {account_id: {"d5": float|None, "d10": ..., "d15": ..., "d21": ...}}
    """
    from utils.daily_fund_service import calculate_period_return_pct

    snapshots = load_snapshot_list()  # 최신 날짜순(desc)
    # account_id -> [(profit, total_assets), ...] 최신순. 값 누락일은 제외(임의 보정 금지).
    series: dict[str, list[tuple[float, float]]] = {}
    for snap in snapshots:
        for account in snap.get("accounts", []):
            account_id = str(account.get("account_id") or "")
            if not account_id:
                continue
            principal = account.get("total_principal")
            total_assets = account.get("total_assets")
            if principal is None or total_assets is None:
                continue
            series.setdefault(account_id, []).append((float(total_assets) - float(principal), float(total_assets)))

    result: dict[str, dict[str, Any]] = {}
    for account_id, points in series.items():
        latest_profit = points[0][0]
        returns: dict[str, Any] = {}
        for days in ACCOUNT_RETURN_TRADING_DAYS:
            if len(points) > days:
                prev_profit, prev_total_assets = points[days]
                returns[f"d{days}"] = round(
                    calculate_period_return_pct(latest_profit - prev_profit, prev_total_assets), 2
                )
            else:
                returns[f"d{days}"] = None
        # 임의 N거래일 수익률(예: 시장 레짐 지속일수)을 프론트가 계산할 수 있게 손익 시계열도 준다.
        # [profit, total_assets] 최신순.
        returns["series"] = [[profit, total_assets] for profit, total_assets in points]
        result[account_id] = returns
    return result
