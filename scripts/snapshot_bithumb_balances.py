"""
Snapshot today's coin total equity from live Bithumb balances (no trades needed).

Computes total_equity = KRW_balance + sum(total_{COIN} * last_close_price_KRW)
and saves it into `daily_equities` for country='coin' and today's date.

Usage:
  python scripts/snapshot_bithumb_balances.py
"""

import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import fetch_ohlcv
from utils.db_manager import save_daily_equity
from utils.env import load_env_if_present
from utils.stock_list_io import get_etfs


def _fetch_bithumb_balance_dict():
    try:
        from utils.exchanges.bithumb_v2 import BithumbV2Client

        v2 = BithumbV2Client()
        # v2.accounts()는 각 통화별 잔액 정보가 담긴 딕셔너리의 리스트를 반환합니다.
        # 예: [{"currency": "KRW", "balance": "1000", "locked": "0"}, {"currency": "BTC", ...}]
        balance_items = v2.accounts()
    except Exception as e:
        print(f"[ERROR] Bithumb v2 accounts failed: {e}")
        return None

    if not isinstance(balance_items, list):
        print(f"[ERROR] Bithumb balance data is not a list: {type(balance_items)}")
        return None

    out = {}

    def _pf(x) -> float:
        try:
            return float(str(x).replace(",", ""))
        except (ValueError, TypeError):
            return 0.0

    for item in balance_items:
        currency = str(item.get("currency") or "").upper()
        if not currency:
            continue

        # Bithumb v2 API는 'balance'(사용 가능)와 'locked'(사용 중) 필드를 반환합니다.
        # 총 잔액은 이 두 값의 합입니다.
        available_balance = _pf(item.get("balance"))
        locked_balance = _pf(item.get("locked"))
        total_balance = available_balance + locked_balance
        if currency == "KRW":
            out["total_krw"] = total_balance
        else:
            out[f"total_{currency}"] = total_balance

    return out


def _get_total_amount_for(symbol: str, bal: dict) -> float:
    if not isinstance(bal, dict):
        return 0.0
    for key in (f"total_{symbol.lower()}", f"total_{symbol.upper()}"):
        if key in bal:
            try:
                return float(str(bal[key]).replace(",", ""))
            except Exception:
                return 0.0
    return 0.0


def main():
    load_env_if_present()
    # Fetch balances first
    bal = _fetch_bithumb_balance_dict()
    if not bal:
        print("[ERROR] Could not fetch Bithumb balances.")
        return

    # Resolve coins universe = DB union accounts
    coin_etfs = get_etfs("coin") or []
    db_coins = {str(s.get("ticker") or "").upper() for s in coin_etfs if s.get("ticker")}
    acct_coins = set()
    for k in list(bal.keys()):
        if isinstance(k, str) and k.lower().startswith("total_"):
            sym = k.split("_", 1)[-1].upper()
            # KRW와 P는 총 평가금액 계산 시 특별 처리되므로, 여기서는 제외합니다.
            if sym not in ["KRW", "P"]:
                acct_coins.add(sym)
    coins = sorted(db_coins.union(acct_coins))
    if not coins:
        print("[WARN] No coins found in DB or accounts; nothing to snapshot.")
        return

    # 총 평가금액 초기화: KRW 잔액과 P 잔액을 합산합니다.
    # 사용자의 요청에 따라 'P'는 원화(KRW)처럼 취급합니다.
    krw_balance = bal.get("total_krw", 0.0)
    p_balance = bal.get("total_P", 0.0)
    total_value = krw_balance + p_balance

    # Price coins using latest 24h close
    for c in coins:
        qty = _get_total_amount_for(c, bal)
        if qty <= 0:
            continue
        price = 0.0
        df = fetch_ohlcv(c, country="coin")
        if df is not None and not df.empty:
            try:
                price = float(df["Close"].iloc[-1])
            except Exception:
                price = 0.0
        total_value += qty * price

    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    ok = save_daily_equity("coin", today, total_value)
    if ok:
        print(
            f"[OK] Saved coin daily equity for {today.strftime('%Y-%m-%d')}: {int(total_value):,} KRW"
        )
    else:
        print("[ERROR] Failed to save daily equity")


if __name__ == "__main__":
    main()
