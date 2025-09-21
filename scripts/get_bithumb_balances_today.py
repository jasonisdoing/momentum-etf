"""
Print today's Bithumb balances (KRW and per-coin totals), with optional KRW valuation and saving.

Usage:
  # Just print balances and KRW valuation
  python scripts/get_bithumb_balances_today.py

  # Include coins with zero balances
  python scripts/get_bithumb_balances_today.py --include-zero

  # Save total equity (KRW) into daily_equities after printing
  python scripts/get_bithumb_balances_today.py --save
"""

import argparse
import os
import sys
from datetime import datetime
from typing import Optional

import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.snapshot_bithumb_balances import (
    _fetch_bithumb_balance_dict as fetch_bithumb_balance_dict,
)
from utils.data_loader import fetch_ohlcv
from utils.db_manager import save_daily_equity
from utils.env import load_env_if_present
from utils.stock_list_io import get_etfs


def get_total_key(symbol: str) -> str:
    return f"total_{symbol.lower()}"


def to_float_safe(x) -> float:
    try:
        return float(str(x).replace(",", ""))
    except Exception:
        return 0.0


def _fetch_bithumb_realtime_price(symbol: str) -> Optional[float]:
    symbol = symbol.upper()
    if symbol in {"KRW", "P"}:
        return 1.0
    url = f"https://api.bithumb.com/public/ticker/{symbol}_KRW"
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and data.get("status") == "0000":
            closing_price = data.get("data", {}).get("closing_price")
            if closing_price is not None:
                return float(str(closing_price).replace(",", ""))
    except Exception:
        return None
    return None


def parse_args():
    p = argparse.ArgumentParser(description="Show today's Bithumb balances and valuation")
    p.add_argument("--include-zero", action="store_true", help="Include coins with zero balances")
    p.add_argument(
        "--save", action="store_true", help="Save total equity (KRW) into daily_equities for today"
    )
    return p.parse_args()


def main():
    args = parse_args()
    load_env_if_present()

    etfs = get_etfs("coin") or []
    db_coins = {str(s.get("ticker") or "").upper() for s in etfs if s.get("ticker")}

    bal = fetch_bithumb_balance_dict()
    if not isinstance(bal, dict):
        print("[ERROR] Could not fetch Bithumb balances")
        return

    acct_coins = set()
    for k in list(bal.keys()):
        if isinstance(k, str) and k.lower().startswith("total_"):
            sym = k.split("_", 1)[-1].upper()
            if sym not in ["KRW", "P"]:
                acct_coins.add(sym)

    coins = sorted(db_coins.union(acct_coins))
    if not coins:
        print("[WARN] No coin tickers found in etf.json or balances")

    krw_balance = to_float_safe(bal.get("total_krw", 0.0))
    p_balance = to_float_safe(bal.get("total_P", 0.0))

    print("Balances (as of today):\n")
    print(f"{'COIN':<8}{'QTY':>16}{'PRICE(KRW)':>16}{'VALUE(KRW)':>16}")

    grand_total_value = krw_balance + p_balance
    for c in coins:
        if c in ["KRW", "P"]:
            continue
        qty = to_float_safe(bal.get(f"total_{c.upper()}"))
        if not args.include_zero and qty <= 1e-9:
            continue

        price = _fetch_bithumb_realtime_price(c) or 0.0
        if price <= 0:
            df = fetch_ohlcv(c, country="coin")
            if df is not None and not df.empty:
                try:
                    price = float(df["Close"].iloc[-1])
                except Exception:
                    price = 0.0

        value = qty * price
        grand_total_value += value
        print(f"{c:<8}{qty:>16.8f}{price:>16,.0f}{value:>16,.0f}")

    print(f"\n{'KRW':<8}{'':>16}{'':>16}{krw_balance:>16,.0f}")
    if p_balance > 0 or (args.include_zero and "P" in db_coins.union(acct_coins)):
        print(f"{'P':<8}{p_balance:>16.8f}{1:>16,.0f}{p_balance:>16,.0f}")
    print(f"{'TOTAL':<8}{'':>16}{'':>16}{grand_total_value:>16,.0f}")

    if args.save:
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        if save_daily_equity("coin", "b1", today, grand_total_value):
            print(
                f"\n[OK] Saved daily_equities for {today.strftime('%Y-%m-%d')}: {int(grand_total_value):,} KRW"
            )
        else:
            print("\n[ERROR] Failed to save daily_equities")


if __name__ == "__main__":
    main()
