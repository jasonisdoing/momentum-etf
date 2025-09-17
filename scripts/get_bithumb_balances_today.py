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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.snapshot_bithumb_balances import (
    _fetch_bithumb_balance_dict as fetch_bithumb_balance_dict,
)
from utils.data_loader import fetch_ohlcv
from utils.db_manager import save_daily_equity
from utils.env import load_env_if_present
from utils.stock_list_io import get_etfs


def get_total_key(symbol: str):
    return f"total_{symbol.lower()}"


def to_float_safe(x):
    try:
        return float(str(x).replace(",", ""))
    except Exception:
        return 0.0


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

    # Coins universe from DB etfs
    etfs = get_etfs("coin") or []
    coins = sorted({str(s.get("ticker") or "").upper() for s in etfs if s.get("ticker")})
    if not coins:
        print("[WARN] No coin tickers found in etf.json")

    bal = fetch_bithumb_balance_dict()
    if not isinstance(bal, dict):
        print("[ERROR] Could not fetch Bithumb balances")
        return

    total_balance = bal.get("total_balance", 0.0)
    print("Balances (as of today):\n")
    print(f"{ 'COIN':<8}{'QTY':>16}{'PRICE(KRW)':>16}{'VALUE(KRW)':>16}")

    grand_total_value = total_balance
    for c in coins:
        qty = 0.0
        # Try lower and upper total key variants
        qty = to_float_safe(bal.get(get_total_key(c))) or to_float_safe(
            bal.get(f"total_{c.upper()}")
        )
        if not args.include_zero and qty <= 0:
            continue
        price = 0.0
        df = fetch_ohlcv(c, country="coin")
        if df is not None and not df.empty:
            try:
                price = float(df["Close"].iloc[-1])
            except Exception:
                price = 0.0
        value = qty * price
        grand_total_value += value
        print(f"{c:<8}{qty:>16.8f}{price:>16,.0f}{value:>16,.0f}")

    print(f"\n{'KRW':<8}{'':>16}{'':>16}{total_balance:>16,.0f}")
    print(f"{ 'TOTAL':<8}{'':>16}{'':>16}{grand_total_value:>16,.0f}")

    if args.save:
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        if save_daily_equity("coin", today, grand_total_value):
            print(
                f"\n[OK] Saved daily_equities for {today.strftime('%Y-%m-%d')}: {int(grand_total_value):,} KRW"
            )
        else:
            print("\n[ERROR] Failed to save daily_equities")


if __name__ == "__main__":
    main()
