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

import os
import sys
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.env import load_env_if_present
from utils.db_manager import get_stocks, save_daily_equity
from utils.data_loader import fetch_ohlcv


def fetch_bithumb_balance_dict():
    # Use v2 accounts; normalize and include balance + locked
    try:
        from utils.exchanges.bithumb_v2 import BithumbV2Client
        v2 = BithumbV2Client()
        items = v2.accounts()
    except Exception as e:
        print(f"[ERROR] Bithumb v2 client/accounts failed: {e}")
        return None
    out = {}
    def _pf(x):
        try:
            return float(str(x).replace(',', ''))
        except Exception:
            return 0.0
    for it in items or []:
        cur = str(it.get('currency') or '').upper()
        if not cur:
            continue
        bal = _pf(it.get('balance'))
        locked = _pf(it.get('locked'))
        total_field = _pf(it.get('total_balance')) or _pf(it.get('quantity'))
        total = bal + locked if (bal or locked) else total_field
        if cur == 'KRW':
            out['total_krw'] = total
        else:
            out[f'total_{cur}'] = total
    return out


def get_total_key(symbol: str):
    return f"total_{symbol.lower()}"


def to_float_safe(x):
    try:
        return float(str(x).replace(',', ''))
    except Exception:
        return 0.0


def parse_args():
    p = argparse.ArgumentParser(description="Show today's Bithumb balances and valuation")
    p.add_argument("--include-zero", action="store_true", help="Include coins with zero balances")
    p.add_argument("--save", action="store_true", help="Save total equity (KRW) into daily_equities for today")
    return p.parse_args()


def main():
    args = parse_args()
    load_env_if_present()

    # Coins universe from DB stocks
    stocks = get_stocks('coin') or []
    coins = sorted({str(s.get('ticker') or '').upper() for s in stocks if s.get('ticker')})
    if not coins:
        print("[WARN] No coin tickers found in DB stocks")

    bal = fetch_bithumb_balance_dict()
    if not isinstance(bal, dict):
        print("[ERROR] Could not fetch Bithumb balances")
        return

    total_krw = 0.0
    for key in ("total_krw", "total_KRW"):
        if key in bal:
            total_krw = to_float_safe(bal[key])
            break
    if total_krw == 0.0:
        # Sum available + in_use if total_krw not provided
        total_krw = to_float_safe(bal.get('available_krw')) + to_float_safe(bal.get('in_use_krw'))

    print("Balances (as of today):\n")
    print(f"{'COIN':<8}{'QTY':>16}{'PRICE(KRW)':>16}{'VALUE(KRW)':>16}")

    grand_total_value = total_krw
    for c in coins:
        qty = 0.0
        # Try lower and upper total key variants
        qty = to_float_safe(bal.get(get_total_key(c))) or to_float_safe(bal.get(f"total_{c.upper()}"))
        if not args.include_zero and qty <= 0:
            continue
        price = 0.0
        df = fetch_ohlcv(c, country='coin')
        if df is not None and not df.empty:
            try:
                price = float(df['Close'].iloc[-1])
            except Exception:
                price = 0.0
        value = qty * price
        grand_total_value += value
        print(f"{c:<8}{qty:>16.8f}{price:>16,.0f}{value:>16,.0f}")

    print(f"\n{'KRW':<8}{'':>16}{'':>16}{total_krw:>16,.0f}")
    print(f"{'TOTAL':<8}{'':>16}{'':>16}{grand_total_value:>16,.0f}")

    if args.save:
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        if save_daily_equity('coin', today, grand_total_value):
            print(f"\n[OK] Saved daily_equities for {today.strftime('%Y-%m-%d')}: {int(grand_total_value):,} KRW")
        else:
            print("\n[ERROR] Failed to save daily_equities")


if __name__ == '__main__':
    main()
