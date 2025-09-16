"""
Sync Bithumb /accounts snapshot to DB trades for country='coin'.

Rules (KOR):
- 첫날: /accounts 결과를 기준으로 각 코인별 BUY 트레이드(수량=총보유, 가격=평단) 생성
- 이후: 직전 스냅샷과 비교하여 수량이 늘면 BUY, 줄면 SELL 트레이드 생성
  - BUY 가격은 평균단가 변화로 역산되면 그 값을 사용, 불가하면 현재/평단으로 대체
  - SELL 가격은 보유 재구성에는 영향 없으므로 현재가 또는 평단으로 기록
- 매 실행 시 최신 스냅샷을 저장하여 다음 비교 기준으로 사용

Usage:
  python scripts/sync_bithumb_accounts_to_trades.py
"""

from __future__ import annotations

import math
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Add project root to Python path for `utils` imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import fetch_ohlcv
from utils.db_manager import get_db_connection, save_trade
from utils.env import load_env_if_present
from utils.stock_list_io import get_etfs


def _now_day() -> datetime:
    """Normalized date (00:00) for snapshot day grouping."""
    return pd.Timestamp.now().normalize().to_pydatetime()


def _now_time() -> datetime:
    """Exact run timestamp for trade event time."""
    return datetime.now()


def _fetch_accounts() -> List[Dict]:
    """Fetch raw Bithumb v2 accounts list."""
    from utils.exchanges.bithumb_v2 import BithumbV2Client

    v2 = BithumbV2Client()
    return v2.accounts() or []


def _normalize_accounts(items: List[Dict]) -> Tuple[float, Dict[str, Dict[str, float]]]:
    """Normalize raw accounts into (krw_balance, per-coin mapping).

    Quantity rule: qty = balance + locked when available; otherwise fall back to
    provided total fields (e.g., total_balance, quantity, available, total).

    Returns:
      krw: float
      coins: { TICKER: {qty: float, avg: float|0.0} }
    """
    krw = 0.0
    coins: Dict[str, Dict[str, float]] = {}
    for it in items:
        cur = str(it.get("currency") or "").upper()
        if not cur:
            continue

        def _pf(x) -> float:
            try:
                return float(str(x).replace(",", ""))
            except Exception:
                return 0.0

        bal = _pf(it.get("balance"))
        locked = _pf(it.get("locked"))
        total_field = (
            _pf(it.get("total_balance"))
            or _pf(it.get("quantity"))
            or _pf(it.get("available"))
            or _pf(it.get("total"))
        )
        qty = bal + locked if (bal or locked) else total_field

        if cur == "KRW":
            krw += qty
            continue

        avg = 0.0
        for k in ("avg_buy_price", "avg_buy_price_krw", "avg_price"):
            if it.get(k) is not None:
                try:
                    avg = float(str(it.get(k)).replace(",", ""))
                except Exception:
                    avg = 0.0
                break
        if qty > 0 or avg > 0:
            coins[cur] = {"qty": qty, "avg": avg}
    return krw, coins


def _last_snapshot(db) -> Optional[Dict]:
    col = db.exchange_account_snapshots
    doc = col.find_one({"country": "coin", "source": "bithumb"}, sort=[("created_at", -1)])
    return doc


def _save_snapshot(db, krw: float, coins: Dict[str, Dict[str, float]]):
    now = datetime.now()
    doc = {
        "country": "coin",
        "source": "bithumb",
        "date": _now_day(),
        "krw": float(krw),
        "coins": {
            k: {"qty": float(v.get("qty", 0.0)), "avg": float(v.get("avg", 0.0))}
            for k, v in coins.items()
        },
        "created_at": now,
    }
    db.exchange_account_snapshots.insert_one(doc)


def _price_close_krw(tkr: str) -> float:
    df = fetch_ohlcv(tkr, country="coin")
    if df is None or df.empty:
        return 0.0
    # Use last close
    try:
        return float(df["Close"].iloc[-1])
    except Exception:
        return 0.0


def _infer_buy_price(
    avg_old: float, q_old: float, avg_new: float, q_new: float, delta: float
) -> Optional[float]:
    """Infer the buy unit price from average price change.
    p = (avg_new*q_new - avg_old*q_old) / delta
    """
    try:
        if delta <= 0:
            return None
        num = (avg_new * q_new) - (avg_old * q_old)
        p = num / delta
        if math.isfinite(p) and p > 0:
            return float(p)
    except Exception:
        pass
    return None


def _universe() -> Tuple[List[str], Dict[str, str]]:
    """Return (tickers, name_by_ticker) for coin universe from DB etfs."""
    etfs = get_etfs("coin") or []
    tickers = []
    names: Dict[str, str] = {}
    for s in etfs:
        t = str(s.get("ticker") or "").upper()
        if not t:
            continue
        tickers.append(t)
        names[t] = str(s.get("name") or "")
    return sorted(tickers), names


def _save_trade(
    ticker: str, action: str, shares: float, price: float, name: str, when: datetime
) -> bool:
    data = {
        "country": "coin",
        "ticker": ticker,
        "name": name,
        "date": when,
        "action": action,
        "shares": float(shares),
        "price": float(price),
        "fees": 0.0,
        "note": "auto-sync from Bithumb accounts",
        "account": "Bithumb",
    }
    return save_trade(data)


def _has_any_trades(db) -> bool:
    try:
        return db.trades.count_documents({"country": "coin", "is_deleted": {"$ne": True}}) > 0
    except Exception:
        return False


def main():
    # Load .env for API keys and DB connection
    try:
        load_env_if_present()
    except Exception:
        pass

    db = get_db_connection()
    if db is None:
        print("[ERROR] No DB connection; aborting")
        return

    # Fetch latest accounts and normalize
    items = _fetch_accounts()
    krw, coins_now = _normalize_accounts(items)

    # Limit to configured universe if available
    tickers, names = _universe()
    if tickers:
        coins_now = {k: v for k, v in coins_now.items() if k in tickers}

    snap_prev = _last_snapshot(db)
    today = _now_day()
    run_ts = _now_time()

    # Seed condition: no previous snapshot OR no existing trades in DB
    if not snap_prev or not _has_any_trades(db):
        # First day: seed BUY trades for all positive balances
        reason = "no previous snapshot" if not snap_prev else "no existing trades"
        print(f"[INFO] Initial seeding due to {reason}; creating BUY trades from accounts")
        for tkr, d in sorted(coins_now.items()):
            q = float(d.get("qty", 0.0))
            if q <= 0:
                continue
            avg = float(d.get("avg", 0.0)) or 0.0
            px = avg or _price_close_krw(tkr)
            if px <= 0:
                # last resort fallback
                px = avg if avg > 0 else 1.0
            _save_trade(tkr, "BUY", q, px, names.get(tkr, ""), run_ts)
        _save_snapshot(db, krw, coins_now)
        print("[OK] Initial trades seeded and snapshot saved.")
        return

    # Subsequent days: diff quantities to create BUY/SELL
    coins_prev: Dict[str, Dict[str, float]] = {
        k: {"qty": float(v.get("qty", 0.0)), "avg": float(v.get("avg", 0.0))}
        for k, v in (snap_prev.get("coins") or {}).items()
    }

    # if today's balances identical to last snapshot, nothing to do
    def _same(a: Dict[str, Dict[str, float]], b: Dict[str, Dict[str, float]]) -> bool:
        keys = set(a.keys()) | set(b.keys())
        for k in keys:
            aq = float((a.get(k) or {}).get("qty", 0.0))
            bq = float((b.get(k) or {}).get("qty", 0.0))
            # tolerate tiny float dust
            if abs(aq - bq) > 1e-10:
                return False
        return True

    if _same(coins_prev, coins_now):
        print("[INFO] Accounts unchanged since last snapshot; no trades created.")
        # still save a new snapshot if date changed to mark the day
        if pd.to_datetime(snap_prev.get("date")).normalize() != pd.to_datetime(today).normalize():
            _save_snapshot(db, krw, coins_now)
            print("[OK] Saved snapshot for today (no trades).")
        return

    # Create trades for each changed coin
    tickers_all = sorted(set(coins_prev.keys()) | set(coins_now.keys()))
    for tkr in tickers_all:
        prev = coins_prev.get(tkr, {"qty": 0.0, "avg": 0.0})
        cur = coins_now.get(tkr, {"qty": 0.0, "avg": 0.0})
        q0 = float(prev.get("qty", 0.0))
        q1 = float(cur.get("qty", 0.0))
        if abs(q1 - q0) <= 1e-10:
            continue
        delta = q1 - q0
        if delta > 0:
            # BUY
            avg0 = float(prev.get("avg", 0.0))
            avg1 = float(cur.get("avg", 0.0))
            px = _infer_buy_price(avg0, q0, avg1, q1, delta) or (avg1 or _price_close_krw(tkr))
            if px <= 0:
                px = avg1 if avg1 > 0 else 1.0
            _save_trade(tkr, "BUY", delta, px, names.get(tkr, ""), run_ts)
        else:
            # SELL
            px = _price_close_krw(tkr) or float(prev.get("avg", 0.0)) or 1.0
            _save_trade(tkr, "SELL", abs(delta), px, names.get(tkr, ""), run_ts)

    _save_snapshot(db, krw, coins_now)
    print("[OK] Trades synced from accounts diff and snapshot saved.")


if __name__ == "__main__":
    main()
