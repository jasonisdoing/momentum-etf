"""
Strategy: 'seykota'
Ed Seykota-style trend-following strategy using moving average crossovers.
"""
import pandas as pd
from typing import Optional, List, Tuple, Dict
from math import ceil

from . import settings as seykota_settings
from utils.data_loader import fetch_ohlcv, get_today_str


def run_portfolio_backtest(
    pairs: List[Tuple[str, str]],
    months_range: Optional[List[int]] = None,
    initial_capital: float = 100_000_000.0,
    core_start_date: Optional[pd.Timestamp] = None,
    top_n: int = 10,
    initial_positions: Optional[dict] = None,
    date_range: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Simulates a Top-N portfolio using a moving average crossover strategy.
    """
    # Settings
    fast_ma_period = int(getattr(seykota_settings, 'SEYKOTA_FAST_MA', 50))
    slow_ma_period = int(getattr(seykota_settings, 'SEYKOTA_SLOW_MA', 150))
    stop_loss = getattr(seykota_settings, 'SEYKOTA_STOP_LOSS_PCT', None)
    cooldown_days = int(getattr(seykota_settings, 'COOLDOWN_DAYS', 0))
    min_pos_pct = float(getattr(seykota_settings, 'MIN_POSITION_PCT', 0.10))
    max_pos_pct = float(getattr(seykota_settings, 'MAX_POSITION_PCT', 0.20))
    trim_on = bool(getattr(seykota_settings, 'ENABLE_MAX_POSITION_TRIM', True))

    # Fetch all series and precompute indicators
    data = {}
    for tkr, _ in pairs:
        df = fetch_ohlcv(tkr, months_range=months_range, date_range=date_range)
        if df is None or len(df) < slow_ma_period:
            continue
        
        close = df['Close']
        fast_ma = close.rolling(window=fast_ma_period).mean()
        slow_ma = close.rolling(window=slow_ma_period).mean()
        ma_score = (fast_ma / slow_ma - 1.0).fillna(0.0)

        data[tkr] = {
            'df': df, 'close': close, 'fast_ma': fast_ma, 'slow_ma': slow_ma,
            'ma_score': ma_score
        }

    if not data:
        return {}

    # Determine common date index
    common_index = None
    for tkr, d in data.items():
        common_index = d['close'].index if common_index is None else common_index.intersection(d['close'].index)
    
    if common_index is None or len(common_index) < slow_ma_period:
        return {}

    # Apply core_start_date and MA warmup period
    if core_start_date is not None:
        common_index = common_index[common_index >= core_start_date]
    
    # Ensure enough data for the longest MA
    warmup_start_index = common_index.searchsorted(common_index[0] + pd.DateOffset(days=slow_ma_period))
    if warmup_start_index >= len(common_index):
        return {}
    common_index = common_index[warmup_start_index:]

    # State
    state = {tkr: {'shares': 0, 'avg_cost': 0.0, 'buy_block_until': -1, 'sell_block_until': -1} for tkr in data.keys()}
    cash = float(initial_capital)
    out_rows = {tkr: [] for tkr in data.keys()}
    out_cash = []

    # Main loop
    for i, dt in enumerate(common_index):
        today_prices = {tkr: float(d['close'].loc[dt]) for tkr, d in data.items() if dt in d['close'].index}
        
        # Sells & Trims
        equity = cash + sum(s['shares'] * today_prices.get(tkr, 0) for tkr, s in state.items())
        for tkr, d in data.items():
            s, price = state[tkr], today_prices.get(tkr)
            decision, trade_amount, trade_profit, trade_pl_pct = None, 0.0, 0.0, 0.0

            if s['shares'] > 0 and price is not None and i >= s['sell_block_until']:
                hold_ret = (price / s['avg_cost'] - 1.0) * 100.0 if s['avg_cost'] > 0 else 0.0

                if stop_loss is not None and hold_ret <= float(stop_loss):
                    decision = 'CUT'
                elif d['fast_ma'].loc[dt] < d['slow_ma'].loc[dt]:
                    decision = 'SELL'

                if decision:
                    qty = s['shares']
                    trade_amount = qty * price
                    if s['avg_cost'] > 0:
                        trade_profit = (price - s['avg_cost']) * qty
                        trade_pl_pct = hold_ret
                    cash += trade_amount
                    s['shares'], s['avg_cost'] = 0, 0.0
                    if cooldown_days > 0: s['buy_block_until'] = i + cooldown_days
                
                elif trim_on and max_pos_pct < 1.0:
                    curr_val, cap_val = s['shares'] * price, max_pos_pct * equity
                    if curr_val > cap_val and price > 0:
                        sell_qty = min(s['shares'], int((curr_val - cap_val) // price))
                        if sell_qty > 0:
                            decision, trade_amount = 'TRIM', sell_qty * price
                            cash += trade_amount
                            s['shares'] -= sell_qty
                            if s['avg_cost'] > 0:
                                trade_profit = (price - s['avg_cost']) * sell_qty
                                trade_pl_pct = hold_ret
                            if cooldown_days > 0: s['buy_block_until'] = i + cooldown_days
            
            decision_out = decision if decision else ('HOLD' if s['shares'] > 0 else 'WAIT')
            out_rows[tkr].append({
                'date': dt, 'price': price, 'shares': s['shares'], 'pv': s['shares'] * (price or 0),
                'decision': decision_out, 'avg_cost': s['avg_cost'], 'trade_amount': trade_amount,
                'trade_profit': trade_profit, 'trade_pl_pct': trade_pl_pct, 'note': '',
                'p1': d['ma_score'].loc[dt], 'p2': 0, 's2_sum': 0, 'st_dir': 0
            })

        # Buys
        held_count = sum(1 for s in state.values() if s['shares'] > 0)
        slots_to_fill = max(0, top_n - held_count)
        if slots_to_fill > 0 and cash > 0:
            cands = []
            for tkr, d in data.items():
                s = state[tkr]
                if s['shares'] == 0 and i >= s['buy_block_until']:
                    if d['fast_ma'].loc[dt] > d['slow_ma'].loc[dt]:
                        cands.append((d['ma_score'].loc[dt], tkr))
            
            cands.sort(reverse=True)

            for k in range(min(slots_to_fill, len(cands))):
                if cash <= 0: break
                _, tkr = cands[k]
                price = today_prices.get(tkr)
                if not price: continue

                equity = cash + sum(s['shares'] * today_prices.get(tkr, 0) for tkr, s in state.items())
                min_val = min_pos_pct * equity
                
                req_qty = ceil(min_val / price) if price > 0 else 0
                if req_qty <= 0: continue

                trade_amount = req_qty * price
                if trade_amount > cash: continue

                s = state[tkr]
                cash -= trade_amount
                s['shares'] += req_qty
                s['avg_cost'] = price
                if cooldown_days > 0: s['sell_block_until'] = max(s['sell_block_until'], i + cooldown_days)

                if out_rows[tkr] and out_rows[tkr][-1]['date'] == dt:
                    row = out_rows[tkr][-1]
                    row.update({'decision': 'BUY', 'trade_amount': trade_amount, 'shares': s['shares'], 'pv': s['shares'] * price, 'avg_cost': s['avg_cost']})

        out_cash.append({'date': dt, 'price': 1.0, 'cash': cash, 'shares': 0, 'pv': cash, 'decision': 'HOLD'})

    result = {}
    for tkr, rows in out_rows.items():
        if rows:
            result[tkr] = pd.DataFrame(rows).set_index('date')
    if out_cash:
        result['CASH'] = pd.DataFrame(out_cash).set_index('date')
    return result


def run_single_ticker_backtest(
    ticker: str,
    df: Optional[pd.DataFrame] = None,
    months_range: Optional[List[int]] = None,
    initial_capital: float = 1_000_000.0,
    core_start_date: Optional[pd.Timestamp] = None,
    date_range: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Simulates a single ticker backtest using a moving average crossover strategy.
    """
    # Settings
    fast_ma_period = int(getattr(seykota_settings, 'SEYKOTA_FAST_MA', 50))
    slow_ma_period = int(getattr(seykota_settings, 'SEYKOTA_SLOW_MA', 150))
    stop_loss = getattr(seykota_settings, 'SEYKOTA_STOP_LOSS_PCT', None)
    cooldown_days = int(getattr(seykota_settings, 'COOLDOWN_DAYS', 0))

    if df is None:
        df = fetch_ohlcv(ticker, months_range=months_range, date_range=date_range)
    if df is None or len(df) < slow_ma_period:
        return pd.DataFrame()

    close = df['Close']
    fast_ma = close.rolling(window=fast_ma_period).mean()
    slow_ma = close.rolling(window=slow_ma_period).mean()

    start_i = slow_ma_period
    if core_start_date is not None:
        try:
            start_i = max(start_i, df.index.searchsorted(core_start_date, side='left'))
        except Exception:
            pass

    # State
    cash = float(initial_capital)
    shares = 0
    avg_cost = 0.0
    buy_block_until = -1
    sell_block_until = -1

    rows = []
    for i in range(start_i, len(df)):
        price = float(close.iloc[i])
        decision, trade_amount, trade_profit, trade_pl_pct = None, 0.0, 0.0, 0.0

        # Sell/Cut logic
        if shares > 0 and i >= sell_block_until:
            hold_ret = (price / avg_cost - 1.0) * 100.0 if avg_cost > 0 else 0.0
            if stop_loss is not None and hold_ret <= float(stop_loss):
                decision = 'CUT'
            elif fast_ma.iloc[i] < slow_ma.iloc[i]:
                decision = 'SELL'
            
            if decision in ('CUT', 'SELL'):
                trade_amount = shares * price
                if avg_cost > 0:
                    trade_profit = (price - avg_cost) * shares
                    trade_pl_pct = hold_ret
                cash += trade_amount
                shares, avg_cost = 0, 0.0
                if cooldown_days > 0: buy_block_until = i + cooldown_days

        # Buy logic
        if decision is None and shares == 0 and i >= buy_block_until:
            if fast_ma.iloc[i] > slow_ma.iloc[i]:
                buy_qty = int(cash // price)
                if buy_qty > 0:
                    trade_amount = buy_qty * price
                    cash -= trade_amount
                    avg_cost = price
                    shares = buy_qty
                    decision = 'BUY'
                    if cooldown_days > 0: sell_block_until = i + cooldown_days

        if decision is None:
            decision = 'HOLD' if shares > 0 else 'WAIT'

        rows.append({
            'date': df.index[i], 'price': price, 'cash': cash, 'shares': shares, 'pv': cash + shares * price,
            'decision': decision, 'avg_cost': avg_cost, 'trade_amount': trade_amount,
            'trade_profit': trade_profit, 'trade_pl_pct': trade_pl_pct, 'note': '',
            'p1': 0, 'p2': 0, 's2_sum': 0, 'st_dir': 0,
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index('date')