"""
Strategy: 'jason'
SuperTrend-based momentum strategy.
이 파일은 'jason' 전략의 핵심 로직을 포함합니다.
"""
import pandas as pd
from typing import Optional, List, Tuple, Dict
from math import ceil

import settings as global_settings
from . import settings as jason_settings
from utils.data_loader import fetch_ohlcv, get_today_str
from utils.indicators import supertrend_direction


def run_portfolio_backtest(
    pairs: List[Tuple[str, str]],
    months_range: Optional[List[int]] = None,
    initial_capital: float = 100_000_000.0,
    core_start_date: Optional[pd.Timestamp] = None,
    top_n: int = 10,
    date_range: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    공유 현금 Top-N 포트폴리오를 시뮬레이션합니다.
    (core.portfolio.portfolio_topn_series에서 이동)
    """
    # Fetch all series
    data = {}
    for tkr, _ in pairs:
        df = fetch_ohlcv(tkr, months_range=months_range, date_range=date_range)
        if df is None or len(df) < 25:
            continue
        close = df['Close']
        # Precompute diagnostics
        p1 = (close / close.shift(5) - 1.0).fillna(0.0).round(3) * 100
        p2 = (close.shift(5) / close.shift(10) - 1.0).fillna(0.0).round(3) * 100
        s2 = p1 + p2
        try:
            st_p = int(getattr(jason_settings, 'ST_ATR_PERIOD', 14))
            st_m = float(getattr(jason_settings, 'ST_ATR_MULTIPLIER', 3.0))
        except Exception:
            st_p, st_m = 14, 3.0
        st_dir = supertrend_direction(df, st_p, st_m) if len(df) > max(2, st_p) else None
        data[tkr] = {
            'df': df, 'close': close, 'p1': p1, 'p2': p2, 's2': s2,
            'st_dir': st_dir if st_dir is not None else pd.Series(0, index=df.index),
        }

    if not data:
        return {}

    # Determine common date index
    common_index = None
    for tkr, d in data.items():
        common_index = d['close'].index if common_index is None else common_index.intersection(d['close'].index)
    if common_index is None or len(common_index) < 11:
        return {}

    # Apply core_start_date
    if core_start_date is not None:
        common_index = common_index[common_index >= core_start_date]
    
    if len(common_index) == 0:
        return {}

    # Settings
    sell_thr = float(getattr(jason_settings, 'SELL_SUM_THRESHOLD', -3.0))
    buy_thr = float(getattr(jason_settings, 'BUY_SUM_THRESHOLD', 3.0))
    stop_loss = getattr(jason_settings, 'HOLDING_STOP_LOSS_PCT', None)
    cooldown_days = int(getattr(global_settings, 'COOLDOWN_DAYS', 0))
    big_drop_pct = float(getattr(global_settings, 'BIG_DROP_PCT', -10.0)) # 이 부분은 global_settings를 사용하는 것이 맞습니다.
    big_drop_block_days = int(getattr(global_settings, 'BIG_DROP_SELL_BLOCK_DAYS', 5))
    min_pos_pct = float(getattr(global_settings, 'MIN_POSITION_PCT', 0.10))
    max_pos_pct = float(getattr(global_settings, 'MAX_POSITION_PCT', 0.15))
    trim_on = bool(getattr(global_settings, 'ENABLE_MAX_POSITION_TRIM', True))

    # State
    state = {tkr: {'shares': 0, 'avg_cost': 0.0, 'buy_block_until': -1, 'sell_block_until': -1} for tkr in data.keys()}
    cash = float(initial_capital)
    out_rows = {tkr: [] for tkr in data.keys()}
    out_cash = []

    # Main loop
    for i, dt in enumerate(common_index):
        today_prices = {tkr: float(d['close'].loc[dt]) for tkr, d in data.items() if dt in d['close'].index}
        
        # 1. Big drop detection
        for tkr, d in data.items():
            pos = d['close'].index.get_loc(dt)
            if pos > 0:
                c0, c_prev = float(d['close'].iloc[pos]), float(d['close'].iloc[pos-1])
                if c_prev > 0 and (c0 / c_prev - 1.0) * 100.0 <= big_drop_pct:
                    state[tkr]['sell_block_until'] = max(state[tkr]['sell_block_until'], i + big_drop_block_days)

        # 2. Sells & Trims
        equity = cash + sum(s['shares'] * today_prices.get(tkr, 0) for tkr, s in state.items())
        for tkr, d in data.items():
            s, price = state[tkr], today_prices.get(tkr)

            decision, trade_amount, trade_profit, trade_pl_pct = None, 0.0, 0.0, 0.0

            # 매도/부분매도 로직은 보유 중이고 거래 가능한 종목에 대해서만 실행
            if s['shares'] > 0 and price is not None and i >= s['sell_block_until']:
                hold_ret = (price / s['avg_cost'] - 1.0) * 100.0 if s['avg_cost'] > 0 else 0.0

                # 매도/손절 조건
                if stop_loss is not None and hold_ret <= float(stop_loss):
                    decision = 'CUT'
                elif (d['s2'].loc[dt] + hold_ret) < sell_thr:
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
                
                # 부분매도(Trim) 조건
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
            
            # Record daily state for this ticker
            decision_out = decision if decision else ('HOLD' if s['shares'] > 0 else 'WAIT')
            note = ''
            if decision_out in ('WAIT', 'HOLD'):
                if s['shares'] > 0 and i < s['sell_block_until']: note = '매도 쿨다운'
                elif s['shares'] == 0 and i < s['buy_block_until']: note = '매수 쿨다운'

            out_rows[tkr].append({
                'date': dt, 'price': price, 'shares': s['shares'], 'pv': s['shares'] * (price if price is not None else 0),
                'decision': decision_out, 'avg_cost': s['avg_cost'], 'trade_amount': trade_amount,
                'trade_profit': trade_profit, 'trade_pl_pct': trade_pl_pct, 'note': note,
                'p1': d['p1'].loc[dt], 'p2': d['p2'].loc[dt], 's2_sum': d['s2'].loc[dt],
                'st_dir': int(d['st_dir'].loc[dt]) if dt in d['st_dir'].index else 0
            })

        # 3. Buys
        held_count = sum(1 for s in state.values() if s['shares'] > 0)
        slots_to_fill = max(0, top_n - held_count)
        if slots_to_fill > 0 and cash > 0:
            cands = []
            for tkr, d in data.items():
                s = state[tkr]
                if s['shares'] == 0 and i >= s['buy_block_until']:
                    s2_val = d['s2'].loc[dt]
                    st_val = int(d['st_dir'].loc[dt]) if dt in d['st_dir'].index else -1
                    if s2_val > buy_thr and st_val > 0:
                        cands.append((s2_val, tkr))
            
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
                s['avg_cost'] = price # Simple avg_cost for new position
                if cooldown_days > 0: s['sell_block_until'] = max(s['sell_block_until'], i + cooldown_days)

                # Update today's row for the bought ticker
                if out_rows[tkr] and out_rows[tkr][-1]['date'] == dt:
                    row = out_rows[tkr][-1]
                    row.update({
                        'decision': 'BUY', 'trade_amount': trade_amount, 'shares': s['shares'],
                        'pv': s['shares'] * price, 'avg_cost': s['avg_cost']
                    })
                else: # Should not happen if sell logic ran first
                    pass

        # 4. Record cash state
        out_cash.append({'date': dt, 'price': 1.0, 'cash': cash, 'shares': 0, 'pv': cash, 'decision': 'HOLD'})

    # Build DataFrames
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
    개별 종목에 대한 일별 시뮬레이션 상태를 반환합니다.
    (core.backtester.simple_daily_series에서 이동)
    """
    if df is None:
        df = fetch_ohlcv(ticker, months_range=months_range, date_range=date_range)
    if df is None or len(df) < 25:
        return pd.DataFrame(columns=['price','cash','shares','pv','decision'], index=pd.DatetimeIndex([]))

    close = df['Close']

    # Determine start index respecting the core start date (after warmup)
    start_i = 20
    if core_start_date is not None:
        try:
            if isinstance(df.index, pd.DatetimeIndex):
                start_i = max(start_i, df.index.searchsorted(core_start_date, side='left'))
        except Exception:
            pass

    # Settings
    sell_thr = float(getattr(jason_settings, 'SELL_SUM_THRESHOLD', -3.0))
    buy_thr = float(getattr(jason_settings, 'BUY_SUM_THRESHOLD', 3.0))
    stop_loss = getattr(jason_settings, 'HOLDING_STOP_LOSS_PCT', None)
    cooldown_days = int(getattr(global_settings, 'COOLDOWN_DAYS', 0))
    try:
        st_p = int(getattr(jason_settings, 'ST_ATR_PERIOD', 14))
        st_m = float(getattr(jason_settings, 'ST_ATR_MULTIPLIER', 3.0))
    except Exception:
        st_p, st_m = 14, 3.0
    st_dir = supertrend_direction(df, st_p, st_m) if len(df) > max(2, st_p) else None

    # State
    cash = float(initial_capital)
    shares = 0
    avg_cost = 0.0
    buy_block_until = -1
    sell_block_until = -1

    rows = []
    for i in range(start_i, len(df)):
        c0 = float(close.iloc[i])
        c5 = float(close.iloc[i-5])
        c10 = float(close.iloc[i-10])
        r21 = (c5/c10 - 1.0) if c10 > 0 else 0.0
        r10 = (c0/c5 - 1.0) if c5 > 0 else 0.0
        p1 = round(r10 * 100, 1)
        p2 = round(r21 * 100, 1)
        s2_sum = p1 + p2

        # Big drop sell block rule
        try:
            big_drop_pct = float(getattr(global_settings, 'BIG_DROP_PCT', -10.0))
            big_drop_block_days = int(getattr(global_settings, 'BIG_DROP_SELL_BLOCK_DAYS', 5))
            c_prev = float(close.iloc[i-1]) if i - 1 >= 0 else None
            if c_prev and c_prev > 0:
                day_chg_pct = (c0 / c_prev - 1.0) * 100.0
                if day_chg_pct <= big_drop_pct:
                    sell_block_until = max(sell_block_until, i + big_drop_block_days)
        except Exception:
            pass

        decision = None
        trade_amount = 0.0
        trade_profit = 0.0
        trade_pl_pct = 0.0
        reason_block = None

        # Sell/Cut logic
        if shares > 0 and i >= sell_block_until:
            curr_hold_ret = (c0 - avg_cost) / avg_cost * 100.0 if avg_cost > 0 else 0.0
            # 1) Stop loss
            if stop_loss is not None and curr_hold_ret <= float(stop_loss):
                decision = 'CUT'
            # 2) Sum threshold
            elif (s2_sum + curr_hold_ret) < sell_thr:
                decision = 'SELL'
            
            if decision in ('CUT', 'SELL'):
                prev_shares = shares
                trade_amount = prev_shares * c0
                if avg_cost > 0:
                    trade_profit = (c0 - avg_cost) * prev_shares
                    trade_pl_pct = (c0 / avg_cost - 1.0) * 100.0
                if decision == 'SELL' and trade_pl_pct < 0: decision = 'CUT' # Re-classify as CUT if loss
                cash += trade_amount
                shares = 0
                avg_cost = 0.0
                if cooldown_days > 0:
                    buy_block_until = i + cooldown_days

        # Buy logic
        if decision is None and shares == 0 and i >= buy_block_until and s2_sum > buy_thr:
            if st_dir is not None and int(st_dir.iloc[i]) <= 0:
                reason_block = 'ST_DOWN'
            if reason_block is None:
                buy_qty = int(cash // c0)
                if buy_qty > 0:
                    trade_amount = buy_qty * c0
                    cash -= trade_amount
                    new_shares = shares + buy_qty
                    avg_cost = ((avg_cost * shares) + (c0 * buy_qty)) / new_shares if new_shares > 0 else 0.0
                    shares = new_shares
                    decision = 'BUY'
                    if cooldown_days > 0:
                        sell_block_until = i + cooldown_days

        if decision is None:
            decision = 'HOLD' if shares > 0 else 'WAIT'

        pv = cash + shares * c0
        note_txt = ''
        if decision in ('WAIT','HOLD'):
            if shares > 0 and (i < sell_block_until): note_txt = '매도 쿨다운'
            elif shares == 0 and (i < buy_block_until): note_txt = '매수 쿨다운'
            elif reason_block == 'ST_DOWN': note_txt = 'ST 하향'

        rows.append({
            'date': df.index[i], 'price': c0, 'cash': cash, 'shares': shares, 'pv': pv,
            'decision': decision, 'avg_cost': avg_cost, 'trade_amount': trade_amount,
            'trade_profit': trade_profit, 'trade_pl_pct': trade_pl_pct, 'note': note_txt,
            'p1': p1, 'p2': p2, 's2_sum': s2_sum,
            'st_dir': int(st_dir.iloc[i]) if st_dir is not None else 0,
        })

    if not rows:
        return pd.DataFrame(columns=['price','cash','shares','pv','decision'], index=pd.DatetimeIndex([]))
    return pd.DataFrame(rows).set_index('date')