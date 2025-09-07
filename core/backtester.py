"""
개별 종목 백테스팅 실행기.
"""
import pandas as pd
from typing import Optional, List

import settings
from utils.data_loader import fetch_ohlcv, get_today_str
from utils.indicators import supertrend_direction

def simple_daily_series(
    ticker: str,
    df: Optional[pd.DataFrame] = None,
    months_range: Optional[List[int]] = None,
    initial_capital: float = 1_000_000.0,
    core_start_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    개별 종목에 대한 일별 시뮬레이션 상태를 반환합니다.
    (logic.a.logic.simple_daily_series에서 이동)
    """
    if df is None:
        df = fetch_ohlcv(ticker, months_range=months_range)
    if df is None or len(df) < 25:
        return pd.DataFrame(columns=['price','cash','shares','pv','decision'], index=pd.DatetimeIndex([]))

    close = df['Close']

    # Determine start index respecting the core start date (after warmup)
    start_i = 20
    if core_start_date is None and months_range is not None and len(months_range) == 2:
        try:
            now = pd.to_datetime(get_today_str())
            core_start_date = now - pd.DateOffset(months=int(months_range[0]))
        except Exception:
            core_start_date = None
    if core_start_date is not None:
        try:
            if isinstance(df.index, pd.DatetimeIndex):
                start_i = max(start_i, df.index.searchsorted(core_start_date, side='left'))
        except Exception:
            pass

    # Settings
    sell_thr = float(getattr(settings, 'SELL_SUM_THRESHOLD', -3.0))
    buy_thr = float(getattr(settings, 'BUY_SUM_THRESHOLD', 3.0))
    stop_loss = getattr(settings, 'HOLDING_STOP_LOSS_PCT', None)
    cooldown_days = int(getattr(settings, 'COOLDOWN_DAYS', 0))
    try:
        st_p = int(getattr(settings, 'ST_ATR_PERIOD', 14))
        st_m = float(getattr(settings, 'ST_ATR_MULTIPLIER', 3.0))
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
            big_drop_pct = float(getattr(settings, 'BIG_DROP_PCT', -10.0))
            big_drop_block_days = int(getattr(settings, 'BIG_DROP_SELL_BLOCK_DAYS', 5))
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