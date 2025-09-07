"""
Strategy: 'dummy'
A simple buy-and-hold strategy for comparison purposes.
"""
import pandas as pd
from typing import Optional, List, Tuple, Dict

from utils.data_loader import fetch_ohlcv, get_today_str


def run_portfolio_backtest(
    pairs: List[Tuple[str, str]],
    months_range: Optional[List[int]] = None,
    initial_capital: float = 100_000_000.0,
    core_start_date: Optional[pd.Timestamp] = None,
    top_n: int = 10,
    initial_positions: Optional[dict] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Simulates a simple buy-and-hold portfolio.
    On the first day, it buys the first `top_n` tickers from the list and holds them.
    """
    # Fetch all series
    data = {}
    for tkr, _ in pairs:
        df = fetch_ohlcv(tkr, months_range=months_range)
        if df is not None and not df.empty:
            data[tkr] = {'df': df, 'close': df['Close']}

    if not data:
        return {}

    # Determine common date index
    common_index = None
    for tkr, d in data.items():
        common_index = d['close'].index if common_index is None else common_index.intersection(d['close'].index)
    
    if common_index is None or len(common_index) == 0:
        return {}

    # Apply core_start_date
    if core_start_date is None and months_range is not None and len(months_range) == 2:
        try:
            now = pd.to_datetime(get_today_str())
            core_start_date = now - pd.DateOffset(months=int(months_range[0]))
        except Exception:
            core_start_date = None
    if core_start_date is not None:
        common_index = common_index[common_index >= core_start_date]
    
    if len(common_index) == 0:
        return {}

    # State
    state = {tkr: {'shares': 0, 'avg_cost': 0.0} for tkr in data.keys()}
    cash = float(initial_capital)
    out_rows = {tkr: [] for tkr in data.keys()}
    out_cash = []

    # Main loop
    for i, dt in enumerate(common_index):
        today_prices = {tkr: float(d['close'].loc[dt]) for tkr, d in data.items() if dt in d['close'].index}

        # On the first day, buy
        if i == 0 and not initial_positions: # 초기 보유 현황이 없을 때만 실행
            tickers_to_buy = [p[0] for p in pairs if p[0] in data][:top_n]
            investment_per_ticker = initial_capital / len(tickers_to_buy) if tickers_to_buy else 0

            for tkr in tickers_to_buy:
                price = today_prices.get(tkr)
                if price and price > 0 and cash >= investment_per_ticker:
                    qty = int(investment_per_ticker // price)
                    if qty > 0:
                        trade_amount = qty * price
                        cash -= trade_amount
                        state[tkr]['shares'] = qty
                        state[tkr]['avg_cost'] = price
        
        # Record daily state for all tickers
        for tkr, d in data.items():
            s = state[tkr]
            price = today_prices.get(tkr)
            decision = 'WAIT'
            trade_amount = 0.0
            if i == 0 and s['shares'] > 0:
                decision = 'BUY'
                trade_amount = s['shares'] * s['avg_cost']
            elif s['shares'] > 0:
                decision = 'HOLD'

            out_rows[tkr].append({
                'date': dt, 'price': price, 'shares': s['shares'], 'pv': s['shares'] * (price if price is not None else 0),
                'decision': decision, 'avg_cost': s['avg_cost'], 'trade_amount': trade_amount,
                'trade_profit': 0.0, 'trade_pl_pct': 0.0, 'note': 'dummy logic',
                'p1': 0, 'p2': 0, 's2_sum': 0, 'st_dir': 0
            })

        # Record cash state
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
) -> pd.DataFrame:
    """
    Simulates a simple buy-and-hold for a single ticker.
    """
    if df is None:
        df = fetch_ohlcv(ticker, months_range=months_range)
    if df is None or df.empty:
        return pd.DataFrame()

    # Determine start index
    start_i = 0
    if core_start_date is not None:
        try:
            if isinstance(df.index, pd.DatetimeIndex):
                start_i = df.index.searchsorted(core_start_date, side='left')
        except Exception:
            pass

    # State
    cash = float(initial_capital)
    shares = 0
    avg_cost = 0.0
    
    rows = []
    for i in range(start_i, len(df)):
        price = float(df['Close'].iloc[i])
        decision = 'WAIT'
        trade_amount = 0.0

        # Buy on the first day
        if i == start_i and cash > 0 and price > 0:
            qty = int(cash // price)
            if qty > 0:
                trade_amount = qty * price
                cash -= trade_amount
                shares = qty
                avg_cost = price
                decision = 'BUY'
        elif shares > 0:
            decision = 'HOLD'

        rows.append({
            'date': df.index[i], 'price': price, 'cash': cash, 'shares': shares, 'pv': cash + shares * price,
            'decision': decision, 'avg_cost': avg_cost, 'trade_amount': trade_amount,
            'trade_profit': 0.0, 'trade_pl_pct': 0.0, 'note': 'dummy logic',
            'p1': 0, 'p2': 0, 's2_sum': 0, 'st_dir': 0,
        })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index('date')