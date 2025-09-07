"""
Strategy: 'donchian'
Richard Donchian-style trend-following strategy using a single moving average.
"""

from math import ceil
from typing import Dict, List, Optional, Tuple

import pandas as pd

import settings as global_settings
from utils.data_loader import fetch_ohlcv

from . import settings as donchian_settings


def run_portfolio_backtest(
    pairs: List[Tuple[str, str]],
    months_range: Optional[List[int]] = None,
    initial_capital: float = 100_000_000.0,
    core_start_date: Optional[pd.Timestamp] = None,
    top_n: int = 10,
    date_range: Optional[List[str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Simulates a Top-N portfolio using a single moving average crossover strategy.
    """
    # Settings
    try:
        # 전략 고유 설정
        ma_period = int(donchian_settings.DONCHIAN_MA_PERIOD)
    except AttributeError as e:
        raise AttributeError(
            f"'{e.name}' 설정이 logics/donchian/settings.py 파일에 반드시 정의되어야 합니다."
        ) from e

    try:
        # 전역 설정
        stop_loss = global_settings.HOLDING_STOP_LOSS_PCT
        cooldown_days = int(global_settings.COOLDOWN_DAYS)
        min_pos_pct = float(global_settings.MIN_POSITION_PCT)
        max_pos_pct = float(global_settings.MAX_POSITION_PCT)
        trim_on = bool(global_settings.ENABLE_MAX_POSITION_TRIM)
    except AttributeError as e:
        raise AttributeError(
            f"'{e.name}' 설정이 전역 settings.py 파일에 반드시 정의되어야 합니다."
        ) from e

    # --- 데이터 로딩 및 지표 계산 ---
    # 전체 기간에 대한 데이터 로딩
    warmup_days = int(ma_period * 1.5)

    adjusted_date_range = None
    if date_range and len(date_range) == 2:
        core_start = pd.to_datetime(date_range[0])
        warmup_start = core_start - pd.DateOffset(days=warmup_days)
        adjusted_date_range = [warmup_start.strftime("%Y-%m-%d"), date_range[1]]

    data = {}
    tickers_to_process = [p[0] for p in pairs]

    for tkr in tickers_to_process:
        df = fetch_ohlcv(tkr, months_range=months_range, date_range=adjusted_date_range)
        if df is None or len(df) < ma_period:
            continue

        close = df["Close"]
        ma = close.rolling(window=ma_period).mean()
        ma_score = (close / ma - 1.0) * 100.0 if ma is not None else 0.0

        data[tkr] = {
            "df": df,
            "close": df["Close"],
            "ma": ma,
            "ma_score": ma_score,
        }

    if not data:
        return {}

    union_index = pd.DatetimeIndex([])
    for tkr, d in data.items():
        union_index = union_index.union(d["close"].index)

    if union_index.empty:
        return {}

    if core_start_date:
        union_index = union_index[union_index >= core_start_date]

    if union_index.empty:
        return {}

    # State
    state = {
        tkr: {
            "shares": 0,
            "avg_cost": 0.0,
            "buy_block_until": -1,
            "sell_block_until": -1,
        }
        for tkr in data.keys()
    }
    cash = float(initial_capital)
    out_rows = {tkr: [] for tkr in data.keys()}
    out_cash = []

    for i, dt in enumerate(union_index):
        tickers_available_today = [tkr for tkr, d in data.items() if dt in d["df"].index]
        today_prices = {
            tkr: float(d["close"].loc[dt]) if pd.notna(d["close"].loc[dt]) else None
            for tkr, d in data.items()
            if dt in d["close"].index
        }

        current_holdings_value = 0
        for tkr_h, s_h in state.items():
            if s_h["shares"] > 0:
                price_h = today_prices.get(tkr_h)
                if pd.notna(price_h):
                    current_holdings_value += s_h["shares"] * price_h

        equity = cash + current_holdings_value

        for tkr, d in data.items():
            s, price = state[tkr], today_prices.get(tkr)
            decision, trade_amount, trade_profit, trade_pl_pct = None, 0.0, 0.0, 0.0

            is_ticker_warming_up = tkr not in tickers_available_today or pd.isna(d["ma"].get(dt))

            if tkr in tickers_available_today:
                if (
                    s["shares"] > 0
                    and pd.notna(price)
                    and i >= s["sell_block_until"]
                    and not is_ticker_warming_up
                ):
                    hold_ret = (price / s["avg_cost"] - 1.0) * 100.0 if s["avg_cost"] > 0 else 0.0

                    if stop_loss is not None and hold_ret <= float(stop_loss):
                        decision = "CUT_STOPLOSS"
                    elif not is_ticker_warming_up and price < d["ma"].loc[dt]:
                        decision = "SELL_TREND"

                    if decision:
                        qty = s["shares"]
                        trade_amount = qty * price
                        if s["avg_cost"] > 0:
                            trade_profit = (price - s["avg_cost"]) * qty
                            trade_pl_pct = hold_ret
                        cash += trade_amount
                        s["shares"], s["avg_cost"] = 0, 0.0
                        if cooldown_days > 0:
                            s["buy_block_until"] = i + cooldown_days

                    elif trim_on and max_pos_pct < 1.0:
                        curr_val, cap_val = s["shares"] * price, max_pos_pct * equity
                        if curr_val > cap_val and price > 0:
                            sell_qty = min(s["shares"], int((curr_val - cap_val) // price))
                            if sell_qty > 0:
                                decision, trade_amount = (
                                    "TRIM_REBALANCE",
                                    sell_qty * price,
                                )
                                cash += trade_amount
                                s["shares"] -= sell_qty
                                if s["avg_cost"] > 0:
                                    trade_profit = (price - s["avg_cost"]) * sell_qty
                                    trade_pl_pct = hold_ret
                                if cooldown_days > 0:
                                    s["buy_block_until"] = i + cooldown_days

            decision_out = decision if decision else ("HOLD" if s["shares"] > 0 else "WAIT")
            note = ""
            if is_ticker_warming_up:
                note = "웜업 기간"
            elif decision_out in ("WAIT", "HOLD"):
                if s["shares"] > 0 and i < s["sell_block_until"]:
                    note = "매도 쿨다운"
                elif s["shares"] == 0 and i < s["buy_block_until"]:
                    note = "매수 쿨다운"

            if tkr in tickers_available_today:
                out_rows[tkr].append(
                    {
                        "date": dt,
                        "price": price,
                        "shares": s["shares"],
                        "pv": s["shares"] * (price if pd.notna(price) else 0),
                        "decision": decision_out,
                        "avg_cost": s["avg_cost"],
                        "trade_amount": trade_amount,
                        "trade_profit": trade_profit,
                        "trade_pl_pct": trade_pl_pct,
                        "note": note,
                        "signal1": d["ma"].loc[dt],
                        "signal2": None,
                        "score": d["ma_score"].loc[dt],
                        "filter": None,
                    }
                )
            else:
                out_rows[tkr].append(
                    {
                        "date": dt,
                        "price": s["avg_cost"],
                        "shares": s["shares"],
                        "pv": s["shares"] * (s["avg_cost"] if pd.notna(s["avg_cost"]) else 0.0),
                        "decision": "HOLD" if s["shares"] > 0 else "WAIT",
                        "avg_cost": s["avg_cost"],
                        "trade_amount": 0.0,
                        "trade_profit": 0.0,
                        "trade_pl_pct": 0.0,
                        "note": "데이터 없음",
                        "signal1": 0,
                        "signal2": None,
                        "score": 0,
                        "filter": None,
                    }
                )

        # Buys
        held_count = sum(1 for s in state.values() if s["shares"] > 0)
        slots_to_fill = max(0, top_n - held_count)
        if slots_to_fill > 0 and cash > 0:
            cands = []
            for tkr in tickers_available_today:
                d = data.get(tkr)
                s = state[tkr]
                price_cand = today_prices.get(tkr)

                ma_cand = d["ma"].get(dt)
                if (
                    s["shares"] == 0
                    and i >= s["buy_block_until"]
                    and pd.notna(ma_cand)
                    and pd.notna(price_cand)
                ):
                    if price_cand > ma_cand:
                        score_cand = d["ma_score"].get(dt, 0.0)
                        cands.append((score_cand, tkr))

            cands.sort(reverse=True)

            for k in range(min(slots_to_fill, len(cands))):
                if cash <= 0:
                    break
                _, tkr = cands[k]

                price = today_prices.get(tkr)
                if pd.isna(price):
                    continue

                equity = cash + current_holdings_value
                min_val = min_pos_pct * equity

                req_qty = ceil(min_val / price) if price > 0 else 0
                if req_qty <= 0:
                    continue

                trade_amount = req_qty * price
                if trade_amount > cash:
                    continue

                s = state[tkr]
                cash -= trade_amount
                s["shares"] += req_qty
                s["avg_cost"] = price
                if cooldown_days > 0:
                    s["sell_block_until"] = max(s["sell_block_until"], i + cooldown_days)

                if out_rows[tkr] and out_rows[tkr][-1]["date"] == dt:
                    row = out_rows[tkr][-1]
                    row.update(
                        {
                            "decision": "BUY",
                            "trade_amount": trade_amount,
                            "shares": s["shares"],
                            "pv": s["shares"] * price,
                            "avg_cost": s["avg_cost"],
                        }
                    )
        else:
            for tkr in tickers_available_today:
                s = state[tkr]
                if s["shares"] == 0:
                    d = data.get(tkr)
                    price = today_prices.get(tkr)
                    ma_note = d["ma"].get(dt)

                    if pd.notna(ma_note) and pd.notna(price) and price > ma_note:
                        if out_rows[tkr] and out_rows[tkr][-1]["date"] == dt:
                            note = "포트폴리오 가득 참" if slots_to_fill <= 0 else "현금 부족"
                            out_rows[tkr][-1]["note"] = note

        out_cash.append(
            {
                "date": dt,
                "price": 1.0,
                "cash": cash,
                "shares": 0,
                "pv": cash,
                "decision": "HOLD",
            }
        )

    result = {}
    for tkr, rows in out_rows.items():
        if rows:
            result[tkr] = pd.DataFrame(rows).set_index("date")
    if out_cash:
        result["CASH"] = pd.DataFrame(out_cash).set_index("date")
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
    Simulates a single ticker backtest using a single moving average crossover strategy.
    """
    try:
        # 전략 고유 설정
        ma_period = int(donchian_settings.DONCHIAN_MA_PERIOD)
    except AttributeError as e:
        raise AttributeError(
            f"'{e.name}' 설정이 logics/donchian/settings.py 파일에 반드시 정의되어야 합니다."
        ) from e

    try:
        # 전역 설정
        stop_loss = global_settings.HOLDING_STOP_LOSS_PCT
        cooldown_days = int(global_settings.COOLDOWN_DAYS)
    except AttributeError as e:
        raise AttributeError(
            f"'{e.name}' 설정이 전역 settings.py 파일에 반드시 정의되어야 합니다."
        ) from e

    if df is None:
        df = fetch_ohlcv(ticker, months_range=months_range, date_range=date_range)
    if df is None or df.empty or len(df) < ma_period:
        return pd.DataFrame()

    close = df["Close"]
    ma = close.rolling(window=ma_period).mean()

    loop_start_index = 0
    if core_start_date is not None:
        try:
            loop_start_index = df.index.searchsorted(core_start_date, side="left")
        except Exception:
            pass

    cash = float(initial_capital)
    shares = 0
    avg_cost = 0.0
    buy_block_until = -1
    sell_block_until = -1

    rows = []
    for i in range(loop_start_index, len(df)):
        price_val = close.iloc[i]
        if pd.isna(price_val):
            continue
        price = float(price_val)
        decision, trade_amount, trade_profit, trade_pl_pct = None, 0.0, 0.0, 0.0

        ma_today = ma.iloc[i]

        if pd.isna(ma_today):
            rows.append(
                {
                    "date": df.index[i],
                    "price": price,
                    "cash": cash,
                    "shares": shares,
                    "pv": cash + shares * price,
                    "decision": "WAIT",
                    "avg_cost": avg_cost,
                    "trade_amount": 0.0,
                    "trade_profit": 0.0,
                    "trade_pl_pct": 0.0,
                    "note": "웜업 기간",
                    "signal1": None,
                    "signal2": None,
                    "score": None,
                    "filter": None,
                }
            )
            continue

        if shares > 0 and i >= sell_block_until:
            hold_ret = (price / avg_cost - 1.0) * 100.0 if avg_cost > 0 else 0.0
            if stop_loss is not None and hold_ret <= float(stop_loss):
                decision = "CUT_STOPLOSS"
            elif price < ma_today:
                decision = "SELL_TREND"

            if decision in ("CUT_STOPLOSS", "SELL_TREND"):
                trade_amount = shares * price
                if avg_cost > 0:
                    trade_profit = (price - avg_cost) * shares
                    trade_pl_pct = hold_ret
                cash += trade_amount
                shares, avg_cost = 0, 0.0
                if cooldown_days > 0:
                    buy_block_until = i + cooldown_days

        if decision is None and shares == 0 and i >= buy_block_until:
            if price > ma_today:
                buy_qty = int(cash // price)
                if buy_qty > 0:
                    trade_amount = buy_qty * price
                    cash -= trade_amount
                    avg_cost, shares = price, buy_qty
                    decision = "BUY"
                    if cooldown_days > 0:
                        sell_block_until = i + cooldown_days

        if decision is None:
            decision = "HOLD" if shares > 0 else "WAIT"

        rows.append(
            {
                "date": df.index[i],
                "price": price,
                "cash": cash,
                "shares": shares,
                "pv": cash + shares * price,
                "decision": decision,
                "avg_cost": avg_cost,
                "trade_amount": trade_amount,
                "trade_profit": trade_profit,
                "trade_pl_pct": trade_pl_pct,
                "note": "",
                "signal1": ma_today,
                "signal2": None,
                "score": ((price / ma_today - 1.0) * 100.0 if ma_today > 0 else 0.0),
                "filter": None,
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("date")
