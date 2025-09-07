"""
Strategy: 'jason'
SuperTrend-based momentum strategy.
이 파일은 'jason' 전략의 핵심 로직을 포함합니다.
"""

from math import ceil
from typing import Dict, List, Optional, Tuple

import pandas as pd

import settings as global_settings
from utils.data_loader import fetch_ohlcv
from utils.indicators import supertrend_direction

from . import settings as jason_settings


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
    tickers_to_process = [p[0] for p in pairs]

    # --- 데이터 로딩 및 지표 계산 ---
    # 웜업 기간(슈퍼트렌드 기간 + 수익률 계산 기간)을 포함하여 시작일을 계산합니다.
    st_p = int(getattr(jason_settings, "ST_ATR_PERIOD", 14))
    warmup_days = int(
        (st_p + 10) * 1.5
    )  # 10일(p2 계산용)과 슈퍼트렌드 기간을 합산하고, 거래일이 아닌 날을 고려하여 1.5배 버퍼를 줍니다.

    adjusted_date_range = None
    if date_range and len(date_range) == 2:
        core_start = pd.to_datetime(date_range[0])
        warmup_start = core_start - pd.DateOffset(days=warmup_days)
        adjusted_date_range = [warmup_start.strftime("%Y-%m-%d"), date_range[1]]

    for tkr in tickers_to_process:
        df = fetch_ohlcv(tkr, months_range=months_range, date_range=adjusted_date_range)

        if df is None or len(df) < 25:
            continue
        close = df["Close"]
        # Precompute diagnostics
        p1 = (close / close.shift(5) - 1.0).fillna(0.0).round(3) * 100
        p2 = (close.shift(5) / close.shift(10) - 1.0).fillna(0.0).round(3) * 100
        s2 = p1 + p2
        try:
            st_p = int(getattr(jason_settings, "ST_ATR_PERIOD", 14))
            st_m = float(getattr(jason_settings, "ST_ATR_MULTIPLIER", 3.0))
        except Exception:
            st_p, st_m = 14, 3.0
        st_dir = supertrend_direction(df, st_p, st_m) if len(df) > max(2, st_p) else None
        data[tkr] = {
            "df": df,
            "close": close,
            "p1": p1,
            "p2": p2,
            "s2": s2,
            "st_dir": st_dir if st_dir is not None else pd.Series(0, index=df.index),
        }

    if not data:
        return {}

    # 모든 종목의 거래일을 합집합하여 전체 백테스트 기간을 설정합니다.
    union_index = pd.DatetimeIndex([])
    for tkr, d in data.items():
        union_index = union_index.union(d["close"].index)

    if union_index.empty:
        return {}

    # 요청된 시작일 이후로 인덱스를 필터링합니다.
    if core_start_date:
        union_index = union_index[union_index >= core_start_date]

    if union_index.empty:
        return {}

    try:
        # Settings
        # 전략 고유 설정
        sell_thr = float(jason_settings.SELL_SUM_THRESHOLD)
        buy_thr = float(jason_settings.BUY_SUM_THRESHOLD)
    except AttributeError as e:
        raise AttributeError(
            f"'{e.name}' 설정이 logics/jason/settings.py 파일에 반드시 정의되어야 합니다."
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

        # Calculate current holdings value once per day
        current_holdings_value = 0
        for tkr_h, s_h in state.items():
            if s_h["shares"] > 0:
                price_h = today_prices.get(tkr_h)
                if pd.notna(price_h):
                    current_holdings_value += s_h["shares"] * price_h

        # 2. Sells, Trims and State Recording
        equity = cash + current_holdings_value

        for tkr, d in data.items():
            s, price = state[tkr], today_prices.get(tkr)
            decision, trade_amount, trade_profit, trade_pl_pct = None, 0.0, 0.0, 0.0

            # Sell/Trim logic only applies to tickers in the active universe
            if (
                tkr in tickers_available_today
            ):  # 이 조건은 이미 루프 시작에서 확인되지만, 명확성을 위해 유지
                if s["shares"] > 0 and pd.notna(price) and i >= s["sell_block_until"]:
                    hold_ret = (price / s["avg_cost"] - 1.0) * 100.0 if s["avg_cost"] > 0 else 0.0

                    # Sell/Cut conditions
                    if stop_loss is not None and hold_ret <= float(stop_loss):
                        decision = "CUT_STOPLOSS"
                    elif dt in d["s2"].index and (d["s2"].loc[dt] + hold_ret) < sell_thr:
                        decision = "SELL_MOMENTUM"

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

                    # Trim condition
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
            if decision_out in ("WAIT", "HOLD"):
                if s["shares"] > 0 and i < s["sell_block_until"]:
                    note = "매도 쿨다운"
                elif s["shares"] == 0 and i < s["buy_block_until"]:
                    note = "매수 쿨다운"

            # 해당 날짜에 데이터가 있는 경우에만 지표를 기록합니다.
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
                        "signal1": d["p1"].loc[dt],
                        "signal2": d["p2"].loc[dt],
                        "score": d["s2"].loc[dt],
                        "filter": (int(d["st_dir"].loc[dt]) if dt in d["st_dir"].index else 0),
                    }
                )
            else:
                # 데이터가 없는 날은 보유 상태만 기록합니다.
                out_rows[tkr].append(
                    {
                        "date": dt,
                        "price": s["avg_cost"],
                        "shares": s["shares"],
                        "pv": s["shares"] * (s["avg_cost"] if pd.notna(s["avg_cost"]) else 0),
                        "decision": "HOLD" if s["shares"] > 0 else "WAIT",
                        "avg_cost": s["avg_cost"],
                        "trade_amount": 0.0,
                        "trade_profit": 0.0,
                        "trade_pl_pct": 0.0,
                        "note": "데이터 없음",
                        "signal1": 0,
                        "signal2": 0,
                        "score": 0,
                        "filter": 0,
                    }
                )

        # 3. Buys
        held_count = sum(1 for s in state.values() if s["shares"] > 0)
        slots_to_fill = max(0, top_n - held_count)
        if slots_to_fill > 0 and cash > 0:  # 매수 여력 있을 때
            cands = []
            for tkr in tickers_available_today:  # 오늘 거래 가능한 종목 중에서만 후보 선정
                d = data.get(tkr)
                s = state[tkr]
                if s["shares"] == 0 and i >= s["buy_block_until"] and dt in d["s2"].index:
                    s2_val = d["s2"].loc[dt]
                    st_val = int(d["st_dir"].loc[dt]) if dt in d["st_dir"].index else -1
                    if s2_val > buy_thr and st_val > 0:
                        cands.append((s2_val, tkr))

            cands.sort(reverse=True)

            for k in range(min(slots_to_fill, len(cands))):
                if cash <= 0:
                    break
                _, tkr = cands[k]

                price = today_prices.get(tkr)
                if pd.isna(price):
                    continue

                # Equity needs to be updated with the new cash value after sells
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
                s["avg_cost"] = price  # Simple avg_cost for new position
                if cooldown_days > 0:
                    s["sell_block_until"] = max(s["sell_block_until"], i + cooldown_days)

                # Update today's row for the bought ticker
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
                else:  # Should not happen if sell logic ran first
                    pass
        else:  # 매수 여력 없을 때 (포트폴리오 가득 참 또는 현금 부족)
            # 매수 신호가 있었으나 무시된 종목에 사유를 기록
            for tkr in tickers_available_today:  # 오늘 거래 가능한 종목 중에서만 확인
                s = state[tkr]
                if s["shares"] == 0:  # 비보유 종목 중에서
                    d = data.get(tkr)
                    if dt in d["s2"].index and d["s2"].loc[dt] > buy_thr:
                        # 오늘 날짜의 기록을 찾아 'note' 업데이트
                        if out_rows[tkr] and out_rows[tkr][-1]["date"] == dt:
                            note = "포트폴리오 가득 참" if slots_to_fill <= 0 else "현금 부족"
                            out_rows[tkr][-1]["note"] = note

        # 4. Record cash state
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

    # Build DataFrames
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
    개별 종목에 대한 일별 시뮬레이션 상태를 반환합니다.
    (core.backtester.simple_daily_series에서 이동)
    """
    if df is None:
        df = fetch_ohlcv(
            ticker, months_range=months_range, date_range=date_range
        )  # 단일 테스트는 웜업 조정 없이 그대로 사용
    if df is None or len(df) < 25:
        return pd.DataFrame(
            columns=["price", "cash", "shares", "pv", "decision"],
            index=pd.DatetimeIndex([]),
        )

    close = df["Close"]

    # Determine start index respecting the core start date (after warmup)
    start_i = 20
    if core_start_date is not None:
        try:
            if isinstance(df.index, pd.DatetimeIndex):
                start_i = max(start_i, df.index.searchsorted(core_start_date, side="left"))
        except Exception:
            pass

    try:
        # Settings
        # 전략 고유 설정
        sell_thr = float(jason_settings.SELL_SUM_THRESHOLD)
        buy_thr = float(jason_settings.BUY_SUM_THRESHOLD)
        st_p = int(jason_settings.ST_ATR_PERIOD)
        st_m = float(jason_settings.ST_ATR_MULTIPLIER)
    except AttributeError as e:
        raise AttributeError(
            f"'{e.name}' 설정이 logics/jason/settings.py 파일에 반드시 정의되어야 합니다."
        ) from e

    try:
        # 전역 설정
        stop_loss = global_settings.HOLDING_STOP_LOSS_PCT
        cooldown_days = int(global_settings.COOLDOWN_DAYS)
    except AttributeError as e:
        raise AttributeError(
            f"'{e.name}' 설정이 전역 settings.py 파일에 반드시 정의되어야 합니다."
        ) from e

    st_dir = supertrend_direction(df, st_p, st_m) if len(df) > max(2, st_p) else None

    # State
    cash = float(initial_capital)
    shares = 0
    avg_cost = 0.0
    buy_block_until = -1
    sell_block_until = -1

    rows = []
    for i in range(start_i, len(df)):
        c0_val = close.iloc[i]
        if pd.isna(c0_val):
            continue  # 데이터가 없는 날은 건너뜁니다.
        c0 = float(c0_val)
        c5 = float(close.iloc[i - 5])
        c10 = float(close.iloc[i - 10])
        r21 = (c5 / c10 - 1.0) if c10 > 0 else 0.0
        r10 = (c0 / c5 - 1.0) if c5 > 0 else 0.0
        p1 = round(r10 * 100, 1)
        p2 = round(r21 * 100, 1)
        s2_sum = p1 + p2

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
                decision = "CUT_STOPLOSS"
            # 2) Sum threshold
            elif (s2_sum + curr_hold_ret) < sell_thr:
                decision = "SELL_MOMENTUM"

            if decision in ("CUT_STOPLOSS", "SELL_MOMENTUM"):
                prev_shares = shares
                trade_amount = prev_shares * c0
                if avg_cost > 0:
                    trade_profit = (c0 - avg_cost) * prev_shares
                    trade_pl_pct = (c0 / avg_cost - 1.0) * 100.0
                cash += trade_amount
                shares = 0
                avg_cost = 0.0
                if cooldown_days > 0:
                    buy_block_until = i + cooldown_days

        # Buy logic
        if decision is None and shares == 0 and i >= buy_block_until and s2_sum > buy_thr:
            if st_dir is not None and int(st_dir.iloc[i]) <= 0:
                reason_block = "ST_DOWN"
            if reason_block is None:
                buy_qty = int(cash // c0)
                if buy_qty > 0:
                    trade_amount = buy_qty * c0
                    cash -= trade_amount
                    new_shares = shares + buy_qty
                    avg_cost = (
                        ((avg_cost * shares) + (c0 * buy_qty)) / new_shares
                        if new_shares > 0
                        else 0.0
                    )
                    shares = new_shares
                    decision = "BUY"
                    if cooldown_days > 0:
                        sell_block_until = i + cooldown_days

        if decision is None:
            decision = "HOLD" if shares > 0 else "WAIT"

        pv = cash + shares * c0
        note_txt = ""
        if decision in ("WAIT", "HOLD"):
            if shares > 0 and (i < sell_block_until):
                note_txt = "매도 쿨다운"
            elif shares == 0 and (i < buy_block_until):
                note_txt = "매수 쿨다운"
            elif reason_block == "ST_DOWN":
                note_txt = "ST 하향"

        rows.append(
            {
                "date": df.index[i],
                "price": c0,
                "cash": cash,
                "shares": shares,
                "pv": pv,
                "decision": decision,
                "avg_cost": avg_cost,
                "trade_amount": trade_amount,
                "trade_profit": trade_profit,
                "trade_pl_pct": trade_pl_pct,
                "note": note_txt,
                "signal1": p1,
                "signal2": p2,
                "score": s2_sum,
                "filter": int(st_dir.iloc[i]) if st_dir is not None else 0,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["price", "cash", "shares", "pv", "decision"],
            index=pd.DatetimeIndex([]),
        )
    return pd.DataFrame(rows).set_index("date")
