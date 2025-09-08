"""
전략: 'donchian'
단일 이동평균선을 사용하는 리처드 돈치안 스타일의 추세추종 전략입니다.
"""

from math import ceil
from typing import Dict, List, Optional, Tuple

import pandas as pd

import settings as global_settings
from utils.data_loader import fetch_ohlcv

from . import settings as donchian_settings


def run_portfolio_backtest(
    ticker_name_pairs: List[Tuple[str, str]],
    months_range: Optional[List[int]] = None,
    initial_capital: float = 100_000_000.0,
    core_start_date: Optional[pd.Timestamp] = None,
    top_n: int = 10,
    date_range: Optional[List[str]] = None,
    country: str = "kor",
) -> Dict[str, pd.DataFrame]:
    """
    단일 이동평균선 교차 전략을 사용하여 Top-N 포트폴리오를 시뮬레이션합니다.
    """
    # Settings
    try:
        # 전략 고유 설정
        ma_period = int(donchian_settings.DONCHIAN_MA_PERIOD)
        entry_delay_days = int(getattr(donchian_settings, "DONCHIAN_ENTRY_DELAY_DAYS", 0))
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
    # 웜업 기간을 포함하여 전체 기간에 대한 데이터를 로딩합니다.
    warmup_days = int(ma_period * 1.5)

    adjusted_date_range = None
    if date_range and len(date_range) == 2:
        core_start = pd.to_datetime(date_range[0])
        warmup_start = core_start - pd.DateOffset(days=warmup_days)
        adjusted_date_range = [warmup_start.strftime("%Y-%m-%d"), date_range[1]]

    data = {}
    tickers_to_process = [p[0] for p in ticker_name_pairs]

    for tkr in tickers_to_process:
        df = fetch_ohlcv(tkr, country=country, date_range=adjusted_date_range)

        # yfinance가 가끔 MultiIndex 컬럼을 반환하는 경우에 대비하여,
        # 컬럼을 단순화하고 중복을 제거합니다.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            df = df.loc[:, ~df.columns.duplicated()]
        if df is None or len(df) < ma_period:
            continue

        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        ma = close.rolling(window=ma_period).mean()
        ma_score = (close / ma - 1.0) * 100.0 if ma is not None else 0.0

        # 이동평균선 위에 주가가 머무른 연속된 일수를 계산합니다.
        buy_signal_active = close > ma
        buy_signal_days = (
            buy_signal_active.groupby((buy_signal_active != buy_signal_active.shift()).cumsum())
            .cumsum()
            .fillna(0)
            .astype(int)
        )

        data[tkr] = {
            "df": df,
            "close": df["Close"],
            "ma": ma,
            "ma_score": ma_score,
            "buy_signal_days": buy_signal_days,
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

    # 시뮬레이션 상태 변수 초기화
    state = {
        tkr: {
            "shares": 0,
            "avg_cost": 0.0,
            "buy_block_until": -1,
            "sell_block_until": -1,
            "peak_high_since_buy": 0.0,
        }
        for tkr in data.keys()
    }
    cash = float(initial_capital)
    out_rows = {tkr: [] for tkr in data.keys()}
    out_cash = []

    # 일별 루프를 돌며 시뮬레이션을 실행합니다.
    for i, dt in enumerate(union_index):
        tickers_available_today = [tkr for tkr, d in data.items() if dt in d["df"].index]
        today_prices = {
            tkr: float(d["close"].loc[dt]) if pd.notna(d["close"].loc[dt]) else None
            for tkr, d in data.items()
            if dt in d["close"].index
        }
        today_highs = {
            tkr: float(d["df"]["High"].loc[dt]) if pd.notna(d["df"]["High"].loc[dt]) else None
            for tkr, d in data.items()
            if dt in d["df"].index
        }

        # 현재 총 보유 자산 가치를 계산합니다.
        current_holdings_value = 0
        for tkr_h, s_h in state.items():
            if s_h["shares"] > 0:
                price_h = today_prices.get(tkr_h)
                if pd.notna(price_h):
                    current_holdings_value += s_h["shares"] * price_h

        # 총 평가금액(현금 + 주식)을 계산합니다.
        equity = cash + current_holdings_value

        for tkr, d in data.items():
            ticker_state, price = state[tkr], today_prices.get(tkr)
            today_high = today_highs.get(tkr)
            decision, trade_amount, trade_profit, trade_pl_pct = None, 0.0, 0.0, 0.0

            is_ticker_warming_up = tkr not in tickers_available_today or pd.isna(d["ma"].get(dt))

            if tkr in tickers_available_today:
                if (
                    ticker_state["shares"] > 0
                    and pd.notna(price)
                    and i >= ticker_state["sell_block_until"]
                    and not is_ticker_warming_up
                ):
                    hold_ret = (
                        (price / ticker_state["avg_cost"] - 1.0) * 100.0
                        if ticker_state["avg_cost"] > 0
                        else 0.0
                    )

                    if stop_loss is not None and hold_ret <= float(stop_loss):
                        decision = "CUT_STOPLOSS"
                    elif not is_ticker_warming_up and price < d["ma"].loc[dt]:
                        decision = "SELL_TREND"

                    # 매도 결정이 내려졌을 경우, 상태를 업데이트합니다.
                    if decision:
                        qty = ticker_state["shares"]
                        trade_amount = qty * price
                        if ticker_state["avg_cost"] > 0:
                            trade_profit = (price - ticker_state["avg_cost"]) * qty
                            trade_pl_pct = hold_ret
                        cash += trade_amount
                        ticker_state["shares"], ticker_state["avg_cost"] = 0, 0.0
                        ticker_state["peak_high_since_buy"] = 0.0
                        if cooldown_days > 0:
                            ticker_state["buy_block_until"] = i + cooldown_days

                    # 최대 비중 초과 시 리밸런싱(부분 매도)
                    elif trim_on and max_pos_pct < 1.0:
                        curr_val, cap_val = ticker_state["shares"] * price, max_pos_pct * equity
                        if curr_val > cap_val and price > 0:
                            sell_qty = min(
                                ticker_state["shares"], int((curr_val - cap_val) // price)
                            )
                            if sell_qty > 0:
                                decision, trade_amount = ("TRIM_REBALANCE", sell_qty * price)
                                cash += trade_amount
                                ticker_state["shares"] -= sell_qty
                                if ticker_state["avg_cost"] > 0:
                                    trade_profit = (price - ticker_state["avg_cost"]) * sell_qty
                                    trade_pl_pct = hold_ret
                                if cooldown_days > 0:
                                    ticker_state["buy_block_until"] = i + cooldown_days

            # 보유 종목의 고점 대비 하락률 계산
            drawdown_from_peak = None
            if ticker_state["shares"] > 0 and pd.notna(price) and pd.notna(today_high):
                # 최고가 업데이트
                ticker_state["peak_high_since_buy"] = max(
                    ticker_state["peak_high_since_buy"], today_high
                )
                if ticker_state["peak_high_since_buy"] > 0:
                    drawdown_from_peak = ((price / ticker_state["peak_high_since_buy"]) - 1.0) * 100.0

            decision_out = (
                decision if decision else ("HOLD" if ticker_state["shares"] > 0 else "WAIT")
            )
            note = ""
            if is_ticker_warming_up:
                note = "웜업 기간"
            elif decision_out in ("WAIT", "HOLD"):
                if ticker_state["shares"] > 0 and i < ticker_state["sell_block_until"]:
                    note = "매도 쿨다운"
                elif ticker_state["shares"] == 0 and i < ticker_state["buy_block_until"]:
                    note = "매수 쿨다운"

            if tkr in tickers_available_today:
                out_rows[tkr].append(
                    {
                        "date": dt,
                        "price": price,
                        "shares": ticker_state["shares"],
                        "pv": ticker_state["shares"] * (price if pd.notna(price) else 0),
                        "decision": decision_out,
                        "avg_cost": ticker_state["avg_cost"],
                        "trade_amount": trade_amount,
                        "trade_profit": trade_profit,
                        "trade_pl_pct": trade_pl_pct,
                        "note": note,
                        "signal1": d["ma"].get(dt),  # 이평선(값)
                        "signal2": drawdown_from_peak,  # 고점대비
                        "score": d["ma_score"].loc[dt],
                        "filter": d["buy_signal_days"].get(dt),
                    }
                )
            else:
                out_rows[tkr].append(
                    {
                        "date": dt,
                        "price": ticker_state["avg_cost"],
                        "shares": ticker_state["shares"],
                        "pv": ticker_state["shares"]
                        * (ticker_state["avg_cost"] if pd.notna(ticker_state["avg_cost"]) else 0.0),
                        "decision": "HOLD" if ticker_state["shares"] > 0 else "WAIT",
                        "avg_cost": ticker_state["avg_cost"],
                        "trade_amount": 0.0,
                        "trade_profit": 0.0,
                        "trade_pl_pct": 0.0,
                        "note": "데이터 없음",
                        "signal1": None,  # 이평선(값)
                        "signal2": None,  # 고점대비
                        "score": None,
                        "filter": None,
                    }
                )

        # 매수 로직
        held_count = sum(1 for s in state.values() if s["shares"] > 0)
        slots_to_fill = max(0, top_n - held_count)
        if slots_to_fill > 0 and cash > 0:
            buy_candidates = []
            for tkr in tickers_available_today:
                d = data.get(tkr)
                ticker_state = state[tkr]
                buy_signal_days_today = d["buy_signal_days"].get(dt, 0)

                if (
                    ticker_state["shares"] == 0
                    and i >= ticker_state["buy_block_until"]
                    and buy_signal_days_today > entry_delay_days
                ):
                    score_cand = d["ma_score"].get(dt, 0.0)
                    buy_candidates.append((score_cand, tkr))

            buy_candidates.sort(reverse=True)

            # 점수가 높은 순으로 매수 후보를 선정하여 매수합니다.
            for k in range(min(slots_to_fill, len(buy_candidates))):
                if cash <= 0:
                    break
                _, tkr = buy_candidates[k]

                price = today_prices.get(tkr)
                today_high = today_highs.get(tkr)
                if pd.isna(price) or pd.isna(today_high):
                    continue

                equity = cash + current_holdings_value
                min_val = min_pos_pct * equity

                req_qty = ceil(min_val / price) if price > 0 else 0
                if req_qty <= 0:
                    continue

                trade_amount = req_qty * price
                if trade_amount > cash:
                    continue

                ticker_state = state[tkr]
                cash -= trade_amount
                ticker_state["shares"] += req_qty
                ticker_state["avg_cost"] = price
                ticker_state["peak_high_since_buy"] = today_high
                if cooldown_days > 0:
                    ticker_state["sell_block_until"] = max(
                        ticker_state["sell_block_until"], i + cooldown_days
                    )

                if out_rows[tkr] and out_rows[tkr][-1]["date"] == dt:
                    row = out_rows[tkr][-1]
                    row.update(
                        {
                            "decision": "BUY",
                            "trade_amount": trade_amount,
                            "shares": ticker_state["shares"],
                            "pv": ticker_state["shares"] * price,
                            "avg_cost": ticker_state["avg_cost"],
                        }
                    )
        else:
            for tkr in tickers_available_today:
                ticker_state = state[tkr]
                if ticker_state["shares"] == 0:
                    d = data.get(tkr)
                    buy_signal_days_today = d["buy_signal_days"].get(dt, 0)

                    if buy_signal_days_today > entry_delay_days:
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
    country: str = "kor",
) -> pd.DataFrame:
    """
    단일 종목에 대해 이동평균선 교차 전략 백테스트를 실행합니다.
    """
    try:
        # 전략 고유 설정
        ma_period = int(donchian_settings.DONCHIAN_MA_PERIOD)
        entry_delay_days = int(getattr(donchian_settings, "DONCHIAN_ENTRY_DELAY_DAYS", 0))
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
        df = fetch_ohlcv(ticker, country=country, date_range=date_range)

    # yfinance가 가끔 MultiIndex 컬럼을 반환하는 경우에 대비하여,
    # 컬럼을 단순화하고 중복을 제거합니다.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        df = df.loc[:, ~df.columns.duplicated()]
    if df is None or df.empty or len(df) < ma_period:
        return pd.DataFrame()

    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    ma = close.rolling(window=ma_period).mean()

    buy_signal_active = close > ma
    buy_signal_days = (
        buy_signal_active.groupby((buy_signal_active != buy_signal_active.shift()).cumsum())
        .cumsum()
        .fillna(0)
        .astype(int)
    )

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
    peak_high_since_buy = 0.0

    rows = []
    for i in range(loop_start_index, len(df)):
        price_val = close.iloc[i]
        high_val = df["High"].iloc[i]
        if pd.isna(price_val) or pd.isna(high_val):
            continue
        price = float(price_val)
        high_today = float(high_val)
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
                    "signal1": ma_today,
                    "signal2": None,
                    "score": None,
                    "filter": buy_signal_days.iloc[i],
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
                peak_high_since_buy = 0.0
                if cooldown_days > 0:
                    buy_block_until = i + cooldown_days

        if decision is None and shares == 0 and i >= buy_block_until:
            buy_signal_days_today = buy_signal_days.iloc[i]
            if buy_signal_days_today > entry_delay_days:
                buy_qty = int(cash // price)
                if buy_qty > 0:
                    trade_amount = buy_qty * price
                    cash -= trade_amount
                    avg_cost, shares = price, buy_qty
                    peak_high_since_buy = high_today
                    decision = "BUY"
                    if cooldown_days > 0:
                        sell_block_until = i + cooldown_days

        if decision is None:
            decision = "HOLD" if shares > 0 else "WAIT"

        # 보유 종목의 고점 대비 하락률 계산
        drawdown_from_peak = None
        if shares > 0:
            peak_high_since_buy = max(peak_high_since_buy, high_today)
            if peak_high_since_buy > 0:
                drawdown_from_peak = ((price / peak_high_since_buy) - 1.0) * 100.0

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
                "signal2": drawdown_from_peak,
                "score": ((price / ma_today - 1.0) * 100.0 if ma_today > 0 else 0.0),
                "filter": buy_signal_days.iloc[i],
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("date")
