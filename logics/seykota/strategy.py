"""
전략: 'seykota'
이동평균선 교차를 사용하는 에드 세이코타 스타일의 추세추종 전략입니다.
"""

from math import ceil
from typing import Dict, List, Optional, Tuple

import pandas as pd

import settings as global_settings
from utils.data_loader import fetch_ohlcv

from . import settings as seykota_settings


def run_portfolio_backtest(
    ticker_name_pairs: List[Tuple[str, str]],
    months_range: Optional[List[int]] = None,
    initial_capital: float = 100_000_000.0,
    core_start_date: Optional[pd.Timestamp] = None,
    top_n: int = 10,
    date_range: Optional[List[str]] = None,
    country: str = "kor",
    prefetched_data: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    이동평균선 교차 전략을 사용하여 Top-N 포트폴리오를 시뮬레이션합니다.
    """
    # Settings
    try:
        # 전략 고유 설정
        fast_ma_period = int(seykota_settings.SEYKOTA_FAST_MA)
        slow_ma_period = int(seykota_settings.SEYKOTA_SLOW_MA)
    except AttributeError as e:
        raise AttributeError(
            f"'{e.name}' 설정이 logics/seykota/settings.py 파일에 반드시 정의되어야 합니다."
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
    # 웜업 기간을 포함하여 데이터 조회 시작일을 계산합니다.
    # 거래일이 아닌 날(주말, 공휴일)을 고려하여 1.5배의 버퍼를 줍니다.
    warmup_days = int(slow_ma_period * 1.5)

    adjusted_date_range = None
    if date_range and len(date_range) == 2:
        core_start = pd.to_datetime(date_range[0])
        warmup_start = core_start - pd.DateOffset(days=warmup_days)
        adjusted_date_range = [warmup_start.strftime("%Y-%m-%d"), date_range[1]]

    # Fetch all series and precompute indicators
    data_by_ticker = {}
    tickers_to_process = [p[0] for p in ticker_name_pairs]

    for tkr in tickers_to_process:
        # 미리 로드된 데이터가 있으면 사용하고, 없으면 새로 조회합니다.
        if prefetched_data and tkr in prefetched_data:
            df = prefetched_data[tkr].copy()
        else:
            df = fetch_ohlcv(tkr, country=country, date_range=adjusted_date_range)

        if df is None:
            continue

        # yfinance가 가끔 MultiIndex 컬럼을 반환하는 경우에 대비하여,
        # 컬럼을 단순화하고 중복을 제거합니다.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            df = df.loc[:, ~df.columns.duplicated()]

        if len(df) < slow_ma_period:
            continue

        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        fast_ma = close.rolling(window=fast_ma_period).mean()
        slow_ma = close.rolling(window=slow_ma_period).mean()
        ma_score = (fast_ma / slow_ma - 1.0).fillna(0.0)

        data_by_ticker[tkr] = {
            "df": df,
            "close": close,
            "fast_ma": fast_ma,
            "slow_ma": slow_ma,
            "ma_score": ma_score,
        }

    if not data_by_ticker:
        return {}

    # 모든 종목의 거래일을 합집합하여 전체 백테스트 기간을 설정합니다.
    union_index = pd.DatetimeIndex([])
    for tkr, d in data_by_ticker.items():
        union_index = union_index.union(d["close"].index)

    if union_index.empty:
        return {}

    # 요청된 시작일 이후로 인덱스를 필터링합니다.
    if core_start_date:
        union_index = union_index[union_index >= core_start_date]

    # 웜업 기간을 고려하여 실제 시작 인덱스를 조정합니다.
    # 이 부분은 루프 내에서 처리하므로, union_index 자체는 그대로 둡니다.
    if union_index.empty:
        return {}

    # 시뮬레이션 상태 변수 초기화
    state = {
        tkr: {
            "shares": 0,
            "avg_cost": 0.0,
            "buy_block_until": -1,
            "sell_block_until": -1,
        }
        for tkr in data_by_ticker.keys()
    }
    cash = float(initial_capital)
    out_rows = {tkr: [] for tkr in data_by_ticker.keys()}
    out_cash = []

    # 일별 루프를 돌며 시뮬레이션을 실행합니다.
    for i, dt in enumerate(union_index):
        tickers_available_today = [tkr for tkr, d in data_by_ticker.items() if dt in d["df"].index]
        today_prices = {
            tkr: float(d["close"].loc[dt]) if pd.notna(d["close"].loc[dt]) else None
            for tkr, d in data_by_ticker.items()
            if dt in d["close"].index
        }

        # Calculate current holdings value once per day
        current_holdings_value = 0.0
        for tkr_h, s_h in state.items():
            if s_h["shares"] > 0:
                price_h = today_prices.get(tkr_h)
                if pd.notna(price_h):
                    current_holdings_value += s_h["shares"] * price_h

        # Sells, Trims and State Recording
        equity = cash + current_holdings_value

        for tkr, d in data_by_ticker.items():
            ticker_state, price = state[tkr], today_prices.get(tkr)
            decision, trade_amount, trade_profit, trade_pl_pct = None, 0.0, 0.0, 0.0

            is_ticker_warming_up = tkr not in tickers_available_today or pd.isna(
                d["slow_ma"].get(dt)
            )

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
                    # MA 값이 유효한 경우에만 데드크로스 판단
                    elif (
                        pd.notna(d["fast_ma"].get(dt))
                        and pd.notna(d["slow_ma"].get(dt))
                        and d["fast_ma"].loc[dt] < d["slow_ma"].loc[dt]
                    ):
                        decision = "SELL_TREND"

                    if decision:
                        qty = ticker_state["shares"]
                        trade_amount = qty * price
                        if ticker_state["avg_cost"] > 0:
                            trade_profit = (price - ticker_state["avg_cost"]) * qty
                            trade_pl_pct = hold_ret
                        cash += trade_amount
                        ticker_state["shares"], ticker_state["avg_cost"] = 0, 0.0
                        if cooldown_days > 0:
                            ticker_state["buy_block_until"] = i + cooldown_days

                    elif trim_on and max_pos_pct < 1.0:
                        curr_val, cap_val = ticker_state["shares"] * price, max_pos_pct * equity
                        if curr_val > cap_val and price > 0:
                            sell_qty = min(
                                ticker_state["shares"], int((curr_val - cap_val) // price)
                            )
                            if sell_qty > 0:
                                decision, trade_amount = (
                                    "TRIM_REBALANCE",
                                    sell_qty * price,
                                )
                                cash += trade_amount
                                ticker_state["shares"] -= sell_qty
                                if ticker_state["avg_cost"] > 0:
                                    trade_profit = (price - ticker_state["avg_cost"]) * sell_qty
                                    trade_pl_pct = hold_ret
                                if cooldown_days > 0:
                                    ticker_state["buy_block_until"] = i + cooldown_days

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
                        "signal1": d["fast_ma"].loc[dt],
                        "signal2": d["slow_ma"].loc[dt],
                        "score": d["ma_score"].loc[dt],
                        "filter": None,
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
                        "signal1": 0,
                        "signal2": 0,
                        "score": 0,
                        "filter": None,
                    }
                )

        # 매수 로직
        held_count = sum(1 for s in state.values() if s["shares"] > 0)
        slots_to_fill = max(0, top_n - held_count)
        if slots_to_fill > 0 and cash > 0:  # 매수 여력 있을 때
            buy_candidates = []
            for tkr in tickers_available_today:  # 오늘 거래 가능한 종목 중에서만 후보 선정
                d = data_by_ticker.get(tkr)
                ticker_state = state[tkr]
                if (
                    ticker_state["shares"] == 0
                    and i >= ticker_state["buy_block_until"]
                    and pd.notna(d["fast_ma"].get(dt))
                    and pd.notna(d["slow_ma"].get(dt))
                ):
                    if d["fast_ma"].loc[dt] > d["slow_ma"].loc[dt]:
                        buy_candidates.append((d["ma_score"].loc[dt], tkr))

            buy_candidates.sort(reverse=True)

            for k in range(min(slots_to_fill, len(buy_candidates))):
                if cash <= 0:
                    break
                _, tkr = buy_candidates[k]

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

                ticker_state = state[tkr]
                cash -= trade_amount
                ticker_state["shares"] += req_qty
                ticker_state["avg_cost"] = price
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
        else:  # 매수 여력 없을 때 (포트폴리오 가득 참 또는 현금 부족)
            # 매수 신호가 있었으나 무시된 종목에 사유를 기록
            for tkr in tickers_available_today:  # 오늘 거래 가능한 종목 중에서만 확인
                ticker_state = state[tkr]
                if ticker_state["shares"] == 0:  # 비보유 종목 중에서
                    d = data_by_ticker.get(tkr)
                    if (
                        pd.notna(d["fast_ma"].get(dt))
                        and pd.notna(d["slow_ma"].get(dt))
                        and d["fast_ma"].loc[dt] > d["slow_ma"].loc[dt]
                    ):
                        # 오늘 날짜의 기록을 찾아 'note' 업데이트
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
        fast_ma_period = int(seykota_settings.SEYKOTA_FAST_MA)
        slow_ma_period = int(seykota_settings.SEYKOTA_SLOW_MA)
    except AttributeError as e:
        raise AttributeError(
            f"'{e.name}' 설정이 logics/seykota/settings.py 파일에 반드시 정의되어야 합니다."
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

    if df is None:
        return pd.DataFrame()

    # yfinance가 가끔 MultiIndex 컬럼을 반환하는 경우에 대비하여,
    # 컬럼을 단순화하고 중복을 제거합니다.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        df = df.loc[:, ~df.columns.duplicated()]
    if len(df) < slow_ma_period:
        return pd.DataFrame()

    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    fast_ma = close.rolling(window=fast_ma_period).mean()
    slow_ma = close.rolling(window=slow_ma_period).mean()

    loop_start_index = 0
    if core_start_date is not None:
        try:
            loop_start_index = df.index.searchsorted(core_start_date, side="left")
        except Exception:
            pass

    # 시뮬레이션 상태 변수 초기화
    cash = float(initial_capital)
    shares = 0
    avg_cost = 0.0
    buy_block_until = -1
    sell_block_until = -1

    rows = []
    for i in range(loop_start_index, len(df)):
        price_val = close.iloc[i]
        if pd.isna(price_val):
            continue  # 데이터가 없는 날은 건너뜁니다.
        price = float(price_val)
        decision, trade_amount, trade_profit, trade_pl_pct = None, 0.0, 0.0, 0.0

        # 웜업 확인: 장기 이동평균이 계산되기 전까지는 거래하지 않습니다.
        if pd.isna(slow_ma.iloc[i]):
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
                    "p1": 0,
                    "p2": 0,
                    "s2_sum": 0,
                    "st_dir": 0,
                }
            )
            continue

        # 매도/손절 로직
        if shares > 0 and i >= sell_block_until:
            hold_ret = (price / avg_cost - 1.0) * 100.0 if avg_cost > 0 else 0.0
            if stop_loss is not None and hold_ret <= float(stop_loss):
                decision = "CUT_STOPLOSS"
            elif fast_ma.iloc[i] < slow_ma.iloc[i]:
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

        # 매수 로직
        if decision is None and shares == 0 and i >= buy_block_until:
            if fast_ma.iloc[i] > slow_ma.iloc[i]:
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
                "signal1": fast_ma.iloc[i],
                "signal2": slow_ma.iloc[i],
                "score": (
                    (fast_ma.iloc[i] / slow_ma.iloc[i] - 1.0) * 100.0
                    if slow_ma.iloc[i] > 0
                    else 0.0
                ),
                "filter": None,
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("date")
