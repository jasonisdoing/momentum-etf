"""
전략: 'jason'
모멘텀 점수와 슈퍼트렌드 지표를 사용하는 전략입니다.
"""

from math import ceil
from typing import Dict, List, Optional, Tuple

import pandas as pd

import settings as global_settings
from utils.data_loader import fetch_ohlcv
from utils.indicators import supertrend_direction

from . import settings as jason_settings


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
    'jason' 전략을 사용하여 Top-N 포트폴리오를 시뮬레이션합니다.
    """
    # Fetch all series
    data = {}
    tickers_to_process = [p[0] for p in ticker_name_pairs]

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
        df = fetch_ohlcv(tkr, country=country, date_range=adjusted_date_range)

        # yfinance가 가끔 MultiIndex 컬럼을 반환하는 경우에 대비하여,
        # 컬럼을 단순화하고 중복을 제거합니다.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            df = df.loc[:, ~df.columns.duplicated()]

        if df is None or len(df) < 25:
            continue
        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        # 모멘텀 점수 계산을 위한 수익률 사전 계산
        return_1w = (close / close.shift(5) - 1.0).fillna(0.0).round(3) * 100
        return_2w = (close.shift(5) / close.shift(10) - 1.0).fillna(0.0).round(3) * 100
        momentum_score = return_1w + return_2w
        try:
            st_p = int(getattr(jason_settings, "ST_ATR_PERIOD", 14))
            st_m = float(getattr(jason_settings, "ST_ATR_MULTIPLIER", 3.0))
        except Exception:
            st_p, st_m = 14, 3.0
        supertrend = supertrend_direction(df, st_p, st_m) if len(df) > max(2, st_p) else None
        data[tkr] = {
            "df": df,
            "close": close,
            "return_1w": return_1w,
            "return_2w": return_2w,
            "momentum_score": momentum_score,
            "supertrend": supertrend if supertrend is not None else pd.Series(0, index=df.index),
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

    # 시뮬레이션 상태 변수 초기화
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

    # 일별 루프를 돌며 시뮬레이션을 실행합니다.
    for i, dt in enumerate(union_index):
        tickers_available_today = [tkr for tkr, d in data.items() if dt in d["df"].index]
        today_prices = {
            tkr: float(d["close"].loc[dt]) if pd.notna(d["close"].loc[dt]) else None
            for tkr, d in data.items()
            if dt in d["close"].index
        }

        # 현재 총 보유 자산 가치를 하루에 한 번 계산합니다.
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
            decision, trade_amount, trade_profit, trade_pl_pct = None, 0.0, 0.0, 0.0

            # 매도/비중축소 로직은 활성 유니버스에 있는 티커에만 적용됩니다.
            if tkr in tickers_available_today:
                if (
                    ticker_state["shares"] > 0
                    and pd.notna(price)
                    and i >= ticker_state["sell_block_until"]
                ):
                    hold_ret = (
                        (price / ticker_state["avg_cost"] - 1.0) * 100.0
                        if ticker_state["avg_cost"] > 0
                        else 0.0
                    )

                    # 매도/손절 조건
                    if stop_loss is not None and hold_ret <= float(stop_loss):
                        decision = "CUT_STOPLOSS"
                    elif (
                        dt in d["momentum_score"].index
                        and (d["momentum_score"].loc[dt] + hold_ret) < sell_thr
                    ):
                        decision = "SELL_MOMENTUM"

                    # 매도 결정이 내려졌을 경우, 상태를 업데이트합니다.
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

                    # 최대 비중 초과 시 리밸런싱(부분 매도)
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
            if decision_out in ("WAIT", "HOLD"):
                if ticker_state["shares"] > 0 and i < ticker_state["sell_block_until"]:
                    note = "매도 쿨다운"
                elif ticker_state["shares"] == 0 and i < ticker_state["buy_block_until"]:
                    note = "매수 쿨다운"

            # 해당 날짜에 데이터가 있는 경우에만 지표를 기록합니다.
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
                        "signal1": d["return_1w"].loc[dt],
                        "signal2": d["return_2w"].loc[dt],
                        "score": d["momentum_score"].loc[dt],
                        "filter": (
                            int(d["supertrend"].loc[dt]) if dt in d["supertrend"].index else 0
                        ),
                    }
                )
            else:
                # 데이터가 없는 날은 보유 상태만 기록합니다.
                out_rows[tkr].append(
                    {
                        "date": dt,
                        "price": ticker_state["avg_cost"],
                        "shares": ticker_state["shares"],
                        "pv": ticker_state["shares"]
                        * (ticker_state["avg_cost"] if pd.notna(ticker_state["avg_cost"]) else 0),
                        "decision": "HOLD" if ticker_state["shares"] > 0 else "WAIT",
                        "avg_cost": ticker_state["avg_cost"],
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

        # 매수 로직
        held_count = sum(1 for s in state.values() if s["shares"] > 0)
        slots_to_fill = max(0, top_n - held_count)
        if slots_to_fill > 0 and cash > 0:  # 매수 여력 있을 때
            buy_candidates = []
            for tkr in tickers_available_today:  # 오늘 거래 가능한 종목 중에서만 후보 선정
                d = data.get(tkr)
                ticker_state = state[tkr]
                if (
                    ticker_state["shares"] == 0
                    and i >= ticker_state["buy_block_until"]
                    and dt in d["momentum_score"].index
                ):
                    score_val = d["momentum_score"].loc[dt]
                    st_val = int(d["supertrend"].loc[dt]) if dt in d["supertrend"].index else -1
                    if score_val > buy_thr and st_val > 0:
                        buy_candidates.append((score_val, tkr))

            buy_candidates.sort(reverse=True)

            # 점수가 높은 순으로 매수 후보를 선정하여 매수합니다.
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

                # Update today's row for the bought ticker
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
                else:  # Should not happen if sell logic ran first
                    pass
        else:  # 매수 여력 없을 때 (포트폴리오 가득 참 또는 현금 부족)
            # 매수 신호가 있었으나 무시된 종목에 사유를 기록
            for tkr in tickers_available_today:  # 오늘 거래 가능한 종목 중에서만 확인
                ticker_state = state[tkr]
                if ticker_state["shares"] == 0:  # 비보유 종목 중에서
                    d = data.get(tkr)
                    if dt in d["momentum_score"].index and d["momentum_score"].loc[dt] > buy_thr:
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

    # 결과 데이터프레임 생성
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
    단일 종목에 대해 'jason' 전략 백테스트를 실행합니다.
    """
    if df is None:
        # 단일 테스트는 웜업 조정 없이 그대로 사용
        df = fetch_ohlcv(
            ticker, country=country, months_range=months_range, date_range=date_range
        )

    # yfinance가 가끔 MultiIndex 컬럼을 반환하는 경우에 대비하여,
    # 컬럼을 단순화하고 중복을 제거합니다.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        df = df.loc[:, ~df.columns.duplicated()]
    if df is None or len(df) < 25:
        return pd.DataFrame(
            columns=["price", "cash", "shares", "pv", "decision"],
            index=pd.DatetimeIndex([]),
        )

    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    # 웜업 기간을 고려하여 실제 시작 인덱스를 결정합니다.
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

    # 시뮬레이션 상태 변수 초기화
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
        return_1w_val = (c0 / c5 - 1.0) if c5 > 0 else 0.0
        return_2w_val = (c5 / c10 - 1.0) if c10 > 0 else 0.0
        return_1w = round(return_1w_val * 100, 1)
        return_2w = round(return_2w_val * 100, 1)
        momentum_score = return_1w + return_2w

        decision = None
        trade_amount = 0.0
        trade_profit = 0.0
        trade_pl_pct = 0.0
        reason_block = None

        # 매도/손절 로직
        if shares > 0 and i >= sell_block_until:
            curr_hold_ret = (c0 - avg_cost) / avg_cost * 100.0 if avg_cost > 0 else 0.0
            # 1) 손절매
            if stop_loss is not None and curr_hold_ret <= float(stop_loss):
                decision = "CUT_STOPLOSS"
            # 2) 모멘텀 소진
            elif (momentum_score + curr_hold_ret) < sell_thr:
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

        # 매수 로직
        if decision is None and shares == 0 and i >= buy_block_until and momentum_score > buy_thr:
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
                "signal1": return_1w,
                "signal2": return_2w,
                "score": momentum_score,
                "filter": int(st_dir.iloc[i]) if st_dir is not None else 0,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["price", "cash", "shares", "pv", "decision"],
            index=pd.DatetimeIndex([]),
        )
    return pd.DataFrame(rows).set_index("date")
