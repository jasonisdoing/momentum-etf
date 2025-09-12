"""
단일 이동평균선을 사용하는 추세추종 전략입니다.
"""

from math import ceil
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.data_loader import fetch_ohlcv

from . import settings


def run_portfolio_backtest(
    stocks: List[Dict],
    months_range: Optional[List[int]] = None,
    initial_capital: float = 100_000_000.0,
    core_start_date: Optional[pd.Timestamp] = None,
    top_n: int = 10,
    date_range: Optional[List[str]] = None,
    country: str = "kor",
    prefetched_data: Optional[Dict[str, pd.DataFrame]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    단일 이동평균선 교차 전략을 사용하여 Top-N 포트폴리오를 시뮬레이션합니다.
    """
    # Settings
    try:
        # 전략 고유 설정
        ma_period_etf = int(settings.MA_PERIOD_FOR_ETF)
        ma_period_stock = int(settings.MA_PERIOD_FOR_STOCK)
        replace_weaker_stock = bool(
            getattr(settings, "REPLACE_WEAKER_STOCK", False)
        )
        replace_threshold = float(
            getattr(settings, "REPLACE_SCORE_THRESHOLD", 0.0)
        )
        max_replacements_per_day = int(
            getattr(settings, "MAX_REPLACEMENTS_PER_DAY", 1)
        )
        # 시장 레짐 필터 설정
        regime_filter_enabled = bool(
            getattr(settings, "MARKET_REGIME_FILTER_ENABLED", False)
        )
        regime_filter_ticker = str(getattr(settings, "MARKET_REGIME_FILTER_TICKER", "^GSPC"))
        regime_filter_ma_period = int(
            getattr(settings, "MARKET_REGIME_FILTER_MA_PERIOD", 20)
        )
        atr_period = int(
            getattr(settings, "ATR_PERIOD_FOR_NORMALIZATION", 20)
        )
    except AttributeError as e:
        raise AttributeError(
            f"'{e.name}' 설정이 logic/settings.py 파일에 반드시 정의되어야 합니다."
        ) from e

    try:
        # 공통 설정
        stop_loss = settings.HOLDING_STOP_LOSS_PCT
        cooldown_days = int(settings.COOLDOWN_DAYS)
    except AttributeError as e:
        raise AttributeError(
            f"'{e.name}' 설정이 logic/settings.py 파일에 반드시 정의되어야 합니다."
        ) from e

    if top_n <= 0:
        raise ValueError("PORTFOLIO_TOPN (top_n)은 0보다 커야 합니다.")
    min_pos_pct = 1.0 / top_n

    # --- 티커 유형(ETF/주식) 구분 ---
    etf_tickers = {stock['ticker'] for stock in stocks if stock.get('type') == 'etf'}

    # --- 데이터 로딩을 위한 날짜 범위 계산 ---
    # 웜업 기간을 가장 긴 MA 기간 기준으로 계산합니다.
    max_ma_period = max(ma_period_etf, ma_period_stock)
    warmup_days = int(max(max_ma_period, atr_period) * 1.5)

    adjusted_date_range = None
    if date_range and len(date_range) == 2:
        core_start = pd.to_datetime(date_range[0])
        warmup_start = core_start - pd.DateOffset(days=warmup_days)
        adjusted_date_range = [warmup_start.strftime("%Y-%m-%d"), date_range[1]]

    # --- 시장 레짐 필터 데이터 로딩 ---
    market_regime_df = None
    if regime_filter_enabled:
        # fetch_ohlcv는 지수 티커를 처리할 수 있도록 수정되었습니다.
        market_regime_df = fetch_ohlcv(
            regime_filter_ticker, country=country, date_range=adjusted_date_range
        )
        if market_regime_df is not None and not market_regime_df.empty:
            market_regime_df["MA"] = (
                market_regime_df["Close"]
                .rolling(window=regime_filter_ma_period)
                .mean()
            )
        else:
            print(f"경고: 시장 레짐 필터 티커({regime_filter_ticker})의 데이터를 가져올 수 없습니다. 필터를 비활성화합니다.")
            regime_filter_enabled = False

    # --- 개별 종목 데이터 로딩 및 지표 계산 ---
    data = {}
    tickers_to_process = [s['ticker'] for s in stocks]

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

        # 티커 유형에 따라 이동평균 기간 결정
        ma_period = ma_period_etf if tkr in etf_tickers else ma_period_stock

        if len(df) < max(ma_period, atr_period):
            continue

        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        from utils.indicators import calculate_atr

        ma = close.rolling(window=ma_period).mean()
        atr = calculate_atr(df, period=atr_period)

        # ma_score: (종가 - 이평선) / ATR. 변동성으로 정규화된 점수.
        ma_score = (close - ma) / atr
        ma_score = ma_score.replace([np.inf, -np.inf], np.nan).fillna(0.0)
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

        # --- 시장 레짐 필터 적용 (Risk-Off 조건 확인) ---
        is_risk_off = False
        if regime_filter_enabled and market_regime_df is not None and dt in market_regime_df.index:
            market_price = market_regime_df.loc[dt, "Close"]
            market_ma = market_regime_df.loc[dt, "MA"]
            if pd.notna(market_price) and pd.notna(market_ma) and market_price < market_ma:
                is_risk_off = True


        # 현재 총 보유 자산 가치를 계산합니다.
        current_holdings_value = 0
        for tkr_h, s_h in state.items():
            if s_h["shares"] > 0:
                price_h = today_prices.get(tkr_h)
                if pd.notna(price_h):
                    current_holdings_value += s_h["shares"] * price_h

        # 총 평가금액(현금 + 주식)을 계산합니다.
        equity = cash + current_holdings_value
        # --- 1. 기본 정보 및 출력 행 생성 ---
        for tkr, d in data.items():
            ticker_state, price = state[tkr], today_prices.get(tkr)
            is_ticker_warming_up = tkr not in tickers_available_today or pd.isna(d["ma"].get(dt))
            
            decision_out = ("HOLD" if ticker_state["shares"] > 0 else "WAIT")
            note = ""
            if is_ticker_warming_up:
                note = "웜업 기간"
            elif decision_out in ("WAIT", "HOLD"):
                if ticker_state["shares"] > 0 and i < ticker_state["sell_block_until"]:
                    note = "매도 쿨다운"
                elif ticker_state["shares"] == 0 and i < ticker_state["buy_block_until"]:
                    note = "매수 쿨다운"

            # Create the row first
            if tkr in tickers_available_today:
                out_rows[tkr].append(
                    {
                        "date": dt,
                        "price": price,
                        "shares": ticker_state["shares"],
                        "pv": ticker_state["shares"] * (price if pd.notna(price) else 0),
                        "decision": decision_out,
                        "avg_cost": ticker_state["avg_cost"],
                        "trade_amount": 0.0, "trade_profit": 0.0, "trade_pl_pct": 0.0,
                        "note": note,
                        "signal1": d["ma"].get(dt),  # 이평선(값)
                        "signal2": None,  # 고점대비
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

        # --- 2. 매도 로직 ---
        # (a) 시장 레짐 필터
        if is_risk_off:
            for tkr_h, ticker_state_h in state.items():
                if ticker_state_h["shares"] > 0:
                    price = today_prices.get(tkr_h)
                    if pd.notna(price):
                        qty = ticker_state_h["shares"]
                        trade_amount = qty * price
                        hold_ret = ((price / ticker_state_h["avg_cost"] - 1.0) * 100.0 if ticker_state_h["avg_cost"] > 0 else 0.0)
                        trade_profit = ((price - ticker_state_h["avg_cost"]) * qty if ticker_state_h["avg_cost"] > 0 else 0.0)
                        
                        cash += trade_amount
                        ticker_state_h["shares"], ticker_state_h["avg_cost"] = 0, 0.0
                        
                        # Update the already created row
                        row = out_rows[tkr_h][-1]
                        row.update({
                            "decision": "SELL_REGIME_FILTER", "trade_amount": trade_amount,
                            "trade_profit": trade_profit, "trade_pl_pct": hold_ret,
                            "shares": 0, "pv": 0, "avg_cost": 0, "note": "시장 위험 회피",
                        })
        # (b) 개별 종목 매도 (시장이 Risk-On일 때만)
        else:
            for tkr, d in data.items():
                ticker_state, price = state[tkr], today_prices.get(tkr)
                is_ticker_warming_up = tkr not in tickers_available_today or pd.isna(d["ma"].get(dt))
                
                if (ticker_state["shares"] > 0 and pd.notna(price) and 
                    i >= ticker_state["sell_block_until"] and not is_ticker_warming_up):
                    
                    decision = None
                    hold_ret = ((price / ticker_state["avg_cost"] - 1.0) * 100.0 if ticker_state["avg_cost"] > 0 else 0.0)
                    
                    if stop_loss is not None and hold_ret <= float(stop_loss):
                        decision = "CUT_STOPLOSS"
                    elif price < d["ma"].loc[dt]:
                        decision = "SELL_TREND"
                    
                    if decision:
                        qty = ticker_state["shares"]
                        trade_amount = qty * price
                        trade_profit = ((price - ticker_state["avg_cost"]) * qty if ticker_state["avg_cost"] > 0 else 0.0)
                        
                        cash += trade_amount
                        ticker_state["shares"], ticker_state["avg_cost"] = 0, 0.0
                        if cooldown_days > 0:
                            ticker_state["buy_block_until"] = i + cooldown_days
                        
                        # Update the row
                        row = out_rows[tkr][-1]
                        row.update({
                            "decision": decision, "trade_amount": trade_amount,
                            "trade_profit": trade_profit, "trade_pl_pct": hold_ret,
                            "shares": 0, "pv": 0, "avg_cost": 0,
                        })

        # --- 3. 매수 로직 (시장이 Risk-On일 때만) ---
        if not is_risk_off:
            # 1. 매수 후보 선정
            buy_candidates = []
            if cash > 0:  # 현금이 있어야만 매수 후보를 고려
                for tkr_cand in tickers_available_today:
                    d_cand = data.get(tkr_cand)
                    ticker_state_cand = state[tkr_cand]
                    buy_signal_days_today = d_cand["buy_signal_days"].get(dt, 0)

                    if (
                        ticker_state_cand["shares"] == 0
                        and i >= ticker_state_cand["buy_block_until"]
                        and buy_signal_days_today > 0
                    ):
                        score_cand = d_cand["ma_score"].get(dt, -float("inf"))
                        if pd.notna(score_cand):
                            buy_candidates.append((score_cand, tkr_cand))
                buy_candidates.sort(reverse=True)

            # 2. 매수 실행 (신규 또는 교체)
            held_count = sum(1 for s in state.values() if s["shares"] > 0)
            slots_to_fill = max(0, top_n - held_count)

            if slots_to_fill > 0 and buy_candidates:
                # 2-1. 신규 매수: 포트폴리오에 빈 슬롯이 있는 경우
                for k in range(min(slots_to_fill, len(buy_candidates))):
                    if cash <= 0:
                        break
                    _, tkr_to_buy = buy_candidates[k]

                    price = today_prices.get(tkr_to_buy)
                    if pd.isna(price):
                        continue

                    equity = cash + current_holdings_value
                    min_val = min_pos_pct * equity
                    req_qty = ceil(min_val / price) if price > 0 else 0
                    if req_qty <= 0:
                        continue

                    trade_amount = req_qty * price
                    if trade_amount <= cash:
                        ticker_state = state[tkr_to_buy]
                        cash -= trade_amount
                        ticker_state["shares"] += req_qty
                        ticker_state["avg_cost"] = price
                        if cooldown_days > 0:
                            ticker_state["sell_block_until"] = max(
                                ticker_state["sell_block_until"], i + cooldown_days
                            )

                        if out_rows[tkr_to_buy] and out_rows[tkr_to_buy][-1]["date"] == dt:
                            row = out_rows[tkr_to_buy][-1]
                            row.update(
                                {
                                    "decision": "BUY",
                                    "trade_amount": trade_amount,
                                    "shares": ticker_state["shares"],
                                    "pv": ticker_state["shares"] * price,
                                    "avg_cost": ticker_state["avg_cost"],
                                }
                            )

            elif slots_to_fill <= 0 and replace_weaker_stock and buy_candidates:
                # 2-2. 교체 매매: 포트폴리오가 가득 찼지만, 더 좋은 종목이 나타난 경우
                held_stocks_with_scores = []
                for tkr_h, ticker_state_h in state.items():
                    if ticker_state_h["shares"] > 0:
                        d_h = data.get(tkr_h)
                        if d_h and dt in d_h["ma_score"].index:
                            score_h = d_h["ma_score"].loc[dt]
                            if pd.notna(score_h):
                                held_stocks_with_scores.append((score_h, tkr_h))

                # 점수 오름차순 정렬 (약한 것부터)
                held_stocks_with_scores.sort()
                # buy_candidates는 이미 점수 내림차순으로 정렬되어 있음 (강한 것부터)

                num_possible_replacements = min(
                    len(buy_candidates), len(held_stocks_with_scores), max_replacements_per_day
                )

                for k in range(num_possible_replacements):
                    best_new_score, best_new_tkr = buy_candidates[k]
                    weakest_held_score, weakest_held_tkr = held_stocks_with_scores[k]

                    # 교체 조건: 새 후보가 기존 보유 종목보다 강하고, 매수/매도할 가격이 유효할 때
                    if best_new_score > weakest_held_score + replace_threshold:
                        sell_price = today_prices.get(weakest_held_tkr)
                        buy_price = today_prices.get(best_new_tkr)

                        if pd.notna(sell_price) and sell_price > 0 and pd.notna(buy_price) and buy_price > 0:
                            # (a) 가장 약한 종목 매도
                            weakest_state = state[weakest_held_tkr]
                            sell_qty = weakest_state["shares"]
                            sell_amount = sell_qty * sell_price
                            hold_ret = ((sell_price / weakest_state["avg_cost"] - 1.0) * 100.0 if weakest_state["avg_cost"] > 0 else 0.0)
                            trade_profit = (sell_price - weakest_state["avg_cost"]) * sell_qty if weakest_state["avg_cost"] > 0 else 0.0

                            cash += sell_amount
                            weakest_state["shares"], weakest_state["avg_cost"] = 0, 0.0
                            if cooldown_days > 0:
                                weakest_state["buy_block_until"] = i + cooldown_days

                            if out_rows[weakest_held_tkr] and out_rows[weakest_held_tkr][-1]["date"] == dt:
                                row = out_rows[weakest_held_tkr][-1]
                                row.update({
                                    "decision": "SELL_REPLACE",
                                    "trade_amount": sell_amount,
                                    "trade_profit": trade_profit,
                                    "trade_pl_pct": hold_ret,
                                    "shares": 0, "pv": 0, "avg_cost": 0,
                                    "note": f"{best_new_tkr}(으)로 교체",
                                })

                            # (b) 가장 강한 새 종목 매수
                            equity = cash + current_holdings_value
                            min_val = min_pos_pct * equity
                            req_qty = ceil(min_val / buy_price) if buy_price > 0 else 0

                            if req_qty > 0:
                                buy_amount = req_qty * buy_price
                                if buy_amount <= cash:
                                    new_ticker_state = state[best_new_tkr]
                                    cash -= buy_amount
                                    new_ticker_state["shares"], new_ticker_state["avg_cost"] = req_qty, buy_price
                                    if cooldown_days > 0:
                                        new_ticker_state["sell_block_until"] = max(new_ticker_state["sell_block_until"], i + cooldown_days)

                                    if out_rows[best_new_tkr] and out_rows[best_new_tkr][-1]["date"] == dt:
                                        row = out_rows[best_new_tkr][-1]
                                        row.update({
                                            "decision": "BUY_REPLACE",
                                            "trade_amount": buy_amount,
                                            "shares": req_qty, "pv": req_qty * buy_price, "avg_cost": buy_price,
                                            "note": f"{weakest_held_tkr}(을)를 대체",
                                        })
                                else:
                                    # 매수 실패 시, 매도만 실행된 상태가 됨. 다음 날 빈 슬롯에 매수 시도.
                                    if out_rows[best_new_tkr] and out_rows[best_new_tkr][-1]["date"] == dt:
                                        out_rows[best_new_tkr][-1]["note"] = "교체매수 현금부족"
                    else:
                        # 점수가 정렬되어 있으므로, 더 이상의 교체는 불가능합니다.
                        break

            # 3. 매수하지 못한 후보에 사유 기록
            # 오늘 매수 또는 교체매수된 종목 목록을 만듭니다.
            bought_tickers_today = {
                tkr for tkr, r in out_rows.items()
                if r and r[-1]["date"] == dt and r[-1]["decision"] in ("BUY", "BUY_REPLACE")
            }
            for _, tkr_cand in buy_candidates:
                if tkr_cand not in bought_tickers_today:
                    if out_rows[tkr_cand] and out_rows[tkr_cand][-1]["date"] == dt:
                        note = "포트폴리오 가득 참" if slots_to_fill <= 0 else "현금 부족"
                        out_rows[tkr_cand][-1]["note"] = note
        else: # is_risk_off
            # 매수 후보가 있더라도, 시장이 위험 회피 상태이므로 매수하지 않음
            # 후보들에게 사유 기록
            buy_candidates = []
            if cash > 0:
                for tkr_cand in tickers_available_today:
                    d_cand = data.get(tkr_cand)
                    ticker_state_cand = state[tkr_cand]
                    buy_signal_days_today = d_cand["buy_signal_days"].get(dt, 0)
                    if (ticker_state_cand["shares"] == 0 and i >= ticker_state_cand["buy_block_until"] and buy_signal_days_today > 0):
                        buy_candidates.append(tkr_cand)

            for tkr_cand in buy_candidates:
                if out_rows[tkr_cand] and out_rows[tkr_cand][-1]["date"] == dt:
                    out_rows[tkr_cand][-1]["note"] = "시장 위험 회피"

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
    stock_type: str = 'stock',
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
        ma_period_etf = int(settings.MA_PERIOD_FOR_ETF)
        ma_period_stock = int(settings.MA_PERIOD_FOR_STOCK)
        atr_period = int(
            getattr(settings, "ATR_PERIOD_FOR_NORMALIZATION", 20)
        )
    except AttributeError as e:
        raise AttributeError(
            f"'{e.name}' 설정이 logic/settings.py 파일에 반드시 정의되어야 합니다."
        ) from e

    try:
        # 공통 설정
        stop_loss = settings.HOLDING_STOP_LOSS_PCT
        cooldown_days = int(settings.COOLDOWN_DAYS)
    except AttributeError as e:
        raise AttributeError(
            f"'{e.name}' 설정이 logic/settings.py 파일에 반드시 정의되어야 합니다."
        ) from e

    # --- 티커 유형(ETF/주식) 구분 ---
    ma_period = ma_period_etf if stock_type == 'etf' else ma_period_stock
    if df is None:
        df = fetch_ohlcv(ticker, country=country, date_range=date_range)

    if df is None or df.empty:
        return pd.DataFrame()

    # yfinance가 가끔 MultiIndex 컬럼을 반환하는 경우에 대비하여,
    # 컬럼을 단순화하고 중복을 제거합니다.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        df = df.loc[:, ~df.columns.duplicated()]
    if len(df) < max(ma_period, atr_period):
        return pd.DataFrame()

    close = df["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]

    from utils.indicators import calculate_atr

    ma = close.rolling(window=ma_period).mean()
    atr = calculate_atr(df, period=atr_period)

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

    rows = []
    for i in range(loop_start_index, len(df)):
        price_val = close.iloc[i]
        if pd.isna(price_val):
            continue
        price = float(price_val)
        decision, trade_amount, trade_profit, trade_pl_pct = None, 0.0, 0.0, 0.0

        ma_today = ma.iloc[i]
        atr_today = atr.iloc[i]

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
                    "score": 0.0,
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
                if cooldown_days > 0:
                    buy_block_until = i + cooldown_days

        if decision is None and shares == 0 and i >= buy_block_until:
            buy_signal_days_today = buy_signal_days.iloc[i]
            if buy_signal_days_today > 0:
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
                "score": (
                    (price - ma_today) / atr_today
                    if pd.notna(ma_today) and pd.notna(atr_today) and atr_today > 0
                    else 0.0
                ),
                "filter": buy_signal_days.iloc[i],
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("date")
