"""Momentum 전략 백테스트 유틸리티."""

from __future__ import annotations

from math import ceil
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from utils.data_loader import fetch_ohlcv

from .rules import passes_min_buy_score, resolve_min_buy_score
from .shared import select_candidates_by_category


def run_portfolio_backtest(
    stocks: List[Dict],
    initial_capital: float = 100_000_000.0,
    core_start_date: Optional[pd.Timestamp] = None,
    top_n: int = 10,
    date_range: Optional[List[str]] = None,
    country: str = "kor",
    prefetched_data: Optional[Dict[str, pd.DataFrame]] = None,
    ma_period: int = 20,
    replace_threshold: float = 0.0,
    regime_filter_enabled: bool = False,
    regime_filter_ticker: str = "^GSPC",
    regime_filter_ma_period: int = 200,
    regime_behavior: str = "sell_all",
    stop_loss_pct: float = -10.0,
    cooldown_days: int = 5,
    min_buy_score: Optional[float] = None,
) -> Dict[str, pd.DataFrame]:
    """
    단일 이동평균선 교차 전략을 사용하여 Top-N 포트폴리오를 시뮬레이션합니다.
    """
    ma_period_etf = ma_period
    ma_period_stock = ma_period
    stop_loss = stop_loss_pct
    min_buy_score = resolve_min_buy_score(min_buy_score)

    valid_regime_behaviors = {"sell_all", "hold_block_buy"}
    if regime_behavior not in valid_regime_behaviors:
        raise ValueError("regime_behavior must be one of {'sell_all', 'hold_block_buy'}")

    if top_n <= 0:
        raise ValueError("PORTFOLIO_TOPN (top_n)은 0보다 커야 합니다.")

    # --- 티커 유형(ETF/주식) 구분 ---
    etf_tickers = {stock["ticker"] for stock in stocks if stock.get("type") == "etf"}

    # --- 데이터 로딩 범위 계산 (웜업 기간 포함) ---
    # 웜업 기간은 이동평균 계산에 필요한 과거 데이터를 확보하기 위함입니다.
    max_ma_period = max(ma_period_etf, ma_period_stock, regime_filter_ma_period)
    warmup_days = int(max_ma_period * 1.5)

    adjusted_date_range = date_range
    if date_range and len(date_range) == 2 and date_range[0] is not None:
        core_start = pd.to_datetime(date_range[0])
        warmup_start = core_start - pd.DateOffset(days=warmup_days)
        adjusted_date_range = [warmup_start.strftime("%Y-%m-%d"), date_range[1]]

    # --- 시장 레짐 필터 데이터 로딩 ---
    market_regime_df = None
    if regime_filter_enabled:
        # 지수 티커를 지원하므로, 국가 코드는 의미상만 전달됩니다.
        market_regime_df = fetch_ohlcv(
            regime_filter_ticker, country=country, date_range=adjusted_date_range
        )
        if market_regime_df is not None and not market_regime_df.empty:
            market_regime_df["MA"] = (
                market_regime_df["Close"].rolling(window=regime_filter_ma_period).mean()
            )
        else:
            print(f"경고: 시장 레짐 필터 티커({regime_filter_ticker})의 데이터를 가져올 수 없습니다. 필터를 비활성화합니다.")
            regime_filter_enabled = False

    # --- 개별 종목 데이터 로딩 및 지표 계산 ---
    # --- 티커별 카테고리 매핑 생성 ---
    ticker_to_category = {stock["ticker"]: stock.get("category") for stock in stocks}
    etf_meta = {stock["ticker"]: stock for stock in stocks if stock.get("ticker")}
    metrics_by_ticker = {}
    tickers_to_process = [s["ticker"] for s in stocks]

    for ticker in tickers_to_process:
        # 미리 로드된 데이터가 있으면 사용하고, 없으면 새로 조회합니다.
        if prefetched_data and ticker in prefetched_data:
            df = prefetched_data[ticker]
        else:
            # prefetched_data가 없으면 date_range를 사용하여 직접 조회합니다.
            # 이 경로는 주로 test.py가 아닌 다른 곳에서 호출될 때 사용됩니다.
            df = fetch_ohlcv(ticker, country=country, date_range=date_range)

        if df is None:
            continue

        # yfinance가 가끔 MultiIndex 컬럼을 반환하는 경우가 있어 컬럼을 단순화/중복 제거
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            df = df.loc[:, ~df.columns.duplicated()]

        # 티커 유형에 따라 이동평균 기간 결정
        ma_period = ma_period_etf if ticker in etf_tickers else ma_period_stock

        if len(df) < ma_period:
            continue

        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        ma = close.rolling(window=ma_period).mean()
        ma_safe = ma.replace(0, np.nan)
        ma_score = ((close / ma_safe) - 1.0) * 100
        ma_score = ma_score.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        # 이동평균선 위에 주가가 머무른 연속된 일수를 계산합니다.
        buy_signal_active = close > ma
        buy_signal_days = (
            buy_signal_active.groupby((buy_signal_active != buy_signal_active.shift()).cumsum())
            .cumsum()
            .fillna(0)
            .astype(int)
        )

        metrics_by_ticker[ticker] = {
            "df": df,
            "close": df["Close"],
            "ma": ma,
            "ma_score": ma_score,
            "buy_signal_days": buy_signal_days,
        }

    if not metrics_by_ticker:
        return {}

    # 모든 종목의 거래일을 합집합하여 전체 백테스트 기간을 설정합니다.
    union_index = pd.DatetimeIndex([])
    for ticker, ticker_metrics in metrics_by_ticker.items():
        union_index = union_index.union(ticker_metrics["close"].index)

    if union_index.empty:
        return {}

    # 요청된 시작일 이후로 인덱스를 필터링합니다.
    if core_start_date:
        union_index = union_index[union_index >= core_start_date]

    if union_index.empty:
        return {}

    # 시뮬레이션 상태 변수 초기화
    position_state = {
        ticker: {
            "shares": 0,
            "avg_cost": 0.0,
            "buy_block_until": -1,
            "sell_block_until": -1,
        }
        for ticker in metrics_by_ticker.keys()
    }
    cash = float(initial_capital)
    daily_records_by_ticker = {ticker: [] for ticker in metrics_by_ticker.keys()}
    out_cash = []

    # 일별 루프를 돌며 시뮬레이션을 실행합니다.
    for i, dt in enumerate(union_index):
        tickers_available_today = [
            ticker
            for ticker, ticker_metrics in metrics_by_ticker.items()
            if dt in ticker_metrics["df"].index
        ]
        today_prices = {
            ticker: (
                float(ticker_metrics["close"].loc[dt])
                if pd.notna(ticker_metrics["close"].loc[dt])
                else None
            )
            for ticker, ticker_metrics in metrics_by_ticker.items()
            if dt in ticker_metrics["close"].index
        }

        # --- 시장 레짐 필터 적용 (리스크 오프 조건 확인) ---
        is_risk_off = False
        if regime_filter_enabled and market_regime_df is not None and dt in market_regime_df.index:
            market_price = market_regime_df.loc[dt, "Close"]
            market_ma = market_regime_df.loc[dt, "MA"]
            if pd.notna(market_price) and pd.notna(market_ma) and market_price < market_ma:
                is_risk_off = True

        force_regime_sell = is_risk_off and regime_behavior == "sell_all"
        allow_individual_sells = (not is_risk_off) or regime_behavior == "hold_block_buy"
        allow_new_buys = not is_risk_off

        # 현재 총 보유 자산 가치를 계산합니다.
        current_holdings_value = 0
        for held_ticker, held_state in position_state.items():
            if held_state["shares"] > 0:
                price_h = today_prices.get(held_ticker)
                if pd.notna(price_h):
                    current_holdings_value += held_state["shares"] * price_h

        # 총 평가금액(현금 + 주식)을 계산합니다.
        equity = cash + current_holdings_value
        # --- 1. 기본 정보 및 출력 행 생성 ---
        for ticker, ticker_metrics in metrics_by_ticker.items():
            position_snapshot = position_state[ticker]
            price = today_prices.get(ticker)
            is_ticker_warming_up = ticker not in tickers_available_today or pd.isna(
                ticker_metrics["ma"].get(dt)
            )

            decision_out = "HOLD" if position_snapshot["shares"] > 0 else "WAIT"
            note = ""
            if is_ticker_warming_up:
                note = "웜업 기간"
            elif decision_out in ("WAIT", "HOLD"):
                if position_snapshot["shares"] > 0 and i < position_snapshot["sell_block_until"]:
                    note = "매도 쿨다운"
                elif position_snapshot["shares"] == 0 and i < position_snapshot["buy_block_until"]:
                    note = "매수 쿨다운"

            # 출력 행을 먼저 구성
            if ticker in tickers_available_today:
                daily_records_by_ticker[ticker].append(
                    {
                        "date": dt,
                        "price": price,
                        "shares": position_snapshot["shares"],
                        "pv": position_snapshot["shares"] * (price if pd.notna(price) else 0),
                        "decision": decision_out,
                        "avg_cost": position_snapshot["avg_cost"],
                        "trade_amount": 0.0,
                        "trade_profit": 0.0,
                        "trade_pl_pct": 0.0,
                        "note": note,
                        "signal1": ticker_metrics["ma"].get(dt),  # 이평선(값)
                        "signal2": None,  # 고점대비
                        "score": ticker_metrics["ma_score"].loc[dt],
                        "filter": ticker_metrics["buy_signal_days"].get(dt),
                    }
                )
            else:
                daily_records_by_ticker[ticker].append(
                    {
                        "date": dt,
                        "price": position_snapshot["avg_cost"],
                        "shares": position_snapshot["shares"],
                        "pv": position_snapshot["shares"]
                        * (
                            position_snapshot["avg_cost"]
                            if pd.notna(position_snapshot["avg_cost"])
                            else 0.0
                        ),
                        "decision": "HOLD" if position_snapshot["shares"] > 0 else "WAIT",
                        "avg_cost": position_snapshot["avg_cost"],
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
        if force_regime_sell:
            for held_ticker, held_state in position_state.items():
                if held_state["shares"] > 0:
                    price = today_prices.get(held_ticker)
                    if pd.notna(price):
                        qty = held_state["shares"]
                        trade_amount = qty * price
                        hold_ret = (
                            (price / held_state["avg_cost"] - 1.0) * 100.0
                            if held_state["avg_cost"] > 0
                            else 0.0
                        )
                        trade_profit = (
                            (price - held_state["avg_cost"]) * qty
                            if held_state["avg_cost"] > 0
                            else 0.0
                        )

                        cash += trade_amount
                        held_state["shares"], held_state["avg_cost"] = 0, 0.0

                        # 이미 만들어둔 행을 업데이트
                        row = daily_records_by_ticker[held_ticker][-1]
                        row.update(
                            {
                                "decision": "SELL_REGIME_FILTER",
                                "trade_amount": trade_amount,
                                "trade_profit": trade_profit,
                                "trade_pl_pct": hold_ret,
                                "shares": 0,
                                "pv": 0,
                                "avg_cost": 0,
                                "note": "시장 위험 회피",
                            }
                        )
        # (b) 개별 종목 매도
        elif allow_individual_sells:
            for ticker, ticker_metrics in metrics_by_ticker.items():
                ticker_state, price = position_state[ticker], today_prices.get(ticker)
                is_ticker_warming_up = ticker not in tickers_available_today or pd.isna(
                    ticker_metrics["ma"].get(dt)
                )

                if (
                    ticker_state["shares"] > 0
                    and pd.notna(price)
                    and i >= ticker_state["sell_block_until"]
                    and not is_ticker_warming_up
                ):
                    decision = None
                    hold_ret = (
                        (price / ticker_state["avg_cost"] - 1.0) * 100.0
                        if ticker_state["avg_cost"] > 0
                        else 0.0
                    )

                    if stop_loss is not None and hold_ret <= float(stop_loss):
                        decision = "CUT_STOPLOSS"
                    elif price < ticker_metrics["ma"].loc[dt]:
                        decision = "SELL_TREND"

                    if decision:
                        qty = ticker_state["shares"]
                        trade_amount = qty * price
                        trade_profit = (
                            (price - ticker_state["avg_cost"]) * qty
                            if ticker_state["avg_cost"] > 0
                            else 0.0
                        )

                        cash += trade_amount
                        ticker_state["shares"], ticker_state["avg_cost"] = 0, 0.0
                        if cooldown_days > 0:
                            ticker_state["buy_block_until"] = i + cooldown_days

                        # 행 업데이트
                        row = daily_records_by_ticker[ticker][-1]
                        row.update(
                            {
                                "decision": decision,
                                "trade_amount": trade_amount,
                                "trade_profit": trade_profit,
                                "trade_pl_pct": hold_ret,
                                "shares": 0,
                                "pv": 0,
                                "avg_cost": 0,
                            }
                        )

        # --- 3. 매수 로직 (리스크 온일 때만) ---
        if allow_new_buys:
            # 1. 매수 후보 선정
            buy_ranked_candidates = []
            for candidate_ticker in tickers_available_today:
                candidate_metrics = metrics_by_ticker.get(candidate_ticker)
                ticker_state_cand = position_state[candidate_ticker]
                buy_signal_days_today = candidate_metrics["buy_signal_days"].get(dt, 0)

                if (
                    ticker_state_cand["shares"] == 0
                    and i >= ticker_state_cand["buy_block_until"]
                    and buy_signal_days_today > 0
                ):
                    score_cand = candidate_metrics["ma_score"].get(dt, -float("inf")) or -float(
                        "inf"
                    )

                    if passes_min_buy_score(score_cand, min_buy_score):
                        buy_ranked_candidates.append((score_cand, candidate_ticker))
            buy_ranked_candidates.sort(reverse=True)

            # 2. 매수 실행 (신규 또는 교체)
            held_count = sum(1 for pos in position_state.values() if pos["shares"] > 0)
            slots_to_fill = max(0, top_n - held_count)

            purchased_today: Set[str] = set()

            if slots_to_fill > 0 and buy_ranked_candidates:
                held_categories = {
                    cat
                    for tkr, state in position_state.items()
                    if state["shares"] > 0 and (cat := ticker_to_category.get(tkr)) and cat != "TBD"
                }

                helper_candidates = [
                    {"tkr": ticker, "score": score} for score, ticker in buy_ranked_candidates
                ]

                selected_candidates, rejected_candidates = select_candidates_by_category(
                    helper_candidates,
                    etf_meta,
                    held_categories=held_categories,
                    max_count=slots_to_fill,
                    skip_held_categories=True,
                )

                for cand, reason in rejected_candidates:
                    if reason != "category_held":
                        continue
                    ticker_rejected = cand.get("tkr")
                    if not ticker_rejected:
                        continue
                    records = daily_records_by_ticker.get(ticker_rejected)
                    if (
                        records
                        and records[-1]["date"] == dt
                        and records[-1].get("decision") == "WAIT"
                    ):
                        records[-1]["note"] = "카테고리 중복"

                for cand in selected_candidates:
                    if cash <= 0:
                        break

                    ticker_to_buy = cand["tkr"]
                    price = today_prices.get(ticker_to_buy)
                    if pd.isna(price):
                        continue

                    equity_base = equity
                    min_val = 1.0 / (top_n * 2.0) * equity_base
                    max_val = 1.0 / top_n * equity_base
                    budget = min(max_val, cash)

                    if budget <= 0 or budget < min_val:
                        continue

                    if country in ("coin", "aus"):
                        req_qty = budget / price if price > 0 else 0
                        trade_amount = budget
                    else:
                        req_qty = ceil(budget / price) if price > 0 else 0
                        req_qty = int(budget // price)
                        trade_amount = req_qty * price
                        if req_qty <= 0 or trade_amount + 1e-9 < min_val:
                            continue

                    if trade_amount <= cash + 1e-9 and req_qty > 0:
                        ticker_state = position_state[ticker_to_buy]
                        cash -= trade_amount
                        ticker_state["shares"] += req_qty
                        ticker_state["avg_cost"] = price
                        if cooldown_days > 0:
                            ticker_state["sell_block_until"] = max(
                                ticker_state["sell_block_until"], i + cooldown_days
                            )

                        category = ticker_to_category.get(ticker_to_buy)
                        if category and category != "TBD":
                            held_categories.add(category)

                        if (
                            daily_records_by_ticker[ticker_to_buy]
                            and daily_records_by_ticker[ticker_to_buy][-1]["date"] == dt
                        ):
                            row = daily_records_by_ticker[ticker_to_buy][-1]
                            row.update(
                                {
                                    "decision": "BUY",
                                    "trade_amount": trade_amount,
                                    "shares": ticker_state["shares"],
                                    "pv": ticker_state["shares"] * price,
                                    "avg_cost": ticker_state["avg_cost"],
                                }
                            )
                        purchased_today.add(ticker_to_buy)

            elif slots_to_fill <= 0 and buy_ranked_candidates:
                helper_candidates = [
                    {"tkr": ticker, "score": score}
                    for score, ticker in buy_ranked_candidates
                    if ticker not in purchased_today
                ]

                replacement_candidates, _ = select_candidates_by_category(
                    helper_candidates,
                    etf_meta,
                    held_categories=None,
                    max_count=None,
                    skip_held_categories=False,
                )

                held_stocks_with_scores = []
                for held_ticker, held_position in position_state.items():
                    if held_position["shares"] > 0:
                        held_metrics = metrics_by_ticker.get(held_ticker)
                        if held_metrics and dt in held_metrics["ma_score"].index:
                            score_h = held_metrics["ma_score"].loc[dt]
                            if pd.notna(score_h):
                                held_stocks_with_scores.append(
                                    {
                                        "ticker": held_ticker,
                                        "score": score_h,
                                        "category": ticker_to_category.get(held_ticker),
                                    }
                                )

                held_stocks_with_scores.sort(key=lambda x: x["score"])

                for candidate in replacement_candidates:
                    replacement_ticker = candidate["tkr"]
                    wait_stock_category = ticker_to_category.get(replacement_ticker)
                    best_new_score_raw = candidate.get("score")
                    try:
                        best_new_score = float(best_new_score_raw)
                    except (TypeError, ValueError):
                        best_new_score = float("-inf")

                    # 교체 대상이 될 수 있는 보유 종목을 찾습니다.
                    # 1. 같은 카테고리의 종목이 있는지 확인
                    held_stock_same_category = next(
                        (
                            s
                            for s in held_stocks_with_scores
                            if s["category"] == wait_stock_category
                        ),
                        None,
                    )

                    weakest_held_stock = (
                        held_stocks_with_scores[0] if held_stocks_with_scores else None
                    )

                    # 교체 여부 및 대상 종목 결정
                    ticker_to_sell = None
                    replacement_note = ""

                    if held_stock_same_category:
                        # 같은 카테고리 종목이 있는 경우: 점수만 비교
                        if best_new_score > held_stock_same_category["score"] + replace_threshold:
                            ticker_to_sell = held_stock_same_category["ticker"]
                            replacement_note = (
                                f"{ticker_to_sell}(을)를 {replacement_ticker}(으)로 교체 (동일 카테고리)"
                            )
                        else:
                            # 점수가 더 높지 않으면 교체하지 않고 다음 대기 종목으로 넘어감
                            if (
                                daily_records_by_ticker[replacement_ticker]
                                and daily_records_by_ticker[replacement_ticker][-1]["date"] == dt
                            ):
                                stock_info = next(
                                    (s for s in stocks if s["ticker"] == replacement_ticker), {}
                                )
                                stock_name = stock_info.get("name", replacement_ticker)
                                daily_records_by_ticker[replacement_ticker][-1][
                                    "note"
                                ] = f"카테고리 중복 - {stock_name}({replacement_ticker})"
                            continue  # 다음 buy_ranked_candidate로 넘어감
                    elif weakest_held_stock:
                        # 같은 카테고리 종목이 없는 경우: 가장 약한 종목과 임계값 포함 비교
                        if best_new_score > weakest_held_stock["score"] + replace_threshold:
                            ticker_to_sell = weakest_held_stock["ticker"]
                            replacement_note = (
                                f"{ticker_to_sell}(을)를 {replacement_ticker}(으)로 교체 (새 카테고리)"
                            )
                        else:
                            # 임계값을 넘지 못하면 교체하지 않고 다음 대기 종목으로 넘어감
                            continue  # 다음 buy_ranked_candidate로 넘어감
                    else:
                        # 보유 종목이 없으면 교체할 수 없음
                        continue  # 다음 buy_ranked_candidate로 넘어감

                    # 교체할 종목이 결정되었으면 매도/매수 진행
                    if ticker_to_sell:
                        sell_price = today_prices.get(ticker_to_sell)
                        buy_price = today_prices.get(replacement_ticker)

                        if (
                            pd.notna(sell_price)
                            and sell_price > 0
                            and pd.notna(buy_price)
                            and buy_price > 0
                        ):
                            # (a) 교체 대상 종목 매도
                            weakest_state = position_state[ticker_to_sell]
                            sell_qty = weakest_state["shares"]
                            sell_amount = sell_qty * sell_price
                            hold_ret = (
                                (sell_price / weakest_state["avg_cost"] - 1.0) * 100.0
                                if weakest_state["avg_cost"] > 0
                                else 0.0
                            )
                            trade_profit = (
                                (sell_price - weakest_state["avg_cost"]) * sell_qty
                                if weakest_state["avg_cost"] > 0
                                else 0.0
                            )

                            cash += sell_amount
                            weakest_state["shares"], weakest_state["avg_cost"] = 0, 0.0
                            if cooldown_days > 0:
                                weakest_state["buy_block_until"] = i + cooldown_days

                            if (
                                daily_records_by_ticker[ticker_to_sell]
                                and daily_records_by_ticker[ticker_to_sell][-1]["date"] == dt
                            ):
                                row = daily_records_by_ticker[ticker_to_sell][-1]
                                row.update(
                                    {
                                        "decision": "SELL_REPLACE",
                                        "trade_amount": trade_amount,
                                        "trade_profit": trade_profit,
                                        "trade_pl_pct": hold_ret,
                                        "shares": 0,
                                        "pv": 0,
                                        "avg_cost": 0,
                                        "note": replacement_note,
                                    }
                                )

                            # (b) 새 종목 매수 (기준 자산 기반 예산)
                            equity_base = equity
                            min_val = 1.0 / (top_n * 2.0) * equity_base
                            max_val = 1.0 / top_n * equity_base
                            budget = min(max_val, cash)
                            if budget <= 0 or budget < min_val:
                                continue
                            # 수량/금액 산정
                            if country in ("coin", "aus"):
                                req_qty = (budget / buy_price) if buy_price > 0 else 0
                                buy_amount = budget
                            else:
                                req_qty = int(budget // buy_price) if buy_price > 0 else 0
                                buy_amount = req_qty * buy_price
                                if req_qty <= 0 or buy_amount + 1e-9 < min_val:
                                    continue

                            # 체결 반영
                            if req_qty > 0 and buy_amount <= cash + 1e-9:
                                new_ticker_state = position_state[replacement_ticker]
                                cash -= buy_amount
                                new_ticker_state["shares"], new_ticker_state["avg_cost"] = (
                                    req_qty,
                                    buy_price,
                                )
                                if cooldown_days > 0:
                                    new_ticker_state["sell_block_until"] = max(
                                        new_ticker_state["sell_block_until"], i + cooldown_days
                                    )

                                # 결과 행 업데이트: 없으면 새로 추가
                                if (
                                    daily_records_by_ticker.get(replacement_ticker)
                                    and daily_records_by_ticker[replacement_ticker]
                                    and daily_records_by_ticker[replacement_ticker][-1]["date"]
                                    == dt
                                ):
                                    row = daily_records_by_ticker[replacement_ticker][-1]
                                    row.update(
                                        {
                                            "decision": "BUY_REPLACE",
                                            "trade_amount": buy_amount,
                                            "shares": req_qty,
                                            "pv": req_qty * buy_price,
                                            "avg_cost": buy_price,
                                            "note": replacement_note,
                                        }
                                    )
                                else:
                                    daily_records_by_ticker.setdefault(
                                        replacement_ticker, []
                                    ).append(
                                        {
                                            "date": dt,
                                            "price": buy_price,
                                            "shares": req_qty,
                                            "pv": req_qty * buy_price,
                                            "decision": "BUY_REPLACE",
                                            "avg_cost": buy_price,
                                            "trade_amount": buy_amount,
                                            "trade_profit": 0.0,
                                            "trade_pl_pct": 0.0,
                                            "note": replacement_note,
                                            "signal1": None,
                                            "signal2": None,
                                            "score": None,
                                            "filter": None,
                                        }
                                    )
                                # 교체가 성공했으므로, held_stocks_with_scores를 업데이트하여 다음 대기 종목 평가에 반영
                                # 매도된 종목 제거
                                held_stocks_with_scores = [
                                    s
                                    for s in held_stocks_with_scores
                                    if s["ticker"] != ticker_to_sell
                                ]
                                # 새로 매수한 종목 추가
                                held_stocks_with_scores.append(
                                    {
                                        "ticker": replacement_ticker,
                                        "score": best_new_score,
                                        "category": wait_stock_category,
                                    }
                                )
                                held_stocks_with_scores.sort(key=lambda x: x["score"])  # 다시 정렬
                                break  # 하나의 대기 종목으로 하나의 교체만 시도하므로, 다음 날로 넘어감
                            else:
                                # 매수 실패 시, 매도만 실행된 상태가 됨. 다음 날 빈 슬롯에 매수 시도.
                                if (
                                    daily_records_by_ticker.get(replacement_ticker)
                                    and daily_records_by_ticker[replacement_ticker]
                                    and daily_records_by_ticker[replacement_ticker][-1]["date"]
                                    == dt
                                ):
                                    daily_records_by_ticker[replacement_ticker][-1][
                                        "note"
                                    ] = "교체매수 현금부족"
                        else:
                            # 가격 정보가 유효하지 않으면 교체하지 않고 다음 대기 종목으로 넘어감
                            continue  # 다음 buy_ranked_candidate로 넘어감

            # 3. 매수하지 못한 후보에 사유 기록
            # 오늘 매수 또는 교체매수된 종목 목록을 만듭니다.
            bought_tickers_today = {
                ticker_symbol
                for ticker_symbol, records in daily_records_by_ticker.items()
                if records
                and records[-1]["date"] == dt
                and records[-1]["decision"] in ("BUY", "BUY_REPLACE")
            }
            for _, candidate_ticker in buy_ranked_candidates:
                if candidate_ticker not in bought_tickers_today:
                    if (
                        daily_records_by_ticker[candidate_ticker]
                        and daily_records_by_ticker[candidate_ticker][-1]["date"] == dt
                    ):
                        note = "포트폴리오 가득 참" if slots_to_fill <= 0 else "현금 부족"
                        daily_records_by_ticker[candidate_ticker][-1]["note"] = note
        else:  # 리스크 오프 상태
            # 매수 후보가 있더라도, 시장이 위험 회피 상태이므로 매수하지 않음
            # 후보들에게 사유 기록
            risk_off_candidates = []
            if cash > 0:
                for candidate_ticker in tickers_available_today:
                    candidate_metrics = metrics_by_ticker.get(candidate_ticker)
                    ticker_state_cand = position_state[candidate_ticker]
                    buy_signal_days_today = candidate_metrics["buy_signal_days"].get(dt, 0)
                    if (
                        ticker_state_cand["shares"] == 0
                        and i >= ticker_state_cand["buy_block_until"]
                        and buy_signal_days_today > 0
                    ):
                        risk_off_candidates.append(candidate_ticker)

            for candidate_ticker in risk_off_candidates:
                if (
                    daily_records_by_ticker[candidate_ticker]
                    and daily_records_by_ticker[candidate_ticker][-1]["date"] == dt
                ):
                    daily_records_by_ticker[candidate_ticker][-1]["note"] = "시장 위험 회피"

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

    result: Dict[str, pd.DataFrame] = {}
    for ticker_symbol, records in daily_records_by_ticker.items():
        if records:
            result[ticker_symbol] = pd.DataFrame(records).set_index("date")
    if out_cash:
        result["CASH"] = pd.DataFrame(out_cash).set_index("date")
    return result


def run_single_ticker_backtest(
    ticker: str,
    stock_type: str = "stock",
    df: Optional[pd.DataFrame] = None,
    initial_capital: float = 1_000_000.0,
    core_start_date: Optional[pd.Timestamp] = None,
    date_range: Optional[List[str]] = None,
    country: str = "kor",
    ma_period: int = 20,
    stop_loss_pct: float = -10.0,
    cooldown_days: int = 5,
    min_buy_score: Optional[float] = None,
) -> pd.DataFrame:
    """
    단일 종목에 대해 이동평균선 교차 전략 백테스트를 실행합니다.
    """
    ma_period_etf = ma_period
    ma_period_stock = ma_period
    stop_loss = stop_loss_pct
    min_buy_score = resolve_min_buy_score(min_buy_score)

    # --- 티커 유형(ETF/주식) 구분 ---
    ma_period = ma_period_etf if stock_type == "etf" else ma_period_stock
    if df is None:
        # df가 제공되지 않으면, date_range를 사용하여 직접 데이터를 조회합니다.
        # date_range가 없으면 기본값(3개월)으로 조회됩니다.
        # test.py에서 호출 시에는 항상 date_range가 전달됩니다.
        df = fetch_ohlcv(ticker, country=country, date_range=date_range)

    if df is None or df.empty:
        return pd.DataFrame()

    # yfinance의 MultiIndex 컬럼을 단순화/중복 제거
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
        df = df.loc[:, ~df.columns.duplicated()]
    if len(df) < ma_period:
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
    shares: float = 0.0
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
            ma_score_today = 0.0
            if pd.notna(ma_today) and ma_today > 0:
                ma_score_today = ((price / ma_today) - 1.0) * 100.0

            passes_score_threshold = True
            if not passes_min_buy_score(ma_score_today, min_buy_score):
                passes_score_threshold = False

            if buy_signal_days_today > 0 and passes_score_threshold:
                if country in ("coin", "aus"):
                    # 소수점 4자리까지 허용
                    buy_qty = round(cash / price, 4) if price > 0 else 0.0
                else:
                    buy_qty = int(cash // price)
                if buy_qty > 0:
                    trade_amount = float(buy_qty) * price
                    cash -= trade_amount
                    avg_cost, shares = price, float(buy_qty)
                    decision = "BUY"
                    if cooldown_days > 0:
                        sell_block_until = i + cooldown_days

        if decision is None:
            decision = "HOLD" if shares > 0 else "WAIT"

        ma_score_today = 0.0
        if pd.notna(ma_today) and ma_today > 0:
            ma_score_today = ((price / ma_today) - 1.0) * 100.0

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
                "score": ma_score_today,
                "filter": buy_signal_days.iloc[i],
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).set_index("date")


__all__ = [
    "run_portfolio_backtest",
    "run_single_ticker_backtest",
]
