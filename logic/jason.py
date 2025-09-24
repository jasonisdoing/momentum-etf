"""
단일 이동평균선을 사용하는 추세추종 전략입니다.
(포트폴리오 Top-N 선택, 교체 매매, 시장 레짐 필터 포함)
"""

from math import ceil
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from utils.data_loader import fetch_ohlcv
from utils.report import format_kr_money

# 슬랙 알림에 사용될 매매 결정(decision) 코드별 표시 설정을 관리합니다.
# - display_name: 슬랙 메시지에 표시될 그룹 헤더
# - order: 그룹 표시 순서 (낮을수록 위)
# - is_recommendation: True이면 @channel 알림을 유발하는 '추천'으로 간주
# - show_slack: True이면 슬랙 알림에 해당 그룹을 포함
DECISION_CONFIG = {
    # 보유  (알림 없음)
    "HOLD": {
        "display_name": "<💼 보유>",
        "order": 1,
        "is_recommendation": False,
        "show_slack": True,
    },
    # 매도 추천 (알림 발생)
    "CUT_STOPLOSS": {
        "display_name": "<🚨 손절매도>",
        "order": 10,
        "is_recommendation": True,
        "show_slack": True,
    },
    "SELL_TREND": {
        "display_name": "<📉 추세이탈 매도>",
        "order": 11,
        "is_recommendation": True,
        "show_slack": True,
    },
    "SELL_REPLACE": {
        "display_name": "<🔄 교체매도>",
        "order": 12,
        "is_recommendation": True,
        "show_slack": True,
    },
    "SELL_REBALANCE": {
        "display_name": "<⚖️ 리밸런스 매도>",
        "order": 13,
        "is_recommendation": True,
        "show_slack": True,
    },
    "SELL_INACTIVE": {
        "display_name": "<🗑️ 비활성 매도>",
        "order": 14,
        "is_recommendation": True,
        "show_slack": True,
    },
    "SELL_REGIME_FILTER": {
        "display_name": "<🛡️ 시장위험회피 매도>",
        "order": 15,
        "is_recommendation": True,
        "show_slack": True,
    },
    # 매수 추천 (알림 발생)
    "BUY_REPLACE": {
        "display_name": "<🔄 교체매수>",
        "order": 20,
        "is_recommendation": True,
        "show_slack": True,
    },
    "BUY": {
        "display_name": "<🚀 신규매수>",
        "order": 21,
        "is_recommendation": True,
        "show_slack": True,
    },
    # 거래 완료 (알림 없음)
    "SOLD": {
        "display_name": "<✅ 매도 완료>",
        "order": 40,
        "is_recommendation": False,
        "show_slack": True,
    },
    # 보유 및 대기 (알림 없음)
    "WAIT": {
        "display_name": "<⏳ 대기>",
        "order": 50,
        "is_recommendation": False,
        "show_slack": False,
    },
}

# 코인 보유 수량에서 0으로 간주할 임계값 (거래소의 dust 처리)
COIN_ZERO_THRESHOLD = 1e-9


def run_portfolio_backtest(
    stocks: List[Dict],
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
    from . import settings

    # 설정값 로드 (필수)
    try:
        # 전략 고유 설정
        ma_period_etf = int(settings.MA_PERIOD)
        ma_period_stock = int(settings.MA_PERIOD)
        replace_weaker_stock = bool(settings.REPLACE_WEAKER_STOCK)
        replace_threshold = float(settings.REPLACE_SCORE_THRESHOLD)
        # 시장 레짐 필터 설정
        regime_filter_enabled = bool(settings.MARKET_REGIME_FILTER_ENABLED)
        regime_filter_ticker = str(settings.MARKET_REGIME_FILTER_TICKER)
        regime_filter_ma_period = int(settings.MARKET_REGIME_FILTER_MA_PERIOD)
    except AttributeError as e:
        raise AttributeError(f"'{e.name}' 설정이 logic/settings.py 파일에 반드시 정의되어야 합니다.") from e

    try:
        # 공통 설정
        stop_loss = settings.HOLDING_STOP_LOSS_PCT
        cooldown_days = int(settings.COOLDOWN_DAYS)
    except AttributeError as e:
        raise AttributeError(f"'{e.name}' 설정이 logic/settings.py 파일에 반드시 정의되어야 합니다.") from e

    if top_n <= 0:
        raise ValueError("PORTFOLIO_TOPN (top_n)은 0보다 커야 합니다.")

    # --- 티커 유형(ETF/주식) 구분 ---
    etf_tickers = {stock["ticker"] for stock in stocks if stock.get("type") == "etf"}

    # --- 티커별 카테고리 매핑 생성 ---
    ticker_to_category = {stock["ticker"]: stock.get("category") for stock in stocks}

    # --- 데이터 로딩 범위 계산 ---
    # 웜업 기간은 필요한 이동평균 기간을 기준으로 산정
    max_ma_period = max(ma_period_etf, ma_period_stock)
    warmup_days = int(max_ma_period * 1.5)

    adjusted_date_range = None
    if date_range and len(date_range) == 2:
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
    metrics_by_ticker = {}
    tickers_to_process = [s["ticker"] for s in stocks]

    for ticker in tickers_to_process:
        # 미리 로드된 데이터가 있으면 사용하고, 없으면 새로 조회합니다.
        if prefetched_data and ticker in prefetched_data:
            df = prefetched_data[ticker].copy()
        else:
            df = fetch_ohlcv(ticker, country=country, date_range=adjusted_date_range)

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
        if is_risk_off:
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
        # (b) 개별 종목 매도 (시장이 리스크 온일 때만)
        else:
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

        # --- 3. 매수 로직 (시장이 리스크 온일 때만) ---
        if not is_risk_off:
            # 1. 매수 후보 선정
            buy_ranked_candidates = []
            if cash > 0:  # 현금이 있어야만 매수 후보를 고려
                for candidate_ticker in tickers_available_today:
                    candidate_metrics = metrics_by_ticker.get(candidate_ticker)
                    ticker_state_cand = position_state[candidate_ticker]
                    buy_signal_days_today = candidate_metrics["buy_signal_days"].get(dt, 0)

                    if (
                        ticker_state_cand["shares"] == 0
                        and i >= ticker_state_cand["buy_block_until"]
                        and buy_signal_days_today > 0
                    ):
                        score_cand = candidate_metrics["ma_score"].get(dt, -float("inf"))
                        if pd.notna(score_cand):
                            buy_ranked_candidates.append((score_cand, candidate_ticker))
                buy_ranked_candidates.sort(reverse=True)

            # 2. 매수 실행 (신규 또는 교체)
            held_count = sum(1 for pos in position_state.values() if pos["shares"] > 0)
            slots_to_fill = max(0, top_n - held_count)

            if slots_to_fill > 0 and buy_ranked_candidates:
                # 2-1. 신규 매수: 포트폴리오에 빈 슬롯이 있는 경우
                for k in range(min(slots_to_fill, len(buy_ranked_candidates))):
                    if cash <= 0:
                        break
                    _, ticker_to_buy = buy_ranked_candidates[k]

                    price = today_prices.get(ticker_to_buy)
                    if pd.isna(price):
                        continue

                    # 예산 산정은 기준 자산(Equity)을 고정하여 일중 처리 순서 영향을 제거합니다.
                    equity_base = equity
                    min_val = 1.0 / (top_n * 2.0) * equity_base
                    max_val = 1.0 / top_n * equity_base
                    budget = min(max_val, cash)

                    # 예산이 최소 비중보다 작으면 스킵
                    if budget <= 0 or budget < min_val:
                        continue

                    if country in ("coin", "aus"):
                        req_qty = budget / price if price > 0 else 0
                        trade_amount = budget
                    else:
                        req_qty = ceil(budget / price) if price > 0 else 0
                        # 정수 수량은 예산 내 최대 구매량으로 계산하되, 최소 비중을 충족하도록 다시 내림 처리합니다.
                        req_qty = int(budget // price)
                        trade_amount = req_qty * price
                        if req_qty <= 0 or trade_amount + 1e-9 < min_val:
                            continue

                    if trade_amount <= cash + 1e-9:
                        ticker_state = position_state[ticker_to_buy]
                        cash -= trade_amount
                        ticker_state["shares"] += req_qty
                        ticker_state["avg_cost"] = price
                        if cooldown_days > 0:
                            ticker_state["sell_block_until"] = max(
                                ticker_state["sell_block_until"], i + cooldown_days
                            )

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

            elif slots_to_fill <= 0 and replace_weaker_stock and buy_ranked_candidates:
                # 2-2. 교체 매매: 포트폴리오가 가득 찼지만, 더 좋은 종목이 나타난 경우
                # 현재 보유 종목 목록 (점수와 티커 포함)
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

                # 대기 종목 (매수 후보) 목록은 이미 점수 내림차순(강한 순)으로 정렬되어 있습니다.
                # held_stocks_with_scores는 점수 오름차순으로 정렬하여 가장 약한 종목을 쉽게 찾을 수 있도록 합니다.
                held_stocks_with_scores.sort(key=lambda x: x["score"])

                # 교체 매매 로직 시작
                # 대기 종목(buy_ranked_candidates)을 점수 높은 순서대로 순회
                for best_new_score, replacement_ticker in buy_ranked_candidates:
                    wait_stock_category = ticker_to_category.get(replacement_ticker)

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
                        if best_new_score > held_stock_same_category["score"]:
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


def generate_daily_signals_for_portfolio(
    country: str,
    account: str,
    base_date: pd.Timestamp,
    portfolio_settings: Dict,
    data_by_tkr: Dict[str, Any],
    holdings: Dict[str, Dict[str, float]],
    etf_meta: Dict[str, Any],
    full_etf_meta: Dict[str, Any],
    regime_info: Optional[Dict],
    current_equity: float,
    total_cash: float,
    pairs: List[Tuple[str, str]],
    consecutive_holding_info: Dict[str, Dict],
    stop_loss: Optional[float],
    COIN_ZERO_THRESHOLD: float,
    DECISION_CONFIG: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    주어진 데이터를 기반으로 포트폴리오의 일일 매매 신호를 생성합니다.
    이 함수는 signals.py에서 호출되어 실제 매매 결정을 내리는 핵심 로직을 제공합니다.
    """

    # 헬퍼 함수 (signals.py에서 가져옴)
    def _format_kr_price(p):
        return f"{int(round(p)):,}"

    def _aud_money_formatter(amount, precision: int):
        return f"${amount:,.{precision}f}"

    def _aud_price_formatter(p, precision: int):
        return f"${p:,.{precision}f}"

    # 계좌 정보에서 통화 및 정밀도 가져오기 (여기서는 직접 접근 불가하므로 인수로 받거나 기본값 사용)
    # 여기서는 임시로 기본값을 사용하거나, portfolio_settings에서 가져오는 것으로 가정합니다.
    currency = portfolio_settings.get("currency", "KRW")
    precision = portfolio_settings.get("precision", 0)

    if currency == "AUD":

        def money_formatter(amount):
            return _aud_money_formatter(amount, precision)

        def price_formatter(p):
            return _aud_price_formatter(p, precision)

    else:  # kor
        money_formatter = format_kr_money
        price_formatter = _format_kr_price

    def format_shares(quantity):
        if country == "coin":
            return f"{quantity:,.8f}".rstrip("0").rstrip(".")
        if country == "aus":
            return f"{quantity:,.4f}".rstrip("0").rstrip(".")
        return f"{int(quantity):,d}"

    # 전략 설정 로드
    try:
        denom = int(portfolio_settings["portfolio_topn"])
        replace_weaker_stock = bool(portfolio_settings["replace_weaker_stock"])
        replace_threshold = float(portfolio_settings["replace_threshold"])
    except (KeyError, ValueError, TypeError) as e:
        raise ValueError(f"포트폴리오 설정값 로드 오류: {e}") from e

    if denom <= 0:
        raise ValueError(f"'{country}' 국가의 최대 보유 종목 수(portfolio_topn)는 0보다 커야 합니다.")

    # 포지션 비중 가이드라인: 모든 국가 동일 규칙 적용
    min_pos = 1.0 / (denom * 2.0)  # 최소 편입 비중
    max_pos = 1.0 / denom  # 목표/최대 비중 # noqa: F841

    # 현재 보유 종목 수 계산
    if country == "coin":
        held_count = sum(
            1
            for v in holdings.values()
            if float((v or {}).get("shares", 0.0)) > COIN_ZERO_THRESHOLD
        )
    else:
        held_count = sum(1 for v in holdings.values() if float((v or {}).get("shares", 0.0)) > 0)

    decisions = []

    for tkr, name in pairs:
        d = data_by_tkr.get(tkr)

        holding_info = holdings.get(tkr, {})
        sh = float(holding_info.get("shares", 0.0))
        ac = float(holding_info.get("avg_cost", 0.0))

        is_effectively_held = (sh > COIN_ZERO_THRESHOLD) if country == "coin" else (sh > 0)

        if not d and not is_effectively_held:
            continue

        if not d:
            d = {
                "price": 0.0,
                "prev_close": 0.0,
                "s1": float("nan"),
                "s2": float("nan"),
                "score": 0.0,
                "filter": 0,
            }

        price = d.get("price", 0.0)
        score = d.get("score", 0.0)

        buy_signal = False
        state = "HOLD" if is_effectively_held else "WAIT"
        phrase = ""
        is_active = full_etf_meta.get(tkr, {}).get("is_active", True)
        if price == 0.0 and is_effectively_held:
            phrase = "가격 데이터 조회 실패"

        buy_date = None
        holding_days = 0
        hold_ret = None

        consecutive_info = consecutive_holding_info.get(tkr)
        buy_date = consecutive_info.get("buy_date") if consecutive_info else None

        if is_effectively_held and buy_date and buy_date <= base_date:
            # Note: get_trading_days is not available here.
            # This part should ideally be handled in signals.py or passed as pre-calculated.
            # For now, using calendar days as a fallback.
            holding_days = (base_date - pd.to_datetime(buy_date).normalize()).days + 1

        hold_ret = (
            ((price / ac) - 1.0) * 100.0
            if (is_effectively_held and ac > 0 and pd.notna(price))
            else None
        )
        if is_effectively_held:
            if (
                stop_loss is not None
                and ac > 0
                and hold_ret is not None
                and hold_ret <= float(stop_loss)
            ):
                state = "CUT_STOPLOSS"
                qty = sh
                prof = (price - ac) * qty if ac > 0 else 0.0
                phrase = f"가격기반손절 {format_shares(qty)}주 @ {price_formatter(price)} 수익 {money_formatter(prof)} 손익률 {f'{hold_ret:+.1f}%'}"
            elif not is_active:
                state = "SELL_INACTIVE"
                qty = sh
                prof = (price - ac) * qty if ac > 0 else 0.0
                phrase = f"비활성 종목 정리 {format_shares(qty)}주 @ {price_formatter(price)} 수익 {money_formatter(prof)} 손익률 {f'{hold_ret:+.1f}%'}"

        if state == "HOLD":
            price_ma, ma = d["price"], d["s1"]
            if not pd.isna(price_ma) and not pd.isna(ma) and price_ma < ma:
                state = "SELL_TREND"
                qty = sh
                prof = (price_ma - ac) * qty if ac > 0 else 0.0
                tag = "추세이탈(이익)" if hold_ret >= 0 else "추세이탈(손실)"
                phrase = f"{tag} {format_shares(qty)}주 @ {price_formatter(price_ma)} 수익 {money_formatter(prof)} 손익률 {f'{hold_ret:+.1f}%'}"

        elif state == "WAIT":
            buy_signal_days_today = d["filter"]
            if buy_signal_days_today > 0:
                buy_signal = True
                phrase = f"추세진입 ({buy_signal_days_today}일째)"

        amount = sh * price if pd.notna(price) else 0.0

        # 일간 수익률 계산
        prev_close = d.get("prev_close", 0.0)
        day_ret = (
            ((price / prev_close) - 1.0) * 100.0
            if pd.notna(price) and pd.notna(prev_close) and prev_close > 0
            else 0.0
        )

        buy_date_display = buy_date.strftime("%Y-%m-%d") if buy_date else "-"
        holding_days_display = str(holding_days) if holding_days > 0 else "-"

        position_weight_pct = (amount / current_equity) * 100.0 if current_equity > 0 else 0.0

        current_row = [
            0,
            tkr,
            state,
            buy_date_display,
            holding_days_display,
            price,
            day_ret,
            sh,
            amount,
            hold_ret if hold_ret is not None else 0.0,
            position_weight_pct,
            (
                f"{d.get('drawdown_from_peak'):.1f}%"
                if d.get("drawdown_from_peak") is not None
                else "-"
            ),
            d.get("score"),
            f"{d['filter']}일" if d.get("filter") is not None else "-",
            phrase,
        ]
        decisions.append(
            {
                "state": state,
                "weight": position_weight_pct,
                "score": score,
                "tkr": tkr,
                "row": current_row,
                "buy_signal": buy_signal,
            }
        )

    universe_tickers = {
        etf["ticker"] for etf in full_etf_meta.values()
    }  # Use full_etf_meta for universe

    is_risk_off = regime_info and regime_info.get("is_risk_off", False)

    if is_risk_off:
        for decision in decisions:
            if decision["state"] == "HOLD":
                decision["state"] = "SELL_REGIME_FILTER"
                decision["row"][2] = "SELL_REGIME_FILTER"

                d_sell = data_by_tkr.get(decision["tkr"])
                if d_sell:
                    sell_price = float(d_sell.get("price", 0))
                    sell_qty = float(d_sell.get("shares", 0))
                    avg_cost = float(d_sell.get("avg_cost", 0))

                    hold_ret = 0.0
                    prof = 0.0
                    if avg_cost > 0 and sell_price > 0:
                        hold_ret = ((sell_price / avg_cost) - 1.0) * 100.0
                        prof = (sell_price - avg_cost) * sell_qty

                    sell_phrase = f"시장위험회피 매도 {format_shares(sell_qty)}주 @ {price_formatter(sell_price)} 수익 {money_formatter(prof)} 손익률 {f'{hold_ret:+.1f}%'}"
                    decision["row"][-1] = sell_phrase

            if decision.get("buy_signal"):
                decision["buy_signal"] = False
                if decision["state"] == "WAIT":
                    original_phrase = decision["row"][-1]
                    if original_phrase and "추세진입" in original_phrase:
                        decision["row"][-1] = f"시장 위험 회피 ({original_phrase})"
                    else:
                        decision["row"][-1] = "시장 위험 회피"
    else:
        # 모든 'WAIT' 상태의 매수 후보 목록을 미리 정의합니다.
        wait_candidates_raw = [
            d
            for d in decisions
            if d["state"] == "WAIT" and d.get("buy_signal") and d["tkr"] in universe_tickers
        ]

        other_sell_states = {"CUT_STOPLOSS", "SELL_TREND", "SELL_INACTIVE"}
        num_already_selling = sum(1 for d in decisions if d["state"] in other_sell_states)

        num_to_sell_for_rebalance = (held_count - num_already_selling) - denom

        if num_to_sell_for_rebalance > 0:
            rebalance_sell_candidates = [d for d in decisions if d["state"] == "HOLD"]
            rebalance_sell_candidates.sort(
                key=lambda x: x.get("score") if pd.notna(x.get("score")) else -float("inf")
            )
            tickers_to_sell = [
                d["tkr"] for d in rebalance_sell_candidates[:num_to_sell_for_rebalance]
            ]

            for decision in decisions:
                if decision["tkr"] in tickers_to_sell:
                    decision["state"] = "SELL_REBALANCE"
                    decision["row"][2] = "SELL_REBALANCE"
                    d_sell = data_by_tkr.get(decision["tkr"])
                    if d_sell:
                        sell_price = float(d_sell.get("price", 0))
                        sell_qty = float(d_sell.get("shares", 0))
                        avg_cost = float(d_sell.get("avg_cost", 0))
                        hold_ret = (
                            ((sell_price / avg_cost) - 1.0) * 100.0
                            if avg_cost > 0 and sell_price > 0
                            else 0.0
                        )
                        prof = (sell_price - avg_cost) * sell_qty if avg_cost > 0 else 0.0
                        sell_phrase = f"리밸런스 매도 {format_shares(sell_qty)}주 @ {price_formatter(sell_price)} 수익 {money_formatter(prof)} 손익률 {f'{hold_ret:+.1f}%'}"
                        decision["row"][-1] = sell_phrase
        else:
            slots_to_fill = denom - held_count
            if slots_to_fill > 0:
                best_wait_by_category = {}
                for cand in wait_candidates_raw:
                    category = etf_meta.get(cand["tkr"], {}).get("category")
                    key = (
                        category
                        if (category and category != "TBD")
                        else f"__individual_{cand['tkr']}"
                    )
                    if (
                        key not in best_wait_by_category
                        or cand["score"] > best_wait_by_category[key]["score"]
                    ):
                        best_wait_by_category[key] = cand

                # 점수 높은 순으로 최종 매수 후보 정렬
                buy_candidates_for_new_buy = sorted(
                    best_wait_by_category.values(),
                    key=lambda x: x["score"],
                    reverse=True,
                )

                final_buy_candidates, recommended_buy_categories = [], set()

                # 현재 보유 종목의 카테고리 (TBD 제외)
                held_categories = {
                    etf_meta.get(tkr, {}).get("category")
                    for tkr, d_holding in holdings.items()
                    if float((d_holding or {}).get("shares", 0.0)) > 0
                    and etf_meta.get(tkr, {}).get("category")
                    and etf_meta.get(tkr, {}).get("category") != "TBD"
                }

                for cand in buy_candidates_for_new_buy:
                    category = etf_meta.get(cand["tkr"], {}).get("category")
                    if category and category != "TBD":
                        # 보유 중인 카테고리와 중복되는지 확인
                        if category in held_categories:
                            cand["row"][-1] = "카테고리 중복 (보유)"
                            continue

                        # 이미 다른 종목이 추천 목록에 있는 카테고리는 건너뜀
                        if category in recommended_buy_categories:
                            cand["row"][-1] = f"카테고리 중복 (추천) - {category}"
                            continue
                        recommended_buy_categories.add(category)
                    final_buy_candidates.append(cand)

                available_cash, buys_made = total_cash, 0
                for cand in final_buy_candidates:
                    # "카테고리 중복"으로 이미 표시된 후보는 건너뜁니다.
                    if "카테고리 중복" in cand["row"][-1]:
                        continue
                    if buys_made < slots_to_fill:
                        d_cand, price = data_by_tkr.get(cand["tkr"]), 0
                        if d_cand:
                            price = d_cand.get("price", 0)
                        if price > 0:
                            min_val, max_val = (
                                min_pos * current_equity,
                                max_pos * current_equity,
                            )
                            budget = min(max_val, available_cash)
                            req_qty, buy_notional = 0, 0.0
                            if budget >= min_val and budget > 0:
                                if country in ("coin", "aus"):
                                    req_qty, buy_notional = budget / price, budget
                                else:
                                    req_qty = int(budget // price)
                                    buy_notional = req_qty * price
                                    if req_qty <= 0 or buy_notional + 1e-9 < min_val:
                                        req_qty, buy_notional = 0, 0.0
                            if req_qty > 0 and buy_notional <= available_cash + 1e-9:
                                cand["state"], cand["row"][2] = "BUY", "BUY"
                                buy_phrase = f"🚀 매수 {format_shares(req_qty)}주 @ {price_formatter(price)} ({money_formatter(buy_notional)})"
                                cand["row"][-1] = buy_phrase
                                available_cash -= buy_notional
                                buys_made += 1
                            else:
                                cand["row"][-1] = "현금 부족"
                        else:
                            cand["row"][-1] = f"가격 정보 없음 ({cand['row'][-1]})"
                    else:
                        if "추세진입" in cand["row"][-1]:
                            cand["row"][-1] = "포트폴리오 가득 참"
            else:
                if replace_weaker_stock:
                    # 1. 매수 후보 필터링: 각 카테고리별 1등만 추출
                    best_wait_by_category = {}
                    for cand in wait_candidates_raw:
                        category = etf_meta.get(cand["tkr"], {}).get("category")
                        # 카테고리가 없거나 'TBD'인 종목은 개별적으로 처리
                        key = (
                            category
                            if (category and category != "TBD")
                            else f"__individual_{cand['tkr']}"
                        )

                        if (
                            key not in best_wait_by_category
                            or cand["score"] > best_wait_by_category[key]["score"]
                        ):
                            best_wait_by_category[key] = cand

                    # 점수 높은 순으로 최종 교체 후보 정렬
                    buy_candidates_for_replacement = sorted(
                        best_wait_by_category.values(),
                        key=lambda x: x["score"],
                        reverse=True,
                    )

                    # 2. 교체 로직 실행
                    current_held_stocks = [d for d in decisions if d["state"] == "HOLD"]
                    current_held_stocks.sort(
                        key=lambda x: x["score"] if pd.notna(x["score"]) else -float("inf")
                    )

                    for best_new in buy_candidates_for_replacement:
                        if not current_held_stocks:
                            break

                        wait_stock_category = etf_meta.get(best_new["tkr"], {}).get("category")

                        # 2-1. 동일 카테고리 보유 종목과 비교
                        held_stock_same_category = next(
                            (
                                s
                                for s in current_held_stocks
                                if wait_stock_category
                                and wait_stock_category != "TBD"
                                and etf_meta.get(s["tkr"], {}).get("category")
                                == wait_stock_category
                            ),
                            None,
                        )

                        ticker_to_sell = None
                        if held_stock_same_category:
                            # 동일 카테고리 보유 종목이 있으면, 점수만 비교 (임계값 미적용)
                            if (
                                pd.notna(best_new["score"])
                                and pd.notna(held_stock_same_category["score"])
                                and best_new["score"] > held_stock_same_category["score"]
                            ):
                                ticker_to_sell = held_stock_same_category["tkr"]
                            else:
                                # 점수가 더 높지 않으면 교체하지 않음
                                continue
                        else:
                            # 2-2. 동일 카테고리가 없으면, 가장 약한 보유 종목과 비교 (임계값 적용)
                            weakest_held = current_held_stocks[0]
                            if (
                                pd.notna(best_new["score"])
                                and pd.notna(weakest_held["score"])
                                and best_new["score"] > weakest_held["score"] + replace_threshold
                            ):
                                ticker_to_sell = weakest_held["tkr"]
                            else:
                                # 임계값을 넘지 못하면 교체하지 않음
                                continue

                        if ticker_to_sell:
                            # 3. 교체 실행
                            d_weakest = data_by_tkr.get(ticker_to_sell)
                            if d_weakest:
                                # (a) 매도 신호 생성
                                sell_price, sell_qty, avg_cost = (
                                    float(d_weakest.get(k, 0))
                                    for k in ["price", "shares", "avg_cost"]
                                )
                                hold_ret = (
                                    ((sell_price / avg_cost) - 1.0) * 100.0
                                    if avg_cost > 0 and sell_price > 0
                                    else 0.0
                                )
                                prof = (sell_price - avg_cost) * sell_qty if avg_cost > 0 else 0.0
                                sell_phrase = f"교체매도 {format_shares(sell_qty)}주 @ {price_formatter(sell_price)} 수익 {money_formatter(prof)} 손익률 {f'{hold_ret:+.1f}%'} ({best_new['tkr']}(으)로 교체)"

                                for d_item in decisions:
                                    if d_item["tkr"] == ticker_to_sell:
                                        d_item["state"], d_item["row"][2], d_item["row"][-1] = (
                                            "SELL_REPLACE",
                                            "SELL_REPLACE",
                                            sell_phrase,
                                        )
                                        break

                            # (b) 매수 신호 생성
                            best_new["state"], best_new["row"][2] = "BUY_REPLACE", "BUY_REPLACE"
                            buy_price = float(data_by_tkr.get(best_new["tkr"], {}).get("price", 0))
                            if buy_price > 0:
                                # 매도 금액만큼 매수 예산 설정
                                sell_value_for_budget = 0.0
                                for d_item in decisions:
                                    if d_item["tkr"] == ticker_to_sell and d_item.get("weight"):
                                        sell_value_for_budget = (
                                            d_item["weight"] / 100.0 * current_equity
                                        )
                                        break
                                if sell_value_for_budget == 0.0 and d_weakest:
                                    sell_value_for_budget = d_weakest.get(
                                        "shares", 0.0
                                    ) * d_weakest.get("price", 0.0)

                                if sell_value_for_budget > 0:  # noqa
                                    buy_qty = (
                                        sell_value_for_budget / buy_price
                                        if country in ("coin", "aus")
                                        else int(sell_value_for_budget // buy_price)
                                    )
                                    buy_notional = buy_qty * buy_price
                                    best_new["row"][
                                        -1
                                    ] = f"매수 {format_shares(buy_qty)}주 @ {price_formatter(buy_price)} ({money_formatter(buy_notional)}) ({ticker_to_sell} 대체)"
                                else:
                                    best_new["row"][-1] = f"{ticker_to_sell}(을)를 대체 (매수 예산 부족)"
                            else:
                                best_new["row"][-1] = f"{ticker_to_sell}(을)를 대체 (가격정보 없음)"

                            # 교체가 일어났으므로, 다음 후보 검증을 위해 상태 업데이트
                            current_held_stocks = [
                                s for s in current_held_stocks if s["tkr"] != ticker_to_sell
                            ]
                            best_new_as_held = best_new.copy()
                            best_new_as_held["state"] = "HOLD"
                            current_held_stocks.append(best_new_as_held)
                            current_held_stocks.sort(
                                key=lambda x: x["score"] if pd.notna(x["score"]) else -float("inf")
                            )

    # --- 최종 필터링: 카테고리별 1등이 아닌 WAIT 종목 제거 ---
    best_wait_by_category = {}
    for cand in wait_candidates_raw:
        category = etf_meta.get(cand["tkr"], {}).get("category")
        key = category if (category and category != "TBD") else f"__individual_{cand['tkr']}"
        if key not in best_wait_by_category or cand["score"] > best_wait_by_category[key]["score"]:
            best_wait_by_category[key] = cand

    best_wait_tickers = {cand["tkr"] for cand in best_wait_by_category.values()}

    # 최종 decisions 리스트에서 카테고리 1등이 아닌 WAIT 종목을 제거합니다.
    final_decisions = []
    for d in decisions:
        # WAIT 상태이고, buy_signal이 있으며, best_wait_tickers에 없는 종목은 제외
        if d["state"] == "WAIT" and d.get("buy_signal") and d["tkr"] not in best_wait_tickers:
            continue
        final_decisions.append(d)

    # 포트폴리오가 가득 찼을 때, 매수 추천되지 않은 WAIT 종목에 사유 기록
    if slots_to_fill <= 0:
        for d in final_decisions:
            if d["state"] == "WAIT" and d.get("buy_signal"):
                # 이미 다른 사유가 기록된 경우는 제외
                if "추세진입" in d["row"][-1]:
                    d["row"][-1] = "포트폴리오 가득 참"

    # 최종 정렬
    def sort_key(decision_dict):
        state = decision_dict["state"]
        score = decision_dict["score"]
        tkr = decision_dict["tkr"]

        state_order = {
            "HOLD": 0,
            "CUT_STOPLOSS": 1,
            "SELL_MOMENTUM": 2,
            "SELL_TREND": 3,
            "SELL_REPLACE": 4,
            "SELL_REBALANCE": 4,
            "SOLD": 5,
            "BUY_REPLACE": 6,
            "BUY": 7,
            "WAIT": 8,
        }
        order = state_order.get(state, 99)

        sort_value = -score
        return (order, sort_value, tkr)

    final_decisions.sort(key=sort_key)

    return final_decisions


def run_single_ticker_backtest(
    ticker: str,
    stock_type: str = "stock",
    df: Optional[pd.DataFrame] = None,
    initial_capital: float = 1_000_000.0,
    core_start_date: Optional[pd.Timestamp] = None,
    date_range: Optional[List[str]] = None,
    country: str = "kor",
) -> pd.DataFrame:
    """
    단일 종목에 대해 이동평균선 교차 전략 백테스트를 실행합니다.
    """
    from . import settings

    try:
        # 전략 고유 설정
        ma_period_etf = int(settings.MA_PERIOD)
        ma_period_stock = int(settings.MA_PERIOD)
    except AttributeError as e:
        raise AttributeError(f"'{e.name}' 설정이 logic/settings.py 파일에 반드시 정의되어야 합니다.") from e

    try:
        # 공통 설정
        stop_loss = settings.HOLDING_STOP_LOSS_PCT
        cooldown_days = int(settings.COOLDOWN_DAYS)
    except AttributeError as e:
        raise AttributeError(f"'{e.name}' 설정이 logic/settings.py 파일에 반드시 정의되어야 합니다.") from e

    # --- 티커 유형(ETF/주식) 구분 ---
    ma_period = ma_period_etf if stock_type == "etf" else ma_period_stock
    if df is None:
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
            if buy_signal_days_today > 0:
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
            ma_score_today = (price / ma_today) - 1.0

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
