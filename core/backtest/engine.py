from typing import Any

import pandas as pd

from utils.logger import get_app_logger

from .data import align_market_data, load_and_process_data
from .domain import BacktestConfig, MarketData, PortfolioState
from .execution import execute_rebalance_buy, execute_rebalance_sell
from .portfolio import calculate_rebalance_targets, get_current_holdings_value


def init_daily_record(state: PortfolioState, market_data: MarketData, i: int, dt: pd.Timestamp) -> None:
    """Initializes the daily record for each ticker."""
    for ticker in market_data.close_prices.keys():
        if ticker not in state.positions:
            state.positions[ticker] = {"shares": 0.0, "avg_cost": 0.0}

        pos = state.positions[ticker]
        price = market_data.close_prices[ticker][i]
        score = market_data.scores[ticker][i]
        rsi = market_data.rsi_scores[ticker][i]
        signal = market_data.buy_signals[ticker][i]

        state.daily_records.setdefault(ticker, []).append(
            {
                "date": dt,
                "shares": pos["shares"],
                "avg_cost": pos["avg_cost"],
                "price": price,
                "pv": pos["shares"] * price if pd.notna(price) else 0.0,
                "decision": "HOLD" if pos["shares"] > 0 else "WAIT",
                "score": score if pd.notna(score) else None,
                "rsi_score": rsi if pd.notna(rsi) else None,
                "filter": signal,
                "note": "",
                "trade_amount": 0.0,
                "trade_profit": 0.0,
                "trade_pl_pct": 0.0,
            }
        )

    # Handle CASH specially
    state.daily_records.setdefault("CASH", []).append(
        {
            "date": dt,
            "shares": state.cash,
            "avg_cost": 1.0,
            "price": 1.0,
            "pv": state.cash,
            "decision": "HOLD",
            "score": None,
            "rsi_score": None,
            "filter": None,
            "note": "",
            "trade_amount": 0.0,
            "trade_profit": 0.0,
            "trade_pl_pct": 0.0,
        }
    )


def compile_daily_series(state: PortfolioState) -> dict[str, pd.DataFrame]:
    """Compiles daily records into DataFrames."""
    results = {}
    for ticker, records in state.daily_records.items():
        results[ticker] = pd.DataFrame(records).set_index("date")
    return results


logger = get_app_logger()


def process_rebalance_day(
    i: int, total_days: int, dt: pd.Timestamp, state: PortfolioState, market_data: Any, config: BacktestConfig
) -> None:
    """Executes rebalancing logic for the day."""
    # 1. Prepare Data
    # available tickers logic equivalent to old engine
    available_tickers = []
    today_scores = {}
    today_prices = {}

    for ticker in market_data.close_prices:
        if market_data.available_mask[ticker][i]:
            available_tickers.append(ticker)
            today_scores[ticker] = market_data.scores[ticker][i]
            today_prices[ticker] = market_data.close_prices[ticker][i]

    # 2. Calculate Targets
    current_equity = state.cash + get_current_holdings_value(state, today_prices)
    targets = calculate_rebalance_targets(config, current_equity, available_tickers, today_scores, today_prices)

    # 3. Execute Overweight Sells (and Exits)
    # Check all current positions + new targets
    # All involved check is implicit in the logic below

    # Sell Logic
    for ticker in list(state.positions.keys()):  # List to avoid runtime error
        if ticker == "CASH":
            continue
        # If not in targets, target is 0
        target = targets.get(ticker, 0.0)
        execute_rebalance_sell(i, total_days, state, ticker, target, market_data, config, dt)

    # Buy Logic
    for ticker, target in targets.items():
        execute_rebalance_buy(i, total_days, state, ticker, target, market_data, config, dt)


def run_portfolio_backtest(
    stocks: list[dict],
    initial_capital: float = 100_000_000.0,
    core_start_date: pd.Timestamp | None = None,
    # ... Many args ...
    # To keep it loosely compatible with old signature or use **kwargs fallback
    **kwargs,
) -> dict[str, pd.DataFrame]:
    """Orchestrates the backtest."""

    # 1. Config Setup
    config = BacktestConfig(
        stocks=stocks,
        start_date=core_start_date,
        end_date=None,  # Derived from data usually
        initial_capital=initial_capital,
        country_code=kwargs.get("country", "kor"),
        top_n=kwargs.get("top_n", 10),
        ma_days=kwargs.get("ma_days", 20),
        bucket_map=kwargs.get("bucket_map", {}),
        bucket_topn=kwargs.get("bucket_topn", 1),
        rebalance_mode=kwargs.get("rebalance_mode", "QUARTERLY"),
        allow_negative_score=kwargs.get("allow_negative_score", True),
        quiet=kwargs.get("quiet", False),
    )

    # 2. Data Loading
    raw_metrics = load_and_process_data(
        config,
        kwargs.get("prefetched_data", {}),
        kwargs.get("prefetched_metrics"),
        kwargs.get("enable_data_sufficiency_check", False),
    )
    market_data = align_market_data(config, raw_metrics)

    if not market_data:
        return {}

    # 3. State Init
    state = PortfolioState(
        cash=config.initial_capital,
        positions={
            t: {"shares": 0.0, "avg_cost": 0.0, "buy_block_until": -1, "sell_block_until": -1}
            for t in market_data.close_prices.keys()
        },
        daily_records={},
        trades=[],
    )
    # Also add CASH to daily records if not present
    if "CASH" not in state.positions:
        state.positions["CASH"] = {"shares": config.initial_capital, "avg_cost": 1.0}

    total_days = len(market_data.union_index)

    # 4. Main Loop
    for i, dt in enumerate(market_data.union_index):
        # Snapshot
        init_daily_record(state, market_data, i, dt)

        # Check Rebalance
        is_quarter_end = dt.month in {3, 6, 9, 12} and (
            i == total_days - 1 or market_data.union_index[i + 1].month != dt.month
        )

        if config.rebalance_mode == "QUARTERLY" and (is_quarter_end or i == 0):
            process_rebalance_day(i, total_days, dt, state, market_data, config)

        # Daily Logic (for non-quarterly) - Skipped for now per request scope

    # 5. Compile Results
    return compile_daily_series(state)
