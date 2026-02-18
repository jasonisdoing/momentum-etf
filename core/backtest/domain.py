from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    stocks: list[dict]
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    initial_capital: float
    country_code: str
    top_n: int
    ma_days: int
    ma_type: str
    bucket_map: dict[str, int]
    bucket_topn: int
    rebalance_mode: str
    quiet: bool


@dataclass
class MarketData:
    """Aligned market data for all tickers."""

    union_index: pd.DatetimeIndex
    # Ticker -> Field -> Array
    # Fields: close, open, high, low, ma, score, rsi, buy_signal
    # access: dates[i], prices[ticker][i]
    close_prices: dict[str, np.ndarray]
    open_prices: dict[str, np.ndarray]
    ma_values: dict[str, np.ndarray]
    scores: dict[str, np.ndarray]
    rsi_scores: dict[str, np.ndarray]
    buy_signals: dict[str, np.ndarray]
    available_mask: dict[str, np.ndarray]

    # Original DataFrames for debugging/details
    raw_frames: dict[str, pd.DataFrame]


@dataclass
class PortfolioState:
    cash: float
    positions: dict[str, dict]  # ticker -> {shares, avg_cost, buy_block_until, sell_block_until}
    daily_records: dict[str, list[dict]]
    trades: list[dict]  # Flat list of all trades? Or per ticker?

    def get_shares(self, ticker: str) -> float:
        return self.positions.get(ticker, {}).get("shares", 0.0)

    def get_avg_cost(self, ticker: str) -> float:
        return self.positions.get(ticker, {}).get("avg_cost", 0.0)


@dataclass
class AccountBacktestResult:
    """Result container mainly for compatibility and comprehensive reporting."""

    account_id: str
    country_code: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    initial_capital: float
    initial_capital_krw: float
    currency: str
    bucket_topn: int
    holdings_limit: int
    summary: dict[str, Any]
    portfolio_timeseries: pd.DataFrame
    ticker_timeseries: dict[str, pd.DataFrame]
    ticker_meta: dict[str, dict[str, Any]]
    evaluated_records: dict[str, dict[str, Any]]
    monthly_returns: pd.Series
    monthly_cum_returns: pd.Series
    yearly_returns: pd.Series
    ticker_summaries: list[dict[str, Any]]
    settings_snapshot: dict[str, Any]
    backtest_start_date: str
    missing_tickers: list[str]

    def to_dict(self) -> dict[str, Any]:
        df = self.portfolio_timeseries.copy()
        df.index = df.index.astype(str)
        return {
            "account_id": self.account_id,
            "country_code": self.country_code,
            "start_date": self.start_date.strftime("%Y-%m-%d"),
            "end_date": self.end_date.strftime("%Y-%m-%d"),
            "initial_capital": float(self.initial_capital),
            "initial_capital_krw": float(self.initial_capital_krw),
            "currency": self.currency,
            "bucket_topn": self.bucket_topn,
            "holdings_limit": self.holdings_limit,
            "summary": self.summary,
            "portfolio_timeseries": df.to_dict(orient="records"),
            "ticker_meta": self.ticker_meta,
            "evaluated_records": self.evaluated_records,
            # Handle float series to dict potentially needing conversion?
            # Pandas to_dict usually handles it.
            "monthly_returns": self.monthly_returns.to_dict(),
            "monthly_cum_returns": self.monthly_cum_returns.to_dict(),
            "yearly_returns": self.yearly_returns.to_dict(),
            "ticker_summaries": self.ticker_summaries,
            "settings_snapshot": self.settings_snapshot,
            "backtest_start_date": self.backtest_start_date,
            "missing_tickers": self.missing_tickers,
        }
