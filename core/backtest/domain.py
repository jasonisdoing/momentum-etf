from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass
class AccountBacktestResult:
    """계정 백테스트 결과 컨테이너."""

    account_id: str
    country_code: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    initial_capital: float
    initial_capital_krw: float
    currency: str
    universe_count: int
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
            "universe_count": self.universe_count,
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
