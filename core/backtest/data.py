from collections.abc import Mapping
from typing import Any

import pandas as pd

from strategies.maps.metrics import process_ticker_data
from utils.logger import get_app_logger

from .domain import BacktestConfig, MarketData

logger = get_app_logger()


def load_and_process_data(
    config: BacktestConfig,
    prefetched_data: dict[str, pd.DataFrame],
    prefetched_metrics: Mapping[str, dict[str, Any]] | None,
    enable_data_sufficiency_check: bool = False,
) -> dict[str, dict]:
    """Loads raw data and calculates metrics for each ticker."""
    metrics_by_ticker = {}
    tickers = [s["ticker"] for s in config.stocks]

    for ticker in tickers:
        df = prefetched_data.get(ticker)
        if df is None:
            logger.warning(f"Data missing for {ticker}")
            continue

        precomputed = prefetched_metrics.get(ticker) if prefetched_metrics else None
        metrics = process_ticker_data(
            ticker,
            df,
            ma_days=config.ma_days,
            ma_type=config.ma_type,
            precomputed_entry=precomputed,
            enable_data_sufficiency_check=enable_data_sufficiency_check,
        )
        if metrics:
            metrics_by_ticker[ticker] = metrics

    return metrics_by_ticker


def align_market_data(config: BacktestConfig, metrics_by_ticker: dict[str, dict]) -> MarketData | None:
    """Aligns all ticker data to a common union index."""
    if not metrics_by_ticker:
        return None

    # 1. Create Union Index
    union_index = pd.DatetimeIndex([])
    for m in metrics_by_ticker.values():
        union_index = union_index.union(m["close"].index)

    if config.start_date:
        union_index = union_index[union_index >= config.start_date]

    if union_index.empty:
        return None

    # 2. Reindex and vectorize
    close_prices = {}
    open_prices = {}
    ma_values = {}
    scores = {}
    rsi_scores = {}
    buy_signals = {}
    available_mask = {}
    raw_frames = {}

    for ticker, m in metrics_by_ticker.items():
        # Reindex
        c = m["close"].reindex(union_index)
        o = m["open"].reindex(union_index)
        ma = m["ma"].reindex(union_index)
        sc = m["ma_score"].reindex(union_index)
        rsi = m.get("rsi_score", pd.Series(dtype=float)).reindex(union_index)
        sig = m["buy_signal_days"].reindex(union_index).fillna(0).astype(int)

        # Store numpy arrays
        close_prices[ticker] = c.to_numpy()
        open_prices[ticker] = o.to_numpy()
        ma_values[ticker] = ma.to_numpy()
        scores[ticker] = sc.to_numpy()
        rsi_scores[ticker] = rsi.to_numpy()
        buy_signals[ticker] = sig.to_numpy()
        available_mask[ticker] = c.notna().to_numpy()

        # Store raw frame if needed (optional, maybe trimmed)
        # raw_frames[ticker] = m.get("raw_df") # logic/metrics might not return raw_df

    return MarketData(
        union_index=union_index,
        close_prices=close_prices,
        open_prices=open_prices,
        ma_values=ma_values,
        scores=scores,
        rsi_scores=rsi_scores,
        buy_signals=buy_signals,
        available_mask=available_mask,
        raw_frames=raw_frames,
    )
