from collections.abc import Collection, Mapping, Sequence
from typing import Any

import pandas as pd

from strategies.maps.rules import StrategyRules
from utils.data_loader import get_exchange_rate_series, get_latest_trading_day
from utils.logger import get_app_logger
from utils.settings_loader import (
    get_account_settings,
    get_strategy_rules,
)
from utils.stock_list_io import get_etfs

from .analysis import (
    build_full_summary,
    build_portfolio_dataframe,
    build_ticker_summaries,
    extract_evaluated_records,
)
from .domain import AccountBacktestResult
from .engine import run_portfolio_backtest

logger = get_app_logger()


def run_account_backtest(
    account_id: str,
    *,
    initial_capital: float | None = None,
    quiet: bool = False,
    prefetched_data: Mapping[str, pd.DataFrame] | None = None,
    override_settings: dict[str, Any] | None = None,
    strategy_override: StrategyRules | None = None,
    excluded_tickers: Collection[str] | None = None,
    prefetched_etf_universe: Sequence[Mapping[str, Any]] | None = None,
    prefetched_metrics: Mapping[str, dict[str, Any]] | None = None,
    trading_calendar: Sequence[pd.Timestamp] | None = None,
    prefetched_fx_series: pd.Series | None = None,
) -> AccountBacktestResult:
    """Runs backtest for a specific account."""

    # 1. Load Settings
    account_settings = get_account_settings(account_id)
    country_code = (account_settings.get("country_code") or "kor").lower()

    if strategy_override:
        rules = strategy_override
    else:
        # Load from config
        base_rules = get_strategy_rules(account_id)
        rules = StrategyRules.from_mapping(base_rules.to_dict())

    # Apply Overrides
    if override_settings and "strategy_overrides" in override_settings:
        pass  # TODO: Implement override logic if needed

    # 2. Capital & Date
    start_date_str = account_settings.get("strategy", {}).get("BACKTEST_START_DATE", "2024-01-01")
    if override_settings and "backtest_start_date" in override_settings:
        start_date_str = override_settings["backtest_start_date"]

    start_date = pd.to_datetime(start_date_str)
    end_date = get_latest_trading_day(country_code)

    init_cap_krw = float(account_settings.get("initial_capital_krw", 100_000_000))
    currency = account_settings.get("currency", "KRW").upper()

    # Resolve Initial Capital
    init_cap_local = None

    # 1. Override/Arg takes precedence
    if initial_capital is not None:
        init_cap_local = initial_capital
    elif account_settings.get("initial_capital"):
        init_cap_local = float(account_settings["initial_capital"])

    # 2. If not set, Convert from KRW if needed
    if init_cap_local is None:
        if currency == "KRW":
            init_cap_local = init_cap_krw
        else:
            # Fetch FX Rate at Start Date
            try:
                # Needed to fetch FX.
                # Optimization: check prefetched first
                fx_rate = 1.0
                if prefetched_fx_series is not None:
                    if start_date in prefetched_fx_series.index:
                        fx_rate = float(prefetched_fx_series.loc[start_date])
                    else:
                        # asof
                        idx = prefetched_fx_series.index.asof(start_date)
                        if pd.notna(idx):
                            fx_rate = float(prefetched_fx_series.loc[idx])
                else:
                    # Fetch from DB/API
                    search_start = start_date - pd.Timedelta(days=10)
                    fx_series = get_exchange_rate_series(search_start, start_date + pd.Timedelta(days=5))
                    if fx_series is not None and not fx_series.empty:
                        idx = fx_series.index.asof(start_date)
                        if pd.notna(idx):
                            fx_rate = float(fx_series.loc[idx])

                if fx_rate > 0:
                    init_cap_local = init_cap_krw / fx_rate
                else:
                    logger.warning(f"Invalid FX rate {fx_rate} for {currency}. Using 1.0")
                    init_cap_local = init_cap_krw

            except Exception as e:
                logger.warning(f"Failed to fetch FX rate for {currency} at {start_date}: {e}. Using KRW value as is.")
                init_cap_local = init_cap_krw

    # 3. Stocks Universe
    if prefetched_etf_universe:
        stocks = [dict(s) for s in prefetched_etf_universe]
    else:
        stocks = get_etfs(account_id)

    if excluded_tickers:
        excl = set(t.upper() for t in excluded_tickers)
        stocks = [s for s in stocks if s["ticker"].upper() not in excl]

    # 4. Bucket Map
    bucket_map = {}
    for s in stocks:
        b = s.get("bucket") or s.get("group")
        if b:
            try:
                bucket_map[s["ticker"]] = int(b)
            except Exception:
                pass

    # 5. Run Engine
    ticker_timeseries = run_portfolio_backtest(
        stocks=stocks,
        initial_capital=init_cap_local,
        core_start_date=start_date,
        country=country_code,
        prefetched_data=prefetched_data,
        prefetched_metrics=prefetched_metrics,
        trading_calendar=trading_calendar,
        # Strategy Params
        ma_days=rules.ma_days,
        ma_type=rules.ma_type,
        stop_loss_pct=rules.stop_loss_pct,
        replace_threshold=rules.replace_threshold,
        rebalance_mode=rules.rebalance_mode,
        bucket_map=bucket_map,
        bucket_topn=rules.bucket_topn,
        allow_negative_score=rules.allow_negative_score,
        quiet=quiet,
    )

    # 6. Analysis
    effective_topn = rules.portfolio_topn
    if rules.rebalance_mode == "QUARTERLY" and rules.bucket_topn > 0:
        effective_topn = rules.bucket_topn * 5  # Approx

    portfolio_df = build_portfolio_dataframe(ticker_timeseries, init_cap_local, effective_topn)

    summary = build_full_summary(
        portfolio_df=portfolio_df,
        start_date=start_date,
        end_date=end_date,
        initial_capital=init_cap_local,
        initial_capital_krw=init_cap_krw,
        currency=currency,
        portfolio_topn=effective_topn,
        account_settings=account_settings,
        prefetched_data=prefetched_data,
        ticker_timeseries=ticker_timeseries,
        ticker_meta={s["ticker"]: s for s in stocks},
    )

    evaluated_records = extract_evaluated_records(ticker_timeseries)

    ticker_meta = {s["ticker"]: s for s in stocks}
    ticker_summaries = build_ticker_summaries(ticker_timeseries, ticker_meta)

    # 7. Result Object
    return AccountBacktestResult(
        account_id=account_id,
        country_code=country_code,
        start_date=start_date,
        end_date=end_date,
        initial_capital=init_cap_local,
        initial_capital_krw=init_cap_krw,
        currency=currency,
        portfolio_topn=effective_topn,
        holdings_limit=effective_topn,
        summary=summary,
        portfolio_timeseries=portfolio_df,
        ticker_timeseries=ticker_timeseries,
        ticker_meta=ticker_meta,
        evaluated_records=evaluated_records,
        monthly_returns=summary.get("monthly_returns"),
        monthly_cum_returns=summary.get("monthly_cum_returns"),
        yearly_returns=summary.get("yearly_returns"),
        ticker_summaries=ticker_summaries,
        settings_snapshot={},
        backtest_start_date=str(start_date_str),
        missing_tickers=[],
    )
