"""국가 기반 백테스트 실행 유틸리티.

계좌 기반 스크립트를 제거하고, 국가 설정과 전략 규칙만을 활용하여 백테스트를 실행합니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, List, Tuple
import math

import pandas as pd

from constants import TEST_INITIAL_CAPITAL, TEST_MONTHS_RANGE
from logic.entry_point import run_portfolio_backtest, StrategyRules
from utils.country_registry import get_common_file_settings
from utils.settings_loader import (
    CountrySettingsError,
    get_country_precision,
    get_country_settings,
    get_country_strategy,
    get_strategy_rules,
)
from utils.data_loader import get_latest_trading_day, fetch_ohlcv
from utils.stock_list_io import get_etfs


def _default_test_months_range() -> int:
    return TEST_MONTHS_RANGE


def _default_initial_capital() -> float:
    return float(TEST_INITIAL_CAPITAL)


@dataclass
class CountryBacktestResult:
    """국가 기반 백테스트 결과."""

    country: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    initial_capital: float
    portfolio_topn: int
    holdings_limit: int
    summary: Dict[str, Any]
    portfolio_timeseries: pd.DataFrame
    ticker_timeseries: Dict[str, pd.DataFrame]
    ticker_meta: Dict[str, Dict[str, Any]]
    evaluated_records: Dict[str, Dict[str, Any]]
    monthly_returns: pd.Series
    monthly_cum_returns: pd.Series
    yearly_returns: pd.Series
    risk_off_periods: List[Tuple[pd.Timestamp, pd.Timestamp]]
    ticker_summaries: List[Dict[str, Any]]
    settings_snapshot: Dict[str, Any]
    months_range: int

    def to_dict(self) -> Dict[str, Any]:
        df = self.portfolio_timeseries.copy()
        df.index = df.index.astype(str)
        return {
            "country": self.country,
            "start_date": self.start_date.strftime("%Y-%m-%d"),
            "end_date": self.end_date.strftime("%Y-%m-%d"),
            "initial_capital": float(self.initial_capital),
            "portfolio_topn": self.portfolio_topn,
            "holdings_limit": self.holdings_limit,
            "summary": self.summary,
            "portfolio_timeseries": df.to_dict(orient="records"),
            "ticker_meta": self.ticker_meta,
            "evaluated_records": self.evaluated_records,
            "monthly_returns": self.monthly_returns.to_dict(),
            "monthly_cum_returns": self.monthly_cum_returns.to_dict(),
            "yearly_returns": self.yearly_returns.to_dict(),
            "risk_off_periods": [
                (s.strftime("%Y-%m-%d"), e.strftime("%Y-%m-%d")) for s, e in self.risk_off_periods
            ],
            "ticker_summaries": self.ticker_summaries,
            "settings_snapshot": self.settings_snapshot,
            "months_range": self.months_range,
        }


def run_country_backtest(
    country: str,
    *,
    months_range: Optional[int] = None,
    initial_capital: Optional[float] = None,
    quiet: bool = False,
    prefetched_data: Optional[Mapping[str, pd.DataFrame]] = None,
    override_settings: Optional[Dict[str, Any]] = None,
    strategy_override: Optional[StrategyRules] = None,  # type: ignore
) -> CountryBacktestResult:
    """국가 코드를 기반으로 백테스트를 실행합니다."""

    def _log(message: str) -> None:
        if not quiet:
            print(message)

    _log(f"[백테스트] {country.upper()} 백테스트를 시작합니다...")

    override_settings = override_settings or {}
    country = (country or "").strip().lower()
    if not country:
        raise CountrySettingsError("국가 코드를 지정해야 합니다.")

    _log("[백테스트] 설정을 로드하는 중...")
    country_settings = get_country_settings(country)
    base_strategy_rules = get_strategy_rules(country)
    strategy_rules = StrategyRules.from_mapping(base_strategy_rules.to_dict())
    precision_settings = get_country_precision(country)
    strategy_settings = dict(get_country_strategy(country))
    common_settings = get_common_file_settings()

    if strategy_override is not None:
        strategy_rules = StrategyRules.from_values(
            ma_period=strategy_override.ma_period,
            portfolio_topn=strategy_override.portfolio_topn,
            replace_threshold=strategy_override.replace_threshold,
        )
        strategy_settings["MA_PERIOD"] = strategy_rules.ma_period
        strategy_settings["PORTFOLIO_TOPN"] = strategy_rules.portfolio_topn
        strategy_settings["REPLACE_SCORE_THRESHOLD"] = strategy_rules.replace_threshold

    months_range = _resolve_months_range(months_range, override_settings)
    end_date = _resolve_end_date(country, override_settings)
    start_date = _resolve_start_date(end_date, months_range, override_settings)

    initial_capital_value = _resolve_initial_capital(
        initial_capital,
        override_settings,
        country_settings,
        precision_settings,
    )

    _log(f"[백테스트] {country.upper()} ETF 목록을 로드하는 중...")
    etf_universe = get_etfs(country)
    if not etf_universe:
        raise CountrySettingsError(f"'data/stocks/{country}.json' 파일에서 종목을 찾을 수 없습니다.")
    _log(f"[백테스트] {len(etf_universe)}개의 ETF를 찾았습니다.")

    ticker_meta = {str(item.get("ticker", "")).upper(): dict(item) for item in etf_universe}
    ticker_meta["CASH"] = {"ticker": "CASH", "name": "현금", "category": "-"}

    portfolio_topn = strategy_rules.portfolio_topn
    holdings_limit = int(strategy_settings.get("MAX_PER_CATEGORY", 0) or 0)
    _log(f"[백테스트] 포트폴리오 TOPN: {portfolio_topn}, 카테고리당 최대 보유 수: {holdings_limit}")

    _log("[백테스트] 백테스트 파라미터를 구성하는 중...")
    backtest_kwargs = _build_backtest_kwargs(
        country=country,
        strategy_rules=strategy_rules,
        common_settings=common_settings,
        strategy_settings=strategy_settings,
        prefetched_data=prefetched_data,
        quiet=quiet,
    )

    date_range = [start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")]

    _log(
        f"[백테스트] {country.upper()} 백테스트 실행 | 기간: {date_range[0]}~{date_range[1]} | "
        f"초기 자본: {initial_capital_value:,.0f}"
    )

    _log("[백테스트] 포트폴리오 백테스트 실행 중...")

    ticker_timeseries = (
        run_portfolio_backtest(
            stocks=etf_universe,
            initial_capital=initial_capital_value,
            core_start_date=start_date,
            top_n=portfolio_topn,
            date_range=date_range,
            country=country,
            **backtest_kwargs,
        )
        or {}
    )
    _log(f"[백테스트] 백테스트 완료. {len(ticker_timeseries)}개 종목의 데이터가 생성되었습니다.")

    if not ticker_timeseries:
        raise RuntimeError("백테스트 결과가 비어 있습니다. 유효한 데이터가 없습니다.")

    portfolio_df = _build_portfolio_timeseries(
        ticker_timeseries,
        initial_capital_value,
        portfolio_topn,
    )

    (
        summary,
        monthly_returns,
        monthly_cum_returns,
        yearly_returns,
        risk_off_periods,
    ) = _build_summary(
        portfolio_df,
        ticker_timeseries=ticker_timeseries,
        country=country,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital_value,
        country_settings=country_settings,
    )

    evaluated_records = _compute_evaluated_records(ticker_timeseries)

    ticker_summaries = _build_ticker_summaries(
        ticker_timeseries,
        ticker_meta,
        start_date,
    )

    _log("[백테스트] 설정 스냅샷을 생성하는 중...")
    settings_snapshot = _build_settings_snapshot(
        country=country,
        strategy_rules=strategy_rules,
        common_settings=common_settings,
        strategy_settings=strategy_settings,
        initial_capital=initial_capital_value,
    )

    return CountryBacktestResult(
        country=country,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital_value,
        portfolio_topn=portfolio_topn,
        holdings_limit=holdings_limit,
        summary=summary,
        portfolio_timeseries=portfolio_df,
        ticker_timeseries=ticker_timeseries,
        ticker_meta=ticker_meta,
        evaluated_records=evaluated_records,
        monthly_returns=monthly_returns,
        monthly_cum_returns=monthly_cum_returns,
        yearly_returns=yearly_returns,
        risk_off_periods=risk_off_periods,
        ticker_summaries=ticker_summaries,
        settings_snapshot=settings_snapshot,
        months_range=months_range,
    )


def _resolve_months_range(months_range: Optional[int], override_settings: Mapping[str, Any]) -> int:
    if months_range is not None:
        return int(months_range)
    if "months_range" in override_settings:
        return int(override_settings["months_range"])
    if "test_months_range" in override_settings:
        return int(override_settings["test_months_range"])
    return _default_test_months_range()


def _resolve_initial_capital(
    initial_capital: Optional[float],
    override_settings: Mapping[str, Any],
    country_settings: Mapping[str, Any],
    precision_settings: Mapping[str, Any],
) -> float:
    if initial_capital is not None:
        return float(initial_capital)
    if "initial_capital" in override_settings:
        return float(override_settings["initial_capital"])

    backtest_config = country_settings.get("backtest", {}) if country_settings else {}
    if isinstance(backtest_config, Mapping) and "initial_capital" in backtest_config:
        return float(backtest_config["initial_capital"])

    currency = str(precision_settings.get("currency", "KRW")).upper()
    if currency == "AUD":
        return 200_000.0
    if currency == "USD":
        return 150_000.0
    return _default_initial_capital()


def _resolve_end_date(country: str, override_settings: Mapping[str, Any]) -> pd.Timestamp:
    if "end_date" in override_settings:
        return pd.to_datetime(override_settings["end_date"])
    return get_latest_trading_day(country)


def _resolve_start_date(
    end_date: pd.Timestamp,
    months_range: int,
    override_settings: Mapping[str, Any],
) -> pd.Timestamp:
    if "start_date" in override_settings:
        return pd.to_datetime(override_settings["start_date"])
    return end_date - pd.DateOffset(months=months_range)


def _build_backtest_kwargs(
    *,
    country: str,
    strategy_rules,
    common_settings: Mapping[str, Any],
    strategy_settings: Mapping[str, Any],
    prefetched_data: Optional[Mapping[str, pd.DataFrame]],
    quiet: bool,
) -> Dict[str, Any]:
    stop_loss_raw = strategy_settings.get("HOLDING_STOP_LOSS_PCT")
    if stop_loss_raw is None:
        raise ValueError("strategy 설정에 'HOLDING_STOP_LOSS_PCT' 값이 필요합니다.")
    stop_loss_pct = -abs(float(stop_loss_raw))
    cooldown_days = int(strategy_settings.get("COOLDOWN_DAYS", 0) or 0)

    kwargs: Dict[str, Any] = {
        "prefetched_data": prefetched_data,
        "ma_period": strategy_rules.ma_period,
        "replace_threshold": strategy_rules.replace_threshold,
        "regime_filter_enabled": bool(common_settings["MARKET_REGIME_FILTER_ENABLED"]),
        "regime_filter_ticker": str(common_settings["MARKET_REGIME_FILTER_TICKER"]),
        "regime_filter_ma_period": int(common_settings["MARKET_REGIME_FILTER_MA_PERIOD"]),
        "stop_loss_pct": stop_loss_pct,
        "cooldown_days": cooldown_days,
        "quiet": quiet,
    }

    clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return clean_kwargs


def _build_portfolio_timeseries(
    ticker_timeseries: Mapping[str, pd.DataFrame],
    initial_capital: float,
    portfolio_topn: int,
) -> pd.DataFrame:
    non_empty = [ts.index for ts in ticker_timeseries.values() if not ts.empty]
    if not non_empty:
        raise RuntimeError("백테스트 결과에 유효한 시계열이 없습니다.")

    common_index = non_empty[0]
    for idx in non_empty[1:]:
        common_index = common_index.intersection(idx)

    if common_index.empty:
        raise RuntimeError("종목들 간에 공통된 거래일이 없습니다.")

    rows = []
    prev_total_value: Optional[float] = None
    for dt in common_index:
        total_value = 0.0
        total_holdings = 0.0
        total_cost = 0.0
        held_count = 0
        cash_value = 0.0

        for ticker, ts in ticker_timeseries.items():
            row = ts.loc[dt]
            pv_val = row.get("pv")
            if pd.notna(pv_val):
                total_value += float(pv_val)

            if ticker == "CASH":
                cash_val = row.get("pv")
                if pd.notna(cash_val):
                    cash_value += float(cash_val)
                continue

            price_val = row.get("price")
            shares_val = row.get("shares")
            avg_cost_val = row.get("avg_cost")

            price = float(price_val) if pd.notna(price_val) else 0.0
            shares = float(shares_val) if pd.notna(shares_val) else 0.0
            avg_cost = float(avg_cost_val) if pd.notna(avg_cost_val) else 0.0

            total_holdings += price * shares
            if shares > 0:
                held_count += 1
                total_cost += avg_cost * shares

        if not math.isfinite(cash_value):
            cash_value = 0.0
        if cash_value <= 0.0:
            diff_cash = total_value - total_holdings
            cash_value = diff_cash if diff_cash > 0 else 0.0

        if prev_total_value is None or prev_total_value <= 0:
            daily_profit_loss = 0.0
            daily_return_pct = 0.0
        else:
            daily_profit_loss = total_value - prev_total_value
            daily_return_pct = 0.0
            if prev_total_value > 0:
                ratio = total_value / prev_total_value
                if math.isfinite(ratio):
                    daily_return_pct = (ratio - 1.0) * 100.0
        prev_total_value = total_value if total_value > 0 else prev_total_value
        if prev_total_value is None:
            prev_total_value = total_value

        cumulative_return_pct = (
            ((total_value / initial_capital) - 1.0) * 100.0 if initial_capital > 0 else 0.0
        )

        eval_profit_loss = total_holdings - total_cost if total_cost > 0 else 0.0
        eval_return_pct = (total_holdings / total_cost - 1.0) * 100.0 if total_cost > 0 else 0.0

        rows.append(
            {
                "date": dt,
                "total_value": total_value,
                "total_cash": cash_value,
                "total_holdings": total_holdings,
                "held_count": held_count,
                "daily_profit_loss": daily_profit_loss if "daily_profit_loss" in locals() else 0.0,
                "daily_return_pct": daily_return_pct,
                "cumulative_return_pct": cumulative_return_pct,
                "evaluation_profit_loss": eval_profit_loss,
                "evaluation_return_pct": eval_return_pct,
                "portfolio_topn": portfolio_topn,
            }
        )

    df = pd.DataFrame(rows)
    df.set_index("date", inplace=True)
    return df


def _build_summary(
    portfolio_df: pd.DataFrame,
    *,
    ticker_timeseries: Mapping[str, pd.DataFrame],
    country: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    initial_capital: float,
    country_settings: Mapping[str, Any],
) -> Tuple[
    Dict[str, Any],
    pd.Series,
    pd.Series,
    pd.Series,
    List[Tuple[pd.Timestamp, pd.Timestamp]],
]:
    final_row = portfolio_df.iloc[-1]
    pv_series = portfolio_df["total_value"].astype(float)
    pv_series.index = pd.to_datetime(pv_series.index)

    years = max((end_date - start_date).days / 365.25, 0.0)
    final_value = float(final_row["total_value"])
    cagr = 0.0
    if years > 0 and initial_capital > 0:
        cagr = (final_value / initial_capital) ** (1 / years) - 1

    running_max = pv_series.cummax()
    drawdown_series = (running_max - pv_series) / running_max.replace({0: pd.NA})
    drawdown_series = drawdown_series.fillna(0.0)
    max_drawdown = float(drawdown_series.max()) if not drawdown_series.empty else 0.0

    daily_returns = pv_series.pct_change().dropna()
    sharpe_ratio = 0.0
    sortino_ratio = 0.0
    if not daily_returns.empty:
        mean_ret = daily_returns.mean()
        std_ret = daily_returns.std()
        if std_ret and math.isfinite(std_ret) and std_ret > 0:
            sharpe_ratio = (mean_ret / std_ret) * (252**0.5)
        downside_returns = daily_returns[daily_returns < 0]
        if not downside_returns.empty:
            downside_std = downside_returns.std()
            if downside_std and math.isfinite(downside_std) and downside_std > 0:
                sortino_ratio = (mean_ret / downside_std) * (252**0.5)

    calmar_ratio = (cagr / max_drawdown) if max_drawdown > 0 else 0.0
    drawdowns_pct = drawdown_series * 100.0
    ulcer_index = 0.0
    if not drawdowns_pct.empty:
        ulcer_index = float((drawdowns_pct.pow(2).mean()) ** 0.5)
    cui = calmar_ratio / ulcer_index if ulcer_index > 0 else 0.0

    benchmark_cum_ret_pct = 0.0
    benchmark_cagr_pct = 0.0
    benchmark_ticker = str(country_settings.get("benchmark_ticker") or "^GSPC")
    benchmark_country = country
    try:
        benchmark_df = fetch_ohlcv(
            benchmark_ticker,
            country=benchmark_country,
            date_range=[start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")],
        )
    except Exception:
        benchmark_df = None

    if benchmark_df is not None and not benchmark_df.empty:
        benchmark_df = benchmark_df.sort_index()
        benchmark_df = benchmark_df.loc[benchmark_df.index.intersection(pv_series.index)]
        if not benchmark_df.empty:
            start_price = float(benchmark_df["Close"].iloc[0])
            end_price = float(benchmark_df["Close"].iloc[-1])
            if start_price > 0:
                benchmark_cum_ret_pct = ((end_price / start_price) - 1) * 100
                if years > 0:
                    benchmark_cagr_pct = ((end_price / start_price) ** (1 / years) - 1) * 100

    monthly_returns = pd.Series(dtype=float)
    monthly_cum_returns = pd.Series(dtype=float)
    yearly_returns = pd.Series(dtype=float)
    if not pv_series.empty:
        start_row = pd.Series([initial_capital], index=[start_date - pd.Timedelta(days=1)])
        pv_series_with_start = pd.concat([start_row, pv_series])
        monthly_returns = pv_series_with_start.resample("ME").last().pct_change().dropna()
        if initial_capital > 0:
            eom_pv = pv_series.resample("ME").last().dropna()
            monthly_cum_returns = (eom_pv / initial_capital - 1).ffill()
        yearly_returns = pv_series_with_start.resample("YE").last().pct_change().dropna()

    risk_off_periods = _detect_risk_off_periods(pv_series.index, ticker_timeseries)

    summary = {
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "initial_capital": float(initial_capital),
        "initial_capital_krw": float(initial_capital),
        "final_value": final_value,
        "cumulative_return_pct": float(final_row["cumulative_return_pct"]),
        "evaluation_return_pct": float(final_row["evaluation_return_pct"]),
        "held_count": int(final_row["held_count"]),
        "cagr_pct": cagr * 100,
        "mdd_pct": max_drawdown * 100,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "calmar_ratio": calmar_ratio,
        "ulcer_index": ulcer_index,
        "cui": cui,
        "benchmark_cum_ret_pct": benchmark_cum_ret_pct,
        "benchmark_cagr_pct": benchmark_cagr_pct,
        "monthly_returns": monthly_returns,
        "monthly_cum_returns": monthly_cum_returns,
        "yearly_returns": yearly_returns,
        "risk_off_periods": risk_off_periods,
    }

    return summary, monthly_returns, monthly_cum_returns, yearly_returns, risk_off_periods


def _compute_evaluated_records(
    ticker_timeseries: Mapping[str, pd.DataFrame]
) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    for ticker, df in ticker_timeseries.items():
        if df is None or df.empty:
            continue

        df_sorted = df.sort_index()
        realized_profit = 0.0
        initial_value: Optional[float] = None

        for _, row in df_sorted.iterrows():
            trade_profit = row.get("trade_profit")
            if isinstance(trade_profit, (int, float)) and math.isfinite(float(trade_profit)):
                realized_profit += float(trade_profit)

            pv_val = row.get("pv")
            pv = (
                float(pv_val)
                if isinstance(pv_val, (int, float)) and math.isfinite(float(pv_val))
                else 0.0
            )
            if initial_value is None and pv > 0:
                initial_value = pv

        records[str(ticker).upper()] = {
            "realized_profit": realized_profit,
            "initial_value": initial_value or 0.0,
        }

    return records


def _detect_risk_off_periods(
    index: pd.Index,
    ticker_timeseries: Mapping[str, pd.DataFrame],
) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    if not isinstance(index, pd.DatetimeIndex):
        index = pd.to_datetime(index)

    risk_off_series = pd.Series(False, index=index)

    for df in ticker_timeseries.values():
        if df is None or df.empty or "note" not in df.columns:
            continue
        note_mask = df["note"].fillna("") == "시장 위험 회피"
        if note_mask.any():
            intersect_index = df.index[note_mask].intersection(risk_off_series.index)
            risk_off_series.loc[intersect_index] = True

    periods: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    in_period = False
    start: Optional[pd.Timestamp] = None
    prev_dt: Optional[pd.Timestamp] = None

    for dt, is_off in risk_off_series.items():
        if is_off and not in_period:
            in_period = True
            start = dt
        elif not is_off and in_period:
            if start is not None and prev_dt is not None:
                periods.append((start, prev_dt))
            in_period = False
            start = None
        prev_dt = dt

    if in_period and start is not None and prev_dt is not None:
        periods.append((start, prev_dt))

    return periods


def _build_ticker_summaries(
    ticker_timeseries: Mapping[str, pd.DataFrame],
    ticker_meta: Mapping[str, Dict[str, Any]],
    core_start_dt: pd.Timestamp,
) -> List[Dict[str, Any]]:
    sell_decisions = {
        "SELL_MOMENTUM",
        "SELL_TREND",
        "CUT_STOPLOSS",
        "SELL_REPLACE",
        "SELL_REGIME_FILTER",
    }

    summaries: List[Dict[str, Any]] = []
    for ticker, df in ticker_timeseries.items():
        ticker_key = str(ticker).upper()
        if ticker_key == "CASH" or df is None or df.empty:
            continue

        df_sorted = df.sort_index()
        trades = (
            df_sorted[df_sorted["decision"].isin(sell_decisions)]
            if "decision" in df_sorted.columns
            else pd.DataFrame()
        )
        realized_profit = (
            float(trades.get("trade_profit", pd.Series(dtype=float)).sum())
            if not trades.empty
            else 0.0
        )
        total_trades = int(len(trades)) if not trades.empty else 0
        winning_trades = (
            int((trades.get("trade_profit", pd.Series(dtype=float)) > 0).sum())
            if not trades.empty
            else 0
        )

        last_row = df_sorted.iloc[-1]
        final_shares = float(last_row.get("shares", 0.0) or 0.0)
        final_price = float(last_row.get("price", 0.0) or 0.0)
        avg_cost = float(last_row.get("avg_cost", 0.0) or 0.0)

        unrealized_profit = 0.0
        if final_shares > 0 and avg_cost > 0:
            unrealized_profit = (final_price - avg_cost) * final_shares

        total_contribution = realized_profit + unrealized_profit

        period_return_pct = 0.0
        listing_date: Optional[str] = None
        if "price" in df_sorted.columns:
            valid_prices = df_sorted[df_sorted["price"] > 0]
            if not valid_prices.empty:
                first_price = float(valid_prices["price"].iloc[0])
                last_price = float(valid_prices["price"].iloc[-1])
                if first_price > 0:
                    period_return_pct = ((last_price / first_price) - 1) * 100.0
                first_date = valid_prices.index[0]
                if isinstance(first_date, pd.Timestamp):
                    listing_date = first_date.strftime("%Y-%m-%d")

        if (
            total_trades == 0
            and final_shares <= 0
            and math.isclose(total_contribution, 0.0, abs_tol=1e-9)
        ):
            continue

        win_rate = (winning_trades / total_trades) * 100.0 if total_trades > 0 else 0.0
        meta = ticker_meta.get(ticker_key, {})

        summaries.append(
            {
                "ticker": ticker_key,
                "name": meta.get("name") or ticker_key,
                "total_trades": total_trades,
                "win_rate": win_rate,
                "realized_profit": realized_profit,
                "unrealized_profit": unrealized_profit,
                "total_contribution": total_contribution,
                "period_return_pct": period_return_pct,
                "listing_date": listing_date,
            }
        )

    summaries.sort(key=lambda x: x["total_contribution"], reverse=True)
    return summaries


def _build_settings_snapshot(
    *,
    country: str,
    strategy_rules: StrategyRules,  # type: ignore
    common_settings: Mapping[str, Any],
    strategy_settings: Mapping[str, Any],
    initial_capital: float,
) -> Dict[str, Any]:
    snapshot = {
        "country": country.upper(),
        "initial_capital": float(initial_capital),
        "strategy_rules": strategy_rules.to_dict(),
        "common_settings": dict(common_settings),
        "strategy_settings": dict(strategy_settings),
    }
    return snapshot
