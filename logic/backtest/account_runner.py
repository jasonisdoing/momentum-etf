"""계정 기반 백테스트 실행 유틸리티."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, List, Tuple
import math

import pandas as pd

from logic.entry_point import run_portfolio_backtest, StrategyRules
from utils.account_registry import get_common_file_settings
from utils.settings_loader import (
    AccountSettingsError,
    get_account_precision,
    get_account_settings,
    get_account_strategy,
    get_strategy_rules,
    get_backtest_months_range,
    get_backtest_initial_capital,
)
from utils.data_loader import (
    get_latest_trading_day,
    fetch_ohlcv,
    get_aud_to_krw_rate,
    get_usd_to_krw_rate,
)
from utils.stock_list_io import get_etfs
from utils.logger import get_app_logger


def _default_test_months_range() -> int:
    return get_backtest_months_range()


def _default_initial_capital() -> float:
    return float(get_backtest_initial_capital())


@dataclass
class InitialCapitalInfo:
    """Container for initial capital values in local currency and KRW."""

    local: float
    krw: float
    fx_rate_to_krw: float
    currency: str


@dataclass
class AccountBacktestResult:
    """계정 기반 백테스트 결과."""

    account_id: str
    country_code: str
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    initial_capital: float
    initial_capital_krw: float
    fx_rate_to_krw: float
    currency: str
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
            "account_id": self.account_id,
            "country_code": self.country_code,
            "start_date": self.start_date.strftime("%Y-%m-%d"),
            "end_date": self.end_date.strftime("%Y-%m-%d"),
            "initial_capital": float(self.initial_capital),
            "initial_capital_krw": float(self.initial_capital_krw),
            "fx_rate_to_krw": float(self.fx_rate_to_krw),
            "currency": self.currency,
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


def run_account_backtest(
    account_id: str,
    *,
    months_range: Optional[int] = None,
    initial_capital: Optional[float] = None,
    quiet: bool = False,
    prefetched_data: Optional[Mapping[str, pd.DataFrame]] = None,
    override_settings: Optional[Dict[str, Any]] = None,
    strategy_override: Optional[StrategyRules] = None,  # type: ignore
) -> AccountBacktestResult:
    """계정 ID를 기반으로 백테스트를 실행합니다."""

    logger = get_app_logger()

    def _log(message: str) -> None:
        if quiet:
            logger.debug(message)
        else:
            logger.info(message)

    override_settings = override_settings or {}
    account_id = (account_id or "").strip().lower()
    if not account_id:
        raise AccountSettingsError("계정 ID를 지정해야 합니다.")

    _log(f"[백테스트] {account_id.upper()} 백테스트를 시작합니다...")

    _log("[백테스트] 설정을 로드하는 중...")
    account_settings = get_account_settings(account_id)
    country_code = (account_settings.get("country_code") or account_id).strip().lower() or "kor"

    base_strategy_rules = get_strategy_rules(account_id)
    strategy_rules = StrategyRules.from_mapping(base_strategy_rules.to_dict())
    precision_settings = get_account_precision(account_id)
    strategy_settings = dict(get_account_strategy(account_id))
    common_settings = get_common_file_settings()

    strategy_overrides_extra = override_settings.get("strategy_overrides")
    if isinstance(strategy_overrides_extra, Mapping):
        strategy_settings.update({str(k): v for k, v in strategy_overrides_extra.items()})

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
    end_date = _resolve_end_date(country_code, override_settings)
    start_date = _resolve_start_date(end_date, months_range, override_settings)

    capital_info = _resolve_initial_capital(
        initial_capital,
        override_settings,
        account_settings,
        precision_settings,
    )
    initial_capital_value = capital_info.local

    _log(f"[백테스트] {account_id.upper()} 계정({country_code.upper()}) ETF 목록을 로드하는 중...")
    etf_universe = get_etfs(country_code)
    if not etf_universe:
        raise AccountSettingsError(f"'data/stocks/{country_code}.json' 파일에서 종목을 찾을 수 없습니다.")
    _log(f"[백테스트] {len(etf_universe)}개의 ETF를 찾았습니다.")

    ticker_meta = {str(item.get("ticker", "")).upper(): dict(item) for item in etf_universe}
    ticker_meta["CASH"] = {"ticker": "CASH", "name": "현금", "category": "-"}

    portfolio_topn = strategy_rules.portfolio_topn
    holdings_limit = int(strategy_settings.get("MAX_PER_CATEGORY", 0) or 0)
    _log(f"[백테스트] 포트폴리오 TOPN: {portfolio_topn}, 카테고리당 최대 보유 수: {holdings_limit}")

    _log("[백테스트] 백테스트 파라미터를 구성하는 중...")
    backtest_kwargs = _build_backtest_kwargs(
        strategy_rules=strategy_rules,
        common_settings=common_settings,
        strategy_settings=strategy_settings,
        prefetched_data=prefetched_data,
        quiet=quiet,
    )

    date_range = [start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")]

    display_currency = (capital_info.currency or "KRW").upper()
    if display_currency != "KRW":
        _log(
            f"[백테스트] {account_id.upper()}({country_code.upper()}) 백테스트 실행 | "
            f"기간: {date_range[0]}~{date_range[1]} | 초기 자본: {initial_capital_value:,.0f} {display_currency}"
            f" (약 {capital_info.krw:,.0f} KRW)"
        )
    else:
        _log(
            f"[백테스트] {account_id.upper()}({country_code.upper()}) 백테스트 실행 | "
            f"기간: {date_range[0]}~{date_range[1]} | 초기 자본: {initial_capital_value:,.0f}"
        )

    _log("[백테스트] 포트폴리오 백테스트 실행 중...")

    ticker_timeseries = (
        run_portfolio_backtest(
            stocks=etf_universe,
            initial_capital=initial_capital_value,
            core_start_date=start_date,
            top_n=portfolio_topn,
            date_range=date_range,
            country=country_code,
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
        country_code=country_code,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital_value,
        initial_capital_krw=capital_info.krw,
        fx_rate_to_krw=capital_info.fx_rate_to_krw,
        currency=display_currency,
        account_settings=account_settings,
        strategy_settings=strategy_settings,
    )

    evaluated_records = _compute_evaluated_records(ticker_timeseries)

    ticker_summaries = _build_ticker_summaries(
        ticker_timeseries,
        ticker_meta,
    )

    _log("[백테스트] 설정 스냅샷을 생성하는 중...")
    settings_snapshot = _build_settings_snapshot(
        account_id=account_id,
        country_code=country_code,
        strategy_rules=strategy_rules,
        common_settings=common_settings,
        strategy_settings=strategy_settings,
        initial_capital=initial_capital_value,
    )

    return AccountBacktestResult(
        account_id=account_id,
        country_code=country_code,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital_value,
        initial_capital_krw=capital_info.krw,
        fx_rate_to_krw=capital_info.fx_rate_to_krw,
        currency=display_currency,
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
    account_settings: Mapping[str, Any],
    precision_settings: Mapping[str, Any],
) -> InitialCapitalInfo:
    logger = get_app_logger()

    def _coerce_positive_float(value: Any) -> Optional[float]:
        try:
            candidate = float(value)
        except (TypeError, ValueError):
            return None
        return candidate if math.isfinite(candidate) and candidate > 0 else None

    currency = str(
        precision_settings.get("currency") or account_settings.get("currency") or "KRW"
    ).upper()

    fx_override = _coerce_positive_float(override_settings.get("fx_rate_to_krw"))
    if fx_override is None:
        fx_override = _coerce_positive_float(account_settings.get("fx_rate_to_krw"))
    if fx_override is None:
        fx_override = _coerce_positive_float(precision_settings.get("fx_rate_to_krw"))

    fx_rate = 1.0
    if currency == "AUD":
        fetched = _coerce_positive_float(get_aud_to_krw_rate())
        fx_rate = fetched or fx_override or 1.0
    elif currency == "USD":
        fetched = _coerce_positive_float(get_usd_to_krw_rate())
        fx_rate = fetched or fx_override or 1.0
    else:
        fx_rate = 1.0

    if fx_rate <= 0 or not math.isfinite(fx_rate):
        fx_rate = 1.0

    if currency != "KRW" and fx_rate == 1.0 and fx_override is None:
        logger.warning(
            "[백테스트] '%s' 통화 환율을 가져오지 못해 KRW와 동일하게 처리합니다.",
            currency,
        )

    backtest_config = account_settings.get("backtest", {}) if account_settings else {}
    if not isinstance(backtest_config, Mapping):
        backtest_config = {}

    local_override = _coerce_positive_float(initial_capital)
    if local_override is None:
        local_override = _coerce_positive_float(override_settings.get("initial_capital"))
    if local_override is None:
        local_override = _coerce_positive_float(backtest_config.get("initial_capital"))

    krw_override = _coerce_positive_float(override_settings.get("initial_capital_krw"))
    if krw_override is None:
        krw_override = _coerce_positive_float(backtest_config.get("initial_capital_krw"))

    if krw_override is None and local_override is not None and currency != "KRW":
        krw_override = local_override * fx_rate

    base_krw = krw_override if krw_override is not None else _default_initial_capital()

    if local_override is not None:
        local_capital = local_override
    else:
        local_capital = base_krw / fx_rate if fx_rate > 0 else base_krw

    return InitialCapitalInfo(
        local=float(local_capital),
        krw=float(base_krw),
        fx_rate_to_krw=float(fx_rate),
        currency=currency,
    )


def _resolve_end_date(country_code: str, override_settings: Mapping[str, Any]) -> pd.Timestamp:
    if "end_date" in override_settings:
        return pd.to_datetime(override_settings["end_date"])
    return get_latest_trading_day(country_code)


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
    strategy_rules,
    common_settings: Mapping[str, Any],
    strategy_settings: Mapping[str, Any],
    prefetched_data: Optional[Mapping[str, pd.DataFrame]],
    quiet: bool,
) -> Dict[str, Any]:
    # 포트폴리오 N개 종목 중 한 종목만 N% 하락해 손절될 경우 전체 손실은 1%가 된다.
    stop_loss_pct = -abs(float(strategy_rules.portfolio_topn))
    cooldown_days = int(strategy_settings.get("COOLDOWN_DAYS", 0) or 0)

    regime_filter_enabled = bool(common_settings.get("MARKET_REGIME_FILTER_ENABLED", True))

    regime_filter_ticker = str(strategy_settings.get("MARKET_REGIME_FILTER_TICKER") or "").strip()
    if not regime_filter_ticker:
        raise ValueError("strategy 설정에 'MARKET_REGIME_FILTER_TICKER' 값이 필요합니다.")

    regime_filter_ma_raw = strategy_settings.get("MARKET_REGIME_FILTER_MA_PERIOD")
    if regime_filter_ma_raw is None:
        raise ValueError("strategy 설정에 'MARKET_REGIME_FILTER_MA_PERIOD' 값이 필요합니다.")
    try:
        regime_filter_ma_period = int(regime_filter_ma_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("'MARKET_REGIME_FILTER_MA_PERIOD' 값은 정수여야 합니다.") from exc

    kwargs: Dict[str, Any] = {
        "prefetched_data": prefetched_data,
        "ma_period": strategy_rules.ma_period,
        "replace_threshold": strategy_rules.replace_threshold,
        "regime_filter_enabled": regime_filter_enabled,
        "regime_filter_ticker": regime_filter_ticker,
        "regime_filter_ma_period": regime_filter_ma_period,
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
    country_code: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    initial_capital: float,
    initial_capital_krw: float,
    fx_rate_to_krw: float,
    currency: str,
    account_settings: Mapping[str, Any],
    strategy_settings: Mapping[str, Any],
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
    initial_capital_local = float(initial_capital)
    initial_capital_krw = float(initial_capital_krw)
    fx_rate_to_krw = float(fx_rate_to_krw) if fx_rate_to_krw else 1.0
    currency = (currency or country_code or "KRW").upper()
    cagr = 0.0
    if years > 0 and initial_capital_local > 0:
        cagr = (final_value / initial_capital_local) ** (1 / years) - 1

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

    def _calc_benchmark_performance(
        *, ticker: str, name: str, country: str
    ) -> Optional[Dict[str, Any]]:
        try:
            benchmark_df = fetch_ohlcv(
                ticker,
                country=country,
                date_range=[start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")],
            )
        except Exception:
            benchmark_df = None

        if benchmark_df is None or benchmark_df.empty:
            return None

        benchmark_df = benchmark_df.sort_index()
        benchmark_df = benchmark_df.loc[benchmark_df.index.intersection(pv_series.index)]
        if benchmark_df.empty:
            return None

        start_price = float(benchmark_df["Close"].iloc[0])
        end_price = float(benchmark_df["Close"].iloc[-1])
        if start_price <= 0:
            return None

        cum_ret_pct = ((end_price / start_price) - 1) * 100
        cagr_pct = 0.0
        if years > 0:
            cagr_pct = ((end_price / start_price) ** (1 / years) - 1) * 100

        return {
            "ticker": ticker,
            "name": name,
            "country": country,
            "cumulative_return_pct": cum_ret_pct,
            "cagr_pct": cagr_pct,
        }

    benchmark_cum_ret_pct = 0.0
    benchmark_cagr_pct = 0.0
    benchmarks_summary: List[Dict[str, Any]] = []

    configured_benchmarks = account_settings.get("benchmarks")
    if isinstance(configured_benchmarks, list) and configured_benchmarks:
        for entry in configured_benchmarks:
            if not isinstance(entry, Mapping):
                continue
            ticker_value = str(entry.get("ticker") or "").strip()
            if not ticker_value:
                continue

            name_value = str(entry.get("name") or ticker_value).strip() or ticker_value
            bench_country = (
                str(entry.get("country") or entry.get("market") or country_code).strip()
                or country_code
            )
            perf = _calc_benchmark_performance(
                ticker=ticker_value,
                name=name_value,
                country=bench_country,
            )
            if perf is not None:
                benchmarks_summary.append(perf)

    if not benchmarks_summary:
        benchmark_ticker = str(account_settings.get("benchmark_ticker") or "^GSPC")
        benchmark_country = country_code
        default_name = str(account_settings.get("benchmark_name") or "S&P 500")
        perf = _calc_benchmark_performance(
            ticker=benchmark_ticker,
            name=default_name,
            country=benchmark_country,
        )
        if perf is not None:
            benchmarks_summary.append(perf)

    if benchmarks_summary:
        benchmark_cum_ret_pct = float(benchmarks_summary[0]["cumulative_return_pct"])
        benchmark_cagr_pct = float(benchmarks_summary[0]["cagr_pct"])

    monthly_returns = pd.Series(dtype=float)
    monthly_cum_returns = pd.Series(dtype=float)
    yearly_returns = pd.Series(dtype=float)
    if not pv_series.empty:
        start_row = pd.Series([initial_capital_local], index=[start_date - pd.Timedelta(days=1)])
        pv_series_with_start = pd.concat([start_row, pv_series])
        monthly_returns = pv_series_with_start.resample("ME").last().pct_change().dropna()
        if initial_capital_local > 0:
            eom_pv = pv_series.resample("ME").last().dropna()
            monthly_cum_returns = (eom_pv / initial_capital_local - 1).ffill()
        yearly_returns = pv_series_with_start.resample("YE").last().pct_change().dropna()

    risk_off_periods = _detect_risk_off_periods(pv_series.index, ticker_timeseries)

    regime_filter_ticker = str(strategy_settings.get("MARKET_REGIME_FILTER_TICKER") or "").strip()
    if not regime_filter_ticker:
        raise ValueError("strategy 설정에 'MARKET_REGIME_FILTER_TICKER' 값이 필요합니다.")

    regime_filter_ma_raw = strategy_settings.get("MARKET_REGIME_FILTER_MA_PERIOD")
    if regime_filter_ma_raw is None:
        raise ValueError("strategy 설정에 'MARKET_REGIME_FILTER_MA_PERIOD' 값이 필요합니다.")
    try:
        regime_filter_ma_period = int(regime_filter_ma_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("'MARKET_REGIME_FILTER_MA_PERIOD' 값은 정수여야 합니다.") from exc

    summary = {
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "initial_capital": initial_capital_local,
        "initial_capital_local": initial_capital_local,
        "initial_capital_krw": initial_capital_krw,
        "final_value": final_value,
        "final_value_local": final_value,
        "final_value_krw": final_value * fx_rate_to_krw,
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
        "benchmarks": benchmarks_summary,
        "benchmark_name": benchmarks_summary[0]["name"] if benchmarks_summary else "S&P 500",
        "regime_filter_enabled": True,
        "regime_filter_ticker": regime_filter_ticker,
        "regime_filter_ma_period": regime_filter_ma_period,
        "monthly_returns": monthly_returns,
        "monthly_cum_returns": monthly_cum_returns,
        "yearly_returns": yearly_returns,
        "risk_off_periods": risk_off_periods,
        "fx_rate_to_krw": fx_rate_to_krw,
        "currency": currency,
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
    account_id: str,
    country_code: str,
    strategy_rules: StrategyRules,  # type: ignore
    common_settings: Mapping[str, Any],
    strategy_settings: Mapping[str, Any],
    initial_capital: float,
) -> Dict[str, Any]:
    snapshot = {
        "account_id": account_id.upper(),
        "country_code": country_code.upper(),
        "initial_capital": float(initial_capital),
        "strategy_rules": strategy_rules.to_dict(),
        "common_settings": dict(common_settings),
        "strategy_settings": dict(strategy_settings),
    }
    return snapshot
