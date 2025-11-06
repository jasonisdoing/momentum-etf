"""계정 기반 백테스트 실행 유틸리티."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Collection, Dict, Mapping, Optional, List, Tuple
import math

import pandas as pd

import config
from logic.entry_point import run_portfolio_backtest, StrategyRules
from utils.account_registry import get_common_file_settings
from utils.settings_loader import (
    AccountSettingsError,
    get_account_precision,
    get_account_settings,
    get_strategy_rules,
)
from utils.data_loader import get_latest_trading_day, fetch_ohlcv
from utils.stock_list_io import get_etfs
from utils.logger import get_app_logger


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
    ticker_summaries: List[Dict[str, Any]]
    settings_snapshot: Dict[str, Any]
    months_range: int
    market_index_data: Optional[pd.DataFrame]
    market_index_ticker: Optional[str]
    missing_tickers: List[str]

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
            "ticker_summaries": self.ticker_summaries,
            "settings_snapshot": self.settings_snapshot,
            "months_range": self.months_range,
            "missing_tickers": self.missing_tickers,
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
    excluded_tickers: Optional[Collection[str]] = None,
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
    account_settings_data = get_account_settings(account_id)
    strategy_settings = dict(account_settings_data.get("strategy", {}).get("tuning", {}))
    common_settings = get_common_file_settings()

    strategy_overrides_extra = override_settings.get("strategy_overrides")
    if isinstance(strategy_overrides_extra, Mapping):
        strategy_settings.update({str(k): v for k, v in strategy_overrides_extra.items()})

    if strategy_override is not None:
        strategy_rules = StrategyRules.from_values(
            ma_period=strategy_override.ma_period,
            portfolio_topn=strategy_override.portfolio_topn,
            replace_threshold=strategy_override.replace_threshold,
            ma_type=strategy_override.ma_type,
            core_holdings=strategy_override.core_holdings,
            stop_loss_pct=strategy_override.stop_loss_pct,
        )
        strategy_settings["MA_PERIOD"] = strategy_rules.ma_period
        strategy_settings["MA_TYPE"] = strategy_rules.ma_type
        strategy_settings["PORTFOLIO_TOPN"] = strategy_rules.portfolio_topn
        strategy_settings["REPLACE_SCORE_THRESHOLD"] = strategy_rules.replace_threshold
        strategy_settings["CORE_HOLDINGS"] = strategy_rules.core_holdings
        if strategy_rules.stop_loss_pct is not None:
            strategy_settings["STOP_LOSS_PCT"] = strategy_rules.stop_loss_pct

    months_range = _resolve_months_range(months_range, override_settings, account_settings)
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
    excluded_upper: set[str] = set()
    if excluded_tickers:
        excluded_upper = {str(ticker).strip().upper() for ticker in excluded_tickers if isinstance(ticker, str) and str(ticker).strip()}

    etf_universe = get_etfs(country_code)
    if not etf_universe:
        raise AccountSettingsError(f"'data/stocks/{country_code}.json' 파일에서 종목을 찾을 수 없습니다.")

    if excluded_upper:
        before_count = len(etf_universe)
        etf_universe = [stock for stock in etf_universe if str(stock.get("ticker", "")).strip().upper() not in excluded_upper]
        removed = before_count - len(etf_universe)
        if removed > 0:
            _log(f"[백테스트] 데이터 부족으로 제외된 {removed}개 종목을 유니버스에서 제거합니다.")
    if not etf_universe:
        raise RuntimeError("백테스트에 사용할 유효한 종목이 없습니다.")

    _log(f"[백테스트] {len(etf_universe)}개의 ETF를 찾았습니다.")

    ticker_meta = {str(item.get("ticker", "")).upper(): dict(item) for item in etf_universe}
    ticker_meta["CASH"] = {"ticker": "CASH", "name": "현금", "category": "-"}

    # 검증은 get_account_strategy에서 이미 완료됨 - 바로 사용
    portfolio_topn = strategy_rules.portfolio_topn
    holdings_limit = strategy_settings.get("MAX_PER_CATEGORY", config.MAX_PER_CATEGORY)
    _log(f"[백테스트] 포트폴리오 TOPN: {portfolio_topn}, 카테고리당 최대 보유 수: {holdings_limit}")

    _log("[백테스트] 백테스트 파라미터를 구성하는 중...")
    backtest_kwargs = _build_backtest_kwargs(
        strategy_rules=strategy_rules,
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
    runtime_missing_tickers: set[str] = set()

    ticker_timeseries = (
        run_portfolio_backtest(
            stocks=etf_universe,
            initial_capital=initial_capital_value,
            core_start_date=start_date,
            top_n=portfolio_topn,
            date_range=date_range,
            country=country_code,
            missing_ticker_sink=runtime_missing_tickers,
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
    ) = _build_summary(
        portfolio_df,
        country_code=country_code,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_capital_value,
        initial_capital_krw=capital_info.krw,
        fx_rate_to_krw=capital_info.fx_rate_to_krw,
        currency=display_currency,
        account_settings=account_settings,
    )

    evaluated_records = _compute_evaluated_records(ticker_timeseries, start_date)

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

    missing_sorted = sorted(runtime_missing_tickers)
    if missing_sorted and not quiet:
        logger.warning(
            "[백테스트] 가격 데이터 부족으로 제외된 종목: %s",
            ", ".join(missing_sorted),
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
        ticker_summaries=ticker_summaries,
        settings_snapshot=settings_snapshot,
        months_range=months_range,
        market_index_data=ticker_timeseries.get("__market_index_data__"),
        market_index_ticker=ticker_timeseries.get("__market_index_ticker__"),
        missing_tickers=missing_sorted,
    )


def _resolve_months_range(
    months_range: Optional[int],
    override_settings: Mapping[str, Any],
    account_settings: Mapping[str, Any],
) -> int:
    if months_range is not None:
        return int(months_range)
    if "months_range" in override_settings:
        return int(override_settings["months_range"])
    if "test_months_range" in override_settings:
        return int(override_settings["test_months_range"])
    account_months = account_settings.get("strategy", {}).get("MONTHS_RANGE") if account_settings else None
    if account_months is not None:
        try:
            return int(account_months)
        except (TypeError, ValueError):
            pass
    raise ValueError("MONTHS_RANGE 설정이 필요합니다. 계정 설정의 strategy.MONTHS_RANGE 값을 확인하세요.")


def _resolve_initial_capital(
    initial_capital: Optional[float],
    override_settings: Mapping[str, Any],
    account_settings: Mapping[str, Any],
    precision_settings: Mapping[str, Any],
) -> InitialCapitalInfo:

    def _coerce_positive_float(value: Any) -> Optional[float]:
        try:
            candidate = float(value)
        except (TypeError, ValueError):
            return None
        return candidate if math.isfinite(candidate) and candidate > 0 else None

    currency = str(precision_settings.get("currency") or account_settings.get("currency") or "KRW").upper()
    if currency not in {"KRW", "KR"}:
        raise ValueError(f"지원하지 않는 통화 코드입니다: {currency}")
    currency = "KRW"

    backtest_config = account_settings.get("backtest", {}) if account_settings else {}
    if not isinstance(backtest_config, Mapping):
        backtest_config = {}

    krw_override = _coerce_positive_float(override_settings.get("initial_capital_krw"))
    if krw_override is None:
        krw_override = _coerce_positive_float(backtest_config.get("initial_capital_krw"))

    if krw_override is None:
        raise ValueError("initial_capital_krw 설정이 필요합니다. 계정 설정의 backtest.initial_capital_krw 값을 확인하세요.")

    fx_rate = 1.0

    local_overrides = [initial_capital, override_settings.get("initial_capital"), backtest_config.get("initial_capital")]
    local_override = None
    for candidate in local_overrides:
        local_override = _coerce_positive_float(candidate)
        if local_override is not None:
            break

    local_capital = float(krw_override)
    if local_override is not None:
        local_capital = float(local_override)

    return InitialCapitalInfo(
        local=float(local_capital),
        krw=float(krw_override),
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
    strategy_settings: Mapping[str, Any],
    prefetched_data: Optional[Mapping[str, pd.DataFrame]],
    quiet: bool,
) -> Dict[str, Any]:
    try:
        configured_stop_loss = (
            float(strategy_rules.stop_loss_pct) if strategy_rules.stop_loss_pct is not None else float(strategy_rules.portfolio_topn)
        )
    except (TypeError, ValueError):
        configured_stop_loss = float(strategy_rules.portfolio_topn)
    stop_loss_threshold = -abs(configured_stop_loss)

    # 필수 설정 검증
    if "COOLDOWN_DAYS" not in strategy_settings:
        raise ValueError("strategy_settings에 COOLDOWN_DAYS 설정이 필요합니다.")
    if "OVERBOUGHT_SELL_THRESHOLD" not in strategy_settings:
        raise ValueError("strategy_settings에 OVERBOUGHT_SELL_THRESHOLD 설정이 필요합니다.")

    cooldown_days = int(strategy_settings["COOLDOWN_DAYS"])
    rsi_sell_threshold = int(strategy_settings["OVERBOUGHT_SELL_THRESHOLD"])

    if not (0 <= rsi_sell_threshold <= 100):
        raise ValueError(f"OVERBOUGHT_SELL_THRESHOLD는 0~100 사이여야 합니다. (현재값: {rsi_sell_threshold})")

    # 시장 레짐 필터 제거됨 (항상 100% 투자)

    kwargs: Dict[str, Any] = {
        "prefetched_data": prefetched_data,
        "ma_period": strategy_rules.ma_period,
        "ma_type": strategy_rules.ma_type,
        "replace_threshold": strategy_rules.replace_threshold,
        "stop_loss_pct": stop_loss_threshold,
        "cooldown_days": cooldown_days,
        "rsi_sell_threshold": rsi_sell_threshold,
        "core_holdings": strategy_rules.core_holdings,
        "quiet": quiet,
    }

    clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return clean_kwargs


def _build_portfolio_timeseries(
    ticker_timeseries: Mapping[str, pd.DataFrame],
    initial_capital: float,
    portfolio_topn: int,
) -> pd.DataFrame:
    # DataFrame만 필터링 (메타데이터 문자열 제외)
    dataframes = [ts for ts in ticker_timeseries.values() if isinstance(ts, pd.DataFrame) and not ts.empty]
    if not dataframes:
        raise RuntimeError("백테스트 결과에 유효한 시계열이 없습니다.")

    non_empty = [ts.index for ts in dataframes]

    # 교집합 대신 합집합 사용 (모든 거래일 포함)
    common_index = non_empty[0]
    for idx in non_empty[1:]:
        common_index = common_index.union(idx)

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
            # DataFrame만 처리 (메타데이터 문자열 제외)
            if not isinstance(ts, pd.DataFrame):
                continue
            # 해당 날짜에 데이터가 없으면 스킵
            if dt not in ts.index:
                continue

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

        cumulative_return_pct = ((total_value / initial_capital) - 1.0) * 100.0 if initial_capital > 0 else 0.0

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
    country_code: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    initial_capital: float,
    initial_capital_krw: float,
    fx_rate_to_krw: float,
    currency: str,
    account_settings: Mapping[str, Any],
) -> Tuple[
    Dict[str, Any],
    pd.Series,
    pd.Series,
    pd.Series,
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
    if not daily_returns.empty:
        mean_ret = daily_returns.mean()
        std_ret = daily_returns.std()
        if std_ret and math.isfinite(std_ret) and std_ret > 0:
            sharpe_ratio = (mean_ret / std_ret) * (252**0.5)

    # Sharpe/MDD 비율 계산 (SDR)
    mdd_pct = max_drawdown * 100
    sharpe_to_mdd = (sharpe_ratio / mdd_pct) if mdd_pct > 0 else 0.0

    def _calc_benchmark_performance(*, ticker: str, name: str, country: str) -> Optional[Dict[str, Any]]:
        benchmark_df = fetch_ohlcv(
            ticker,
            country=country,
            date_range=[start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")],
        )

        if benchmark_df is None or benchmark_df.empty:
            return None

        benchmark_df = benchmark_df.sort_index()
        benchmark_df = benchmark_df.loc[benchmark_df.index.intersection(pv_series.index)]
        if benchmark_df.empty:
            return None

        # 시작 가격: 백테스트 시작일 종가 (슬리피지 없음)
        start_price = float(benchmark_df["Close"].iloc[0])

        # 종료 가격: 최신 거래일 종가 (슬리피지 없음)
        end_price = float(benchmark_df["Close"].iloc[-1])

        if start_price <= 0:
            return None

        cum_ret_pct = ((end_price / start_price) - 1) * 100
        cagr_pct = 0.0
        if years > 0:
            cagr_pct = ((end_price / start_price) ** (1 / years) - 1) * 100

        # 벤치마크 Sharpe 및 MDD 계산
        bench_series = benchmark_df["Close"].astype(float)
        bench_running_max = bench_series.cummax()
        bench_drawdown_series = (bench_running_max - bench_series) / bench_running_max.replace({0: pd.NA})
        bench_drawdown_series = bench_drawdown_series.fillna(0.0)
        bench_mdd = float(bench_drawdown_series.max()) if not bench_drawdown_series.empty else 0.0
        bench_mdd_pct = bench_mdd * 100

        bench_daily_returns = bench_series.pct_change().dropna()
        bench_sharpe = 0.0
        if not bench_daily_returns.empty:
            bench_mean_ret = bench_daily_returns.mean()
            bench_std_ret = bench_daily_returns.std()
            if bench_std_ret and math.isfinite(bench_std_ret) and bench_std_ret > 0:
                bench_sharpe = (bench_mean_ret / bench_std_ret) * (252**0.5)

        bench_sharpe_to_mdd = (bench_sharpe / bench_mdd_pct) if bench_mdd_pct > 0 else 0.0

        return {
            "ticker": ticker,
            "name": name,
            "country": country,
            "cumulative_return_pct": cum_ret_pct,
            "cagr_pct": cagr_pct,
            "sharpe": bench_sharpe,
            "mdd": bench_mdd_pct,
            "sharpe_to_mdd": bench_sharpe_to_mdd,
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
            bench_country = str(entry.get("country") or entry.get("market") or country_code).strip() or country_code
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

    weekly_summary_rows: List[Dict[str, Any]] = []
    monthly_returns = pd.Series(dtype=float)
    monthly_cum_returns = pd.Series(dtype=float)
    yearly_returns = pd.Series(dtype=float)
    if not pv_series.empty:
        start_row = pd.Series([initial_capital_local], index=[start_date - pd.Timedelta(days=1)])
        pv_series_with_start = pd.concat([start_row, pv_series])
        weekly_values = pv_series_with_start.resample("W-FRI").last().dropna()
        if not weekly_values.empty:
            weekly_return_pct = weekly_values.pct_change().mul(100).fillna(0.0)
            if initial_capital_local > 0:
                weekly_cum_pct = (weekly_values / initial_capital_local - 1).mul(100)
            else:
                weekly_cum_pct = pd.Series([0.0] * len(weekly_values), index=weekly_values.index)
            for dt, value in weekly_values.items():
                weekly_summary_rows.append(
                    {
                        "week_end": dt.strftime("%Y-%m-%d"),
                        "value": float(value),
                        "weekly_return_pct": float(weekly_return_pct.loc[dt]),
                        "cumulative_return_pct": float(weekly_cum_pct.loc[dt]),
                    }
                )
        monthly_returns = pv_series_with_start.resample("ME").last().pct_change().dropna()
        if initial_capital_local > 0:
            eom_pv = pv_series.resample("ME").last().dropna()
            monthly_cum_returns = (eom_pv / initial_capital_local - 1).ffill()
        yearly_returns = pv_series_with_start.resample("YE").last().pct_change().dropna()

    summary = {
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "initial_capital": initial_capital_local,
        "initial_capital_local": initial_capital_local,
        "initial_capital_krw": initial_capital_krw,
        "final_value": final_value,
        "final_value_local": final_value,
        "final_value_krw": final_value * fx_rate_to_krw,
        "period_return": float(final_row["cumulative_return_pct"]),
        "evaluation_return_pct": float(final_row["evaluation_return_pct"]),
        "held_count": int(final_row["held_count"]),
        "cagr": cagr * 100,
        "mdd": mdd_pct,
        "sharpe": sharpe_ratio,
        "sharpe_to_mdd": sharpe_to_mdd,
        "benchmark_cum_ret_pct": benchmark_cum_ret_pct,
        "benchmark_cagr_pct": benchmark_cagr_pct,
        "benchmarks": benchmarks_summary,
        "benchmark_name": benchmarks_summary[0]["name"] if benchmarks_summary else "S&P 500",
        "weekly_summary": weekly_summary_rows,
        "monthly_returns": monthly_returns,
        "monthly_cum_returns": monthly_cum_returns,
        "yearly_returns": yearly_returns,
        "fx_rate_to_krw": fx_rate_to_krw,
        "currency": currency,
    }

    return summary, monthly_returns, monthly_cum_returns, yearly_returns


def _compute_evaluated_records(
    ticker_timeseries: Mapping[str, pd.DataFrame],
    start_date: pd.Timestamp,
) -> Dict[str, Dict[str, Any]]:
    """백테스트 시작일 이후의 거래 손익만 집계합니다.

    주의: 이 함수는 현재 사용되지 않습니다.
    누적 손익은 reporting.py에서 cost_basis 기준으로 직접 계산됩니다.
    """
    records: Dict[str, Dict[str, Any]] = {}
    start_date_norm = start_date.normalize()

    for ticker, df in ticker_timeseries.items():
        # DataFrame만 처리 (메타데이터 문자열 제외)
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue

        df_sorted = df.sort_index()
        # 백테스트 시작일 이후 데이터만 필터링
        df_filtered = df_sorted[df_sorted.index >= start_date_norm]

        realized_profit = 0.0
        initial_value: Optional[float] = None

        for idx, row in df_filtered.iterrows():
            trade_profit = row.get("trade_profit")
            if isinstance(trade_profit, (int, float)) and math.isfinite(float(trade_profit)):
                realized_profit += float(trade_profit)

            pv_val = row.get("pv")
            pv = float(pv_val) if isinstance(pv_val, (int, float)) and math.isfinite(float(pv_val)) else 0.0
            if initial_value is None and pv > 0:
                initial_value = pv

        records[str(ticker).upper()] = {
            "realized_profit": realized_profit,
            "initial_value": initial_value or 0.0,
        }

    return records


def _build_ticker_summaries(
    ticker_timeseries: Mapping[str, pd.DataFrame],
    ticker_meta: Mapping[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    sell_decisions = {
        "SELL_MOMENTUM",
        "SELL_TREND",
        "CUT_STOPLOSS",
        "SELL_REPLACE",
    }

    summaries: List[Dict[str, Any]] = []
    for ticker, df in ticker_timeseries.items():
        ticker_key = str(ticker).upper()
        # DataFrame만 처리 (메타데이터 문자열 제외)
        if ticker_key == "CASH" or not isinstance(df, pd.DataFrame) or df.empty:
            continue

        df_sorted = df.sort_index()
        if "decision" in df_sorted.columns:
            trades_mask = df_sorted["decision"].isin(sell_decisions)
        else:
            trades_mask = pd.Series(False, index=df_sorted.index)

        trades = df_sorted[trades_mask] if trades_mask.any() else pd.DataFrame()
        realized_profit = float(trades.get("trade_profit", pd.Series(dtype=float)).sum()) if not trades.empty else 0.0
        total_trades = int(len(trades)) if not trades.empty else 0
        winning_trades = int((trades.get("trade_profit", pd.Series(dtype=float)) > 0).sum()) if not trades.empty else 0

        last_row = df_sorted.iloc[-1]
        final_shares = float(last_row.get("shares", 0.0))
        final_price = float(last_row.get("price", 0.0))
        avg_cost = float(last_row.get("avg_cost", 0.0))

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

        if total_trades == 0 and final_shares <= 0 and math.isclose(total_contribution, 0.0, abs_tol=1e-9):
            continue

        # 점수가 음수인 종목 제외
        last_score = float(last_row.get("score", 0.0))
        if last_score < 0:
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
