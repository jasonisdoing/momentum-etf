"""계정 기반 백테스트 실행 유틸리티."""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from config import TRADING_DAYS_PER_MONTH
from core.backtest.engine import run_portfolio_backtest
from strategies.maps.rules import StrategyRules
from utils.account_registry import get_common_file_settings
from utils.data_loader import get_exchange_rate_series, get_latest_trading_day, get_trading_days
from utils.logger import get_app_logger
from utils.settings_loader import (
    AccountSettingsError,
    get_account_precision,
    get_account_settings,
    get_strategy_rules,
    resolve_strategy_params,
)
from utils.stock_list_io import get_etfs


@dataclass
class InitialCapitalInfo:
    """Container for initial capital values in local currency and KRW."""

    local: float
    krw: float
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
            "monthly_returns": self.monthly_returns.to_dict(),
            "monthly_cum_returns": self.monthly_cum_returns.to_dict(),
            "yearly_returns": self.yearly_returns.to_dict(),
            "ticker_summaries": self.ticker_summaries,
            "settings_snapshot": self.settings_snapshot,
            "backtest_start_date": self.backtest_start_date,
            "missing_tickers": self.missing_tickers,
        }


def run_account_backtest(
    account_id: str,
    *,
    initial_capital: float | None = None,
    quiet: bool = False,
    prefetched_data: Mapping[str, pd.DataFrame] | None = None,
    override_settings: dict[str, Any] | None = None,
    strategy_override: StrategyRules | None = None,  # type: ignore
    excluded_tickers: Collection[str] | None = None,
    prefetched_etf_universe: Sequence[Mapping[str, Any]] | None = None,
    prefetched_metrics: Mapping[str, dict[str, Any]] | None = None,
    trading_calendar: Sequence[pd.Timestamp] | None = None,
    prefetched_fx_series: pd.Series | None = None,
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

    # 튜닝 최적화: 모든 데이터가 프리패치되어 있으면 설정 로드 건너뛰기
    is_tuning_fast_path = (
        quiet
        and prefetched_data is not None
        and prefetched_etf_universe is not None
        and prefetched_metrics is not None
        and trading_calendar is not None
        and strategy_override is not None
    )

    if not is_tuning_fast_path:
        _log(f"[백테스트] {account_id.upper()} 백테스트를 시작합니다...")
        _log("[백테스트] 설정을 로드하는 중...")

    account_settings = get_account_settings(account_id)
    country_code = (account_settings.get("country_code") or account_id).strip().lower() or "kor"

    if is_tuning_fast_path:
        # 튜닝 고속 경로: 최소한의 설정만 로드
        strategy_rules = strategy_override
        strategy_settings = dict(resolve_strategy_params(account_settings.get("strategy", {})))
        precision_settings = get_account_precision(account_id)
        common_settings = {}
    else:
        base_strategy_rules = get_strategy_rules(account_id)
        strategy_rules = StrategyRules.from_mapping(base_strategy_rules.to_dict())
        precision_settings = get_account_precision(account_id)
        account_settings_data = get_account_settings(account_id)
        strategy_source = account_settings_data.get("strategy", {})
        strategy_settings = dict(resolve_strategy_params(strategy_source))
        common_settings = get_common_file_settings()

    strategy_overrides_extra = override_settings.get("strategy_overrides")
    if isinstance(strategy_overrides_extra, Mapping):
        strategy_settings.update({str(k): v for k, v in strategy_overrides_extra.items()})

    if strategy_override is not None:
        strategy_rules = StrategyRules.from_values(
            ma_days=strategy_override.ma_days,
            bucket_topn=strategy_override.bucket_topn,
            ma_type=strategy_override.ma_type,
            enable_data_sufficiency_check=strategy_override.enable_data_sufficiency_check,
            rebalance_mode=strategy_override.rebalance_mode,
        )
        strategy_settings["MA_MONTH"] = strategy_rules.ma_days // TRADING_DAYS_PER_MONTH
        strategy_settings["MA_TYPE"] = strategy_rules.ma_type
        strategy_settings["BUCKET_TOPN"] = strategy_rules.bucket_topn
        strategy_settings["MA_TYPE"] = strategy_rules.ma_type

    backtest_start_date_str = _resolve_backtest_start_date(None, override_settings, account_settings)
    end_date = _resolve_end_date(country_code, override_settings)
    start_date = pd.to_datetime(backtest_start_date_str)

    capital_info = _resolve_initial_capital(
        initial_capital,
        override_settings,
        account_settings,
        precision_settings,
        start_date=start_date,
        prefetched_fx_series=prefetched_fx_series,
    )
    initial_capital_value = capital_info.local

    if not is_tuning_fast_path:
        _log(f"[백테스트] {account_id.upper()} 계정({country_code.upper()}) ETF 목록을 로드하는 중...")
    excluded_upper: set[str] = set()
    if excluded_tickers:
        excluded_upper = {
            str(ticker).strip().upper()
            for ticker in excluded_tickers
            if isinstance(ticker, str) and str(ticker).strip()
        }

    if prefetched_etf_universe is not None:
        etf_universe = [dict(stock) for stock in prefetched_etf_universe if isinstance(stock, Mapping)]
        if not is_tuning_fast_path:
            _log(f"[백테스트] 사전 추려진 ETF 대표군 {len(etf_universe)}개를 재사용합니다.")
    else:
        etf_universe = get_etfs(account_id)
    if not etf_universe:
        raise AccountSettingsError(f"계정 '{account_id}'에 대한 종목 설정(stocks.json)을 찾을 수 없습니다.")

    if excluded_upper:
        before_count = len(etf_universe)
        etf_universe = [
            stock for stock in etf_universe if str(stock.get("ticker", "")).strip().upper() not in excluded_upper
        ]
        removed = before_count - len(etf_universe)
        if removed > 0 and not is_tuning_fast_path:
            _log(f"[백테스트] 데이터 부족으로 제외된 {removed}개 종목을 유니버스에서 제거합니다.")
    if not etf_universe:
        raise RuntimeError("백테스트에 사용할 유효한 종목이 없습니다.")

    if not is_tuning_fast_path:
        _log(f"[백테스트] {len(etf_universe)}개의 ETF를 찾았습니다.")

    ticker_meta = {str(item.get("ticker", "")).upper(): dict(item) for item in etf_universe}
    ticker_meta["CASH"] = {"ticker": "CASH", "name": "현금"}

    # 검증은 get_account_strategy에서 이미 완료됨 - 바로 사용
    bucket_topn = strategy_rules.bucket_topn
    if not is_tuning_fast_path:
        _log(f"[백테스트] 포트폴리오 TOPN: {bucket_topn}")

    if not is_tuning_fast_path:
        _log("[백테스트] 백테스트 파라미터를 구성하는 중...")
    backtest_kwargs = _build_backtest_kwargs(
        strategy_rules=strategy_rules,
        strategy_settings=strategy_settings,
        prefetched_data=prefetched_data,
        prefetched_metrics=prefetched_metrics,
        quiet=quiet,
    )

    date_range = [start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")]

    display_currency = (capital_info.currency or "KRW").upper()

    if not is_tuning_fast_path:
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

    if trading_calendar:
        calendar_arg: list[pd.Timestamp] | None = []
        for raw in trading_calendar:
            try:
                dt = pd.Timestamp(raw)
            except Exception:
                continue
            if pd.isna(dt):
                continue
            if start_date <= dt <= end_date:
                calendar_arg.append(dt.normalize())
        if not calendar_arg:
            calendar_arg = None
    else:
        calendar_arg = get_trading_days(date_range[0], date_range[1], country_code)
        if not calendar_arg:
            raise RuntimeError(
                f"{account_id.upper()} 기간 {date_range[0]}~{date_range[1]}의 거래일 정보를 로드하지 못했습니다."
            )

    # 버켓 맵 구성 및 전체 top_n 계산
    bucket_map = {str(item.get("ticker", "")).upper(): item.get("bucket", 1) for item in etf_universe}
    unique_buckets = {item.get("bucket", 1) for item in etf_universe}
    num_buckets = len(unique_buckets) if unique_buckets else 1
    total_top_n = bucket_topn * num_buckets

    ticker_timeseries = (
        run_portfolio_backtest(
            stocks=etf_universe,
            initial_capital=initial_capital_value,
            core_start_date=start_date,
            top_n=total_top_n,
            bucket_topn=bucket_topn,  # 버켓당 제한 전달
            bucket_map=bucket_map,  # 버켓 맵 전달
            date_range=date_range,
            country=country_code,
            missing_ticker_sink=runtime_missing_tickers,
            **backtest_kwargs,
            trading_calendar=calendar_arg,
        )
        or {}
    )
    _log(f"[백테스트] 백테스트 완료. {len(ticker_timeseries)}개 종목의 데이터가 생성되었습니다.")

    if not ticker_timeseries:
        raise RuntimeError("백테스트 결과가 비어 있습니다. 유효한 데이터가 없습니다.")

    portfolio_df = _build_portfolio_timeseries(
        ticker_timeseries,
        initial_capital_value,
        bucket_topn,
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
        currency=display_currency,
        bucket_topn=bucket_topn,
        holdings_limit=total_top_n,  # Pass total limit
        account_settings=account_settings,
        prefetched_data=prefetched_data,
        ticker_timeseries=ticker_timeseries,
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
        currency=display_currency,
        bucket_topn=bucket_topn,
        holdings_limit=total_top_n,
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
        backtest_start_date=backtest_start_date_str,
        missing_tickers=missing_sorted,
    )


def _resolve_backtest_start_date(
    start_date: str | None,
    override_settings: Mapping[str, Any],
    account_settings: Mapping[str, Any],
) -> str:
    """백테스트 시작일을 결정합니다.

    우선순위:
    1. 직접 전달된 start_date
    2. override_settings의 backtest_start_date
    3. account_settings의 strategy.BACKTEST_START_DATE
    """
    if start_date is not None:
        return str(start_date)
    if "backtest_start_date" in override_settings:
        return str(override_settings["backtest_start_date"])
    if "start_date" in override_settings:
        return str(override_settings["start_date"])
    account_start = account_settings.get("strategy", {}).get("BACKTEST_START_DATE") if account_settings else None
    if account_start is not None:
        return str(account_start)
    raise ValueError("BACKTEST_START_DATE 설정이 필요합니다. 계정 설정의 strategy.BACKTEST_START_DATE 값을 확인하세요.")


def _resolve_initial_capital(
    initial_capital: float | None,
    override_settings: Mapping[str, Any],
    account_settings: Mapping[str, Any],
    precision_settings: Mapping[str, Any],
    start_date: pd.Timestamp | None = None,
    prefetched_fx_series: pd.Series | None = None,
) -> InitialCapitalInfo:
    def _coerce_positive_float(value: Any) -> float | None:
        try:
            candidate = float(value)
        except (TypeError, ValueError):
            return None
        return candidate if math.isfinite(candidate) and candidate > 0 else None

    currency = str(precision_settings.get("currency") or account_settings.get("currency") or "KRW").upper()
    # if currency not in {"KRW", "KR"}:
    #     raise ValueError(f"지원하지 않는 통화 코드입니다: {currency}")
    if currency == "KR":
        currency = "KRW"

    krw_override = _coerce_positive_float(override_settings.get("initial_capital_krw"))
    if krw_override is None:
        krw_override = _coerce_positive_float(account_settings.get("initial_capital_krw") if account_settings else None)

    if krw_override is None:
        raise ValueError("initial_capital_krw 설정이 필요합니다. 계정 설정의 'initial_capital_krw' 값을 확인하세요.")

    fx_rate = 1.0

    local_overrides = [
        initial_capital,
        override_settings.get("initial_capital"),
        account_settings.get("initial_capital") if account_settings else None,
    ]
    local_override = None
    for candidate in local_overrides:
        local_override = _coerce_positive_float(candidate)
        if local_override is not None:
            break

    fx_rate = 1.0
    if currency != "KRW" and start_date is not None:
        try:
            # Prefetch된 환율 시리즈가 있으면 사용
            if prefetched_fx_series is not None and not prefetched_fx_series.empty:
                fx_rate = float(prefetched_fx_series.asof(start_date))
                if pd.isna(fx_rate):
                    fx_rate = float(prefetched_fx_series.iloc[-1])
            else:
                # Fallback: 실시간 조회
                # 시작일 기준 환율 조회 (약간의 range를 두어 데이터 확보)
                search_start = start_date - pd.Timedelta(days=5)
                search_end = start_date + pd.Timedelta(days=1)
                fx_series = get_exchange_rate_series(search_start, search_end)
                if fx_series is not None and not fx_series.empty:
                    # start_date와 가장 가까운(이전 포함) 날짜의 환율 사용
                    # ffill 후 마지막 값 사용 or asof 사용
                    fx_rate = float(fx_series.asof(start_date))
                    if pd.isna(fx_rate):
                        fx_rate = float(fx_series.iloc[-1])
        except Exception:
            fx_rate = 1.0  # fallback

    local_capital = float(krw_override)
    if currency != "KRW" and fx_rate > 0:
        local_capital = float(krw_override) / fx_rate

    if local_override is not None:
        local_capital = float(local_override)

    return InitialCapitalInfo(
        local=float(local_capital),
        krw=float(krw_override),
        currency=currency,
    )


def _resolve_end_date(country_code: str, override_settings: Mapping[str, Any]) -> pd.Timestamp:
    if "end_date" in override_settings:
        return pd.to_datetime(override_settings["end_date"])
    return get_latest_trading_day(country_code)


def _build_backtest_kwargs(
    *,
    strategy_rules,
    strategy_settings: Mapping[str, Any],
    prefetched_data: Mapping[str, pd.DataFrame] | None,
    prefetched_metrics: Mapping[str, dict[str, Any]] | None,
    quiet: bool,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "prefetched_data": prefetched_data,
        "prefetched_metrics": prefetched_metrics,
        "ma_days": strategy_rules.ma_days,
        "ma_type": strategy_rules.ma_type,
        "rebalance_mode": strategy_rules.rebalance_mode,
        "quiet": quiet,
        "enable_data_sufficiency_check": strategy_rules.enable_data_sufficiency_check,
    }

    clean_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return clean_kwargs


def _build_portfolio_timeseries(
    ticker_timeseries: Mapping[str, pd.DataFrame],
    initial_capital: float,
    bucket_topn: int,
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
    prev_total_value: float | None = None
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
                "bucket_topn": bucket_topn,
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
    currency: str,
    bucket_topn: int,
    holdings_limit: int,
    account_settings: Mapping[str, Any],
    prefetched_data: Mapping[str, pd.DataFrame] | None = None,
    ticker_timeseries: dict[str, pd.DataFrame] | None = None,
) -> tuple[
    dict[str, Any],
    pd.Series,
    pd.Series,
    pd.Series,
]:
    final_row = portfolio_df.iloc[-1]
    pv_series = portfolio_df["total_value"].astype(float)
    pv_series.index = pd.to_datetime(pv_series.index)

    initial_capital_local = initial_capital
    final_value = float(pv_series.iloc[-1])

    # 환율 데이터 로드 및 KRW 가치 계산
    pv_series_krw = pv_series.copy()
    initial_capital_val_for_cagr = initial_capital
    if currency != "KRW":
        try:
            fx_series = get_exchange_rate_series(start_date, end_date)
            if fx_series is not None and not fx_series.empty:
                # 인덱스 정렬 및 보간
                fx_series = fx_series.reindex(pv_series.index).fillna(method="ffill").fillna(method="bfill")
                pv_series_krw = pv_series * fx_series

                # CAGR 계산을 위한 초기 자본금은 KRW 기준 사용
                initial_capital_val_for_cagr = initial_capital_krw

                # 리포트에 환율 정보 추가 (옵션)
                # logger.info(f"환율 적용 완료: {len(fx_series)}일치")
        except Exception:
            # 환율 조회 실패 시 경고 로그?
            pass

    years = max((end_date - start_date).days / 365.25, 0.0)

    # 평가액(KRW 기준) 사용
    final_value_krw = float(pv_series_krw.iloc[-1])

    cagr = 0.0
    if years > 0 and initial_capital_val_for_cagr > 0:
        cagr = (final_value_krw / initial_capital_val_for_cagr) ** (1 / years) - 1

    running_max = pv_series_krw.cummax()
    drawdown_series = (running_max - pv_series_krw) / running_max.replace({0: pd.NA})
    drawdown_series = drawdown_series.fillna(0.0)
    max_drawdown = float(drawdown_series.max()) if not drawdown_series.empty else 0.0

    daily_returns = pv_series_krw.pct_change().dropna()
    sharpe_ratio = 0.0
    if not daily_returns.empty:
        mean_ret = daily_returns.mean()
        std_ret = daily_returns.std()
        if std_ret and math.isfinite(std_ret) and std_ret > 0:
            sharpe_ratio = (mean_ret / std_ret) * (252**0.5)

    # Sharpe/MDD 비율 계산 (SDR)
    mdd_pct = max_drawdown * 100
    sharpe_to_mdd = (sharpe_ratio / mdd_pct) if mdd_pct > 0 else 0.0

    def _load_benchmark_frame(ticker: str) -> pd.DataFrame | None:
        candidates: list[str] = []
        norm = str(ticker or "").strip()
        if not norm:
            return None
        candidates.extend({norm, norm.upper(), norm.lower()})

        if prefetched_data:
            for candidate in candidates:
                frame = prefetched_data.get(candidate)
                if isinstance(frame, pd.DataFrame) and not frame.empty:
                    return frame

        return None

    def _calc_benchmark_performance(*, ticker: str, name: str, country: str) -> dict[str, Any] | None:
        benchmark_df = _load_benchmark_frame(ticker)
        if benchmark_df is None or benchmark_df.empty:
            local_logger = get_app_logger()
            local_logger.warning("[백테스트] 벤치마크 '%s' 데이터를 프리패치에서 찾을 수 없습니다.", ticker)
            return None

        benchmark_df = benchmark_df.sort_index()
        benchmark_df.index = pd.to_datetime(benchmark_df.index)
        mask = (benchmark_df.index >= start_date) & (benchmark_df.index <= end_date)
        benchmark_df = benchmark_df.loc[mask]
        benchmark_df = benchmark_df.loc[benchmark_df.index.intersection(pv_series.index)]
        if benchmark_df.empty or "Close" not in benchmark_df.columns:
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

        # 월별 수익률 계산 (리포팅용)
        bench_monthly_prices = bench_series.resample("ME").last()
        bench_monthly_returns = bench_monthly_prices.pct_change()

        # 첫 달 수익률 보정: pct_change는 전월 데이터가 없어 NaN이 되므로, 시작 가격 기준으로 계산
        if not bench_monthly_returns.empty and start_price > 0:
            first_val = float(bench_monthly_prices.iloc[0])
            first_ret = (first_val / start_price) - 1.0

            if pd.isna(bench_monthly_returns.iloc[0]):
                bench_monthly_returns.iloc[0] = first_ret

        bench_monthly_returns = bench_monthly_returns.dropna()

        return {
            "ticker": ticker,
            "name": name,
            "country": country,
            "cumulative_return_pct": cum_ret_pct,
            "cagr_pct": cagr_pct,
            "sharpe": bench_sharpe,
            "mdd": bench_mdd_pct,
            "sharpe_to_mdd": bench_sharpe_to_mdd,
            "monthly_returns": bench_monthly_returns,
        }

    benchmark_cum_ret_pct = 0.0
    benchmark_cagr_pct = 0.0
    benchmarks_summary: list[dict[str, Any]] = []

    # 1. 단일 벤치마크 (우선순위)
    bench_conf = account_settings.get("benchmark")

    # 2. 레거시 지원 (리스트 형태)
    if not bench_conf:
        legacy_list = account_settings.get("benchmarks")
        if isinstance(legacy_list, list) and legacy_list:
            bench_conf = legacy_list[0]

    if isinstance(bench_conf, Mapping):
        ticker_value = str(bench_conf.get("ticker") or "").strip()
        if ticker_value:
            name_value = str(bench_conf.get("name") or ticker_value).strip() or ticker_value
            bench_country = (
                str(bench_conf.get("country") or bench_conf.get("market") or country_code).strip() or country_code
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

    weekly_summary_rows: list[dict[str, Any]] = []
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
                # 해당 날짜의 보유종목 수 가져오기
                held_count = 0
                max_topn = holdings_limit
                actual_date = dt

                # 해당 날짜가 portfolio_df에 없으면 가장 가까운 이전 날짜 찾기
                if dt not in portfolio_df.index:
                    # portfolio_df에서 dt 이전의 가장 가까운 날짜 찾기
                    earlier_dates = portfolio_df.index[portfolio_df.index <= dt]
                    if len(earlier_dates) > 0:
                        actual_date = earlier_dates[-1]
                        held_count = (
                            int(portfolio_df.loc[actual_date, "held_count"])
                            if pd.notna(portfolio_df.loc[actual_date, "held_count"])
                            else 0
                        )
                else:
                    held_count = (
                        int(portfolio_df.loc[dt, "held_count"]) if pd.notna(portfolio_df.loc[dt, "held_count"]) else 0
                    )

                # 날짜 포맷: 금요일이 아니면 요일 표시
                weekday_map = {0: "월", 1: "화", 2: "수", 3: "목", 4: "금", 5: "토", 6: "일"}
                weekday = weekday_map.get(actual_date.weekday(), "")
                if actual_date.weekday() == 4:  # 금요일
                    date_display = actual_date.strftime("%Y-%m-%d")
                else:
                    date_display = f"{actual_date.strftime('%Y-%m-%d')}({weekday})"

                weekly_summary_rows.append(
                    {
                        "week_end": date_display,
                        "value": float(value),
                        "held_count": held_count,
                        "max_topn": max_topn,
                        "weekly_return_pct": float(weekly_return_pct.loc[dt]),
                        "cumulative_return_pct": float(weekly_cum_pct.loc[dt]),
                    }
                )

        monthly_returns = pv_series_with_start.resample("ME").last().pct_change().dropna()
        if initial_capital_local > 0:
            eom_pv = pv_series.resample("ME").last().dropna()
            monthly_cum_returns = (eom_pv / initial_capital_local - 1).ffill()
        yearly_returns = pv_series_with_start.resample("YE").last().pct_change()
        # 첫 해 수익률 보정: pct_change는 전년도 데이터가 없어 NaN이 되므로, 초기 자본금 기준으로 계산
        if not yearly_returns.empty and initial_capital_local > 0:
            yearly_prices = pv_series_with_start.resample("YE").last()
            # 첫 번째 연도의 기말 평가액
            first_val = float(yearly_prices.iloc[0])
            # 수익률 = (기말 / 기초) - 1
            first_ret = (first_val / initial_capital_local) - 1.0

            # 첫 번째 값이 NaN이면 채워넣기
            if pd.isna(yearly_returns.iloc[0]):
                yearly_returns.iloc[0] = first_ret

        yearly_returns = yearly_returns.dropna()

    # Turnover calculation (전체 거래 횟수) - if 블록 밖에서 항상 실행
    total_turnover = 0
    if ticker_timeseries:
        trade_decisions = {"BUY", "BUY_REPLACE", "SELL_REPLACE"}
        for t_data in ticker_timeseries.values():
            if isinstance(t_data, pd.DataFrame) and "decision" in t_data.columns:
                trade_count = t_data["decision"].isin(trade_decisions).sum()
                total_turnover += int(trade_count)

    summary = {
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "initial_capital": initial_capital_local,
        "initial_capital_local": initial_capital_local,
        "initial_capital_krw": initial_capital_krw,
        "final_value": final_value,
        "final_value_local": final_value,
        "final_value_krw": final_value_krw,
        "period_return": float(final_row["cumulative_return_pct"]),
        "evaluation_return_pct": float(final_row["evaluation_return_pct"]),
        "held_count": int(final_row["held_count"]),
        "turnover": total_turnover,
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
        "benchmark_monthly_returns": {
            (b.get("name") or b.get("ticker")): b.get("monthly_returns")
            for b in benchmarks_summary
            if b.get("monthly_returns") is not None and not b["monthly_returns"].empty
        },
        "currency": currency,
    }

    return summary, monthly_returns, monthly_cum_returns, yearly_returns


def _compute_evaluated_records(
    ticker_timeseries: Mapping[str, pd.DataFrame],
    start_date: pd.Timestamp,
) -> dict[str, dict[str, Any]]:
    """백테스트 시작일 이후의 거래 손익만 집계합니다.

    주의: 이 함수는 현재 사용되지 않습니다.
    누적 손익은 reporting.py에서 cost_basis 기준으로 직접 계산됩니다.
    """
    records: dict[str, dict[str, Any]] = {}
    start_date_norm = start_date.normalize()

    for ticker, df in ticker_timeseries.items():
        # DataFrame만 처리 (메타데이터 문자열 제외)
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue

        df_sorted = df.sort_index()
        # 백테스트 시작일 이후 데이터만 필터링
        df_filtered = df_sorted[df_sorted.index >= start_date_norm]

        realized_profit = 0.0
        initial_value: float | None = None

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
    ticker_meta: Mapping[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    sell_decisions = {
        "SELL_REPLACE",
    }

    summaries: list[dict[str, Any]] = []
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
        listing_date: str | None = None
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

        win_rate = (winning_trades / total_trades) * 100.0 if total_trades > 0 else 0.0
        meta = ticker_meta.get(ticker_key, {})

        summaries.append(
            {
                "ticker": ticker_key,
                "name": meta.get("name") or ticker_key,
                "total_trades": total_trades,
                "winning_trades": winning_trades,
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
) -> dict[str, Any]:
    snapshot = {
        "account_id": account_id.upper(),
        "country_code": country_code.upper(),
        "initial_capital": float(initial_capital),
        "strategy_rules": strategy_rules.to_dict(),
        "common_settings": dict(common_settings),
        "strategy_settings": dict(strategy_settings),
    }
    return snapshot
