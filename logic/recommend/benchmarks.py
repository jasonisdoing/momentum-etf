"""Benchmarks logic extracted from root signals module.

This module computes benchmark cumulative returns and excess returns versus the
portfolio as of a given date.
"""
from __future__ import annotations

from typing import Optional, List, Dict, Any

import pandas as pd

try:  # optional dependency
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None

from utils.data_loader import fetch_ohlcv, get_trading_days

try:
    from utils.db_manager import get_portfolio_snapshot as _get_portfolio_snapshot  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - optional dependency removed
    _get_portfolio_snapshot = None  # type: ignore[assignment]
from utils.country_registry import get_country_settings


def _normalize_yfinance_df(df_y: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Normalize yfinance dataframe to expected format (Close column, no tz)."""
    if df_y is None or df_y.empty:
        return None
    if isinstance(df_y.columns, pd.MultiIndex):
        df_y.columns = df_y.columns.get_level_values(0)
        df_y = df_y.loc[:, ~df_y.columns.duplicated()]
    if getattr(df_y.index, "tz", None) is not None:
        df_y.index = df_y.index.tz_localize(None)
    if "Adj Close" in df_y.columns:
        df_y = df_y.rename(columns={"Adj Close": "Close"})
    elif "Close" not in df_y.columns:
        return None
    return df_y


def _determine_benchmark_country(ticker: str) -> str:
    """Infer country code from benchmark ticker.

    - 6-digit numeric -> kor
    - contains .AX -> aus
    - default -> kor
    """
    if ticker.isdigit() and len(ticker) == 6:
        return "kor"
    if ".AX" in ticker.upper():
        return "aus"
    return "kor"


def _calculate_single_benchmark(
    benchmark_ticker: str,
    benchmark_name: str,
    benchmark_country: str,
    initial_date: pd.Timestamp,
    base_date: pd.Timestamp,
) -> Dict[str, Any]:
    """Calculate single benchmark cumulative return info.

    Returns dict with keys: ticker, name, cum_ret_pct or error.
    """
    base_result = {"ticker": benchmark_ticker, "name": benchmark_name}

    if base_date < initial_date:
        base_result[
            "error"
        ] = f"조회 종료일({base_date.strftime('%Y-%m-%d')})이 시작일({initial_date.strftime('%Y-%m-%d')})보다 빠릅니다."
        return base_result

    try:
        df_benchmark = fetch_ohlcv(
            benchmark_ticker,
            country=benchmark_country,
            date_range=[
                initial_date.strftime("%Y-%m-%d"),
                base_date.strftime("%Y-%m-%d"),
            ],
        )
    except Exception:
        # fallback to previous trading day window
        prev_day_search_end = base_date - pd.Timedelta(days=1)
        prev_day_search_start = prev_day_search_end - pd.Timedelta(days=14)
        previous_trading_days = get_trading_days(
            prev_day_search_start.strftime("%Y-%m-%d"),
            prev_day_search_end.strftime("%Y-%m-%d"),
            benchmark_country,
        )
        if previous_trading_days:
            previous_trading_day = previous_trading_days[-1]
            df_benchmark = fetch_ohlcv(
                benchmark_ticker,
                country=benchmark_country,
                date_range=[
                    initial_date.strftime("%Y-%m-%d"),
                    previous_trading_day.strftime("%Y-%m-%d"),
                ],
            )
        else:
            df_benchmark = None

    # generic fallback using yfinance when primary data source is unavailable
    if (df_benchmark is None or df_benchmark.empty) and yf is not None:
        try:
            y_ticker = (
                "BTC-USD"
                if benchmark_ticker.upper() == "BTC"
                else f"{benchmark_ticker.upper()}-USD"
            )
            df_y = yf.download(
                y_ticker,
                start=initial_date.strftime("%Y-%m-%d"),
                end=(base_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                auto_adjust=True,
                progress=False,
            )
            df_benchmark = _normalize_yfinance_df(df_y)
        except Exception:
            pass

    if df_benchmark is None or df_benchmark.empty:
        base_result["error"] = "데이터 조회 실패"
        return base_result

    start_prices = df_benchmark[df_benchmark.index >= initial_date]["Close"]
    if start_prices.empty:
        base_result["error"] = "시작 가격 조회 실패"
        return base_result
    benchmark_start_price = start_prices.iloc[0]

    end_prices = df_benchmark[df_benchmark.index <= base_date]["Close"]
    if end_prices.empty:
        base_result["error"] = "종료 가격 조회 실패"
        return base_result
    benchmark_end_price = end_prices.iloc[-1]

    if pd.isna(benchmark_start_price) or pd.isna(benchmark_end_price) or benchmark_start_price <= 0:
        base_result["error"] = "가격 정보 오류"
        return base_result

    benchmark_cum_ret_pct = ((benchmark_end_price / benchmark_start_price) - 1.0) * 100.0
    base_result["cum_ret_pct"] = benchmark_cum_ret_pct
    base_result["error"] = None
    return base_result


def _is_trading_day(country: str, a_date: pd.Timestamp) -> bool:
    try:
        days = get_trading_days(a_date.strftime("%Y-%m-%d"), a_date.strftime("%Y-%m-%d"), country)
        return any(pd.Timestamp(d).date() == a_date.date() for d in days)
    except Exception:
        return False


def calculate_benchmark_comparison(
    country: str, date_str: Optional[str] = None
) -> Optional[List[Dict[str, Any]]]:
    """국가 단위 포트폴리오와 벤치마크 성과를 비교합니다."""
    from utils.transaction_manager import get_transactions_up_to_date

    if not country:
        return None

    # 국가 설정 로드
    try:
        # 국가 설정 가져오기
        country_settings = get_country_settings(country)
        if not country_settings:
            raise ValueError(f"등록되지 않은 국가입니다: {country}")

        # 초기 자본금 및 시작일 설정 (국가 설정 파일에서 가져옴)
        initial_capital_krw = float(country_settings.get("initial_capital_krw", 0))
        initial_date_str = country_settings.get("initial_date")
        if not initial_date_str:
            raise ValueError(f"{country} 국가 설정에 initial_date가 설정되지 않았습니다.")
        initial_date = pd.to_datetime(initial_date_str).normalize()

        # 벤치마크 티커 목록 가져오기
        benchmarks_to_compare = country_settings.get("benchmarks_tickers", [])
        if not isinstance(benchmarks_to_compare, list):
            benchmarks_to_compare = []

    except Exception as e:
        return [{"name": "벤치마크", "error": str(e)}]

    if not benchmarks_to_compare or initial_capital_krw <= 0:
        return None

    if not callable(_get_portfolio_snapshot):
        return None

    portfolio_data = _get_portfolio_snapshot(country, date_str=date_str)
    if not portfolio_data:
        return None

    base_date = pd.to_datetime(portfolio_data["date"]).normalize()

    # recalculate holdings value to align with generate_signal_report logic
    current_equity = float(portfolio_data.get("total_equity", 0.0))
    recalculated_holdings_value = 0.0
    for h in portfolio_data.get("holdings") or []:
        ticker = h.get("ticker")
        shares = h.get("shares", 0.0)
        if not ticker or not shares > 0:
            continue
        df = fetch_ohlcv(ticker, country=country, months_back=1, base_date=base_date)
        price = 0.0
        if df is not None and not df.empty:
            prices_until_base = df[df.index <= base_date]["Close"]
            if not prices_until_base.empty:
                price = prices_until_base.iloc[-1]
        if price > 0:
            recalculated_holdings_value += shares * price

    if country == "aus":
        intl_info = portfolio_data.get("international_shares")
        if isinstance(intl_info, dict):
            recalculated_holdings_value += float(intl_info.get("value", 0.0))

    equity_for_calc = current_equity
    if recalculated_holdings_value > 0 and (
        current_equity == 0 or recalculated_holdings_value > current_equity
    ):
        equity_for_calc = recalculated_holdings_value

    # adjust base_date to previous trading day if needed
    if not _is_trading_day(country, base_date):
        start_search = (base_date - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
        end_search = base_date.strftime("%Y-%m-%d")
        trading_days = get_trading_days(start_search, end_search, country)
        if trading_days:
            base_date = pd.to_datetime(trading_days[-1]).normalize()

    if initial_date > base_date:
        error_msg = f"초기 기준일({initial_date.strftime('%Y-%m-%d')})이 조회일({base_date.strftime('%Y-%m-%d')})보다 미래입니다."
        return [{"name": "벤치마크", "error": error_msg}]

    injections = get_transactions_up_to_date(country, base_date, "capital_injection")
    withdrawals = get_transactions_up_to_date(country, base_date, "cash_withdrawal")

    total_injections = sum(inj.get("amount", 0.0) for inj in injections)
    total_withdrawals = sum(wd.get("amount", 0.0) for wd in withdrawals)

    adjusted_capital_base = initial_capital_krw + total_injections
    adjusted_equity = equity_for_calc + total_withdrawals

    portfolio_cum_ret_pct = (
        ((adjusted_equity / adjusted_capital_base) - 1.0) * 100.0
        if adjusted_capital_base > 0
        else 0.0
    )

    results: List[Dict[str, Any]] = []
    for bm_info in benchmarks_to_compare:
        bm_ticker = bm_info.get("ticker")
        bm_name = bm_info.get("name")
        if not bm_ticker or not bm_name:
            continue

        bm_country = _determine_benchmark_country(bm_ticker)
        bm_result = _calculate_single_benchmark(
            benchmark_ticker=bm_ticker,
            benchmark_name=bm_name,
            benchmark_country=bm_country,
            initial_date=initial_date,
            base_date=base_date,
        )
        if bm_result:
            bm_result["ticker"] = bm_ticker
            if not bm_result.get("error"):
                bm_result["excess_return_pct"] = portfolio_cum_ret_pct - bm_result["cum_ret_pct"]
            results.append(bm_result)

    return results if results else None


__all__ = ["calculate_benchmark_comparison"]
