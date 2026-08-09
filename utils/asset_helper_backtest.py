"""자산 헬퍼 백테스트 — 리밸런싱 시뮬레이션과 금요일 비중 이력.

`utils/asset_helper_service.py` 에서 분리(이동만, 로직 불변). 선정·정리 헬퍼는
서비스와 시장 데이터 층을 재사용한다.
"""


from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from core.strategy.scoring import build_composite_rank_scores
from utils.logger import get_app_logger
from utils.perf_metrics import curve_metrics, mdd_span

logger = get_app_logger()

from utils.asset_helper_market_data import (
    _convert_close_frame_to_krw,
    _load_close_frame,
    _resolve_backtest_currency,
)
from utils.asset_helper_service import (
    _allocate_deviation_filtered_weights,
    _build_asset_helper_ma_rule,
    _clean_backtest_settings,
    _clean_settings,
    _clean_tickers,
    _filter_rank_excluded_tickers,
    _with_account_asset_helper_basis,
)


def _select_rebalance_dates(index: pd.DatetimeIndex, rebalance: str) -> list[pd.Timestamp]:
    if index.empty:
        return []
    if rebalance == "none":
        return [index[0]]

    freq_map = {"weekly": "W", "monthly": "M", "quarterly": "Q", "yearly": "Y"}
    periods = index.to_period(freq_map[rebalance])
    selected: dict[Any, pd.Timestamp] = {}
    for date, period in zip(index, periods):
        selected[period] = date
    dates = sorted(set(selected.values()))
    if index[0] not in dates:
        dates.insert(0, index[0])
    return dates


def _select_friday_history_dates(index: pd.DatetimeIndex) -> set[pd.Timestamp]:
    if index.empty:
        return set()

    max_date = index[-1].normalize()
    periods = index.to_period("W-FRI")
    selected: dict[Any, pd.Timestamp] = {}
    for date, period in zip(index, periods):
        period_end = period.end_time.normalize()
        if period_end <= max_date:
            selected[period] = date
    res = set(selected.values())
    res.add(index[-1])
    return res


def _build_asset_helper_weight_engine(
    close_frame: pd.DataFrame,
    tickers: list[dict[str, Any]],
    settings: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    ma_rules = [_build_asset_helper_ma_rule(settings)]
    composite_frame, trend_by_order, _ = build_composite_rank_scores(close_frame, ma_rules)
    return composite_frame, trend_by_order


def _calculate_asset_helper_weights_on_date(
    eval_date: pd.Timestamp,
    tickers: list[dict[str, Any]],
    settings: dict[str, Any],
    composite_frame: pd.DataFrame,
    trend_frame: pd.DataFrame,
) -> dict[str, float] | None:
    eligible_dates = composite_frame.index[composite_frame.index <= eval_date]
    if eligible_dates.empty:
        return None

    score_date = eligible_dates.max()
    composite_row = composite_frame.loc[score_date]
    trend_row = trend_frame.loc[score_date] if score_date in trend_frame.index else pd.Series(dtype=float)
    deviation_pct_by_ticker: dict[str, float | None] = {}
    for item in tickers:
        ticker = str(item.get("ticker") or "").strip().upper()
        score_value = composite_row.get(ticker)
        deviation_pct = trend_row.get(ticker)
        if pd.isna(score_value) or pd.isna(deviation_pct):
            deviation_pct_by_ticker[ticker] = None
        elif float(deviation_pct) > 0:
            deviation_pct_by_ticker[ticker] = float(deviation_pct)
        else:
            deviation_pct_by_ticker[ticker] = float(deviation_pct)
    return _allocate_deviation_filtered_weights(deviation_pct_by_ticker, float(settings["STOCK_MAX_WEIGHT"]))


def _resolve_slippage_by_ticker(clean_tickers: list[dict[str, Any]]) -> dict[str, tuple[float, float]]:
    """종목별 (매수, 매도) 슬리피지 비율 — 종목풀 DB 설정을 그대로 사용한다.

    ticker_type 미상이거나 종목풀 슬리피지 설정이 없으면 임의 기본값 없이 명시적 에러.
    """
    from utils.settings_loader import get_ticker_type_settings

    rates: dict[str, tuple[float, float]] = {}
    problems: list[str] = []
    settings_cache: dict[str, dict[str, Any]] = {}
    for item in clean_tickers:
        ticker = str(item["ticker"])
        ticker_type = str(item.get("ticker_type") or "").strip().lower()
        if ticker_type:
            if ticker_type not in settings_cache:
                settings_cache[ticker_type] = get_ticker_type_settings(ticker_type)
        config = settings_cache.get(ticker_type, {})
        if (
            not ticker_type
            or config.get("BUY_SLIPPAGE_PCT") in (None, "")
            or config.get("SELL_SLIPPAGE_PCT") in (None, "")
        ):
            problems.append(f"{ticker}({ticker_type or '종목풀 미상'})")
            continue
        rates[ticker] = (float(config["BUY_SLIPPAGE_PCT"]) / 100.0, float(config["SELL_SLIPPAGE_PCT"]) / 100.0)
    if problems:
        raise ValueError(
            f"종목풀 슬리피지 설정이 없는 종목이 있습니다: {', '.join(problems)}. "
            "/pools-settings 에서 해당 종목풀의 매수/매도 슬리피지를 저장하세요."
        )
    return rates


def run_asset_helper_backtest(
    tickers: list[dict[str, Any]],
    settings: dict[str, Any] | None = None,
    backtest_settings: dict[str, Any] | None = None,
    weight_mode: str = "variable",
) -> dict[str, Any]:
    # weight_mode="fixed"(고정 보유)는 이평선·종목풀 연결이 필요 없다. 그 외는 동일 슬롯 + 개별 이평선 필터.
    # 백테스트는 전달받은 "현재 종목"만 검증한다 — trend 라도 풀 전체를 재선정하지 않는다
    # (풀 백테스트는 종목풀 화면에 별도로 있음).
    clean_tickers = _clean_tickers(tickers)
    clean_settings = _with_account_asset_helper_basis(_clean_settings(settings), weight_mode=weight_mode)
    # 고정 비중(사용자가 직접 고른 종목)은 순위 고정(exclude_from_ranking) 제외를 적용하지 않는다.
    # 변동 모드에서만 순위 유니버스에서 고정 종목을 뺀다(비중 계산부와 동일 규칙).
    if weight_mode == "fixed":
        excluded_fixed_tickers = []
    else:
        clean_tickers, excluded_fixed_tickers = _filter_rank_excluded_tickers(clean_tickers, clean_settings)
    if len(clean_tickers) < 3:
        suffix = f" 고정 종목 제외: {', '.join(excluded_fixed_tickers)}" if excluded_fixed_tickers else ""
        raise ValueError(f"백테스트에는 고정 종목 제외 후 확인된 종목이 3개 이상 필요합니다.{suffix}")
    clean_backtest = _clean_backtest_settings(backtest_settings, require_benchmark=False)
    # 결과 비교용 벤치마크 — 계좌 설정(/account-settings)의 벤치마크를 쓴다(계좌별로 다름).
    account_id = str(clean_settings.get("ACCOUNT_ID") or "").strip()
    if not account_id:
        raise ValueError("백테스트에는 적용 계좌가 필요합니다.")
    from utils.settings_loader import get_account_settings

    _account_bench = get_account_settings(account_id).get("benchmark")
    if not isinstance(_account_bench, dict) or not str(_account_bench.get("ticker") or "").strip():
        raise ValueError(f"계좌 '{account_id}' 설정에 벤치마크가 없습니다. 계좌 설정에서 등록해주세요.")
    benchmark = {
        "ticker": str(_account_bench["ticker"]).strip().upper(),
        "name": str(_account_bench.get("name") or _account_bench["ticker"]).strip(),
    }
    months = int(clean_backtest["months"])
    rebalance = str(clean_backtest["rebalance"])
    initial_capital_krw = float(int(clean_backtest["initial_amount_manwon"]) * 10_000)

    ticker_order = [item["ticker"] for item in clean_tickers]
    combined_tickers = clean_tickers + [benchmark]
    close_frame, missing = _load_close_frame(combined_tickers)
    if close_frame.empty:
        raise ValueError("백테스트 가격 캐시가 없습니다.")

    missing_required = [ticker for ticker in ticker_order + [benchmark["ticker"]] if ticker not in close_frame.columns]
    if missing_required:
        raise ValueError(f"백테스트 가격 캐시 누락: {', '.join(missing_required)}")

    # 시점별 환율 반영: 비원화(미국 USD·호주 AUD) 종목 종가를 원화로 환산한 뒤 시뮬레이션한다.
    # 벤치마크는 country_code 가 없으므로 티커 형식/메타로 통화를 추정한다.
    currency_by_ticker = {
        str(item["ticker"]).strip().upper(): _resolve_backtest_currency(
            item["ticker"], item.get("country_code"), item.get("ticker_type")
        )
        for item in clean_tickers
    }
    currency_by_ticker[benchmark["ticker"]] = _resolve_backtest_currency(benchmark["ticker"], None, None)
    close_frame = _convert_close_frame_to_krw(close_frame, currency_by_ticker)

    today = pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None).normalize()
    start_target = (today - pd.DateOffset(months=months)).normalize()
    candidate_close = close_frame[ticker_order].sort_index()

    simulation_columns = list(dict.fromkeys(ticker_order + [benchmark["ticker"]]))
    simulation_frame = close_frame[simulation_columns].sort_index()
    simulation_frame = simulation_frame[simulation_frame.index >= start_target].dropna(how="all").ffill()
    simulation_frame = simulation_frame.dropna(subset=[benchmark["ticker"]])
    if simulation_frame.empty or len(simulation_frame) < 2:
        raise ValueError("백테스트 기간의 가격 데이터가 부족합니다.")

    requested_rebalance_dates = _select_rebalance_dates(simulation_frame.index, rebalance)
    weights_by_date: dict[pd.Timestamp, dict[str, float]] = {}

    if weight_mode == "fixed":
        fixed_weights = {item["ticker"]: float(item.get("fixed_weight_pct") or 0.0) / 100.0 for item in clean_tickers}
        sum_fixed = sum(fixed_weights.values())
        if sum_fixed > 1.0:
            for tk in fixed_weights:
                fixed_weights[tk] /= sum_fixed
            cash_weight = 0.0
        else:
            cash_weight = 1.0 - sum_fixed
        fixed_weights["__CASH__"] = cash_weight

        weights_by_date = {date: fixed_weights for date in requested_rebalance_dates}
    else:
        composite_frame, trend_by_order = _build_asset_helper_weight_engine(candidate_close, clean_tickers, clean_settings)
        trend_frame = trend_by_order[1]

        for date in requested_rebalance_dates:
            weights = _calculate_asset_helper_weights_on_date(
                date,
                clean_tickers,
                clean_settings,
                composite_frame,
                trend_frame,
            )
            if weights is not None:
                weights_by_date[date] = weights

    if not weights_by_date:
        raise ValueError("백테스트 기간에 계산 가능한 비중이 없습니다.")

    start_date = min(weights_by_date)
    sim = simulation_frame[simulation_frame.index >= start_date].copy()
    if len(sim) < 2:
        raise ValueError("백테스트 시뮬레이션 기간이 부족합니다.")

    # 슬리피지(종목풀별 DB 설정) — 매수/매도 금액에 비용으로 차감한다.
    slippage_by_ticker = _resolve_slippage_by_ticker(clean_tickers)
    total_slippage_cost = 0.0

    def allocate_with_slippage(
        total_value: float, weights: dict[str, float], current_values: dict[str, float]
    ) -> tuple[dict[str, float], float, float, dict[str, float]]:
        """총자산을 목표비중으로 재배치하며 종목별 매수/매도 금액 × 슬리피지율을 비용으로 차감한다."""
        cost = 0.0
        costs_by_ticker: dict[str, float] = {}
        for ticker in ticker_order:
            target = total_value * float(weights.get(ticker, 0.0))
            delta = target - float(current_values.get(ticker, 0.0))
            buy_ratio, sell_ratio = slippage_by_ticker[ticker]
            ticker_cost = 0.0
            if delta > 0:
                ticker_cost = delta * buy_ratio
            elif delta < 0:
                ticker_cost = (-delta) * sell_ratio
            costs_by_ticker[ticker] = ticker_cost
            cost += ticker_cost
        net_total = total_value - cost
        values = {ticker: net_total * float(weights.get(ticker, 0.0)) for ticker in ticker_order}
        cash = net_total * float(weights.get("__CASH__", 0.0))
        return values, cash, cost, costs_by_ticker

    current_weights = weights_by_date[start_date]
    profit_by_ticker = {ticker: 0.0 for ticker in ticker_order}
    # 최초 편입 — 전액 현금에서 매수하므로 매수 슬리피지가 발생한다.
    asset_values, cash_value, initial_cost, initial_costs_by_ticker = allocate_with_slippage(
        initial_capital_krw, current_weights, {ticker: 0.0 for ticker in ticker_order}
    )
    total_slippage_cost += initial_cost
    for ticker, ticker_cost in initial_costs_by_ticker.items():
        profit_by_ticker[ticker] -= ticker_cost
    curve_values: list[float] = [initial_capital_krw]
    weight_history: list[dict[str, Any]] = []
    friday_history_dates = _select_friday_history_dates(sim.index)

    # 종목별/현금의 일별 실제 비중(%) 이력을 추적합니다.
    weights_history_by_ticker: dict[str, list[float]] = {ticker: [] for ticker in ticker_order}
    weights_history_by_ticker["__CASH__"] = []

    def record_current_weights():
        total_val = sum(asset_values.values()) + cash_value
        if total_val > 0:
            for t in ticker_order:
                weights_history_by_ticker[t].append((asset_values[t] / total_val) * 100.0)
            weights_history_by_ticker["__CASH__"].append((cash_value / total_val) * 100.0)

    # 1. start_date 시점의 비중 기록
    record_current_weights()

    def append_weight_history(date: pd.Timestamp, values: dict[str, float], cash: float) -> None:
        row: dict[str, Any] = {"date": date.date().isoformat()}
        for item in clean_tickers:
            ticker = item["ticker"]
            row[ticker] = round(float(values.get(ticker, 0.0)), 0)
        row["__CASH__"] = round(float(cash), 0)
        weight_history.append(row)

    if start_date in friday_history_dates:
        append_weight_history(start_date, asset_values, cash_value)
    sim_index = list(sim.index)
    for idx in range(1, len(sim_index)):
        prev_date = sim_index[idx - 1]
        date = sim_index[idx]
        previous_prices = sim.loc[prev_date, ticker_order]
        current_prices = sim.loc[date, ticker_order]
        for ticker in ticker_order:
            previous_price = float(previous_prices[ticker])
            current_price = float(current_prices[ticker])
            if previous_price > 0 and current_price > 0:
                previous_value = float(asset_values.get(ticker, 0.0))
                current_value = previous_value * (current_price / previous_price)
                profit_by_ticker[ticker] += current_value - previous_value
                asset_values[ticker] = current_value

        if date in weights_by_date and date != start_date:
            current_weights = weights_by_date[date]
            total_value = sum(asset_values.values()) + cash_value
            asset_values, cash_value, rebalance_cost, rebalance_costs_by_ticker = allocate_with_slippage(
                total_value, current_weights, asset_values
            )
            total_slippage_cost += rebalance_cost
            for ticker, ticker_cost in rebalance_costs_by_ticker.items():
                profit_by_ticker[ticker] -= ticker_cost

        curve_values.append(sum(asset_values.values()) + cash_value)
        # 매일 최종 자산 상태 기준 비중 기록
        record_current_weights()
        if date in friday_history_dates:
            append_weight_history(date, asset_values, cash_value)

    # 비중 이력 리스트의 최소/최대비중을 각 종목별로 집계합니다.
    min_weights = {}
    max_weights = {}
    for t in weights_history_by_ticker:
        lst = weights_history_by_ticker[t]
        if lst:
            min_weights[t] = min(lst)
            max_weights[t] = max(lst)
        else:
            min_weights[t] = 0.0
            max_weights[t] = 0.0

    curve = np.asarray(curve_values, dtype=np.float64)
    summary = curve_metrics(initial_capital_krw, curve)

    bench_series = pd.to_numeric(sim[benchmark["ticker"]], errors="coerce").dropna()
    bench_start = float(bench_series.iloc[0])
    bench_curve = (bench_series / bench_start * initial_capital_krw).to_numpy(dtype=np.float64)
    benchmark_summary = curve_metrics(initial_capital_krw, bench_curve)

    positions: list[dict[str, Any]] = []
    for item in clean_tickers:
        ticker = item["ticker"]
        series = pd.to_numeric(sim[ticker], errors="coerce").dropna()
        if len(series) < 2:
            continue
        start_price = float(series.iloc[0])
        end_price = float(series.iloc[-1])
        metrics = curve_metrics(start_price, series.to_numpy(dtype=np.float64))
        mdd_peak, mdd_trough, _ = mdd_span(series.to_numpy(dtype=np.float64))
        positions.append(
            {
                "ticker": ticker,
                "name": item.get("name") or ticker,
                "bucket": item.get("bucket"),
                "buy_date": series.index[0].date().isoformat(),
                "late_entry": series.index[0] > start_date,
                "shares": 0,
                "buy_price": round(start_price, 4),
                "last_price": round(end_price, 4),
                "return_pct": round((end_price / start_price - 1.0) * 100.0, 2) if start_price > 0 else 0.0,
                "mdd_pct": round(metrics["mdd_pct"], 2),
                "mdd_start": series.index[mdd_peak].date().isoformat(),
                "mdd_end": series.index[mdd_trough].date().isoformat(),
                "sortino": round(metrics["sortino"], 2),
                "profit": round(float(profit_by_ticker.get(ticker, 0.0)), 0),
                "value": round(float(asset_values.get(ticker, 0.0)), 0),
                "min_weight": round(min_weights.get(ticker, 0.0), 1),
                "max_weight": round(max_weights.get(ticker, 0.0), 1),
            }
        )

    return {
        "months": months,
        "rebalance": rebalance,
        "buy_date": start_date.date().isoformat(),
        "end_date": sim.index[-1].date().isoformat(),
        "has_late_entry": any(position["late_entry"] for position in positions),
        "initial_capital": round(initial_capital_krw, 0),
        "final_value": round(float(curve[-1]), 0),
        "slippage": {
            "total_cost": round(total_slippage_cost, 0),
            "total_cost_pct": round(total_slippage_cost / initial_capital_krw * 100.0, 2),
        },
        "summary": {key: round(value, 2) for key, value in summary.items()},
        "benchmark": {
            **benchmark,
            "summary": {key: round(value, 2) for key, value in benchmark_summary.items()},
        },
        "positions": positions,
        "cash_min_weight": round(min_weights.get("__CASH__", 0.0), 1),
        "cash_max_weight": round(max_weights.get("__CASH__", 0.0), 1),
        "chart": {
            "dates": [date.date().isoformat() for date in sim.index],
            "portfolio_value": [round(float(value), 0) for value in curve],
            "benchmark_value": [round(float(value), 0) for value in bench_curve],
            "portfolio_pct": [round((value / initial_capital_krw - 1.0) * 100.0, 3) for value in curve],
            "benchmark_pct": [round((value / initial_capital_krw - 1.0) * 100.0, 3) for value in bench_curve],
        },
        "weight_history": weight_history,
        "weight_items": [
            {
                "key": item["ticker"],
                "label": item.get("name") or item["ticker"],
                "bucket": item.get("bucket"),
            }
            for item in clean_tickers
        ]
        + [{"key": "__CASH__", "label": "현금"}],
        "missing_tickers": missing,
        "excluded_fixed_tickers": excluded_fixed_tickers,
    }
