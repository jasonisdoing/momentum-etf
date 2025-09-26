#!/usr/bin/env python
"""Compare binary vs linear regime allocation for a given account."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from datetime import datetime
from typing import Dict

import numpy as np
import pandas as pd

from logic.strategies.momentum.backtest import run_portfolio_backtest
from utils.account_registry import (
    get_account_file_settings,
    get_account_info,
    get_strategy_rules_for_account,
    get_common_file_settings,
)
from utils.data_loader import fetch_ohlcv
from utils.stock_list_io import get_etfs
from test import TEST_MONTHS_RANGE


def _parse_date(value: str | None) -> str | None:
    if not value:
        return None
    return pd.to_datetime(value).strftime("%Y-%m-%d")


def _aggregate_equity(portfolio: Dict[str, pd.DataFrame]) -> pd.Series:
    total_series: pd.Series | None = None
    for df in portfolio.values():
        if df.empty or "pv" not in df.columns:
            continue
        pv = df["pv"].astype(float)
        total_series = pv if total_series is None else total_series.add(pv, fill_value=0.0)
    if total_series is None:
        raise RuntimeError("포트폴리오 데이터를 집계할 수 없습니다.")
    return total_series.sort_index()


def _calculate_metrics(equity: pd.Series) -> Dict[str, float]:
    equity = equity.sort_index()
    returns = equity.pct_change().fillna(0.0)
    cumulative = (1 + returns).cumprod()
    max_cum = cumulative.cummax()
    drawdown = cumulative / max_cum - 1.0

    start_date = equity.index[0]
    end_date = equity.index[-1]
    days = (end_date - start_date).days or 1

    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    cagr = (1.0 + total_return) ** (365.25 / days) - 1.0

    return {
        "start": start_date.strftime("%Y-%m-%d"),
        "end": end_date.strftime("%Y-%m-%d"),
        "total_return": total_return,
        "cagr": cagr,
        "mdd": drawdown.min(),
    }


def _build_exposure(close: pd.Series, ma: pd.Series, threshold_pct: float, mode: str) -> pd.Series:
    ratio = close / ma - 1.0
    ratio = ratio.fillna(-np.inf)

    if mode == "binary":
        threshold = float(threshold_pct or 0.0)
        exposure = (ratio >= threshold).astype(float)
    elif mode == "linear":
        exposure = ratio.copy()
        exposure[ratio <= 0.0] = 0.0
        exposure[ratio >= threshold_pct] = 1.0
        mask = (ratio > 0.0) & (ratio < threshold_pct)
        exposure[mask] = (ratio[mask] / threshold_pct).clip(0.0, 1.0)
    else:
        raise ValueError("mode must be 'binary' or 'linear'")

    return exposure


def _apply_exposure(equity: pd.Series, exposure: pd.Series) -> pd.Series:
    equity = equity.sort_index()
    exposure = exposure.reindex(equity.index).ffill().fillna(0.0)
    returns = equity.pct_change().fillna(0.0)
    adjusted_returns = returns * exposure
    adjusted_equity = equity.iloc[0] * (1 + adjusted_returns).cumprod()
    adjusted_equity.iloc[0] = equity.iloc[0]
    return adjusted_equity


def compare_allocation(account: str, start: str | None, end: str | None) -> None:
    account_settings = get_account_file_settings(account)
    account_info = get_account_info(account)
    if not account_info:
        raise SystemExit(f"등록되지 않은 계좌입니다: {account}")
    country = str(account_info.get("country") or "").strip()
    if not country:
        raise SystemExit(f"'{account}' 계좌에 국가 정보가 없습니다.")

    rules = get_strategy_rules_for_account(account)
    common = get_common_file_settings()

    end_dt = pd.to_datetime(_parse_date(end) or datetime.now().strftime("%Y-%m-%d")).normalize()
    months_range = TEST_MONTHS_RANGE if TEST_MONTHS_RANGE else 12
    default_start_dt = (end_dt - pd.DateOffset(months=months_range)).normalize()
    start_dt = pd.to_datetime(
        _parse_date(start) or default_start_dt.strftime("%Y-%m-%d")
    ).normalize()

    start_date = start_dt.strftime("%Y-%m-%d")
    end_date = end_dt.strftime("%Y-%m-%d")

    stocks = get_etfs(country)

    print(f"포트폴리오 백테스트 실행 ({account}, {start_date}~{end_date})...")
    portfolio = run_portfolio_backtest(
        stocks=stocks,
        initial_capital=account_settings["initial_capital_krw"],
        top_n=rules.portfolio_topn,
        ma_period=rules.ma_period,
        replace_weaker_stock=rules.replace_weaker_stock,
        replace_threshold=rules.replace_threshold,
        country=country,
        stop_loss_pct=-abs(float(common["HOLDING_STOP_LOSS_PCT"])),
        cooldown_days=int(common["COOLDOWN_DAYS"]),
        regime_filter_enabled=False,
        min_buy_score=rules.min_buy_score,
        date_range=[start_date, end_date],
    )

    equity_series = _aggregate_equity(portfolio)

    print("현재 전략(레짐 필터 적용) 백테스트 실행...")
    portfolio_current = run_portfolio_backtest(
        stocks=stocks,
        initial_capital=account_settings["initial_capital_krw"],
        top_n=rules.portfolio_topn,
        ma_period=rules.ma_period,
        replace_weaker_stock=rules.replace_weaker_stock,
        replace_threshold=rules.replace_threshold,
        country=country,
        stop_loss_pct=-abs(float(common["HOLDING_STOP_LOSS_PCT"])),
        cooldown_days=int(common["COOLDOWN_DAYS"]),
        regime_filter_enabled=bool(common.get("MARKET_REGIME_FILTER_ENABLED", False)),
        regime_filter_ticker=str(common.get("MARKET_REGIME_FILTER_TICKER", "^GSPC")),
        regime_filter_ma_period=int(common.get("MARKET_REGIME_FILTER_MA_PERIOD", 20)),
        min_buy_score=rules.min_buy_score,
        date_range=[start_date, end_date],
    )

    equity_current = _aggregate_equity(portfolio_current)

    print("S&P 500 데이터 로딩 및 20일 이평 계산...")
    sp500 = fetch_ohlcv("^GSPC", country="kor", date_range=[start_date, end_date])
    if sp500 is None or sp500.empty:
        raise RuntimeError("S&P 500 데이터를 가져올 수 없습니다.")
    sp500 = sp500.sort_index()
    sp500_ma = sp500["Close"].rolling(window=20).mean()

    scenario_specs: list[tuple[str, str, float | None]] = [
        ("완전투자", "full", None),
        ("1% 아래로 떨어졌을때", "gap_minus_1_00", -0.01),
        ("0.5% 아래로 떨어졌을때", "gap_minus_0_50", -0.005),
        ("현재투자", "current_actual", None),  # 실제 전략 값 별도 처리
        ("-0.5% 아래로 떨어졌을때", "gap_plus_0_50", 0.005),
        ("-1% 아래로 떨어졌을때", "gap_plus_1_00", 0.01),
        ("-1.5% 아래로 떨어졌을때", "gap_plus_1_50", 0.015),
        ("-2% 아래로 떨어졌을때", "gap_plus_2_00", 0.02),
    ]

    results: list[tuple[str, Dict[str, float]]] = []
    result_df = pd.DataFrame(index=equity_series.index)
    result_df["equity_full"] = equity_series

    for label, key, threshold in scenario_specs:
        if key == "full":
            description = "레짐 미적용, 항상 100%"
            metrics = _calculate_metrics(equity_series)
            results.append((f"{label} ({description})", metrics))
            continue

        if key == "current_actual":
            description = "실제 전략, 괴리율 ≥ 0% 시 100%, 미만 시 전량 현금"
            result_df["equity_current_actual"] = equity_current.reindex(result_df.index)
            metrics = _calculate_metrics(equity_current)
            results.append((f"{label} ({description})", metrics))
            continue

        threshold_value = float(threshold)
        exposure = _build_exposure(sp500["Close"], sp500_ma, threshold_value, mode="binary")
        equity_adjusted = _apply_exposure(equity_series, exposure)
        result_df[f"equity_{key}"] = equity_adjusted.reindex(result_df.index)
        result_df[f"exposure_{key}"] = exposure.reindex(result_df.index)

        description = (
            f"괴리율이 {threshold_value * 100:+.2f}% 미만이면 전량 현금"
            if threshold_value >= 0
            else f"괴리율이 {threshold_value * 100:+.2f}% 아래까지 보유"
        )
        metrics = _calculate_metrics(equity_adjusted)
        results.append((f"{label} ({description})", metrics))

    def _format(metrics: Dict[str, float]) -> str:
        return (
            f"기간: {metrics['start']}~{metrics['end']} | "
            f"누적수익률: {metrics['total_return'] * 100:.2f}% | "
            f"CAGR: {metrics['cagr'] * 100:.2f}% | "
            f"MDD: {metrics['mdd'] * 100:.2f}%"
        )

    print("\n=== 결과 비교 ===")
    for label, metrics in results:
        print(label + ":")
        print("  " + _format(metrics))

    output_path = f"regime_allocation_compare_{account}_{start_date}_{end_date}.csv"
    result_df.to_csv(output_path, encoding="utf-8-sig")
    print(f"\n상세 결과를 CSV로 저장했습니다: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Regime allocation 비교 백테스트")
    parser.add_argument("account", default="m1", nargs="?", help="계좌 코드 (기본: m1)")
    parser.add_argument("--start", help="시작일 (YYYY-MM-DD)")
    parser.add_argument("--end", help="종료일 (YYYY-MM-DD)")
    args = parser.parse_args()

    compare_allocation(args.account, args.start, args.end)


if __name__ == "__main__":
    main()
