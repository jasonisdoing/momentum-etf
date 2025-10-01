#!/usr/bin/env python
"""Compare regime exit behaviors for a given account."""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from datetime import datetime
from typing import Dict

import pandas as pd

from logic.strategies.momentum.backtest import run_portfolio_backtest
from utils.account_registry import (
    get_country_settings,
    get_strategy_rules_for_country,
    get_common_file_settings,
)
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


def compare_exit_behaviors(country: str, start: str | None, end: str | None) -> None:
    # 국가 설정 가져오기
    country_settings = get_country_settings(country)
    if not country_settings:
        raise SystemExit(f"등록되지 않은 국가입니다: {country}")

    # 공통 설정 가져오기
    common = get_common_file_settings()

    # 전략 규칙 가져오기
    rules = get_strategy_rules_for_country(country)

    # 날짜 설정
    end_dt = pd.to_datetime(_parse_date(end) or datetime.now().strftime("%Y-%m-%d")).normalize()
    months_range = TEST_MONTHS_RANGE if TEST_MONTHS_RANGE else 12
    default_start_dt = (end_dt - pd.DateOffset(months=months_range)).normalize()
    start_dt = pd.to_datetime(
        _parse_date(start) or default_start_dt.strftime("%Y-%m-%d")
    ).normalize()

    start_date = start_dt.strftime("%Y-%m-%d")
    end_date = end_dt.strftime("%Y-%m-%d")

    # ETF 목록 가져오기
    stocks = get_etfs(country)

    print(f"레짐 출구 비교 백테스트 실행 ({country}, {start_date}~{end_date})...")

    # 국가 설정에서 cooldown_days 가져오기
    strategy_settings = country_settings.get("strategy", {})
    cooldown_days = int(strategy_settings.get("COOLDOWN_DAYS", 0))

    # 초기 자본 가져오기
    initial_capital = float(country_settings.get("initial_capital_krw", 100000000))  # 기본값 1억원

    base_kwargs = dict(
        stocks=stocks,
        initial_capital=initial_capital,
        top_n=rules.portfolio_topn,
        ma_period=rules.ma_period,
        replace_threshold=rules.replace_threshold,
        country=country,
        stop_loss_pct=-abs(float(common["HOLDING_STOP_LOSS_PCT"])),
        cooldown_days=cooldown_days,
        regime_filter_enabled=bool(common.get("MARKET_REGIME_FILTER_ENABLED", False)),
        regime_filter_ticker=str(common.get("MARKET_REGIME_FILTER_TICKER", "^GSPC")),
        regime_filter_ma_period=int(common.get("MARKET_REGIME_FILTER_MA_PERIOD", 20)),
        date_range=[start_date, end_date],
    )

    if not base_kwargs["regime_filter_enabled"]:
        print("경고: 공통 설정에서 레짐 필터가 비활성화되어 비교 결과가 동일할 수 있습니다.")

    portfolio_sell_all = run_portfolio_backtest(**base_kwargs, regime_behavior="sell_all")
    equity_sell_all = _aggregate_equity(portfolio_sell_all)

    portfolio_hold_block = run_portfolio_backtest(**base_kwargs, regime_behavior="hold_block_buy")
    equity_hold_block = _aggregate_equity(portfolio_hold_block)

    metrics_sell_all = _calculate_metrics(equity_sell_all)
    metrics_hold_block = _calculate_metrics(equity_hold_block)

    def _format(metrics: Dict[str, float]) -> str:
        return (
            f"기간: {metrics['start']}~{metrics['end']} | "
            f"누적수익률: {metrics['total_return'] * 100:.2f}% | "
            f"CAGR: {metrics['cagr'] * 100:.2f}% | "
            f"MDD: {metrics['mdd'] * 100:.2f}%"
        )

    print("\n=== 레짐 출구 비교 ===")
    print("현재투자 (괴리율 < 0% 익일 전량 매도):")
    print("  " + _format(metrics_sell_all))
    print("보유 유지 (괴리율 < 0% 시 신규 매수 중단, 기존 종목은 추세 이탈까지 보유):")
    print("  " + _format(metrics_hold_block))

    result_df = pd.DataFrame(
        {
            "equity_sell_all": equity_sell_all,
            "equity_hold_block": equity_hold_block,
        }
    )

    output_path = f"regime_exit_compare_{country}_{start_date}_{end_date}.csv"
    result_df.to_csv(output_path, encoding="utf-8-sig")
    print(f"\n상세 결과를 CSV로 저장했습니다: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Regime exit behavior 비교")
    parser.add_argument("--start", help="시작일 (YYYY-MM-DD)")
    parser.add_argument("--end", help="종료일 (YYYY-MM-DD)")
    args = parser.parse_args()
    compare_exit_behaviors(args.country, args.start, args.end)


if __name__ == "__main__":
    main()
