#!/usr/bin/env python3
"""Quick checker for market regime status (moving-average comparison).

This script helps diagnose discrepancies between local and server regime
calculations by explicitly fetching price data, computing the moving average,
and printing the derived metrics.
"""

from __future__ import annotations

import argparse
import sys
from typing import Any, Dict, Optional, Tuple

import pandas as pd

sys.path.append(".")

from logic.recommend.market import _compute_market_regime_status  # type: ignore
from utils.account_registry import load_account_configs, pick_default_account
from utils.settings_loader import get_market_regime_settings


def _find_account(account_id: Optional[str]) -> Dict[str, Any]:
    accounts = load_account_configs()
    if not accounts:
        raise RuntimeError("계정 설정을 찾을 수 없습니다. data/settings/account/*.json을 확인하세요.")

    if account_id:
        account_id = account_id.strip().lower()
        for account in accounts:
            if account["account_id"] == account_id:
                return account
        raise RuntimeError(f"지정한 account_id '{account_id}' 에 대한 설정을 찾을 수 없습니다.")

    return pick_default_account(accounts)


def _resolve_ratio(account: Dict[str, Any], fallback: Optional[int]) -> Optional[int]:
    settings = account.get("settings") or {}
    strategy_cfg = settings.get("strategy") or {}
    static_cfg = strategy_cfg.get("static") or {}
    tuning_cfg = strategy_cfg.get("tuning") or {}

    for candidate in (
        static_cfg.get("MARKET_REGIME_RISK_OFF_EQUITY_RATIO"),
        tuning_cfg.get("MARKET_REGIME_RISK_OFF_EQUITY_RATIO"),
        strategy_cfg.get("MARKET_REGIME_RISK_OFF_EQUITY_RATIO"),
    ):
        try:
            if candidate is not None:
                value = int(candidate)
                if 0 <= value <= 100:
                    return value
        except (TypeError, ValueError):
            continue

    return fallback


def compute_status(ticker: str, country: str, ma_period: int, delay_days: int) -> Tuple[Optional[Dict[str, Any]], str]:
    info, message = _compute_market_regime_status(
        ticker,
        ma_period=ma_period,
        country=country,
        delay_days=delay_days,
    )
    return info, message


def main() -> None:
    parser = argparse.ArgumentParser(description="시장 레짐 상태 계산기")
    parser.add_argument("--account", help="기본 설정으로 사용할 account id (없으면 기본 계정 사용)")
    parser.add_argument("--ticker", help="시장 레짐 계산에 사용할 티커 (기본값은 공통 설정)")
    parser.add_argument("--country", help="데이터 소스 국가 코드 (예: us, kor). 기본값은 공통 설정")
    parser.add_argument(
        "--ma-period",
        type=int,
        help="이동평균 기간 (1-200). 기본값은 공통 설정의 기간",
    )
    parser.add_argument(
        "--delay-days",
        type=int,
        help="시장 레짐 지연 일수. 지정하지 않으면 공통 설정값 사용",
    )

    args = parser.parse_args()

    ticker_default, ma_default, country_default, delay_default, ratio_default = get_market_regime_settings()

    account = _find_account(args.account)
    ticker = args.ticker or ticker_default
    country = (args.country or country_default or "").strip().lower() or "us"
    ma_period = int(args.ma_period or ma_default)
    delay_days = int(args.delay_days if args.delay_days is not None else delay_default or 0)

    if ma_period <= 0:
        raise ValueError("MA 기간은 1 이상이어야 합니다.")

    risk_off_ratio = _resolve_ratio(account, ratio_default)

    print("=== Regime Check ===")
    print(f"Account:      {account['account_id']}")
    print(f"Ticker:       {ticker}")
    print(f"Country:      {country}")
    print(f"MA period:    {ma_period}")
    print(f"Risk-off %:   {risk_off_ratio if risk_off_ratio is not None else 'N/A'}")
    print(f"Delay days:   {delay_days}")
    print()

    info, message = compute_status(ticker, country, ma_period, delay_days)

    if info is None:
        print("Regime info could not be computed.")
        print("Message:", message)
        return

    def _fmt_date(value: Any) -> str:
        if value is None:
            return "-"
        if hasattr(value, "strftime"):
            try:
                return value.strftime("%Y-%m-%d")
            except Exception:
                return str(value)
        return str(value)

    prox_pct = float(info.get("proximity_pct", 0.0))
    status_label = info.get("status_label", "-")
    last_start = _fmt_date(info.get("last_risk_off_start"))
    last_end = _fmt_date(info.get("last_risk_off_end"))

    print("Status label: ", status_label)
    print("Above MA?:    ", "Yes" if prox_pct >= 0 else "No")
    print("Divergence %: ", f"{prox_pct:+.3f}%")
    print("Last risk-off:", f"{last_start} -> {last_end}")
    print("Raw message:  ", message)

    if info.get("ticker") == ticker:
        print("\nCalculated fields:")
        data_summary = {
            "ticker": info.get("ticker"),
            "ma_period": info.get("ma_period"),
            "country": info.get("country"),
            "status": info.get("status"),
        }
        print(pd.Series(data_summary).to_string())


if __name__ == "__main__":
    main()
