#!/usr/bin/env python3
"""매수 시점별 체결 가격 찾는 스크립트"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Project imports (lazy to avoid heavy modules when unused)
from logic import momentum as strategy_module
from utils.account_registry import (
    get_account_file_settings,
    get_common_file_settings,
    get_strategy_rules_for_account,
)
from utils.data_loader import get_latest_trading_day
from utils.stock_list_io import get_etfs as get_etfs_from_files

# Decision tags used in portfolio output
BUY_DECISIONS = {"BUY", "BUY_REPLACE"}
SELL_DECISIONS = {"SELL_TREND", "SELL_REPLACE", "SELL_REGIME_FILTER", "CUT_STOPLOSS"}

INTERVAL_SECONDS = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "10m": 600,
    "30m": 1800,
    "1h": 3600,
    "6h": 21600,
    "12h": 43200,
    "24h": 86400,
}


@dataclass(frozen=True)
class TradeEvent:
    ticker: str
    date: datetime
    close_price: float
    notional: float  # KRW size at close


def parse_time_list(times: Optional[str], step_minutes: int) -> List[str]:
    if times:
        parsed = []
        for fragment in times.split(","):
            fragment = fragment.strip()
            if not fragment:
                continue
            try:
                datetime.strptime(fragment, "%H:%M")
            except ValueError as exc:  # noqa: PERF203
                raise SystemExit(f"잘못된 시간 형식입니다: {fragment} (HH:MM)") from exc
            parsed.append(fragment)
        if not parsed:
            raise SystemExit("평가할 시간이 비어 있습니다.")
        return parsed
    if step_minutes <= 0:
        raise SystemExit("step_minutes 값은 0보다 커야 합니다.")
    times_out: List[str] = []
    for minutes in range(0, 24 * 60, step_minutes):
        hour = minutes // 60
        minute = minutes % 60
        times_out.append(f"{hour:02d}:{minute:02d}")
    if "24:00" in times_out:
        times_out.remove("24:00")
    return times_out


def fetch_coin_candles(
    ticker: str,
    interval: str,
    start_dt: datetime,
    end_dt: datetime,
    *,
    max_count: int,
    session: Optional[requests.Session] = None,
) -> pd.DataFrame:
    if interval not in INTERVAL_SECONDS:
        raise SystemExit(f"지원하지 않는 interval 입니다: {interval}")
    seconds = INTERVAL_SECONDS[interval]
    span_seconds = max((end_dt - start_dt).total_seconds(), seconds)
    count_needed = int(math.ceil(span_seconds / seconds)) + 10
    count = max(200, min(count_needed, max_count))
    url = f"https://api.bithumb.com/public/candlestick/{ticker.upper()}_KRW/{interval}"
    params = {"count": count}
    sess = session or requests.Session()
    try:
        resp = sess.get(url, params=params, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as exc:  # noqa: PERF203
        raise SystemExit(f"{ticker} {interval} 캔들 조회 실패: {exc}") from exc
    payload = resp.json() or {}
    data = payload.get("data") or []
    if not data:
        raise SystemExit(f"{ticker} {interval} 데이터가 비어 있습니다.")
    rows: List[Tuple[datetime, float, float, float, float, float]] = []
    for arr in data:
        try:
            ts = int(arr[0])
            o = float(arr[1])
            c = float(arr[2])
            h = float(arr[3])
            low = float(arr[4])
            v = float(arr[5])
            dt = datetime.fromtimestamp(ts / 1000.0)
        except Exception:
            continue
        rows.append((dt, o, h, low, c, v))
    if not rows:
        raise SystemExit(f"{ticker} {interval} 데이터 파싱 실패")
    df = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
    df.set_index("Date", inplace=True)
    df.sort_index(inplace=True)
    window = df[(df.index >= start_dt) & (df.index <= end_dt)]
    if window.empty:
        earliest = df.index.min()
        latest = df.index.max()
        raise SystemExit(
            f"{ticker} {interval} 데이터에 목표 구간이 없습니다."
            f" (가장 오래된 데이터: {earliest}, 최신: {latest})."
            " count 값을 늘려서 다시 시도하세요."
        )
    return window


def price_at(candles: pd.DataFrame, dt: datetime, interval: str) -> Optional[float]:
    if candles.empty:
        return None
    seconds = INTERVAL_SECONDS[interval]
    idx = candles.index.searchsorted(dt, side="right") - 1
    if idx < 0:
        return None
    ts = candles.index[idx]
    if (dt - ts).total_seconds() > seconds * 1.5:
        return None
    return float(candles.iloc[idx]["Close"])


def collect_trade_events(
    portfolio: Dict[str, pd.DataFrame]
) -> Tuple[List[TradeEvent], List[TradeEvent]]:
    buys: List[TradeEvent] = []
    sells: List[TradeEvent] = []
    for ticker, df in portfolio.items():
        if ticker == "CASH" or df is None or df.empty:
            continue
        df_sorted = df.sort_index()
        for dt, row in df_sorted.iterrows():
            decision = str(row.get("decision") or "").upper()
            trade_amount = float(row.get("trade_amount") or 0.0)
            price = float(row.get("price") or 0.0)
            if decision in BUY_DECISIONS and trade_amount > 0 and price > 0:
                buys.append(TradeEvent(ticker, dt, price, trade_amount))
            elif decision in SELL_DECISIONS and trade_amount > 0 and price > 0:
                sells.append(TradeEvent(ticker, dt, price, trade_amount))
    return buys, sells


def summarise_slippage(
    events: Sequence[TradeEvent],
    candles_by_ticker: Dict[str, pd.DataFrame],
    times: Sequence[str],
    *,
    interval: str,
) -> pd.DataFrame:
    rows = []
    if not events:
        return pd.DataFrame(
            columns=[
                "time",
                "trades",
                "missing",
                "weighted_pct",
                "mean_pct",
                "median_pct",
                "total_krw",
                "avg_krw",
            ]
        )
    for label in times:
        hh_mm = datetime.strptime(label, "%H:%M").time()
        weighted_sum = 0.0
        total_weight = 0.0
        pct_values: List[float] = []
        delta_values: List[float] = []
        count = 0
        missing = 0
        for ev in events:
            base_date = ev.date.date()
            target_dt = datetime.combine(base_date, hh_mm)
            candles = candles_by_ticker.get(ev.ticker)
            if candles is None:
                missing += 1
                continue
            px = price_at(candles, target_dt, interval)
            if px is None:
                missing += 1
                continue
            slippage = (px / ev.close_price) - 1.0
            pct_values.append(slippage * 100.0)
            delta = ev.notional * slippage
            delta_values.append(delta)
            weighted_sum += slippage * ev.notional
            total_weight += ev.notional
            count += 1
        weighted_pct = (weighted_sum / total_weight) * 100.0 if total_weight else np.nan
        mean_pct = float(np.mean(pct_values)) if pct_values else np.nan
        median_pct = float(np.median(pct_values)) if pct_values else np.nan
        total_delta = float(np.sum(delta_values)) if delta_values else 0.0
        avg_delta = float(np.mean(delta_values)) if delta_values else 0.0
        rows.append(
            {
                "time": label,
                "trades": count,
                "missing": missing,
                "weighted_pct": weighted_pct,
                "mean_pct": mean_pct,
                "median_pct": median_pct,
                "total_krw": total_delta,
                "avg_krw": avg_delta,
            }
        )
    df = pd.DataFrame(rows)
    df.sort_values(by="weighted_pct", inplace=True)
    return df


def load_portfolio_results(
    account: str,
    country: str,
    start_date: datetime,
    end_date: datetime,
) -> Dict[str, pd.DataFrame]:
    strategy_rules = get_strategy_rules_for_account(account)
    account_settings = get_account_file_settings(account)
    common = get_common_file_settings()
    stocks = get_etfs_from_files(country)
    date_range = [start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")]
    portfolio = strategy_module.run_portfolio_backtest(
        stocks=stocks,
        initial_capital=account_settings["initial_capital_krw"],
        core_start_date=start_date,
        top_n=strategy_rules.portfolio_topn,
        date_range=date_range,
        country=country,
        ma_period=strategy_rules.ma_period,
        replace_threshold=strategy_rules.replace_threshold,
        regime_filter_enabled=common["MARKET_REGIME_FILTER_ENABLED"],
        regime_filter_ticker=common["MARKET_REGIME_FILTER_TICKER"],
        regime_filter_ma_period=int(common["MARKET_REGIME_FILTER_MA_PERIOD"]),
        stop_loss_pct=-abs(float(common["HOLDING_STOP_LOSS_PCT"])),
        cooldown_days=account_settings.get("cooldown_days", 0),
        min_buy_score=strategy_rules.min_buy_score,
    )
    if not portfolio:
        raise SystemExit("백테스트 결과가 비어 있습니다.")
    return portfolio


def determine_date_range(
    start: Optional[str],
    end: Optional[str],
    months: int,
    country: str,
) -> Tuple[datetime, datetime]:
    if start and end:
        start_dt = datetime.strptime(start, "%Y-%m-%d")
        end_dt = datetime.strptime(end, "%Y-%m-%d")
        return start_dt, end_dt
    latest = get_latest_trading_day(country)
    end_dt = latest.replace(hour=0, minute=0, second=0, microsecond=0)
    start_dt = (end_dt - pd.DateOffset(months=months)).to_pydatetime()
    return start_dt, end_dt


def ensure_pandas_date(dt: datetime) -> datetime:
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="코인 매수/매도 시점 민감도 분석")
    parser.add_argument("--account", default="b1", help="분석할 계좌 코드")
    parser.add_argument("--country", default="coin", help="국가 코드 (기본 coin)")
    parser.add_argument("--start", help="시작일 (YYYY-MM-DD)")
    parser.add_argument("--end", help="종료일 (YYYY-MM-DD)")
    parser.add_argument("--months", type=int, default=12, help="시작/종료일 미지정 시 최근 개월 수")
    parser.add_argument(
        "--interval", default="1h", choices=sorted(INTERVAL_SECONDS.keys()), help="캔들 간격"
    )
    parser.add_argument("--times", help="콤마로 구분한 시각 목록 (HH:MM). 미지정 시 step 분 간격")
    parser.add_argument("--step-minutes", type=int, default=60, help="기본 시간 리스트 생성 간격 (분)")
    parser.add_argument("--max-count", type=int, default=10000, help="API 호출 시 가져올 최대 캔들 수")
    parser.add_argument("--skip-sell", action="store_true", help="매도 시점 분석 생략")
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    country = args.country or "coin"
    if country != "coin":
        raise SystemExit("현재 스크립트는 coin 계좌 전용입니다.")

    start_dt, end_dt = determine_date_range(args.start, args.end, args.months, country)
    start_dt = ensure_pandas_date(start_dt)
    end_dt = ensure_pandas_date(end_dt)
    times = parse_time_list(args.times, args.step_minutes)

    portfolio = load_portfolio_results(args.account, country, start_dt, end_dt)
    buy_events, sell_events = collect_trade_events(portfolio)
    if not buy_events:
        raise SystemExit("매수 체결 이벤트가 없습니다.")

    min_day = min(ev.date for ev in buy_events + sell_events) - timedelta(days=1)
    max_day = max(ev.date for ev in buy_events + sell_events) + timedelta(days=1)

    candles_by_ticker: Dict[str, pd.DataFrame] = {}
    session = requests.Session()
    for ticker in sorted({ev.ticker for ev in buy_events + sell_events}):
        candles_by_ticker[ticker] = fetch_coin_candles(
            ticker,
            args.interval,
            min_day,
            max_day,
            max_count=args.max_count,
            session=session,
        )

    buy_summary = summarise_slippage(buy_events, candles_by_ticker, times, interval=args.interval)
    print("\n==== 매수 시점별 체결 가격 편차 (음수일수록 유리) ====")
    if buy_summary.empty:
        print("매수 분석 결과가 없습니다.")
    else:
        print(
            buy_summary[
                [
                    "time",
                    "trades",
                    "missing",
                    "weighted_pct",
                    "mean_pct",
                    "median_pct",
                    "total_krw",
                    "avg_krw",
                ]
            ].to_string(index=False, float_format=lambda v: f"{v:,.4f}")
        )

    if not args.skip_sell and sell_events:
        sell_summary = summarise_slippage(
            sell_events, candles_by_ticker, times, interval=args.interval
        )
        sell_summary.sort_values(by="weighted_pct", ascending=False, inplace=True)
        print("\n==== 매도 시점별 체결 가격 편차 (양수일수록 유리) ====")
        print(
            sell_summary[
                [
                    "time",
                    "trades",
                    "missing",
                    "weighted_pct",
                    "mean_pct",
                    "median_pct",
                    "total_krw",
                    "avg_krw",
                ]
            ].to_string(index=False, float_format=lambda v: f"{v:,.4f}")
        )

    print("\n* weighted_pct: 체결 금액 가중 평균, total_krw: 해당 시각으로 거래 시 총 비용/수익 변화 (KRW)")


if __name__ == "__main__":
    main()
