#!/usr/bin/env python
"""종목군(account_id token) 기준 RANK 순위 추출 스크립트."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import pandas as pd

# 프로젝트 루트를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TRADING_DAYS_PER_MONTH
from core.strategy.metrics import process_ticker_data
from utils.data_loader import get_latest_trading_day, prepare_price_data
from utils.env import load_env_if_present
from utils.stock_list_io import get_etfs


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RANK 종목군 순위 추출")
    parser.add_argument("pool_id", help="종목군 ID (예: kor_kr, kor_us, tax, aus)")
    parser.add_argument("--country", required=True, help="국가 코드 (kor/us/au)")
    parser.add_argument("--months", type=int, default=12, help="RANK MA 개월 수")
    parser.add_argument("--ma-type", default="HMA", help="이동평균 타입 (SMA/EMA/WMA/DEMA/TEMA/HMA/ALMA)")
    parser.add_argument("--top", type=int, default=20, help="출력 개수")
    parser.add_argument("--allow-remote-fetch", action="store_true", help="캐시 누락 시 원격 조회 허용")
    return parser


def main() -> int:
    load_env_if_present()
    args = _build_parser().parse_args()

    pool_id = args.pool_id.strip().lower()
    country = args.country.strip().lower()
    ma_days = int(args.months) * TRADING_DAYS_PER_MONTH

    universe = get_etfs(pool_id)
    if not universe:
        print(f"[오류] pool_id='{pool_id}' 종목이 비어있습니다.")
        return 1

    tickers = sorted({str(item.get("ticker") or "").strip().upper() for item in universe if item.get("ticker")})
    end_date = get_latest_trading_day(country)
    if not isinstance(end_date, pd.Timestamp):
        end_date = pd.Timestamp.now().normalize()
    start_date = (end_date - pd.DateOffset(days=max(ma_days * 3, 360))).strftime("%Y-%m-%d")

    prices_map, missing = prepare_price_data(
        tickers=tickers,
        country=country,
        start_date=start_date,
        end_date=end_date.strftime("%Y-%m-%d"),
        warmup_days=0,
        account_id=pool_id,
        allow_remote_fetch=bool(args.allow_remote_fetch),
    )

    if missing:
        print(f"[경고] 가격 누락 {len(missing)}개: {', '.join(sorted(missing))}")

    name_map = {str(item.get("ticker") or "").strip().upper(): str(item.get("name") or "") for item in universe}
    rows: list[dict[str, object]] = []

    for ticker in tickers:
        df = prices_map.get(ticker)
        if df is None or df.empty:
            continue

        metrics = process_ticker_data(
            ticker=ticker,
            df=df,
            ma_days=ma_days,
            ma_type=args.ma_type,
            enable_data_sufficiency_check=False,
        )
        if not metrics:
            continue

        score_series = metrics.get("ma_score")
        close_series = metrics.get("close")
        ma_series = metrics.get("ma")
        if score_series is None or close_series is None or ma_series is None:
            continue

        try:
            score = float(score_series.iloc[-1])
            close = float(close_series.iloc[-1])
            ma_val = float(ma_series.iloc[-1])
        except Exception:
            continue

        rows.append(
            {
                "ticker": ticker,
                "name": name_map.get(ticker, ""),
                "score": score,
                "close": close,
                "ma": ma_val,
            }
        )

    if not rows:
        print("[오류] 점수 계산 가능한 종목이 없습니다.")
        return 1

    rows.sort(key=lambda x: float(x["score"]), reverse=True)
    top_n = max(1, int(args.top))

    print(
        f"\nRANK | pool={pool_id} country={country} ma={args.ma_type.upper()}({args.months}m) "
        f"base={end_date.strftime('%Y-%m-%d')} generated={datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    print("-" * 100)
    for idx, row in enumerate(rows[:top_n], 1):
        print(
            f"{idx:>2}. {row['ticker']:<10} {str(row['name'])[:36]:<36} "
            f"score={float(row['score']):>7.2f}% close={float(row['close']):>10.2f} ma={float(row['ma']):>10.2f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
