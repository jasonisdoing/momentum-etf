from __future__ import annotations

import argparse
import traceback

from pykrx import stock
# 476030

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="pykrx ETF 구성종목(PDF) 조회를 직접 테스트합니다.",
    )
    parser.add_argument("ticker", help="ETF 티커. 예: 117700")
    parser.add_argument(
        "--date",
        dest="date",
        default=None,
        help="조회 일자 YYYYMMDD. 예: 20250407",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ticker = str(args.ticker).strip()
    date = str(args.date).strip() if args.date else None

    print(f"ticker={ticker}")
    print(f"date={date or '(생략)'}")
    print("-" * 80)

    try:
        df = stock.get_etf_portfolio_deposit_file(ticker, date)
    except Exception as error:
        print(f"호출 실패: {type(error).__name__}: {error}")
        print("-" * 80)
        traceback.print_exc()
        return 1

    print(f"rows={len(df)}")
    print(f"columns={list(df.columns)}")
    print("-" * 80)

    if df is None or df.empty:
        print("빈 DataFrame 입니다.")
        return 0

    print(df.head(20).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
