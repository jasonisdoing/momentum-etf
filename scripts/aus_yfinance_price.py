"""호주 ETF 전체의 yfinance 가격/NAV를 확인하는 간단한 스크립트."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

try:
    import yfinance as yf
except ImportError as exc:  # pragma: no cover - 실행 환경에 yfinance 없음
    raise RuntimeError("yfinance 패키지가 필요합니다. `pip install yfinance`로 설치해주세요.") from exc

DEFAULT_TICKER = "ASX:GDX"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
AUS_DATA_PATH = PROJECT_ROOT / "data" / "stocks" / "aus.json"


def _safe_float(value: object) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_yf_symbol(ticker: str) -> str:
    ticker = ticker.strip().upper()
    if ticker.startswith("ASX:"):
        ticker = ticker.split(":", 1)[1]
    if not ticker.endswith(".AX"):
        ticker = f"{ticker}.AX"
    return ticker


def _load_all_aus_tickers() -> List[Tuple[str, str]]:
    if not AUS_DATA_PATH.exists():
        raise FileNotFoundError(f"호주 ETF 목록 파일이 없습니다: {AUS_DATA_PATH}")

    with AUS_DATA_PATH.open("r", encoding="utf-8") as fp:
        raw = json.load(fp)

    tickers: List[Tuple[str, str]] = []
    if isinstance(raw, list):
        for block in raw:
            category = str((block or {}).get("category") or "").strip()
            items = (block or {}).get("tickers") or []
            for item in items:
                ticker = str((item or {}).get("ticker") or "").strip()
                if ticker:
                    tickers.append((ticker.upper(), category))
    # 중복 제거
    unique_pairs = {}
    for ticker, category in tickers:
        unique_pairs.setdefault(ticker, category)
    return sorted(unique_pairs.items())


def fetch_price_nav(ticker: str) -> Tuple[Optional[float], Optional[float], Optional[str]]:
    yf_symbol = _to_yf_symbol(ticker)
    instrument = yf.Ticker(yf_symbol)
    info = instrument.info or {}

    price = _safe_float(info.get("regularMarketPrice"))
    nav = _safe_float(info.get("navPrice"))
    currency = info.get("currency")
    return price, nav, currency


def format_value(value: Optional[float]) -> str:
    if value is None:
        return "-"
    return f"{value:,.4f}"


def iter_targets(single_ticker: Optional[str], include_all: bool) -> Sequence[Tuple[str, str]]:
    if single_ticker:
        return [(single_ticker.upper(), "")]
    if include_all or not single_ticker:
        return _load_all_aus_tickers()
    return [(DEFAULT_TICKER, "")]


def main() -> None:
    parser = argparse.ArgumentParser(description="호주 ETF yfinance 가격/NAV 탐색")
    parser.add_argument(
        "--ticker",
        help="특정 티커만 조회 (예: ASX:IOZ). 지정하지 않으면 --all 동작과 동일합니다.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="호주 ETF 전체를 순회합니다 (기본값).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.2,
        help="요청 사이의 지연 시간(초). 기본 0.2초.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="조회할 티커 수 제한 (디버깅용).",
    )
    args = parser.parse_args()

    try:
        tickers = list(iter_targets(args.ticker, args.all or args.ticker is None))
    except Exception as exc:  # pragma: no cover - 파일 로드 실패
        print(f"[ERROR] 티커 목록 로드 실패: {exc}", file=sys.stderr)
        sys.exit(1)

    if args.limit is not None:
        tickers = tickers[: args.limit]

    print(f"# 조회 대상: {len(tickers)}개 티커")
    print("티커,통화,가격,Nav,category")

    for idx, (ticker, category) in enumerate(tickers, start=1):
        try:
            price, nav, currency = fetch_price_nav(ticker)
        except Exception as exc:  # pragma: no cover - yfinance 예외 방어
            print(f"{ticker},ERROR,{exc},-")
            continue

        price_str = format_value(price)
        nav_str = format_value(nav)
        currency_str = currency or "-"
        category_str = category or "-"
        print(f"{ticker},{currency_str},{price_str},{nav_str},{category_str}")

        if args.delay and idx < len(tickers):
            time.sleep(args.delay)


if __name__ == "__main__":
    main()
