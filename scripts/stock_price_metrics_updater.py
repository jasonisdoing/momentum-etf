"""
배치 A — 종목 가격 지표(거래량·기간수익률·backtest_stats) 갱신.
OHLCV 캐시만 사용하며 외부 메타 API(네이버 ETF 상세·KIS)는 호출하지 않는다.

Usage:
  python scripts/stock_price_metrics_updater.py
  python scripts/stock_price_metrics_updater.py <ticker_type>
"""

import sys
from pathlib import Path

# Add project root to sys.path to allow imports from utils
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import argparse

from utils.logger import get_app_logger
from utils.stock_meta_updater import update_stock_price_metrics


def main():
    parser = argparse.ArgumentParser(description="Update stock price metrics (batch A).")
    parser.add_argument("target", nargs="?", help="Ticker Pool ID (optional)")
    args = parser.parse_args()

    target = (args.target or "").strip().lower()  # None if not provided
    logger = get_app_logger()

    try:
        if target:
            logger.info(f"[배치 A] Target ticker pool specified: {target}")
            update_stock_price_metrics(target)
        else:
            logger.info("[배치 A] No target specified. Updating all configured ticker pools.")
            update_stock_price_metrics(None)
    except Exception as e:
        logger.error(f"Failed to update stock price metrics: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
