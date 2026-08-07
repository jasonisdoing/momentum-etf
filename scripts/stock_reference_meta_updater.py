"""
배치 B — 종목 식별·상세 메타 갱신.
이름/상장일/마켓/업종 + ETF 상세 캐시(holdings·배당·ETFBase) + KIS 국내 ETF 마스터.

Usage:
  python scripts/stock_reference_meta_updater.py
  python scripts/stock_reference_meta_updater.py <ticker_type>
"""

import sys
from pathlib import Path

# Add project root to sys.path to allow imports from utils
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

import argparse

from utils.logger import get_app_logger
from utils.stock_meta_updater import update_stock_reference_metadata


def main():
    parser = argparse.ArgumentParser(description="Update stock reference metadata (batch B).")
    parser.add_argument("target", nargs="?", help="Ticker Pool ID (optional)")
    args = parser.parse_args()

    target = (args.target or "").strip().lower()  # None if not provided
    logger = get_app_logger()

    try:
        if target:
            logger.info(f"[배치 B] Target ticker pool specified: {target}")
            update_stock_reference_metadata(target)
        else:
            logger.info("[배치 B] No target specified. Updating all configured ticker pools.")
            update_stock_reference_metadata(None)
    except Exception as e:
        logger.error(f"Failed to update stock reference metadata: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
