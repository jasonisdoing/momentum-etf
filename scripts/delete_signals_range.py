"""
Delete signals documents for a given country and date range (inclusive).

Usage examples:
  # Delete KOR signals from 2025-09-01 through 2025-09-12 (inclusive)
  python scripts/delete_signals_range.py --country kor --start 2025-09-01 --end 2025-09-12

  # Dry-run (show how many would be deleted)
  python scripts/delete_signals_range.py --country kor --start 2025-09-01 --end 2025-09-12 --dry-run
"""

import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.db_manager import get_db_connection
from utils.env import load_env_if_present


def parse_args():
    p = argparse.ArgumentParser(description="Delete signals in a date range (inclusive)")
    p.add_argument("--country", type=str, required=True, help="Country code, e.g., kor/aus")
    p.add_argument("--start", type=str, required=True, help="Start date YYYY-MM-DD (inclusive)")
    p.add_argument("--end", type=str, required=True, help="End date YYYY-MM-DD (inclusive)")
    p.add_argument("--dry-run", action="store_true", help="Do not delete, only report count")
    return p.parse_args()


def to_day_start(date_str: str) -> datetime:
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def to_day_end(date_str: str) -> datetime:
    # inclusive end: last microsecond of the day
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return dt.replace(hour=23, minute=59, second=59, microsecond=999999)


def main():
    args = parse_args()
    load_env_if_present()
    db = get_db_connection()
    if db is None:
        print("[ERROR] MongoDB connection failed")
        return

    try:
        start_dt = to_day_start(args.start)
        end_dt = to_day_end(args.end)
    except ValueError:
        print("[ERROR] --start/--end must be YYYY-MM-DD")
        return

    query = {
        "country": args.country,
        "date": {"$gte": start_dt, "$lte": end_dt},
    }
    if args.dry_run:
        count = db.signals.count_documents(query)
        print(
            f"[DRY-RUN] Would delete {count} signals for {args.country} from {args.start} to {args.end}"
        )
        return

    res = db.signals.delete_many(query)
    print(
        f"[RESULT] Deleted {res.deleted_count} signals for {args.country} from {args.start} to {args.end}"
    )


if __name__ == "__main__":
    main()
