#!/usr/bin/env python3
"""Remove legacy Australian (AUS / a1) documents from MongoDB collections."""

from __future__ import annotations

import os
import sys
from typing import Dict, Iterable

import argparse

from pymongo import MongoClient
from pymongo.errors import PyMongoError, ServerSelectionTimeoutError
from pymongo.collection import Collection


DEFAULT_DB = "momentum_etf_db"
DEFAULT_URI = "mongodb://localhost:27017"
TARGET_DB = os.getenv("MONGO_DB_NAME", DEFAULT_DB)
MONGO_URI = os.getenv("MONGO_URI", DEFAULT_URI)
ACCOUNT_KEYS: Iterable[str] = (
    "account",
    "account_id",
    "accountId",
    "account_code",
    "metadata.account",
    "metadata.account_id",
    "meta.account",
    "meta.account_id",
)
COUNTRY_KEYS: Iterable[str] = (
    "country",
    "country_code",
    "countryCode",
    "metadata.country",
    "metadata.country_code",
    "meta.country",
    "meta.country_code",
)
TARGET_ACCOUNTS = ("a1", "A1")
TARGET_COUNTRY_REGEX = [{"$regex": pattern, "$options": "i"} for pattern in ("^aus$", "^australia$")]


def build_cleanup_filter() -> Dict:
    """Construct a filter that matches Australian data regardless of schema variations."""
    conditions = []

    for key in ACCOUNT_KEYS:
        conditions.append({key: {"$in": TARGET_ACCOUNTS}})

    for key in COUNTRY_KEYS:
        for regex in TARGET_COUNTRY_REGEX:
            conditions.append({key: regex})

    if not conditions:
        raise RuntimeError("No filter conditions were generated; check configuration.")

    return {"$or": conditions}


def purge_collection(collection: Collection, filter_query: Dict) -> int:
    """Delete matching documents from the given collection and return deleted count."""
    to_delete = collection.count_documents(filter_query)
    if to_delete == 0:
        return 0

    result = collection.delete_many(filter_query)
    return result.deleted_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remove AUS/a1 documents from MongoDB collections.")
    parser.add_argument("--uri", default=MONGO_URI, help="MongoDB connection URI (default: %(default)s)")
    parser.add_argument("--db", default=TARGET_DB, help="Database name (default: %(default)s)")
    parser.add_argument("--dry-run", action="store_true", help="Show counts without deleting documents.")
    parser.add_argument("--timeout", type=float, default=20.0, help="Connection timeout in seconds (default: %(default)s)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        client = MongoClient(
            args.uri,
            serverSelectionTimeoutMS=int(args.timeout * 1000),
        )
        # Force connection check
        client.admin.command("ping")
    except (ServerSelectionTimeoutError, PyMongoError, OSError) as exc:
        print(
            "Failed to connect to MongoDB. Verify the URI/host or run this script where the database is reachable.\n"
            f"  URI: {args.uri}\n"
            f"  Error: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    db = client[args.db]

    filter_query = build_cleanup_filter()

    targets = (
        "stock_recommendations",
        "trades",
    )

    total_removed = 0
    for name in targets:
        collection = db[name]
        if args.dry_run:
            count = collection.count_documents(filter_query)
            print(f"[{name}] matched {count} documents (dry-run, nothing deleted).")
            total_removed += count
        else:
            removed = purge_collection(collection, filter_query)
            print(f"[{name}] deleted {removed} documents.")
            total_removed += removed

    action = "matched" if args.dry_run else "removed"
    print(f"Cleanup complete. Total documents {action}: {total_removed}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - safety net for CLI usage
        print(f"Error while purging AUS data: {exc}", file=sys.stderr)
        sys.exit(1)
