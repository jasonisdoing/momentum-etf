#!/usr/bin/env python
"""daily_snapshots 루트 TOTAL 금액을 명시적으로 복구한다."""

from __future__ import annotations

import argparse
import datetime
import os
import sys
from zoneinfo import ZoneInfo


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.db_manager import get_db_connection


KST = ZoneInfo("Asia/Seoul")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="daily_snapshots의 TOTAL 금액을 날짜 기준으로 복구합니다.")
    parser.add_argument("--date", required=True, help="복구할 snapshot_date (YYYY-MM-DD)")
    parser.add_argument("--total-assets", required=True, type=int, help="복구할 총 자산(KRW)")
    parser.add_argument("--total-principal", type=int, help="복구할 투자 원금(KRW)")
    parser.add_argument("--cash-balance", type=int, help="복구할 현금 잔고(KRW)")
    parser.add_argument("--valuation-krw", type=int, help="복구할 평가액(KRW)")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    db = get_db_connection()
    if db is None:
        raise RuntimeError("DB 연결 실패")

    doc = db.daily_snapshots.find_one({"snapshot_date": args.date})
    if not doc:
        raise RuntimeError(f"snapshot_date={args.date} 문서를 찾지 못했습니다.")

    update_doc: dict[str, object] = {
        "total_assets": int(args.total_assets),
        "updated_at": datetime.datetime.now(KST),
    }
    if args.total_principal is not None:
        update_doc["total_principal"] = int(args.total_principal)
    if args.cash_balance is not None:
        update_doc["cash_balance"] = int(args.cash_balance)
    if args.valuation_krw is not None:
        update_doc["valuation_krw"] = int(args.valuation_krw)

    db.daily_snapshots.update_one({"_id": doc["_id"]}, {"$set": update_doc})

    print(
        {
            "snapshot_date": args.date,
            "total_assets_before": doc.get("total_assets"),
            "total_assets_after": int(args.total_assets),
            "updated_fields": sorted(update_doc.keys()),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
