"""stock_cache_meta_history → previous_stock_cache_meta 이관 (1회성).

날짜별 스냅샷을 전부 쌓던 구조를 **티커당 직전 1건**만 두는 구조로 바꿨다.
이 스크립트는 기존 히스토리에서 티커별 최신 1건을 옮기고 옛 컬렉션을 지운다.

'최신 1건'은 현재 `stock_cache_meta` 와 같은 날짜일 수 있다. 그 경우 비교 기준이
자기 자신이 되므로, **현재 스냅샷 날짜와 다른 것 중 가장 최근**을 고른다.
백업하지 않는다 — 배치가 매일 다시 채우는 파생 데이터다.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from utils.db_manager import get_db_connection  # noqa: E402
from utils.env import load_env_if_present  # noqa: E402

HISTORY = "stock_cache_meta_history"
PREVIOUS = "previous_stock_cache_meta"


def main() -> int:
    load_env_if_present()
    db = get_db_connection()
    if db is None:
        raise RuntimeError("DB 연결에 실패했습니다.")

    if HISTORY not in db.list_collection_names():
        print(f"[migrate] {HISTORY} 가 없습니다. 이관할 것이 없습니다.")
        return 0

    # 현재 메타의 귀속 날짜 — 이 날짜와 같은 히스토리는 '직전'이 아니다.
    current_dates = {
        (d.get("ticker_type"), d.get("ticker")): str(d.get("snapshot_date") or "")
        for d in db["stock_cache_meta"].find({}, {"_id": 0, "ticker_type": 1, "ticker": 1, "snapshot_date": 1})
    }

    latest: dict[tuple[str, str], dict] = {}
    for doc in db[HISTORY].find({}, {"_id": 0}).sort("date", 1):
        key = (doc.get("ticker_type"), doc.get("ticker"))
        if str(doc.get("date") or "") == current_dates.get(key, ""):
            continue
        latest[key] = doc  # 오름차순이라 마지막이 최신

    if latest:
        db[PREVIOUS].delete_many({})
        db[PREVIOUS].insert_many(list(latest.values()))
        db[PREVIOUS].create_index(
            [("ticker_type", 1), ("ticker", 1)], unique=True, name="ticker_previous_unique"
        )
    print(f"[migrate] 직전값 이관: {len(latest)}건")

    dropped = db[HISTORY].estimated_document_count()
    db[HISTORY].drop()
    print(f"[migrate] {HISTORY} 삭제: {dropped:,}건")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
