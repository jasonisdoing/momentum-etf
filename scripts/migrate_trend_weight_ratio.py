"""TREND_WEIGHT_RATIO 풀별 이관 마이그레이션 (1회성).

전역(__global__) SCORE_TREND_WEIGHT_RATIO 를 풀별 pool_settings.TREND_WEIGHT_RATIO 로 옮기고,
backtest_config 각 풀 문서에 탐색 리스트 TREND_WEIGHT_RATIO 를 채운다.

기본은 dry-run(비교확인)이며, 실제 반영은 --apply 를 붙여 실행한다.

    python scripts/migrate_trend_weight_ratio.py            # 현황만 출력
    python scripts/migrate_trend_weight_ratio.py --apply    # 실제 반영

- 풀별 단일값: 전역 문서의 값(없으면 --ratio 필수)을 TREND_WEIGHT_RATIO 가 없는 풀에만 설정.
- 백테스트 탐색 리스트: TREND_WEIGHT_RATIO 가 없는 backtest_config 문서에 --search 값 설정.
- --apply 시 전역(__global__) 문서는 삭제한다(메커니즘 제거됨).
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.env import load_env_if_present

load_env_if_present()

DEFAULT_SEARCH = [50, 60, 70, 80]


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="TREND_WEIGHT_RATIO 풀별 이관")
    parser.add_argument("--apply", action="store_true", help="실제 DB 반영 (없으면 dry-run)")
    parser.add_argument("--ratio", type=int, default=None, help="풀별 단일값 (기본: __global__ 문서 값)")
    parser.add_argument(
        "--search",
        type=str,
        default=",".join(str(v) for v in DEFAULT_SEARCH),
        help="백테스트 탐색 리스트 (기본: 50,60,70,80)",
    )
    args = parser.parse_args(argv[1:])

    from utils.db_manager import get_db_connection

    db = get_db_connection()
    if db is None:
        print("DB 연결 실패", file=sys.stderr)
        return 1

    # 1) 기준 비율 결정: --ratio > __global__ 문서
    global_doc = db["pool_settings"].find_one({"_id": "__global__"}) or {}
    ratio = args.ratio if args.ratio is not None else global_doc.get("SCORE_TREND_WEIGHT_RATIO")
    if ratio is None:
        print("전역(__global__) 값이 없습니다. --ratio 로 명시해주세요.", file=sys.stderr)
        return 2
    ratio = int(ratio)
    if not (0 <= ratio <= 100):
        print(f"ratio 는 0~100 이어야 합니다: {ratio}", file=sys.stderr)
        return 2

    search_values = [int(v) for v in str(args.search).split(",") if str(v).strip()]
    now = datetime.now(timezone.utc)
    print(f"기준 비율(풀별 단일값): {ratio} | 탐색 리스트: {search_values} | apply={args.apply}")

    # 2) pool_settings: TREND_WEIGHT_RATIO 없는 풀에만 설정
    print("\n[pool_settings]")
    for doc in db["pool_settings"].find({}):
        pid = str(doc.get("_id"))
        if pid == "__global__":
            continue
        current = doc.get("TREND_WEIGHT_RATIO")
        if current is not None:
            print(f"  {pid}: 이미 있음 ({current}) — 건너뜀")
            continue
        print(f"  {pid}: 없음 → {ratio} 설정{' (dry-run)' if not args.apply else ''}")
        if args.apply:
            db["pool_settings"].update_one(
                {"_id": pid},
                {"$set": {"TREND_WEIGHT_RATIO": ratio, "updated_at": now, "save_method": "마이그레이션"}},
            )

    # 3) backtest_config: TREND_WEIGHT_RATIO 리스트 없는 문서에 설정
    print("\n[backtest_config]")
    for doc in db["backtest_config"].find({}):
        pid = str(doc.get("_id"))
        current = doc.get("TREND_WEIGHT_RATIO")
        if isinstance(current, list) and current:
            print(f"  {pid}: 이미 있음 ({current}) — 건너뜀")
            continue
        print(f"  {pid}: 없음 → {search_values} 설정{' (dry-run)' if not args.apply else ''}")
        if args.apply:
            db["backtest_config"].update_one(
                {"_id": pid},
                {"$set": {"TREND_WEIGHT_RATIO": search_values, "updated_at": now}},
            )

    # 4) 전역 문서 정리
    if global_doc:
        print(f"\n[__global__] SCORE_TREND_WEIGHT_RATIO={global_doc.get('SCORE_TREND_WEIGHT_RATIO')} → 삭제{' (dry-run)' if not args.apply else ''}")
        if args.apply:
            db["pool_settings"].delete_one({"_id": "__global__"})

    if args.apply:
        from utils.backtest_config_store import invalidate_cache
        from utils.pool_settings_store import invalidate_overlay_cache

        invalidate_overlay_cache()
        invalidate_cache()
        print("\n완료: 캐시 무효화까지 반영했습니다.")
    else:
        print("\ndry-run 완료: 반영하려면 --apply 를 붙여 다시 실행하세요.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
