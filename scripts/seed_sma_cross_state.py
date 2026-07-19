"""sma_cross 시장별 추천 상태(보유 시작일) 시드 — 1회성.

- 한국(sma_cross_kor): 기존 switch 전략 상태의 보유 시작일을 그대로 이관해 보유일 연속성을 준다.
- 미국(sma_cross_us): 이관할 상태가 없으므로 오늘(최근 확정 종가일) 판정 기준으로 새로 만든다.

안전 절차: 실행 전 현재 상태를 백업(JSON 출력) → 비교 → upsert. 이미 보유 시작일이 있으면
덮어쓰지 않는다(--force 로만 덮어씀). 배치가 아직 없으므로 이 시드가 최초 상태가 된다.
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from leverage.config_store import load_leverage_state, save_leverage_state  # noqa: E402
from utils.leverage_sma_service import compute_sma_cross_view  # noqa: E402


def _dump(label: str, state: dict) -> None:
    print(f"[{label}] {json.dumps(state, ensure_ascii=False, default=str)}")


def seed_korea(force: bool) -> None:
    switch = load_leverage_state("switch")
    cur = load_leverage_state("sma_cross_kor")
    _dump("backup switch", switch)
    _dump("backup sma_cross_kor(before)", cur)

    if cur.get("holding_start_date") and not force:
        print("→ 한국: 이미 보유 시작일이 있어 건너뜀 (덮으려면 --force)")
        return
    start = switch.get("holding_start_date")
    if not start:
        print("→ 한국: switch 상태에 holding_start_date 가 없어 이관 불가")
        return
    new_state = {
        "date": switch.get("date"),
        "target": switch.get("target"),
        "target_name": switch.get("target_name"),
        "side": "leverage" if switch.get("target") not in (None, "CASH") else "defense",
        "holding_start_date": start,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    save_leverage_state("sma_cross_kor", new_state)
    _dump("saved sma_cross_kor", load_leverage_state("sma_cross_kor"))


def seed_us(force: bool) -> None:
    cur = load_leverage_state("sma_cross_us")
    _dump("backup sma_cross_us(before)", cur)
    if cur.get("holding_start_date") and not force:
        print("→ 미국: 이미 보유 시작일이 있어 건너뜀 (덮으려면 --force)")
        return

    view = compute_sma_cross_view("sma_cross_us")
    rec = view.get("recommendation")
    if rec is None:
        print("→ 미국: 판정 데이터가 없어 시드 불가(설정 확인 필요)")
        return
    new_state = {
        "date": rec["as_of"],
        "target": rec["target_ticker"],
        "target_name": rec["target_name"],
        "side": rec["side"],
        "holding_start_date": rec["as_of"],  # 오늘(최근 확정 종가일)부터 카운트 시작
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    save_leverage_state("sma_cross_us", new_state)
    _dump("saved sma_cross_us", load_leverage_state("sma_cross_us"))


def main() -> None:
    parser = argparse.ArgumentParser(description="sma_cross 시장별 보유 상태 시드")
    parser.add_argument("--force", action="store_true", help="기존 보유 시작일이 있어도 덮어쓴다")
    parser.add_argument("--market", choices=["kor", "us", "all"], default="all")
    args = parser.parse_args()

    if args.market in ("kor", "all"):
        seed_korea(args.force)
    if args.market in ("us", "all"):
        seed_us(args.force)


if __name__ == "__main__":
    main()
