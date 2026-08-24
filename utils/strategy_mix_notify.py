"""합성 전략 오늘의 액션 슬랙 알람 — 화면과 같은 지시(`actions.groups`)를 보낸다.

계좌 설정에서 `mix_pool` + `mix_slack_enabled` 가 켜진 계좌만 감시한다.
발송 조건은 **새 지시 또는 수량 증가**다 — 지시가 줄거나 사라지는 변화(체결 반영,
증권사 동기화)는 조용히 넘긴다. 매번 전체 목록을 보내면 체결할 때마다 알람이 와서
소음이 된다. 상태는 DB(system_config)에 남긴다(신고가 알림과 같은 패턴).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from utils.logger import get_app_logger

logger = get_app_logger()

_STATE_KEY = "strategy_mix_notify_state"


def _db():
    from utils.db_manager import get_db_connection

    db = get_db_connection()
    if db is None:
        raise RuntimeError("DB 연결에 실패했습니다.")
    return db


def _load_state(pool: str) -> dict[str, int]:
    doc = _db().system_config.find_one({"_id": _STATE_KEY}) or {}
    return (doc.get("pools") or {}).get(pool, {}).get("quantities") or {}


def _save_state(pool: str, quantities: dict[str, int]) -> None:
    _db().system_config.update_one(
        {"_id": _STATE_KEY},
        {"$set": {f"pools.{pool}": {"quantities": quantities, "updated_at": datetime.utcnow().isoformat()}}},
        upsert=True,
    )


def _format_message(pool: str, account_name: str, groups: list[dict[str, Any]]) -> str:
    lines = [f"🧭 합성 오늘의 액션 — {account_name} ({pool})"]
    for group in groups:
        lines.append(f"*{group['title']}*")
        for item in group["items"]:
            emoji = "🔴" if item["side"] == "buy" else "🔵"
            lines.append(f"{emoji} {item['title']} — {item['text']}")
    return "\n".join(lines)


def _watch_targets() -> list[dict[str, Any]]:
    """감시 대상 — mix_pool 과 슬랙 알람이 모두 설정된 계좌."""
    from utils.account_settings_store import load_account_docs

    targets = []
    for doc in load_account_docs():
        pool = str(doc.get("mix_pool") or "").strip().lower()
        if pool and bool(doc.get("mix_slack_enabled")):
            targets.append(
                {"account_id": doc["account_id"], "name": doc.get("name") or doc["account_id"], "pool": pool}
            )
    return targets


def notify_all() -> dict[str, Any]:
    """감시 계좌 순회 — 새 지시·수량 증가가 있을 때만 슬랙 1건 발송."""
    from utils.notification import send_slack_message_v2
    from utils.strategy_mix_service import mix_positions

    results = []
    sent = 0
    for target in _watch_targets():
        pool = target["pool"]
        try:
            positions = mix_positions(pool)
        except Exception as exc:
            logger.warning("[MIX-NOTIFY] %s 계산 실패: %s", pool, exc)
            results.append({"pool": pool, "error": str(exc)})
            continue
        # (예상) 그룹도 포함 — 처음 등장할 때 1건 발송되고, 종가 확정 후 확정 지시로 바뀌면
        # 키가 달라져 다시 1건 나간다. 경계에서 빠졌다 재진입하면 재발송된다(예상 알림의 비용).
        groups = positions["actions"]["groups"]
        current = {item["key"]: int(item.get("quantity") or 0) for group in groups for item in group["items"]}
        stored = _load_state(pool)
        grown = {key: qty for key, qty in current.items() if qty > int(stored.get(key, 0))}
        # 저장은 항상 한다 — 지시가 줄어든 것도 다음 비교의 기준이 되어야, 같은 지시가
        # 다시 커졌을 때(재발) 알림이 나간다.
        _save_state(pool, current)
        if not grown:
            results.append({"pool": pool, "changed": False, "items": len(current)})
            continue
        send_slack_message_v2(_format_message(pool, target["name"], groups))
        sent += 1
        results.append({"pool": pool, "changed": True, "new_or_grown": len(grown), "items": len(current)})
    return {"sent": sent, "targets": results}


def send_test(account_id: str) -> dict[str, Any]:
    """테스트 발송 — 변화 여부와 무관하게 지금 액션을 즉시 보낸다(상태는 건드리지 않는다)."""
    from utils.notification import send_slack_message_v2
    from utils.settings_loader import get_account_settings
    from utils.strategy_mix_service import mix_positions

    settings = get_account_settings(account_id)
    pool = str(settings.get("mix_pool") or "").strip().lower()
    if not pool:
        raise ValueError(f"'{account_id}' 에 합성 종목풀(mix_pool)이 설정돼 있지 않습니다.")
    positions = mix_positions(pool)
    groups = positions["actions"]["groups"]
    name = str(settings.get("name") or account_id)
    message = _format_message(pool, name, groups) if groups else f"🧭 합성 오늘의 액션 — {name} ({pool})\n(지시 없음)"
    send_slack_message_v2("[테스트] " + message)
    item_count = sum(len(group["items"]) for group in groups)
    return {"sent": True, "items": item_count}
