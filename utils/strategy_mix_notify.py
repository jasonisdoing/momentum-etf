"""합성 전략 오늘의 액션 슬랙 알람 — 화면과 같은 지시(`actions.groups`)를 보낸다.

계좌 설정에서 모멘텀·신고가 종목풀이 둘 다 지정되고 `mix_slack_enabled` 가 켜진 계좌만 감시한다.
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


def _load_state(account_id: str) -> dict[str, int]:
    doc = _db().system_config.find_one({"_id": _STATE_KEY}) or {}
    return (doc.get("accounts") or {}).get(account_id, {}).get("quantities") or {}


def _save_state(account_id: str, quantities: dict[str, int]) -> None:
    _db().system_config.update_one(
        {"_id": _STATE_KEY},
        {"$set": {f"accounts.{account_id}": {"quantities": quantities, "updated_at": datetime.utcnow().isoformat()}}},
        upsert=True,
    )


def _format_message(account_name: str, groups: list[dict[str, Any]]) -> str:
    lines = [f"🧭 합성 오늘의 액션 — {account_name}"]
    for group in groups:
        lines.append(f"*{group['title']}*")
        for item in group["items"]:
            emoji = "🔴" if item["side"] == "buy" else "🔵"
            lines.append(f"{emoji} {item['title']} — {item['text']}")
    return "\n".join(lines)


def _watch_targets(country: str | None = None) -> list[dict[str, Any]]:
    """감시 대상 — 슬리브별 풀과 슬랙 알람이 모두 설정된 계좌. ``country`` 를 주면 그 국가만
    (한국·미국 장 시간이 달라 배치를 국가별로 나눠 돌린다).

    국가는 계좌 값을 쓴다 — 두 풀이 계좌와 같은 국가인 것은 계좌 설정 저장이 보장한다.
    """
    from utils.account_settings_store import load_account_docs

    targets = []
    for doc in load_account_docs():
        if not doc.get("mix_a_pool") or not doc.get("mix_b_pool"):
            continue
        if not bool(doc.get("mix_slack_enabled")):
            continue
        if country and str(doc.get("country_code") or "").strip().lower() != country:
            continue
        targets.append({"account_id": doc["account_id"], "name": doc.get("name") or doc["account_id"]})
    return targets


def notify_all(country: str | None = None) -> dict[str, Any]:
    """감시 계좌 순회 — 새 지시·수량 증가가 있을 때만 슬랙 1건 발송."""
    from utils.notification import send_slack_message_v2
    from utils.strategy_mix_service import mix_positions

    results = []
    sent = 0
    for target in _watch_targets(country):
        account_id = target["account_id"]
        try:
            positions = mix_positions(account_id)
        except Exception as exc:
            logger.warning("[MIX-NOTIFY] %s 계산 실패: %s", account_id, exc)
            results.append({"account_id": account_id, "name": target["name"], "error": str(exc)})
            continue
        # (예상) 그룹도 포함 — 처음 등장할 때 1건 발송되고, 종가 확정 후 확정 지시로 바뀌면
        # 키가 달라져 다시 1건 나간다. 경계에서 빠졌다 재진입하면 재발송된다(예상 알림의 비용).
        groups = positions["actions"]["groups"]
        current = {item["key"]: int(item.get("quantity") or 0) for group in groups for item in group["items"]}
        stored = _load_state(account_id)
        grown = {key: qty for key, qty in current.items() if qty > int(stored.get(key, 0))}
        # 저장은 항상 한다 — 지시가 줄어든 것도 다음 비교의 기준이 되어야, 같은 지시가
        # 다시 커졌을 때(재발) 알림이 나간다.
        _save_state(account_id, current)
        if not grown:
            results.append(
                {"account_id": account_id, "name": target["name"], "changed": False, "items": len(current)}
            )
            continue
        send_slack_message_v2(_format_message(target["name"], groups))
        sent += 1
        results.append(
            {
                "account_id": account_id,
                "name": target["name"],
                "changed": True,
                "new_or_grown": len(grown),
                "items": len(current),
            }
        )
    return {"sent": sent, "targets": results}


def send_test(account_id: str) -> dict[str, Any]:
    """테스트 발송 — 변화 여부와 무관하게 지금 액션을 즉시 보낸다(상태는 건드리지 않는다)."""
    from utils.notification import send_slack_message_v2
    from utils.settings_loader import get_account_settings
    from utils.strategy_mix_service import mix_positions

    settings = get_account_settings(account_id)
    if not settings.get("mix_a_pool") or not settings.get("mix_b_pool"):
        raise ValueError(f"'{account_id}' 에 합성 A·B 슬리브가 모두 설정돼 있지 않습니다.")
    positions = mix_positions(account_id)
    groups = positions["actions"]["groups"]
    name = str(settings.get("name") or account_id)
    message = _format_message(name, groups) if groups else f"🧭 합성 오늘의 액션 — {name}\n(지시 없음)"
    send_slack_message_v2("[테스트] " + message)
    item_count = sum(len(group["items"]) for group in groups)
    return {"sent": True, "items": item_count}
