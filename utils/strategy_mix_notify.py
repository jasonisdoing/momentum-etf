"""합성 전략 오늘의 액션 슬랙 알람 — 화면과 같은 지시(`actions.groups`)를 보낸다.

계좌 설정에서 모멘텀·신고가 종목풀이 둘 다 지정되고 `mix_slack_enabled` 가 켜진 계좌만 감시한다.
발송 조건은 **새 지시 또는 수량 증가**다 — 지시가 줄거나 사라지는 변화(체결 반영,
증권사 동기화)는 조용히 넘긴다. 매번 전체 목록을 보내면 체결할 때마다 알람이 와서
소음이 된다. 상태는 DB(system_config)에 남긴다(신고가 알림과 같은 패턴).

**체결 확인**(2026-10) — 보낸 지시는 대기 이력(``pending``)으로 남기고, 매 배치마다 계좌 보유가
지시 수량만큼 **전량** 바뀌었으면 확인 한 줄을 보낸다(전략 사고팔기와 같은 공용 흐름,
``utils.order_confirm``). 지시가 사라진 것만으로는 체결로 보지 않는다 — 합성은 예상 진입이
빠지거나 목표 수량이 바뀌어도 지시가 줄거나 사라진다. 이력은 종목·방향마다 하나라 같은
종목의 새 지시가 오면 덮어쓰고, 반대 방향 지시가 오면 지운다.

**오늘의 액션(`section == "today"`)만 보낸다**(2026-09) — 내일의 액션(다음 거래일,
장중 잠정 판정)은 지시가 계속 바뀌어 반복 알림이 됐다. 그 시장 현지 날짜가 실행일이
되는 아침 배치에서 오늘의 액션이 되어 새 지시로 감지·발송된다(개장 전 주문 타이밍).
화면은 오늘·내일 두 섹션을 모두 보여준다.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from utils.logger import get_app_logger
from utils.order_confirm import build_confirm_line, pop_filled

logger = get_app_logger()

_STATE_KEY = "strategy_mix_notify_state"


def _db():
    from utils.db_manager import get_db_connection

    db = get_db_connection()
    if db is None:
        raise RuntimeError("DB 연결에 실패했습니다.")
    return db


def _load_state(account_id: str) -> dict[str, Any]:
    """계좌의 알림 상태 — ``quantities``(지시 키 → 수량), ``pending``(종목:방향 → 체결 대기 지시)."""
    doc = _db().system_config.find_one({"_id": _STATE_KEY}) or {}
    return (doc.get("accounts") or {}).get(account_id) or {}


def _save_state(account_id: str, quantities: dict[str, int], pending: dict[str, dict[str, Any]]) -> None:
    _db().system_config.update_one(
        {"_id": _STATE_KEY},
        {
            "$set": {
                f"accounts.{account_id}": {
                    "quantities": quantities,
                    "pending": pending,
                    "updated_at": datetime.utcnow().isoformat(),
                }
            }
        },
        upsert=True,
    )


def _held_by_ticker(positions: dict[str, Any]) -> dict[str, int]:
    """계좌 실제 보유 수량 — 합성 보유 표(계좌 보유를 읽은 행)에서 꺼낸다."""
    return {row["ticker"]: int(float(row.get("held_quantity") or 0)) for row in positions["holdings"]}


def _order_filled(entry: dict[str, Any], held: dict[str, int]) -> bool:
    """지시 수량이 **전량** 보유에 반영됐는지 — 일부 체결은 아직 대기다.

    표에 없는 종목은 계좌에도 없다는 뜻이라 보유 0 이다(전량 매도 후 행이 빠진다).
    """
    now = held.get(entry["ticker"], 0)
    if entry["action"] == "buy":
        return now >= entry["held_at_send"] + entry["quantity"]
    return now <= entry["held_at_send"] - entry["quantity"]


def _remember_orders(
    pending: dict[str, dict[str, Any]], groups: list[dict[str, Any]], positions: dict[str, Any]
) -> None:
    """보낸 지시를 체결 대기로 남긴다 — 종목·방향마다 하나(새 지시가 덮고, 반대 방향은 지운다)."""
    held = _held_by_ticker(positions)
    names = {row["ticker"]: row.get("name") or row["ticker"] for row in positions["holdings"]}
    for group in groups:
        for item in group["items"]:
            ticker, side = item["ticker"], item["side"]
            opposite = "sell" if side == "buy" else "buy"
            pending.pop(f"{ticker}:{opposite}", None)
            pending[f"{ticker}:{side}"] = {
                "ticker": ticker,
                "name": names.get(ticker, ticker),
                "action": side,
                "quantity": int(item["quantity"]),
                "held_at_send": held.get(ticker, 0),
                "at": datetime.utcnow().isoformat(),
            }


def _confirm_line(entry: dict[str, Any], held: dict[str, int]) -> str:
    """`✅ MRNA 8주 매수 확인 · 보유 0주 → 8주`."""
    label = f"{entry['name']}({entry['ticker']}) {entry['quantity']:,}주"
    detail = f"보유 {entry['held_at_send']:,}주 → {held.get(entry['ticker'], 0):,}주"
    return build_confirm_line(label, entry["action"], detail)


def _format_message(account_name: str, groups: list[dict[str, Any]]) -> str:
    lines = [f"🧭 합성 오늘의 액션 — {account_name}"]
    for group in groups:
        lines.append(f"*{group['title']}*")
        if group.get("funding_warning"):
            lines.append(f"⚠️ {group['funding_warning']}")
        for item in group["items"]:
            emoji = "🔴" if item["side"] == "buy" else "🔵"
            lines.append(f"{emoji} {item['title']} — {item['text']}")
    return "\n".join(lines)


def _watch_targets(country: str | None = None) -> list[dict[str, Any]]:
    """감시 대상 — 슬리브별 풀과 슬랙 알람이 모두 설정된 계좌. ``country`` 를 주면 그 국가만
    (한국·미국 장 시간이 달라 배치를 국가별로 나눠 돌린다).

    국가는 계좌 값을 쓴다 — 슬리브 풀이 계좌와 같은 국가인 것은 계좌 설정 저장이 보장한다.
    """
    from utils.account_settings_store import load_account_docs, mix_sleeves_of

    targets = []
    for doc in load_account_docs():
        sleeves = mix_sleeves_of(doc)
        if not sleeves or not all(row["strategy"] and row["pool"] for row in sleeves):
            continue
        if not bool(doc.get("mix_slack_enabled")):
            continue
        if country and str(doc.get("country_code") or "").strip().lower() != country:
            continue
        targets.append({"account_id": doc["account_id"], "name": doc.get("name") or doc["account_id"]})
    return targets


def notify_all(country: str | None = None) -> dict[str, Any]:
    """감시 계좌 순회 — 새 지시·수량 증가, 또는 보낸 지시의 체결 확인이 있을 때만 슬랙 1건 발송."""
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
        # 오늘의 액션만 발송·비교한다 — 내일의 액션(장중 잠정 판정)은 제외. 상태 비교에서도
        # 빼야, 실행일 아침에 오늘의 액션이 되는 순간 새 지시로 발송된다.
        groups = [group for group in positions["actions"]["groups"] if group.get("section") == "today"]
        current = {item["key"]: int(item.get("quantity") or 0) for group in groups for item in group["items"]}
        state = _load_state(account_id)
        stored = state.get("quantities") or {}
        pending = dict(state.get("pending") or {})
        held = _held_by_ticker(positions)
        # ① 보낸 지시의 체결 확인 — 신호 유무와 무관하게 매번 본다.
        confirmed = pop_filled(pending, lambda entry: _order_filled(entry, held))
        grown = {key: qty for key, qty in current.items() if qty > int(stored.get(key, 0))}
        lines = [_confirm_line(entry, held) for entry in confirmed]
        if grown:
            lines.append(_format_message(target["name"], groups))
            _remember_orders(pending, groups, positions)
        if lines:
            send_slack_message_v2("\n".join(lines))
            sent += 1
        # 저장은 항상 한다 — 지시가 줄어든 것도 다음 비교의 기준이 되어야, 같은 지시가
        # 다시 커졌을 때(재발) 알림이 나간다.
        _save_state(account_id, current, pending)
        results.append(
            {
                "account_id": account_id,
                "name": target["name"],
                "changed": bool(grown),
                "new_or_grown": len(grown),
                "confirmed": len(confirmed),
                "items": len(current),
            }
        )
    return {"sent": sent, "targets": results}


def send_test(account_id: str) -> dict[str, Any]:
    """테스트 발송 — 변화 여부와 무관하게 지금 액션을 즉시 보낸다(상태는 건드리지 않는다)."""
    from utils.account_settings_store import mix_sleeves_of
    from utils.notification import send_slack_message_v2
    from utils.settings_loader import get_account_settings
    from utils.strategy_mix_service import mix_positions

    settings = get_account_settings(account_id)
    sleeves = mix_sleeves_of(settings.get("settings") or settings)
    if not sleeves or not all(row["strategy"] and row["pool"] for row in sleeves):
        raise ValueError(f"'{account_id}' 에 합성 슬리브가 설정돼 있지 않습니다.")
    positions = mix_positions(account_id)
    # 실제 알람과 같은 범위 — 오늘의 액션만.
    groups = [group for group in positions["actions"]["groups"] if group.get("section") == "today"]
    name = str(settings.get("name") or account_id)
    message = _format_message(name, groups) if groups else f"🧭 합성 오늘의 액션 — {name}\n(지시 없음)"
    send_slack_message_v2("[테스트] " + message)
    item_count = sum(len(group["items"]) for group in groups)
    return {"sent": True, "items": item_count}
