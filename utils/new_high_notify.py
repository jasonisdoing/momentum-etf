"""신고가 돌파 — 진입·매도 예정 슬랙 알림 (화면 테스트 버튼·장중 감시 배치 공용).

10분 주기 배치가 같은 내용을 반복 발송하면 도배가 되므로, 풀별로 **마지막 발송 상태**를
DB(``system_config.new_high_notify_state``)에 남기고 달라진 것만 알린다.

- 새로 생긴 진입 예정 / 매도 예정 → 발송
- 직전에 알렸던 예정이 사라짐 → "해제" 로 발송 (장중 되밀림 — 안 알리면 낡은 정보로 주문한다)
- 판정일이 바뀌면(새 세션) 상태를 비우고 새로 시작 — 지난 세션 항목의 해제 알림은 내지 않는다
- ``force=True``(화면 테스트 버튼)는 비교 없이 현재 상태 전체를 즉시 발송한다

발송 여부와 무관하게 계산 결과 요약을 돌려줘 배치 로그·화면 토스트가 같은 근거를 쓴다.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from utils.logger import get_app_logger

logger = get_app_logger()

_STATE_KEY = "new_high_notify_state"


def _db():
    from utils.new_high_service import _db as service_db

    return service_db()


def _load_pool_state(pool: str) -> dict[str, Any]:
    doc = _db().system_config.find_one({"_id": _STATE_KEY}) or {}
    return (doc.get("pools") or {}).get(pool) or {}


def _save_pool_state(pool: str, state: dict[str, Any]) -> None:
    _db().system_config.update_one(
        {"_id": _STATE_KEY},
        {"$set": {f"pools.{pool}": {**state, "updated_at": datetime.utcnow().isoformat()}}},
        upsert=True,
    )


def _format_price(value: Any) -> str:
    try:
        price = float(value)
    except (TypeError, ValueError):
        return "-"
    return f"{price:,.0f}" if price >= 1000 else f"{price:,.2f}"


def snapshot_from_positions(positions: dict[str, Any]) -> dict[str, Any]:
    """current_positions 결과에서 알림 비교에 쓰는 상태를 뽑는다.

    entries/sells 는 {티커: 표시문구} 맵 — 키 집합으로 변화를 판정하고 값은 메시지에 쓴다.
    """
    entries: dict[str, str] = {}
    for row in positions.get("planned_entries") or []:
        ticker = str(row.get("ticker") or "").strip()
        if not ticker:
            continue
        mult = row.get("value_mult")
        mult_text = f" · 거래대금 {float(mult):.1f}배" if isinstance(mult, (int, float)) else ""
        change = row.get("change_pct")
        change_text = f" ({float(change):+.2f}%)" if isinstance(change, (int, float)) else ""
        entries[ticker] = (
            f"{row.get('name') or ticker}({ticker}) {_format_price(row.get('price'))}{change_text}{mult_text}"
        )

    sells: dict[str, str] = {}
    for row in positions.get("holdings") or []:
        if str(row.get("status") or "") != "sell":
            continue
        ticker = str(row.get("ticker") or "").strip()
        if not ticker:
            continue
        ret = row.get("return_pct")
        ret_text = f" 수익률 {float(ret):+.2f}%" if isinstance(ret, (int, float)) else ""
        reason = str(row.get("exit_reason") or "이탈")
        sells[ticker] = f"{row.get('name') or ticker}({ticker}){ret_text} — {reason}"

    return {"as_of": str(positions.get("as_of") or ""), "entries": entries, "sells": sells}


def _pool_label(pool: str) -> str:
    from utils.settings_loader import get_ticker_type_settings

    try:
        settings = get_ticker_type_settings(pool) or {}
    except Exception:
        settings = {}
    return str(settings.get("name") or pool)


def notify_pool(pool: str, *, force: bool = False) -> dict[str, Any]:
    """한 풀의 진입·매도 예정 변화를 계산해 필요하면 슬랙으로 보낸다.

    반환: {"pool", "sent", "reason", "entries", "sells", ...} — 배치 로그·화면 토스트 공용.
    """
    from utils.new_high_backtest import current_positions
    from utils.new_high_service import load_settings_map, validate_settings

    saved = load_settings_map().get(pool)
    if not saved:
        return {"pool": pool, "sent": False, "reason": "저장된 설정 없음"}
    settings = validate_settings({**saved, "pool": pool})
    if not force and not settings.get("slack_enabled"):
        return {"pool": pool, "sent": False, "reason": "슬랙 알람 꺼짐"}

    positions = current_positions(settings)
    snapshot = snapshot_from_positions(positions)
    stored = _load_pool_state(pool)

    # 판정일이 바뀌면 새 세션 — 지난 세션과 비교하지 않는다(해제 알림 없음).
    same_session = str(stored.get("as_of") or "") == snapshot["as_of"]
    prev_entries = dict(stored.get("entries") or {}) if same_session else {}
    prev_sells = dict(stored.get("sells") or {}) if same_session else {}

    added_entries = {t: v for t, v in snapshot["entries"].items() if t not in prev_entries}
    added_sells = {t: v for t, v in snapshot["sells"].items() if t not in prev_sells}
    removed_entries = {t: v for t, v in prev_entries.items() if t not in snapshot["entries"]}
    removed_sells = {t: v for t, v in prev_sells.items() if t not in snapshot["sells"]}
    changed = bool(added_entries or added_sells or removed_entries or removed_sells)

    lines: list[str] = []
    if force:
        # 테스트 발송 — 비교 없이 현재 상태 전체.
        for text in snapshot["entries"].values():
            lines.append(f"🔴 진입 예정: {text}")
        for text in snapshot["sells"].values():
            lines.append(f"🔵 매도 예정: {text}")
        if not lines:
            lines.append("진입·매도 예정 없음")
    elif changed:
        for text in added_entries.values():
            lines.append(f"🔴 진입 예정: {text}")
        for text in added_sells.values():
            lines.append(f"🔵 매도 예정: {text}")
        for text in removed_entries.values():
            lines.append(f"⚪ 진입 예정 해제: {text}")
        for text in removed_sells.values():
            lines.append(f"⚪ 매도 예정 해제: {text}")

    sent = False
    if lines and (force or changed):
        basis = "장중" if positions.get("live") else "마감"
        header = f"🚀 신고가 돌파 · {_pool_label(pool)} — {snapshot['as_of']} {basis} 기준"
        footer = "장중 기준이라 종가 확정 시 달라질 수 있습니다." if positions.get("live") else ""
        if force:
            header += " (수동 테스트)"
        from utils.notification import send_slack_message_v2

        message = "\n".join([header, *lines] + ([footer] if footer else []))
        send_slack_message_v2(message)
        sent = True
        logger.info("[NEW-HIGH NOTIFY] %s 발송 %d줄 (force=%s)", pool, len(lines), force)

    # 발송 여부와 무관하게 현재 상태를 저장한다 — 다음 비교의 기준.
    _save_pool_state(pool, snapshot)

    return {
        "pool": pool,
        "sent": sent,
        "reason": None if sent else ("변화 없음" if not force else "발송됨"),
        "as_of": snapshot["as_of"],
        "live": bool(positions.get("live")),
        "entries": len(snapshot["entries"]),
        "sells": len(snapshot["sells"]),
        "added": len(added_entries) + len(added_sells),
        "removed": len(removed_entries) + len(removed_sells),
    }
