"""사고팔기 슬랙 신호 스레드의 메시지 기록과 삭제.

신호·주문·알림 메시지를 이 모듈의 `send_thread_message` 로 보내면 스레드별 메시지 ts 가 컬렉션에 남는다.
`delete_signal_thread` 는 주문이 없는 신호만 슬랙 메시지(스레드 전체)와 우리 쪽 기록을 함께 지운다 —
메시지 ts 가 없는 옛 신호(기록 도입 전)는 지우지 않는다.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from utils.db_manager import get_db_connection
from utils.notification import send_slack_message_v2, update_slack_message
from utils.settings_loader import get_slack_channel

COLLECTION = "strategy_trade_slack_messages"
# 신호가 살아 있는 동안만 필요하다 — 오래된 기록은 DB 가 스스로 지운다.
_RETENTION_SECONDS = 7 * 24 * 3600
# 이미 지워진 메시지는 실패로 보지 않는다(사용자가 슬랙에서 직접 지웠을 수 있다).
_ALREADY_GONE = "message_not_found"

_index_ready = False


def _messages():
    global _index_ready
    db = get_db_connection()
    if db is None:
        raise RuntimeError("MongoDB 연결에 실패했습니다.")
    collection = db[COLLECTION]
    if not _index_ready:
        collection.create_index("created_at", expireAfterSeconds=_RETENTION_SECONDS)
        collection.create_index("thread_ts")
        _index_ready = True
    return collection


def send_thread_message(
    text: str, blocks: list[dict[str, Any]] | None = None, thread_ts: str | None = None
) -> str | None:
    """사고팔기 슬랙 메시지를 보내고 그 ts 를 스레드 기록에 남긴다 — `thread_ts` 가 없으면 새 스레드의 뿌리 메시지다."""
    ts = send_slack_message_v2(text, blocks, thread_ts)
    if ts:
        _messages().insert_one(
            {
                "channel": get_slack_channel(),
                "thread_ts": thread_ts or ts,
                "ts": ts,
                "created_at": datetime.now(UTC),
            }
        )
    return ts


def record_signal_thread(signal_ts: str, entries: list[dict[str, Any]]) -> None:
    """새 신호를 스레드에 연결한다. 같은 회차에서 다음 신호가 나가도 이 기록은 남는다."""
    _messages().update_one(
        {"thread_ts": signal_ts, "ts": signal_ts},
        {"$set": {"signal_entries": entries}},
    )


def mark_signal_thread_used(signal_ts: str) -> None:
    _messages().update_one({"thread_ts": signal_ts, "ts": signal_ts}, {"$set": {"signal_used": True}})


def reserve_signal_for_order(signal_ts: str) -> bool:
    """주문 요청과 신호 회수 중 먼저 기록된 쪽이 진행한다."""
    root = {"thread_ts": signal_ts, "ts": signal_ts}
    updated = _messages().update_one({**root, "withdrawal_text": {"$exists": False}}, {"$set": {"signal_used": True}})
    return (
        bool(updated.matched_count)
        or _messages().find_one({**root, "withdrawal_text": {"$exists": True}}, {"_id": 1}) is None
    )


def open_signal_threads(active_roots: set[str]) -> list[dict[str, Any]]:
    return list(
        _messages().find(
            {
                "signal_entries": {"$exists": True},
                "signal_used": {"$ne": True},
                "$or": [{"withdrawal_done": {"$ne": True}}, {"thread_ts": {"$in": list(active_roots)}}],
            },
            {"_id": 0, "thread_ts": 1, "signal_entries": 1, "withdrawal_text": 1, "withdrawal_done": 1},
        )
    )


def _signal_thread_has_orders(signal_ts: str) -> bool:
    from utils.strategy_trade_orders import _orders

    if _orders().find_one({"slack.thread_ts": signal_ts}, {"_id": 1}) is not None:
        mark_signal_thread_used(signal_ts)
        return True
    return False


def claim_signal_withdrawal(signal_ts: str, note: str) -> bool:
    if _signal_thread_has_orders(signal_ts):
        return False
    return bool(
        _messages()
        .update_one(
            {
                "thread_ts": signal_ts,
                "ts": signal_ts,
                "signal_used": {"$ne": True},
                "withdrawal_text": {"$exists": False},
            },
            {"$set": {"withdrawal_text": note}},
        )
        .matched_count
    )


def finish_signal_withdrawal(signal_ts: str, note: str) -> None:
    """본문에 회수를 기록하고 앱이 보낸 답글만 제거한다."""
    root = {"thread_ts": signal_ts, "ts": signal_ts}
    record = _messages().find_one(root, {"channel": 1})
    if record is None:
        raise ValueError("회수할 슬랙 메시지 기록을 찾을 수 없습니다.")
    try:
        update_slack_message(note, channel=record["channel"], message_ts=signal_ts, blocks=[])
    except SlackApiError as exc:
        if exc.response.get("error") != _ALREADY_GONE:
            raise ValueError("회수한 슬랙 본문을 수정하지 못했습니다.") from exc
    _delete_slack_thread(signal_ts, keep_root=True)
    _forget_thread(signal_ts, keep_root=True)
    _messages().update_one(root, {"$set": {"withdrawal_done": True}})


def expire_signal_thread(signal_ts: str) -> bool:
    """주문 이력이 없는 미사용 신호 스레드를 지운다."""
    if _signal_thread_has_orders(signal_ts):
        return False
    root = {"thread_ts": signal_ts, "ts": signal_ts}
    if _messages().find_one(
        {**root, "$or": [{"signal_used": True}, {"withdrawal_text": {"$exists": True}}]}, {"_id": 1}
    ):
        return False
    _delete_slack_thread(signal_ts)
    _forget_thread(signal_ts)
    return True


def _require_no_orders(signal_ts: str) -> None:
    """이 신호의 스레드로 낸 주문이 있으면 지우지 않는다 — 접수·체결 추적과 스레드 완료 표시가 끊긴다.

    같은 날 같은 전략·회차·구분의 다른 신호로 낸 주문은 상관없다(회차는 하루에도 여러 번 되풀이된다).
    접수 전 거부는 주문이 아니다.
    """
    from utils.strategy_trade_orders import _orders

    today = datetime.now().strftime("%Y-%m-%d")
    found = _orders().find_one(
        {"created_at": {"$gte": today}, "status": {"$ne": "rejected"}, "slack.thread_ts": signal_ts}
    )
    if found is not None:
        raise ValueError("이 신호로 낸 주문이 있어 삭제할 수 없습니다.")


def _delete_slack_thread(signal_ts: str, *, keep_root: bool = False) -> None:
    """스레드의 기록된 메시지를 답글부터 지우고 뿌리를 마지막에 지운다 — 실패한 것이 있으면 에러로 알린다."""
    records = list(_messages().find({"thread_ts": signal_ts}))
    targets = {record["ts"]: record["channel"] for record in records if record["ts"] != signal_ts}
    # 답글은 최신 것부터, 뿌리는 마지막.
    ordered = [(ts, targets[ts]) for ts in sorted(targets, reverse=True)]
    if not keep_root:
        ordered.append((signal_ts, next((r["channel"] for r in records if r["ts"] == signal_ts), get_slack_channel())))
    client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])
    failed: list[str] = []
    for ts, channel in ordered:
        if ts == signal_ts and failed:
            break
        try:
            client.chat_delete(channel=channel, ts=ts)
        except SlackApiError as exc:
            error = str(exc.response.get("error"))
            if error != _ALREADY_GONE:
                failed.append(error)
    if failed:
        raise ValueError(f"슬랙 메시지를 모두 지우지 못했습니다: {', '.join(sorted(set(failed)))}")


def _forget_thread(signal_ts: str, *, keep_root: bool = False) -> None:
    """스레드의 메시지 기록과 스레드 추적 기록을 지운다."""
    from utils.strategy_trade_orders import _SETTINGS_KEY, _config

    query: dict[str, Any] = {"thread_ts": signal_ts}
    if keep_root:
        query["ts"] = {"$ne": signal_ts}
    _messages().delete_many(query)
    _config().update_one({"_id": _SETTINGS_KEY}, {"$pull": {"signal_threads": {"thread_ts": signal_ts}}})


def delete_signal_thread(key: str) -> dict[str, Any]:
    """발송 기록 한 건(`key`)의 신호를 지운다 — 주문이 없을 때만. 슬랙 스레드 전체와 같은 스레드의 발송 기록을 함께 지운다."""
    from utils.strategy_trade_notify import _load_sent_log, _save_sent_log

    sent_log = _load_sent_log()
    entry = sent_log.get(key)
    if entry is None:
        raise ValueError("이미 지워진 신호입니다.")
    signal_ts = entry.get("signal_ts")
    if not signal_ts:
        raise ValueError("슬랙 메시지 기록이 없는 신호는 지울 수 없습니다.")
    if _messages().find_one(
        {"ts": signal_ts, "thread_ts": signal_ts, "withdrawal_text": {"$exists": True}}, {"_id": 1}
    ):
        raise ValueError("이미 회수된 신호입니다.")
    doomed = [k for k, value in sent_log.items() if value.get("signal_ts") == signal_ts]
    _require_no_orders(signal_ts)
    _delete_slack_thread(signal_ts)
    _forget_thread(signal_ts)
    for doomed_key in doomed:
        del sent_log[doomed_key]
    _save_sent_log(sent_log)
    return {"records_deleted": len(doomed)}
