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
from utils.notification import send_slack_message_v2
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


def _require_no_orders(entries: list[dict[str, Any]]) -> None:
    """지울 신호들 가운데 주문을 낸 것이 있으면 지우지 않는다 — 접수·체결 추적과 스레드 완료 표시가 끊긴다. 접수 전 거부는 주문이 아니다."""
    from utils.strategy_trade_orders import _orders

    same_signal: list[dict[str, Any]] = [
        {"strategy_id": entry["strategy_id"], "round": entry["round"], "side": entry["action"]} for entry in entries
    ]
    signal_ts = {entry["signal_ts"] for entry in entries if entry.get("signal_ts")}
    if signal_ts:
        same_signal.append({"slack.thread_ts": {"$in": sorted(signal_ts)}})
    today = datetime.now().strftime("%Y-%m-%d")
    found = _orders().find_one({"created_at": {"$gte": today}, "status": {"$ne": "rejected"}, "$or": same_signal})
    if found is not None:
        raise ValueError("이 신호로 낸 주문이 있어 삭제할 수 없습니다.")


def _delete_slack_thread(signal_ts: str) -> None:
    """스레드의 기록된 메시지를 답글부터 지우고 뿌리를 마지막에 지운다 — 실패한 것이 있으면 에러로 알린다."""
    records = list(_messages().find({"thread_ts": signal_ts}))
    targets = {record["ts"]: record["channel"] for record in records if record["ts"] != signal_ts}
    # 답글은 최신 것부터, 뿌리는 마지막.
    ordered = [(ts, targets[ts]) for ts in sorted(targets, reverse=True)]
    ordered.append((signal_ts, next((r["channel"] for r in records if r["ts"] == signal_ts), get_slack_channel())))
    client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])
    failed: list[str] = []
    for ts, channel in ordered:
        try:
            client.chat_delete(channel=channel, ts=ts)
        except SlackApiError as exc:
            error = str(exc.response.get("error"))
            if error != _ALREADY_GONE:
                failed.append(error)
    if failed:
        raise ValueError(f"슬랙 메시지를 모두 지우지 못했습니다: {', '.join(sorted(set(failed)))}")


def _forget_thread(signal_ts: str) -> None:
    """스레드의 메시지 기록과 스레드 추적 기록을 지운다."""
    from utils.strategy_trade_orders import _SETTINGS_KEY, _config

    _messages().delete_many({"thread_ts": signal_ts})
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
    doomed = [k for k, value in sent_log.items() if value.get("signal_ts") == signal_ts]
    _require_no_orders([sent_log[k] for k in doomed])
    _delete_slack_thread(signal_ts)
    _forget_thread(signal_ts)
    for doomed_key in doomed:
        del sent_log[doomed_key]
    _save_sent_log(sent_log)
    return {"records_deleted": len(doomed)}
