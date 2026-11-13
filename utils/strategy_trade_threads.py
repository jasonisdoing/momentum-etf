"""사고팔기 슬랙 신호 스레드의 메시지 기록과 삭제.

신호·주문·알림 메시지를 이 모듈의 `send_thread_message` 로 보내면 스레드별 메시지 ts 가 컬렉션에 남는다.
신호 회수는 모두 `withdraw_signal` 하나를 거친다 — 주문이 없는 신호만 본문을 표준 한 줄로 바꿔 기록을 남기고
앱이 단 답글은 지운다.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from utils.db_manager import get_db_connection
from utils.notification import send_slack_message_v2, update_slack_message
from utils.settings_loader import get_slack_channel

KST = ZoneInfo("Asia/Seoul")
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


#: 회수 사유 — 슬랙 회수 한 줄에 들어가는 표시 문구의 단일 소스.
WITHDRAW_REASONS: dict[str, str] = {
    "price_returned": "가격 복귀",
    "settings_changed": "설정 변경",
    "removed": "사고팔기에서 제외",
    "session_closed": "장 마감 미사용",
    "manual": "화면에서 회수",
}


def withdrawal_line(entries: list[dict[str, Any]], reason: str, detail: str | None, now: datetime) -> str:
    """회수 표준 한 줄 — `↩️ 회수 · 종목 1호 매수 · 사유[ · 상세] · MM-DD HH:MM`(한국시간)."""
    items = ", ".join(
        "임의 매도"
        if entry.get("arbitrary_id")
        else f"{entry['round']}호 {'매수' if entry['action'] == 'buy' else '매도'}"
        for entry in entries
    )
    parts = ["↩️ 회수", f"{entries[0]['name']} {items}", WITHDRAW_REASONS[reason], *([detail] if detail else [])]
    return " · ".join([*parts, f"{now.astimezone(KST):%m-%d %H:%M}"])


def withdraw_signal(signal_ts: str, reason: str, detail: str | None) -> bool:
    """신호를 회수한다 — 사고팔기 슬랙 회수는 모두 이 함수 하나를 거친다.

    본문을 표준 한 줄(`withdrawal_line`)로 바꿔 회수 기록을 남기고 앱이 단 답글은 지운다. 이 신호 스레드로 낸 주문이
    있거나 이미 쓰인·회수된 신호면 건드리지 않고 False 를 돌려준다. 회수를 먼저 기록(선점)한 뒤 슬랙을 고치므로,
    중간에 끊겨도 `resume_withdrawal` 이 마저 끝낸다.
    """
    if reason not in WITHDRAW_REASONS:
        raise ValueError(f"알 수 없는 회수 사유입니다: {reason}")
    root = {"thread_ts": signal_ts, "ts": signal_ts}
    record = _messages().find_one(root, {"signal_entries": 1})
    if record is None or not record.get("signal_entries"):
        raise ValueError("회수할 슬랙 신호 기록을 찾을 수 없습니다.")
    if _signal_thread_has_orders(signal_ts):
        return False
    note = withdrawal_line(record["signal_entries"], reason, detail, datetime.now(UTC))
    claimed = _messages().update_one(
        {**root, "signal_used": {"$ne": True}, "withdrawal_text": {"$exists": False}},
        {"$set": {"withdrawal_text": note}},
    )
    if not claimed.matched_count:
        return False
    resume_withdrawal(signal_ts)
    return True


def resume_withdrawal(signal_ts: str) -> None:
    """회수를 선점한 신호의 본문을 회수 한 줄로 바꾸고 앱이 보낸 답글만 지운다."""
    root = {"thread_ts": signal_ts, "ts": signal_ts}
    record = _messages().find_one(root, {"channel": 1, "withdrawal_text": 1})
    if record is None or not record.get("withdrawal_text"):
        raise ValueError("회수를 기록한 슬랙 메시지를 찾을 수 없습니다.")
    try:
        update_slack_message(record["withdrawal_text"], channel=record["channel"], message_ts=signal_ts, blocks=[])
    except SlackApiError as exc:
        if exc.response.get("error") != _ALREADY_GONE:
            raise ValueError("회수한 슬랙 본문을 수정하지 못했습니다.") from exc
    _delete_replies(signal_ts)
    _forget_replies(signal_ts)
    _messages().update_one(root, {"$set": {"withdrawal_done": True}})


def _delete_replies(signal_ts: str) -> None:
    """스레드에 기록된 앱 답글을 최신 것부터 지운다(본문은 남긴다) — 실패한 것이 있으면 에러로 알린다."""
    records = _messages().find({"thread_ts": signal_ts, "ts": {"$ne": signal_ts}})
    targets = sorted(((record["ts"], record["channel"]) for record in records), reverse=True)
    client = WebClient(token=os.environ["SLACK_BOT_TOKEN"])
    failed: list[str] = []
    for ts, channel in targets:
        try:
            client.chat_delete(channel=channel, ts=ts)
        except SlackApiError as exc:
            error = str(exc.response.get("error"))
            if error != _ALREADY_GONE:
                failed.append(error)
    if failed:
        raise ValueError(f"슬랙 답글을 모두 지우지 못했습니다: {', '.join(sorted(set(failed)))}")


def _forget_replies(signal_ts: str) -> None:
    """스레드의 답글 기록과 스레드 추적 기록을 지운다 — 본문 기록(회수 기록)은 남긴다."""
    from utils.strategy_trade_orders import _SETTINGS_KEY, _config

    _messages().delete_many({"thread_ts": signal_ts, "ts": {"$ne": signal_ts}})
    _config().update_one({"_id": _SETTINGS_KEY}, {"$pull": {"signal_threads": {"thread_ts": signal_ts}}})


def withdraw_signal_by_key(key: str) -> dict[str, Any]:
    """화면에서 발송 기록 한 건(`key`)의 신호를 회수한다 — 같은 메시지로 나간 발송 기록도 함께 지운다."""
    from utils.strategy_trade_notify import _load_sent_log, _save_sent_log

    sent_log = _load_sent_log()
    entry = sent_log.get(key)
    if entry is None:
        raise ValueError("이미 정리된 신호입니다.")
    signal_ts = entry.get("signal_ts")
    if not signal_ts:
        raise ValueError("슬랙 메시지 기록이 없는 신호는 회수할 수 없습니다.")
    if not withdraw_signal(signal_ts, "manual", None):
        raise ValueError("이 신호로 낸 주문이 있거나 이미 쓰인·회수된 신호라 회수할 수 없습니다.")
    doomed = [k for k, value in sent_log.items() if value.get("signal_ts") == signal_ts]
    for doomed_key in doomed:
        del sent_log[doomed_key]
    _save_sent_log(sent_log)
    return {"records_deleted": len(doomed)}
