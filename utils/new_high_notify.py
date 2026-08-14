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


def _pool_order_and_label(pool: str) -> tuple[int, str]:
    """(정렬용 번호, "3. 코스피 + 코스닥" 형식 라벨) — 종목풀 설정의 order·name 을 쓴다."""
    from utils.settings_loader import get_ticker_type_settings

    try:
        settings = get_ticker_type_settings(pool) or {}
    except Exception:
        settings = {}
    name = str(settings.get("name") or pool)
    try:
        order = int(settings.get("order"))
    except (TypeError, ValueError):
        order = 999
    return order, f"{order}. {name}"


def _collect_pool(pool: str, *, force: bool) -> dict[str, Any] | None:
    """한 풀을 계산해 섹션(제목+줄들)과 저장할 상태를 만든다. 대상이 아니면 None.

    force=False 면 직전 발송 상태와 비교해 **변화 줄만** 담고, 변화가 없으면 줄이 비어
    발송 대상에서 빠진다. force=True(테스트)는 비교 없이 현재 예정 전체를 담는다.
    """
    from utils.new_high_backtest import current_positions
    from utils.new_high_service import load_settings_map, validate_settings

    saved = load_settings_map().get(pool)
    if not saved:
        return None
    settings = validate_settings({**saved, "pool": pool})
    if not force and not settings.get("slack_enabled"):
        return None

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

    lines: list[str] = []
    if force:
        lines += [f"🚀 진입 예정: {text}" for text in snapshot["entries"].values()]
        lines += [f"🔻 매도 예정: {text}" for text in snapshot["sells"].values()]
        if not lines:
            lines.append("진입·매도 예정 없음")
    else:
        lines += [f"🚀 진입 예정: {text}" for text in added_entries.values()]
        lines += [f"🔻 매도 예정: {text}" for text in added_sells.values()]
        lines += [f"⚪ 진입 예정 해제: {text}" for text in removed_entries.values()]
        lines += [f"⚪ 매도 예정 해제: {text}" for text in removed_sells.values()]

    order, label = _pool_order_and_label(pool)
    return {
        "pool": pool,
        "order": order,
        "label": label,
        "lines": lines,
        "snapshot": snapshot,
        "live": bool(positions.get("live")),
        "added": len(added_entries) + len(added_sells),
        "removed": len(removed_entries) + len(removed_sells),
    }


def _send_sections(sections: list[dict[str, Any]]) -> None:
    """풀 번호순 섹션들을 *신고가 돌파* 제목 아래 한 건의 메시지로 발송한다."""
    from utils.notification import send_slack_message_v2

    parts = ["*신고가 돌파*"]
    for section in sorted(sections, key=lambda s: s["order"]):
        parts.append("\n".join([section["label"], *section["lines"]]))
    send_slack_message_v2("\n\n".join(parts))


def notify_pool(pool: str, *, force: bool = False) -> dict[str, Any]:
    """한 풀만 계산·발송한다 (화면 '지금 발송(테스트)' 버튼)."""
    collected = _collect_pool(pool, force=force)
    if collected is None:
        return {"pool": pool, "sent": False, "reason": "저장된 설정 없음 또는 슬랙 알람 꺼짐"}

    sent = False
    if collected["lines"]:
        _send_sections([collected])
        sent = True
        logger.info("[NEW-HIGH NOTIFY] %s 발송 %d줄 (force=%s)", pool, len(collected["lines"]), force)

    # 발송 여부와 무관하게 현재 상태를 저장한다 — 다음 비교의 기준.
    _save_pool_state(pool, collected["snapshot"])
    return {
        "pool": pool,
        "sent": sent,
        "reason": None if sent else "변화 없음",
        "as_of": collected["snapshot"]["as_of"],
        "live": collected["live"],
        "entries": len(collected["snapshot"]["entries"]),
        "sells": len(collected["snapshot"]["sells"]),
        "added": collected["added"],
        "removed": collected["removed"],
    }


# 감시 창 여유 — 개장 전 10분(장전 상태 반영)부터 마감 후 90분(마감 후 가격 캐시 배치가
# 확정 종가를 채우는 :20 실행을 덮는다)까지를 감시한다. 창 밖에서는 가격이 안 움직여
# 변화가 있을 수 없으므로 계산 자체를 건너뛴다.
_WINDOW_BEFORE_OPEN_MIN = 10
_WINDOW_AFTER_CLOSE_MIN = 90


def _within_notify_window(pool: str) -> bool:
    """이 풀의 시장이 감시 창 안인지 — 창 밖이면 계산을 건너뛴다(비용 절약)."""
    import pandas as pd

    from config import MARKET_SCHEDULES
    from utils.settings_loader import get_ticker_type_settings

    country = str((get_ticker_type_settings(pool) or {}).get("country_code") or "").strip().lower()
    schedule = (MARKET_SCHEDULES or {}).get(country)
    if not isinstance(schedule, dict):
        return False
    tz_name = str(schedule.get("timezone") or "").strip()
    open_time, close_time = schedule.get("open"), schedule.get("close")
    if not tz_name or open_time is None or close_time is None:
        return False
    try:
        now_local = pd.Timestamp.now(tz=tz_name)
        opens_at = pd.Timestamp(f"{now_local.date()} {open_time.hour:02d}:{open_time.minute:02d}", tz=tz_name)
        closes_at = pd.Timestamp(f"{now_local.date()} {close_time.hour:02d}:{close_time.minute:02d}", tz=tz_name)
    except Exception:
        return False
    window_start = opens_at - pd.Timedelta(minutes=_WINDOW_BEFORE_OPEN_MIN)
    window_end = closes_at + pd.Timedelta(minutes=_WINDOW_AFTER_CLOSE_MIN)
    return window_start <= now_local <= window_end


def notify_all(*, force: bool = False) -> dict[str, Any]:
    """슬랙 알람을 켠 모든 풀을 번호순으로 묶어 한 건으로 발송한다 (장중 감시 배치용).

    변화가 없는 풀·설정이 없는 풀·알람을 끈 풀은 메시지에서 빠진다.
    """
    from utils.new_high_service import available_pools

    sections: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for pool in available_pools():
        if not force and not _within_notify_window(pool):
            continue
        try:
            collected = _collect_pool(pool, force=force)
        except Exception as exc:  # 한 풀 실패가 다른 풀 알림을 막지 않게
            logger.warning("[NEW-HIGH NOTIFY] %s 계산 실패: %s", pool, exc)
            summaries.append({"pool": pool, "error": str(exc)[:120]})
            continue
        if collected is None:
            continue
        summaries.append(
            {
                "pool": pool,
                "changed": bool(collected["lines"]),
                "added": collected["added"],
                "removed": collected["removed"],
            }
        )
        if collected["lines"]:
            sections.append(collected)
        _save_pool_state(pool, collected["snapshot"])

    if sections:
        _send_sections(sections)
        logger.info("[NEW-HIGH NOTIFY] 통합 발송 — %d개 풀", len(sections))
    return {"sent": bool(sections), "pools": summaries, "section_count": len(sections)}
