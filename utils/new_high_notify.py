"""신고가 전략 전용 슬랙 알람 — 풀별 설정(`slack_enabled`)이 켜진 풀을 장중 10분 간격으로 감시한다.

내용: **모든 계좌에 없는(미보유)** 종목을 돌파성공 / 돌파(미달) / 터치 후 밀림 / 임박
네 그룹으로 나눠 보낸다. 분류는 화면과 같은 기준 — 거래대금 하한은 장중에 시간 비례로
환산한 목표(하루 목표 × 장 경과 비율)와 비교한다.

발송 규칙(트리거): 어떤 종목의 **실시간 비율(시간 환산 배수)이 새 정수 단(1배·2배·3배…)에
처음 도달했을 때만** 전체 내용을 1회 발송한다. 단은 하루 안에서 올라가기만 한다 —
2배를 찍고 1배로 줄었다가 다시 2배가 되어도 재발송하지 않고, 3배가 되면 그 종목이
트리거가 되어 다시 나간다. 트리거된 종목의 실시간 비율은 볼드로 표시한다.
구성(종목 출입)만 바뀌는 것은 발송 사유가 아니다 — 10분마다 반복되는 소음을 막는다.
단 기록은 DB(system_config)에 날짜와 함께 남기고, 날이 바뀌면 비운다.

섹션은 앞으로 몇 개 더 붙는다 — `_classify` 에 그룹을 추가하면 표시·트리거가 그대로
따라온다.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

from utils.logger import get_app_logger

logger = get_app_logger()

_STATE_KEY = "new_high_notify_state"


def _db():
    from utils.db_manager import get_db_connection

    db = get_db_connection()
    if db is None:
        raise RuntimeError("DB 연결에 실패했습니다.")
    return db


def _load_levels(pool: str, date: str) -> dict[str, int]:
    """오늘 도달한 거래대금 정수 단(티커→최고 단). 다른 날짜의 기록은 버린다."""
    doc = _db().system_config.find_one({"_id": _STATE_KEY}) or {}
    stored = (doc.get("pools") or {}).get(pool) or {}
    if str(stored.get("date") or "") != date:
        return {}
    return {str(k): int(v) for k, v in (stored.get("levels") or {}).items()}


def _save_levels(pool: str, date: str, levels: dict[str, int]) -> None:
    _db().system_config.update_one(
        {"_id": _STATE_KEY},
        {
            "$set": {
                f"pools.{pool}": {"date": date, "levels": levels, "updated_at": datetime.utcnow().isoformat()}
            }
        },
        upsert=True,
    )


def _pool_display_name(pool: str) -> str:
    from utils.settings_loader import get_ticker_type_settings

    name = str((get_ticker_type_settings(pool) or {}).get("name") or "").strip()
    return f"{name}({pool})" if name else pool


def _row_mult(row: dict[str, Any]) -> float | None:
    """표시에 쓰는 거래대금 배수 — 본값(장중=실시간 누적, 확정 후=KRX)."""
    value = row.get("value_mult")
    return float(value) if value is not None else None


def _row_pace_level(row: dict[str, Any]) -> int:
    """실시간 비율(시간 환산 배수)의 정수 단(내림) — 트리거·괄호 표시의 기준. 없으면 0."""
    pace = row.get("value_mult_live")
    if pace is None:
        return 0
    return int(math.floor(float(pace)))


def _row_line(row: dict[str, Any], bold: bool) -> str:
    """한 종목 줄 — 「티커 이름 · 일간% · 종가 대비 % · 거래대금 배수(실시간 비율 N배)」.

    종가 대비는 항상 볼드(화면 컬럼과 같은 강조). 실시간 비율은 정수 단(내림)으로
    보여주고, ``bold`` (이번 발송의 트리거 종목)면 그 부분만 볼드로 감싼다.
    """
    change = row.get("change_pct")
    gap = row.get("gap_pct")
    mult = _row_mult(row)
    mult_text = None
    if mult is not None:
        mult_text = f"거래대금 {mult:.1f}배"
        if row.get("value_mult_live") is not None:
            pace_text = f"실시간 비율 {_row_pace_level(row)}배"
            mult_text += f"({f'*{pace_text}*' if bold else pace_text})"
    parts = [
        f"*{row['ticker']} {row.get('name') or ''}*".strip(),
        f"{'+' if (change or 0) >= 0 else ''}{change:.2f}%" if change is not None else None,
        f"*종가 대비 {'+' if gap >= 0 else ''}{gap:.2f}%*" if gap is not None else None,
        mult_text,
    ]
    return "🔺 " + " · ".join(part for part in parts if part)


def _classify(positions: dict[str, Any]) -> list[dict[str, Any]]:
    """화면(describeStage)과 같은 우선순위로 네 그룹에 배분한다. 전 계좌 미보유 종목만."""
    success: list[dict[str, Any]] = []
    blocked: list[dict[str, Any]] = []
    pullback: list[dict[str, Any]] = []
    imminent: list[dict[str, Any]] = []
    for row in [*(positions.get("breakouts") or []), *(positions.get("candidates") or [])]:
        if row.get("account_held"):  # 어느 계좌든 실보유면 제외 — 신규 진입감만 알린다
            continue
        gap = row.get("gap_pct")
        if gap is None:
            continue
        if gap >= 0:
            (success if row.get("qualifies") else blocked).append(row)
        elif row.get("touched"):
            # 화면과 같은 우선순위 — 오늘 고가가 선을 건드렸으면 3% 이내라도 임박이 아니다.
            pullback.append(row)
        elif gap > -3:
            imminent.append(row)
    return [
        {"key": "breakout_success", "title": "돌파성공", "rows": success},
        {"key": "breakout_blocked", "title": "돌파(미달)", "rows": blocked},
        {"key": "pullback", "title": "터치 후 밀림", "rows": pullback},
        {"key": "imminent", "title": "임박", "rows": imminent},
    ]


def _format_message(
    pool: str, groups: list[dict[str, Any]], settings: dict[str, Any], triggered: set[str]
) -> str:
    from utils.new_high_service import kor_session_elapsed_fraction, live_min_value_mult

    lines = [f"📈 신고가 알람 — {_pool_display_name(pool)}"]
    min_mult = settings.get("min_value_mult")
    required = live_min_value_mult(min_mult)
    if required is not None:
        lines.append(
            f"거래대금 목표 {required:.1f}배 = 하한 {min_mult:g}배 × 장 경과 {kor_session_elapsed_fraction() * 100:.0f}%"
        )
    for group in groups:
        if not group["rows"]:
            continue
        lines.append(f"*{group['title']}*")
        lines.extend(_row_line(row, str(row["ticker"]) in triggered) for row in group["rows"])
    return "\n".join(lines)


def _watch_targets(country: str | None = None) -> list[str]:
    """감시 대상 풀 — 신고가 설정에서 슬랙 알람을 켠 풀. ``country`` 를 주면 그 국가만."""
    from utils.new_high_service import load_settings_map, pool_country

    targets = []
    for pool, stored in load_settings_map().items():
        if not bool((stored or {}).get("slack_enabled")):
            continue
        if country and pool_country(pool) != country:
            continue
        targets.append(pool)
    return targets


def _pool_settings(pool: str) -> dict[str, Any]:
    from utils.new_high_service import load_settings_map, validate_settings

    return validate_settings({"pool": pool, **(load_settings_map().get(pool) or {})})


def notify_pool(pool: str) -> dict[str, Any]:
    """한 풀 판정 — 거래대금 배수가 새 정수 단에 도달한 종목이 있을 때만 슬랙 1건."""
    from utils.new_high_backtest import current_positions
    from utils.notification import send_slack_message_v2

    settings = _pool_settings(pool)
    positions = current_positions(settings)
    if not positions.get("live"):
        # 장중이 아니면(휴장·장전) 판정할 실시간 값이 없다 — 상태도 건드리지 않는다.
        return {"pool": pool, "skipped": True, "reason": "장중 아님"}

    groups = _classify(positions)
    today = str(positions.get("quote_at") or "")[:10] or str(positions.get("as_of") or "")
    stored = _load_levels(pool, today)

    # 오늘의 단 — 목록에 있는 종목의 **실시간 비율**(시간 환산 배수)을 정수로 내림(1 미만은 0).
    # 단은 올라가기만 한다(래칫). 비율은 장 초반에 높다가 내려오는 성질이라, 아침 첫 발송에서
    # 래칫이 높게 잡히고 이후에는 페이스가 더 가속한 종목만 다시 트리거된다.
    triggered: set[str] = set()
    levels = dict(stored)
    for group in groups:
        for row in group["rows"]:
            level = _row_pace_level(row)
            ticker = str(row["ticker"])
            if level >= 1 and level > stored.get(ticker, 0):
                triggered.add(ticker)
            if level > levels.get(ticker, 0):
                levels[ticker] = level

    if levels != stored:
        _save_levels(pool, today, levels)
    items = sum(len(group["rows"]) for group in groups)
    if not triggered:
        return {"pool": pool, "sent": False, "items": items}
    send_slack_message_v2(_format_message(pool, groups, settings, triggered))
    return {"pool": pool, "sent": True, "items": items, "triggered": sorted(triggered)}


def notify_all(country: str | None = None) -> dict[str, Any]:
    """감시 풀 순회 — 새 정수 단에 도달한 종목이 생긴 풀만 발송한다."""
    results: list[dict[str, Any]] = []
    sent = 0
    for pool in _watch_targets(country):
        try:
            row = notify_pool(pool)
        except Exception as exc:
            logger.warning("[신고가 알람] %s 판정 실패: %s", pool, exc)
            row = {"pool": pool, "error": str(exc)}
        if row.get("sent"):
            sent += 1
        results.append(row)
    return {"sent": sent, "targets": results}


def send_test(pool: str) -> dict[str, Any]:
    """테스트 발송 — 트리거 여부와 무관하게 지금 내용을 즉시 보낸다(단 기록은 건드리지 않는다)."""
    from utils.new_high_backtest import current_positions
    from utils.notification import send_slack_message_v2

    settings = _pool_settings(pool)
    positions = current_positions(settings)
    groups = _classify(positions)
    items = sum(len(group["rows"]) for group in groups)
    message = _format_message(pool, groups, settings, triggered=set())
    if items == 0:
        message += "\n(대상 없음)"
    send_slack_message_v2("[테스트] " + message)
    return {"sent": True, "items": items}
