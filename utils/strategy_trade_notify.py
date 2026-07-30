"""전략 사고팔기 슬랙 알림.

거래시간 중 10분마다 실행돼, **매수·매도 지정가에 현재가가 닿은 회차가 있을 때만**
슬랙을 보낸다. 트리거가 없으면 아무것도 보내지 않는다.

중복 방지
--------
10분 간격으로 판정하므로 같은 조건이 유지되면 하루에 수십 번 발송될 수 있다.
그래서 ``회차-동작`` 단위로 **하루 1회만** 보낸다. 발송 이력은
``system_config.strategy_trade_settings.sent_log`` 에 ``{"4-buy": "2026-07-30"}``
형태로 남긴다. 이력은 발송에 성공했을 때만 갱신한다.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from utils.strategy_trade_service import load_settings, load_strategy_trade_view

KST = ZoneInfo("Asia/Seoul")
_CONFIG_COLLECTION = "system_config"
_SETTINGS_KEY = "strategy_trade_settings"


def _load_sent_log() -> dict[str, str]:
    from utils.db_manager import get_db_connection

    db = get_db_connection()
    if db is None:
        return {}
    doc = db[_CONFIG_COLLECTION].find_one({"_id": _SETTINGS_KEY}) or {}
    stored = doc.get("sent_log")
    if not isinstance(stored, dict):
        return {}
    return {str(k): str(v) for k, v in stored.items()}


def _save_sent_log(sent_log: dict[str, str]) -> None:
    from utils.db_manager import get_db_connection

    db = get_db_connection()
    if db is None:
        raise RuntimeError("DB 연결에 실패했습니다.")
    db[_CONFIG_COLLECTION].update_one(
        {"_id": _SETTINGS_KEY},
        {"$set": {"sent_log": sent_log, "sent_log_updated_at": datetime.now(KST).isoformat()}},
        upsert=True,
    )


def collect_triggers(view: dict[str, Any]) -> list[dict[str, Any]]:
    """지정가에 닿은 회차를 모은다. 없으면 빈 리스트."""
    triggers: list[dict[str, Any]] = []
    for row in view["rounds"]:
        if row["held"] and row["sell_reached"]:
            triggers.append(
                {
                    "action": "sell",
                    "round": row["round"],
                    "ticker": row["ticker"],
                    "name": row["name"],
                    "limit_price": row["sell_limit"],
                    "close": row["close"],
                    "index_level": row["sell_index"],
                    "profit_pct": row["profit_pct"],
                }
            )
        # 매수는 '다음 진입 대상' 회차만 의미가 있다(회차는 순차 진행).
        elif row["is_next"] and row["buy_reached"]:
            triggers.append(
                {
                    "action": "buy",
                    "round": row["round"],
                    "ticker": row["ticker"],
                    "name": row["name"],
                    "limit_price": row["buy_limit"],
                    "close": row["close"],
                    "index_level": row["buy_index"],
                    "profit_pct": None,
                }
            )
    return triggers


def build_message(view: dict[str, Any], triggers: list[dict[str, Any]]) -> str:
    """슬랙 본문. 트리거된 회차와 현재 보유 요약을 담는다."""
    now = datetime.now(KST).strftime("%Y-%m-%d %H:%M")
    index = view["index"]
    status = view["status"]

    lines = [f"*⚡ 전략 사고팔기 신호 ({now})*"]
    for trigger in triggers:
        if trigger["action"] == "sell":
            lines.append(
                f"🔴 *매도* {trigger['round']}호 {trigger['ticker']} {trigger['name']} — "
                f"목표 {trigger['limit_price']:,.0f} 도달 (현재 {trigger['close']:,.0f}, "
                f"{trigger['profit_pct']:+.2f}%)"
            )
        else:
            lines.append(
                f"🔵 *매수* {trigger['round']}호 {trigger['ticker']} {trigger['name']} — "
                f"지정가 {trigger['limit_price']:,.0f} 도달 (현재 {trigger['close']:,.0f})"
            )

    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━")
    lines.append(f"{index['name']} {index['close']:,.2f} · 보유 {status['held_count']}/{view['config']['rounds']}회차")
    if status["held_count"] > 0 and status["profit_amount"] is not None:
        lines.append(
            f"평가 {status['valuation_amount']:,}원 · 손익 {status['profit_amount']:,}원 "
            f"({status['profit_pct']:+.2f}%)"
        )
    return "\n".join(lines)


def notify_strategy_trade(*, force: bool = False) -> dict[str, Any]:
    """트리거를 판정해 필요할 때만 슬랙을 보낸다.

    Args:
        force: 스위치·중복 방지를 무시하고 현재 상태를 발송한다(테스트 버튼용).

    Returns:
        ``{"sent": bool, "reason": str, "triggers": [...]}``
    """
    from utils.notification import send_slack_message_v2

    settings = load_settings()
    if not force and not settings["slack_enabled"]:
        return {"sent": False, "reason": "슬랙 알림이 꺼져 있습니다.", "triggers": []}

    view = load_strategy_trade_view()
    triggers = collect_triggers(view)

    if force:
        # 테스트 발송 — 트리거가 없으면 현재 현황만 보낸다.
        message = (
            build_message(view, triggers)
            if triggers
            else _build_status_only_message(view)
        )
        sent = send_slack_message_v2(message)
        return {
            "sent": bool(sent),
            "reason": "테스트 발송" if sent else "슬랙 전송에 실패했습니다.",
            "triggers": triggers,
        }

    if not triggers:
        return {"sent": False, "reason": "매수·매도 신호가 없습니다.", "triggers": []}

    today = datetime.now(KST).strftime("%Y-%m-%d")
    sent_log = _load_sent_log()
    fresh = [t for t in triggers if sent_log.get(f"{t['round']}-{t['action']}") != today]
    if not fresh:
        return {"sent": False, "reason": "오늘 이미 발송한 신호입니다.", "triggers": triggers}

    message = build_message(view, fresh)
    if not send_slack_message_v2(message):
        return {"sent": False, "reason": "슬랙 전송에 실패했습니다.", "triggers": fresh}

    for trigger in fresh:
        sent_log[f"{trigger['round']}-{trigger['action']}"] = today
    # 지난 날짜 이력은 정리해 문서가 커지지 않게 한다.
    sent_log = {key: value for key, value in sent_log.items() if value == today}
    _save_sent_log(sent_log)

    return {"sent": True, "reason": f"신호 {len(fresh)}건 발송", "triggers": fresh}


def _build_status_only_message(view: dict[str, Any]) -> str:
    """트리거가 없을 때의 테스트 발송 본문 — 현재 대기 조건을 알려준다."""
    now = datetime.now(KST).strftime("%Y-%m-%d %H:%M")
    index = view["index"]
    status = view["status"]
    config = view["config"]

    lines = [
        f"*🧪 전략 사고팔기 테스트 ({now})*",
        "현재 매수·매도 신호는 없습니다.",
        f"{index['name']} {index['close']:,.2f} · 보유 {status['held_count']}/{config['rounds']}회차",
    ]
    for row in view["rounds"]:
        if row["held"]:
            lines.append(
                f"· {row['round']}호 {row['name']} 매도 {row['sell_limit']:,.0f} "
                f"(현재 {row['close']:,.0f}, {row['profit_pct']:+.2f}%)"
            )
        elif row["is_next"] and row["buy_limit"] is not None:
            lines.append(
                f"· {row['round']}호 {row['name']} 매수 {row['buy_limit']:,.0f} "
                f"(현재 {row['close']:,.0f})"
            )
    return "\n".join(lines)
