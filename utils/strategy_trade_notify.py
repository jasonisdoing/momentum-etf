"""전략 사고팔기 슬랙 알림.

거래시간 중 10분마다 실행돼, **매수·매도 지정가에 현재가가 닿은 회차가 있을 때만**
슬랙을 보낸다. 트리거가 없으면 아무것도 보내지 않는다.

중복 방지
--------
10분 간격으로 판정하므로 같은 조건이 유지되면 하루에 수십 번 발송될 수 있다.
그래서 ``전략:회차-동작`` 단위로 **하루 1회만** 보낸다. 발송 이력은
``system_config.strategy_trade_settings.sent_log`` 에 ``{"kospi200:4-buy": "2026-07-30"}``
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


def collect_triggers(strategy_view: dict[str, Any]) -> list[dict[str, Any]]:
    """전략 하나에서 지정가에 닿은 회차를 모은다. 없으면 빈 리스트."""
    sid = strategy_view["strategy_id"]
    label = strategy_view["label"]
    triggers: list[dict[str, Any]] = []
    for row in strategy_view["rounds"]:
        if row["held"] and row["sell_reached"]:
            triggers.append(
                {
                    "strategy_id": sid,
                    "strategy_label": label,
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
                    "strategy_id": sid,
                    "strategy_label": label,
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
    """슬랙 본문 — 전략별 섹션으로 전 회차 현황을 나열하고 조치가 필요한 줄만 표시한다.

    회차 수는 신호를 **실행한 뒤**의 상태로 적는다(매수 신호면 +1, 매도면 −1).
    트리거가 없으면(테스트 발송) 같은 형식에서 '필요!' 표시만 빠진다.
    """
    now = datetime.now(KST).strftime("%Y-%m-%d %H:%M")

    # 실제 신호와 테스트 발송을 상단에서 바로 구분한다.
    # 실제 신호는 주문이 필요하므로 <!channel> 로 채널 전체에 알린다(테스트는 제외).
    if triggers:
        lines = [f"<!channel> *⚡ 전략 사고팔기 — 실제 신호* ({now})"]
    else:
        lines = [f"*🧪 전략 사고팔기 테스트* ({now})", "현재 매수·매도 신호는 없습니다."]

    for strategy_view in view["strategies"]:
        sid = strategy_view["strategy_id"]
        index = strategy_view["index"]
        status = strategy_view["status"]
        rounds_total = strategy_view["config"]["rounds"]

        triggered = {(t["round"], t["action"]) for t in triggers if t["strategy_id"] == sid}
        buy_count = sum(1 for t in triggers if t["strategy_id"] == sid and t["action"] == "buy")
        sell_count = sum(1 for t in triggers if t["strategy_id"] == sid and t["action"] == "sell")
        after_held = status["held_count"] + buy_count - sell_count

        # 가장 중요한 정보 — 몇 % 움직이면 사고/파는지 헤더에 바로 보여준다.
        parts = []
        drop = status.get("next_buy_drop_pct")
        if drop is None:
            parts.append("매수 없음(회차 소진)")
        elif drop >= 0:
            parts.append(f"📉 *{status['next_round']}호 매수가 도달!*")
        else:
            parts.append(f"📉 {abs(drop):.2f}% 내리면 {status['next_round']}호 매수")
        rise = status.get("next_sell_rise_pct")
        if rise is not None:
            if rise <= 0:
                parts.append(f"📈 *{status['next_sell_round']}호 매도가 도달!*")
            else:
                parts.append(f"📈 {rise:.2f}% 오르면 {status['next_sell_round']}호 매도")
        lines.append(
            f"*[{strategy_view['label']}]* {index['name']} {index['close']:,.2f} · {after_held}/{rounds_total}회차 · "
            + " · ".join(parts)
        )
        # 1~6호 전부 나열 — 상태를 이모지로 즉시 구분한다.
        #   🔴 지금 팔 것 / 🔵 지금 살 것 / 🟢 보유 중 / ⏳ 다음 매수 대기 / ⚪ 이후 회차
        for row in strategy_view["rounds"]:
            close_text = f"{row['close']:,.0f}" if row["close"] is not None else "-"
            if row["held"]:
                is_triggered = (row["round"], "sell") in triggered
                profit_text = f"{row['profit_pct']:+.2f}%" if row["profit_pct"] is not None else "-"
                if is_triggered:
                    lines.append(
                        f"🔴 {row['round']}호 {row['name']} *지금 매도!* "
                        f"목표 {row['sell_limit']:,.0f} 도달 (현재 {close_text}, {profit_text})"
                    )
                else:
                    lines.append(
                        f"🟢 {row['round']}호 {row['name']} 보유 중 ({profit_text}) · "
                        f"매도 목표 {row['sell_limit']:,.0f} (현재 {close_text})"
                    )
            elif row["is_next"]:
                is_triggered = (row["round"], "buy") in triggered
                buy_text = f"{row['buy_limit']:,.0f}" if row["buy_limit"] is not None else "-"
                if is_triggered:
                    lines.append(
                        f"🔵 {row['round']}호 {row['name']} *지금 매수!* "
                        f"지정가 {buy_text} 도달 (현재 {close_text})"
                    )
                else:
                    lines.append(
                        f"⏳ {row['round']}호 {row['name']} 다음 매수 대기 · "
                        f"지정가 {buy_text} (현재 {close_text})"
                    )
            else:
                buy_text = f"{row['buy_limit']:,.0f}" if row["buy_limit"] is not None else "-"
                lines.append(f"⚪ {row['round']}호 {row['name']} 이후 회차 · 매수 예정가 {buy_text}")

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
    triggers = [t for s in view["strategies"] for t in collect_triggers(s)]

    if force:
        # 테스트 발송 — 트리거가 없으면 현재 현황만 보낸다.
        sent = send_slack_message_v2(build_message(view, triggers))
        return {
            "sent": bool(sent),
            "reason": "테스트 발송" if sent else "슬랙 전송에 실패했습니다.",
            "triggers": triggers,
        }

    if not triggers:
        return {"sent": False, "reason": "매수·매도 신호가 없습니다.", "triggers": []}

    today = datetime.now(KST).strftime("%Y-%m-%d")
    sent_log = _load_sent_log()
    fresh = [t for t in triggers if sent_log.get(f"{t['strategy_id']}:{t['round']}-{t['action']}") != today]
    if not fresh:
        return {"sent": False, "reason": "오늘 이미 발송한 신호입니다.", "triggers": triggers}

    message = build_message(view, fresh)
    if not send_slack_message_v2(message):
        return {"sent": False, "reason": "슬랙 전송에 실패했습니다.", "triggers": fresh}

    for trigger in fresh:
        sent_log[f"{trigger['strategy_id']}:{trigger['round']}-{trigger['action']}"] = today
    # 지난 날짜 이력은 정리해 문서가 커지지 않게 한다.
    sent_log = {key: value for key, value in sent_log.items() if value == today}
    _save_sent_log(sent_log)

    return {"sent": True, "reason": f"신호 {len(fresh)}건 발송", "triggers": fresh}
