"""전략 사고팔기 슬랙 알림.

거래시간 중 10분마다 실행돼, **매수·매도 지정가에 현재가가 닿은 회차가 있을 때만**
슬랙을 보낸다. 트리거가 없으면 아무것도 보내지 않는다.

중복 방지 · 체결 확인
--------------------
10분 간격으로 판정하므로 같은 조건이 유지되면 하루에 수십 번 발송될 수 있다. 그래서
``전략:회차-동작`` 단위로 처음 한 번만 전체 메시지를 보내고, 신호가 살아 있는 동안에는
한 시간 간격으로 한 줄 리마인더만 보낸다. 발송 이력은
``system_config.strategy_trade_settings.sent_log`` 에
``{"kospi200:2-buy": {"at": ..., "ticker": ..., "name": ..., "round": ..., "index_text": ...}}``
형태로 남기며, 발송에 성공했을 때만 갱신한다.

이력에 남은 주문이 실제로 체결되면 확인 한 줄을 보내고 이력을 지운다. 이 확인은
**신호가 없어도** 매번 검사한다 — 주문을 넣는 순간 그 회차가 보유로 바뀌어 바로 그 신호가
사라지기 때문에, 신호가 있을 때만 검사하면 확인은 영영 나가지 못한다.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from utils.strategy_trade_service import load_settings, load_strategy_trade_view

KST = ZoneInfo("Asia/Seoul")
_CONFIG_COLLECTION = "system_config"
_SETTINGS_KEY = "strategy_trade_settings"


def _load_sent_log() -> dict[str, dict[str, Any]]:
    from utils.db_manager import get_db_connection

    db = get_db_connection()
    if db is None:
        return {}
    doc = db[_CONFIG_COLLECTION].find_one({"_id": _SETTINGS_KEY}) or {}
    stored = doc.get("sent_log")
    if not isinstance(stored, dict):
        return {}
    # 옛 형식(값이 날짜 문자열)은 확인 판정에 쓸 티커가 없어 버린다 — 하루면 자연히 정리된다.
    return {str(k): v for k, v in stored.items() if isinstance(v, dict)}


def _save_sent_log(sent_log: dict[str, dict[str, Any]]) -> None:
    from utils.db_manager import get_db_connection

    db = get_db_connection()
    if db is None:
        raise RuntimeError("DB 연결에 실패했습니다.")
    db[_CONFIG_COLLECTION].update_one(
        {"_id": _SETTINGS_KEY},
        {"$set": {"sent_log": sent_log, "sent_log_updated_at": datetime.now(KST).isoformat()}},
        upsert=True,
    )


# 같은 신호가 계속 살아 있을 때 다시 알리는 주기. 10분마다 도는 배치가 매번 보내면
# 소음이라 첫 발송 뒤에는 이 간격으로만 한 줄 리마인더를 보낸다.
_REPEAT_INTERVAL_SECONDS = 60 * 60


def _trigger_key(trigger: dict[str, Any]) -> str:
    return f"{trigger['strategy_id']}:{trigger['round']}-{trigger['action']}"


def _parse_sent_at(value: str | None) -> datetime | None:
    """발송 시각. 날짜만 담긴 옛 기록은 그날 자정으로 읽어 하루 뒤 자연히 정리되게 둔다."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=KST)


def _level(value: float | None) -> str:
    """지수 환산값 — 없으면 '-'."""
    return f"{value:,.2f}" if value is not None else "-"


def _index_text(index: dict[str, Any]) -> str:
    """`6,472.32 (-2.34%)` — 지수 종가와 일간 변동률."""
    change = index.get("change_pct")
    base = f"{index['close']:,.2f}"
    return f"{base} ({change:+.2f}%)" if change is not None else base


def build_confirm_line(pending: dict[str, Any], index_text: str) -> str:
    """확인 한 줄 — 요청한 주문이 실제로 체결됐을 때 한 번만 보낸다.

    조치가 이미 끝난 뒤라 `@channel` 은 붙이지 않는다. 회차 번호는 평균단가에 따라
    다시 매겨지므로 **알림 당시 번호**를 그대로 쓰고, 종목명으로 무엇인지 알린다.
    """
    word = "매도 확인" if pending["action"] == "sell" else "매수 확인"
    asked = pending.get("index_text")
    tail = f" · 요청 시 {asked} → 지금 {index_text}" if asked else f" · {index_text}"
    return f"✅ *{pending['round']}호 {pending['name']} {word}*{tail}"


def _held_state(view: dict[str, Any]) -> dict[str, float | None]:
    """보유 중인 티커 → 평균단가. 미보유 티커는 키가 없다."""
    return {
        row["ticker"]: row.get("avg_price")
        for strategy in view["strategies"]
        for row in strategy["rounds"]
        if row["held"]
    }


def build_repeat_line(trigger: dict[str, Any]) -> str:
    """재알림 한 줄 — 무엇을 얼마에 사고팔지만 담는다. 본문은 첫 발송에서 이미 보냈다."""
    emoji = "🔴" if trigger["action"] == "sell" else "🔵"
    word = "지금 매도!" if trigger["action"] == "sell" else "지금 매수!"
    return (
        f"<!channel> {emoji} *{trigger['round']}호 {trigger['name']} {word}* "
        f"{trigger['index_name']} {_level(trigger['index_level'])} 도달 "
        f"(현재 {_index_text(trigger['index'])})"
    )


def collect_triggers(strategy_view: dict[str, Any]) -> list[dict[str, Any]]:
    """전략 하나에서 지정가에 닿은 회차를 모은다. 없으면 빈 리스트."""
    sid = strategy_view["strategy_id"]
    label = strategy_view["label"]
    index = strategy_view["index"]
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
                    "index_name": index["name"],
                    "index": index,
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
                    "index_name": index["name"],
                    "index": index,
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
            f"*[{strategy_view['label']}]* {index['name']} {_index_text(index)} · {after_held}/{rounds_total}회차 · "
            + " · ".join(parts)
        )
        # 1~6호 전부 나열 — 상태를 이모지로 즉시 구분한다.
        #   🔴 지금 팔 것 / 🔵 지금 살 것 / 🟢 보유 중 / ⏳ 다음 매수 대기 / ⚪ 이후 회차
        # 회차별 줄 — 종목 가격이 아니라 **지수 환산값**으로 적는다. ETF 단가는 종목마다
        # 달라 서로 비교가 안 되고, 판단은 결국 지수가 어디까지 왔느냐로 하기 때문이다.
        for row in strategy_view["rounds"]:
            if row["held"]:
                is_triggered = (row["round"], "sell") in triggered
                profit_text = f"{row['profit_pct']:+.2f}%" if row["profit_pct"] is not None else "-"
                if is_triggered:
                    lines.append(
                        f"🔴 {row['round']}호 {row['name']} *지금 매도!* "
                        f"{index['name']} {_level(row['sell_index'])} 도달 ({profit_text})"
                    )
                else:
                    lines.append(
                        f"🟢 {row['round']}호 {row['name']} 보유 중 ({profit_text}) · "
                        f"매도 목표 {index['name']} {_level(row['sell_index'])}"
                    )
            elif row["is_next"]:
                is_triggered = (row["round"], "buy") in triggered
                if is_triggered:
                    lines.append(
                        f"🔵 {row['round']}호 {row['name']} *지금 매수!* "
                        f"{index['name']} {_level(row['buy_index'])} 도달"
                    )
                else:
                    lines.append(
                        f"⏳ {row['round']}호 {row['name']} 다음 매수 대기 · "
                        f"{index['name']} {_level(row['buy_index'])}"
                    )
            else:
                lines.append(
                    f"⚪ {row['round']}호 {row['name']} 이후 회차 · "
                    f"매수 예정 {index['name']} {_level(row['buy_index'])}"
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
    triggers = [t for s in view["strategies"] for t in collect_triggers(s)]

    if force:
        # 테스트 발송 — 트리거가 없으면 현재 현황만 보낸다.
        sent = send_slack_message_v2(build_message(view, triggers))
        return {
            "sent": bool(sent),
            "reason": "테스트 발송" if sent else "슬랙 전송에 실패했습니다.",
            "triggers": triggers,
        }

    now = datetime.now(KST)
    today = now.strftime("%Y-%m-%d")
    sent_log = _load_sent_log()

    # 처음 보는 신호는 전 회차 현황이 담긴 본문으로, 이미 보낸 신호는 아직 살아 있는 동안
    # 한 시간마다 한 줄 리마인더로 보낸다. 지정가에 닿아 있는데 주문을 안 넣은 채로
    # 하루가 지나가는 것을 막기 위한 것이다.
    # ① 요청해 둔 주문이 체결됐는지 먼저 본다. 체결됐으면 확인 한 줄을 보내고 기록을 지운다 —
    #    지워야 다음에 같은 회차 신호가 나면 다시 전체 메시지로 나간다.
    #    이 검사는 트리거 유무와 무관하게 돌아야 한다: 주문을 넣는 순간 그 회차가 보유로 바뀌어
    #    바로 그 트리거가 사라지므로, 신호가 있을 때만 검사하면 확인은 영영 나가지 못한다.
    held = _held_state(view)
    index_by_ticker = {
        row["ticker"]: strategy["index"]
        for strategy in view["strategies"]
        for row in strategy["rounds"]
    }
    confirmed: list[dict[str, Any]] = []
    for key, pending in list(sent_log.items()):
        ticker = pending.get("ticker")
        if not ticker:
            continue
        done = (ticker in held) if pending.get("action") == "buy" else (ticker not in held)
        if done:
            index = index_by_ticker.get(ticker)
            confirmed.append({**pending, "now_text": _index_text(index) if index else "-"})
            sent_log.pop(key, None)

    # ② 남은 신호 — 처음이면 전체 메시지, 이미 보냈으면 한 시간 간격으로 한 줄 재알림.
    fresh: list[dict[str, Any]] = []
    repeats: list[dict[str, Any]] = []
    for trigger in triggers:
        last = _parse_sent_at((sent_log.get(_trigger_key(trigger)) or {}).get("at"))
        if last is None or last.strftime("%Y-%m-%d") != today:
            fresh.append(trigger)
        elif (now - last).total_seconds() >= _REPEAT_INTERVAL_SECONDS:
            repeats.append(trigger)

    lines: list[str] = []
    for pending in confirmed:
        lines.append(build_confirm_line(pending, pending["now_text"]))
    if fresh:
        lines.append(build_message(view, fresh))
    else:
        lines.extend(build_repeat_line(trigger) for trigger in repeats)
    if not lines:
        if not triggers:
            # 확인할 체결도 신호도 없다 — 10분마다 도는 배치라 여기서 DB 를 건드리면 무의미한 쓰기만 쌓인다.
            return {"sent": False, "reason": "매수·매도 신호가 없습니다.", "triggers": []}
        _save_sent_log(sent_log)
        return {"sent": False, "reason": "오늘 이미 발송한 신호입니다.", "triggers": triggers}

    if not send_slack_message_v2("\n".join(lines)):
        return {"sent": False, "reason": "슬랙 전송에 실패했습니다.", "triggers": fresh or repeats}

    stamp = now.isoformat()
    for trigger in fresh + repeats:
        sent_log[_trigger_key(trigger)] = {
            "at": stamp,
            "action": trigger["action"],
            "ticker": trigger["ticker"],
            "name": trigger["name"],
            "round": trigger["round"],
            "index_text": f"{trigger['index_name']} {_index_text(trigger['index'])}",
        }
    # 지난 날짜 이력은 정리해 문서가 커지지 않게 한다.
    sent_log = {k: v for k, v in sent_log.items() if str(v.get("at", "")).startswith(today)}
    _save_sent_log(sent_log)

    parts = []
    if confirmed:
        parts.append(f"체결확인 {len(confirmed)}건")
    if fresh:
        parts.append(f"신호 {len(fresh)}건")
    if repeats:
        parts.append(f"재알림 {len(repeats)}건")
    return {"sent": True, "reason": " · ".join(parts) + " 발송", "triggers": fresh or repeats or confirmed}
