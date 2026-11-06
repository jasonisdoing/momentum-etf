"""전략 사고팔기 슬랙 알림.

거래시간 중 10분마다 실행돼, **매수·매도 지정가에 현재가가 닿은 회차가 있을 때만**
슬랙을 보낸다. 트리거가 없으면 아무것도 보내지 않는다. 메시지는 **전략(종목)마다 따로**
보내고, 보낼 것(신호·재알림·체결 확인)이 있는 전략에만 보낸다. 보유 0 인 전략은 장중에 1호
진입선에 못 닿았어도 15:00 이후 배치에서 1호 장 마감 매수 신호가 난다(서비스가 판정).

중복 방지 · 체결 확인
--------------------
10분 간격으로 판정하므로 같은 조건이 유지되면 하루에 수십 번 발송될 수 있다. 그래서
``전략:회차-동작`` 단위로 처음 한 번만 전체 메시지를 보내고, 신호가 살아 있는 동안에는
한 시간 간격으로 한 줄 리마인더만 보낸다 — 리마인더는 처음 신호를 보낸 메시지(``signal_ts``)의
스레드에 단다. 발송 이력은
``system_config.strategy_trade_settings.sent_log`` 에
``{"kospi200:2-buy": {"at": ..., "strategy_id": ..., "ticker": ..., "name": ..., "round": ...,
"index_text": ..., "signal_ts": ..., "held_count": ...}}`` 형태로 남기며, 발송에 성공했을 때만 갱신한다.

이력에 남은 주문이 실제로 체결되면 확인 한 줄을 보내고 이력을 지운다. 이 확인은
**신호가 없어도** 매번 검사한다 — 주문을 넣는 순간 그 회차가 보유로 바뀌어 바로 그 신호가
사라지기 때문에, 신호가 있을 때만 검사하면 확인은 영영 나가지 못한다.

체결 판정은 **보유 회차 수**로 한다. 전략마다 종목이 하나뿐이라(같은 종목을 회차로
나눠 담는다) 티커 증감으로는 어느 회차가 사고팔렸는지 알 수 없다 — 매수 N호는 보유
회차 수가 N 에 닿았는지, 매도 N호는 N 미만으로 줄었는지로 본다.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from utils.logger import get_app_logger
from utils.order_confirm import build_confirm_line, pop_filled
from utils.strategy_trade_service import (
    load_settings,
    load_views_for_pools,
    refresh_all_ledgers,
    sleeve_for_pool,
)

KST = ZoneInfo("Asia/Seoul")
logger = get_app_logger()
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
    """가격 표기(원 단위) — 없으면 '-'."""
    return f"{value:,.0f}" if value is not None else "-"


def _index_text(index: dict[str, Any]) -> str:
    """`262,870 (-2.34%)` — 종목 현재가와 일간 변동률."""
    change = index.get("change_pct")
    base = f"{index['close']:,.0f}"
    return f"{base} ({change:+.2f}%)" if change is not None else base


def _held_counts(view: dict[str, Any]) -> dict[str, int]:
    """전략별 보유 회차 수 — 전략마다 종목이 하나라 체결 확인은 회차 수 증감으로 본다."""
    return {
        strategy["strategy_id"]: int(strategy["status"]["held_count"])
        for strategy in view["strategies"]
        if "error" not in strategy
    }


def _round_filled(pending: dict[str, Any], held_count: int) -> bool:
    """대기 중인 주문이 체결됐는지 판정한다.

    매수 N호는 **보유 회차 수가 N 에 닿았는지**, 매도 N호는 **N 미만으로 줄었는지**로
    본다 — 같은 종목을 회차로 나눠 담으므로(마지막 회차부터 팔림) 회차 수가 곧 상태다.
    """
    round_no = pending.get("round")
    if not isinstance(round_no, int):
        return False
    if pending.get("action") != "buy":
        return held_count < round_no
    return held_count >= round_no


def _confirm_line(pending: dict[str, Any], index: dict[str, Any] | None) -> str:
    """`✅ 2호 KODEX 200 매수 확인 · 요청 시 … → 지금 …`."""
    now_text = _index_text(index) if index else "-"
    asked = pending.get("index_text")
    detail = f"요청 시 {asked} → 지금 {now_text}" if asked else now_text
    return build_confirm_line(f"{pending['round']}호 {pending['name']}", pending["action"], detail)


def _reach_text(trigger: dict[str, Any]) -> str:
    """신호가 난 까닭 — 지정가 도달, 또는 장중 미도달 1호의 장 마감 매수."""
    return "장 마감 매수" if trigger.get("close_entry") else f"{_level(trigger['index_level'])} 도달"


def build_repeat_line(trigger: dict[str, Any]) -> str:
    """재알림 한 줄 — 무엇을 얼마에 사고팔지만 담는다. 본문은 첫 발송에서 이미 보냈다."""
    emoji = "🔴" if trigger["action"] == "sell" else "🔵"
    word = "지금 매도!" if trigger["action"] == "sell" else "지금 매수!"
    return (
        f"<!channel> {emoji} *{trigger['round']}호 {trigger['name']} {trigger['quantity']:,}주 {word}* "
        f"{_reach_text(trigger)} "
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
                    "quantity": strategy_view["config"]["round_quantity"],
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
                    "quantity": strategy_view["config"]["round_quantity"],
                    "index_level": row["buy_index"],
                    "index_name": index["name"],
                    "index": index,
                    "profit_pct": None,
                    "close_entry": row["close_entry"],
                }
            )
    return triggers


def mark_sent_rounds(view: dict[str, Any]) -> dict[str, Any]:
    """화면용 — 회차마다 오늘 슬랙 신호가 나간 방향·발송 기록 키·슬랙 메시지 기록 유무(`slack_signals`)와
    아직 끝나지 않은 주문의 방향(`order_sides`)을 행에 적는다. 회차 원장을 못 만든 전략(`error`)은 건너뛴다.

    지정가에 닿아 있는지와 무관하다 — 신호·주문이 나간 뒤 가격이 물러나도 표시가 남아야 한다.
    체결된 주문과 이미 체결된 신호의 슬랙 기록은 더 표시하지 않는다 — 기록은 배치가 체결을 확인해 지우기까지
    최대 한 배치 주기 남으므로, 화면도 배치와 같은 판정(`_round_filled`)으로 먼저 걸러낸다.
    """
    from utils.strategy_trade_orders import active_order_keys

    today = datetime.now(KST).strftime("%Y-%m-%d")
    held_counts = _held_counts(view)
    sent_log = _load_sent_log()
    for strategy_id in held_counts:
        _keep_latest_buy(sent_log, strategy_id)
    # (전략, 회차, 구분) → 발송 기록 — 화면이 기록 키로 신호를 지운다.
    slack_keys = {
        (entry.get("strategy_id"), entry.get("round"), entry.get("action")): {
            "key": key,
            "has_ts": bool(entry.get("signal_ts")),
        }
        for key, entry in sent_log.items()
        # 이 화면에 없는 전략의 기록은 어차피 행에 닿지 않는다 — 보유 수를 모르니 판정 없이 뺀다.
        if entry.get("strategy_id") in held_counts
        and (sent_at := _parse_sent_at(entry.get("at"))) is not None
        and sent_at.strftime("%Y-%m-%d") == today
        and not _round_filled(entry, held_counts[entry["strategy_id"]])
    }
    strategy_ids = [strategy_view["strategy_id"] for strategy_view in view["strategies"]]
    order_keys = active_order_keys(strategy_ids)
    for strategy_view in view["strategies"]:
        for row in strategy_view["rounds"]:
            key = (strategy_view["strategy_id"], row["round"])
            row["slack_signals"] = [
                {"side": side, **slack_keys[(*key, side)]} for side in ("buy", "sell") if (*key, side) in slack_keys
            ]
            row["order_sides"] = [side for side in ("buy", "sell") if (*key, side) in order_keys]
    return view


def _keep_latest_buy(sent_log: dict[str, dict[str, Any]], strategy_id: str) -> None:
    """한 종목의 매수 신호 기록은 가장 최근 하나만 남긴다.

    매수 신호는 다음 회차 하나뿐이라, 기록이 둘 이상이면 이전 것은 신호 뒤에 보유가 바뀌어 무효가 된 신호다.
    매도는 지정가에 닿은 보유 회차마다 신호가 나가므로 여러 개가 정상이다.
    """
    buys = {
        key: entry
        for key, entry in sent_log.items()
        if entry.get("strategy_id") == strategy_id and entry.get("action") == "buy"
    }
    if len(buys) <= 1:
        return
    newest = max(buys, key=lambda key: str(buys[key].get("at")))
    for key in buys:
        if key != newest:
            del sent_log[key]


def _already_sent_note(round_no: int, action: str, fresh: set[tuple[int, str]]) -> str:
    """지정가에 닿아 있지만 이번에 새로 보내는 신호가 아닌 회차 — 앞서 알린 신호임을 줄에 덧붙인다."""
    return "" if (round_no, action) in fresh else " · 앞서 알림"


def build_message(strategy_view: dict[str, Any], triggers: list[dict[str, Any]]) -> str:
    """슬랙 본문 — **전략 하나**의 전 회차 현황을 나열하고 조치가 필요한 줄만 표시한다.

    종목마다 메시지를 따로 보낸다(필요한 종목만 보내고, 이후 종목별 주문 버튼을 붙이기 쉽게).
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

    index = strategy_view["index"]
    status = strategy_view["status"]
    rounds_total = strategy_view["config"]["rounds"]
    quantity = f"{strategy_view['config']['round_quantity']:,}주"

    triggered = {(t["round"], t["action"]) for t in triggers}
    buy_count = sum(1 for t in triggers if t["action"] == "buy")
    sell_count = sum(1 for t in triggers if t["action"] == "sell")
    after_held = status["held_count"] + buy_count - sell_count

    # 가장 중요한 정보 — 몇 % 움직이면 사고/파는지 헤더에 바로 보여준다.
    parts = []
    drop = status.get("next_buy_drop_pct")
    next_row = next((row for row in strategy_view["rounds"] if row["is_next"]), None)
    if drop is None:
        parts.append("매수 없음(회차 소진)")
    elif status["limit_blocked"]:
        parts.append(f"⏸ {status['next_round']}호 한도 도달 — 추가 매수 중단")
    elif next_row is not None and next_row["close_entry"]:
        parts.append(f"📉 *{status['next_round']}호 장 마감 매수!*")
    elif drop >= 0 and status["sell_pending"]:
        parts.append(f"📉 {status['next_round']}호 매수가 도달 — {status['next_sell_round']}호 매도 체결 뒤 매수")
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
        f"*[{strategy_view['label']}]* {_index_text(index)} · {after_held}/{rounds_total}회차 · " + " · ".join(parts)
    )
    # 1~6호 전부 나열 — 상태를 이모지로 즉시 구분한다.
    #   🔴 지금 팔 것 / 🔵 지금 살 것 / 🟢 보유 중 / ⏳ 다음 매수 대기 / ⚪ 이후 회차
    # 회차별 줄 — 종목 가격이 아니라 **지수 환산값**으로 적는다. ETF 단가는 종목마다
    # 달라 서로 비교가 안 되고, 판단은 결국 지수가 어디까지 왔느냐로 하기 때문이다.
    for row in strategy_view["rounds"]:
        if row["held"]:
            profit_text = f"{row['profit_pct']:+.2f}%" if row["profit_pct"] is not None else "-"
            if row["sell_reached"]:
                lines.append(
                    f"🔴 {row['round']}호 {quantity} *지금 매도!* {_level(row['sell_index'])} 도달 ({profit_text})"
                    f"{_already_sent_note(row['round'], 'sell', triggered)}"
                )
            else:
                lines.append(
                    f"🟢 {row['round']}호 {quantity} 보유 중 ({profit_text}) · 매도 목표 {_level(row['sell_index'])}"
                )
        elif row["is_next"] and row["over_limit"]:
            lines.append(f"⏸ {row['round']}호 {quantity} 한도 도달 — 추가 매수 중단")
        elif row["is_next"]:
            if row["buy_reached"]:
                reach = (
                    f"장 마감 매수 · 현재가 {_level(row['close'])}"
                    if row["close_entry"]
                    else f"{_level(row['buy_index'])} 도달"
                )
                lines.append(
                    f"🔵 {row['round']}호 {quantity} *지금 매수!* {reach}"
                    f"{_already_sent_note(row['round'], 'buy', triggered)}"
                )
            else:
                # 보유 0 이면 장중에 못 닿아도 장 마감 매수가 있다.
                close_note = (
                    f" · 미도달 시 {status['first_entry_close_time']} 매수" if status["waiting_first_entry"] else ""
                )
                lines.append(f"⏳ {row['round']}호 {quantity} 다음 매수 대기 · {_level(row['buy_index'])}{close_note}")
        else:
            lines.append(f"⚪ {row['round']}호 {quantity} 이후 회차 · 매수 예정 {_level(row['buy_index'])}")

    return "\n".join(lines)


def _pools_with_strategies() -> list[str]:
    return sorted({entry["pool"] for entry in load_settings()["strategies"].values()})


def send_test(account_id: str) -> dict[str, Any]:
    """그 계좌에 붙은 사고팔기 슬리브의 현재 상태를 즉시 보낸다(합성 화면의 슬랙 테스트에서 부른다).

    스위치·중복 방지와 무관하고 상태는 건드리지 않는다. 신호가 없으면 현황만 보낸다.
    """
    from utils.strategy_trade_threads import send_thread_message

    pools = [
        pool
        for pool in _pools_with_strategies()
        if (sleeve := sleeve_for_pool(pool)) is not None and sleeve["account_id"] == account_id
    ]
    if not pools:
        return {"sent": False, "reason": "이 계좌에 사고팔기 슬리브가 없습니다.", "triggers": []}
    view = load_views_for_pools(pools, fresh=False)
    triggers: list[dict[str, Any]] = []
    sent = bool(view["strategies"])
    for strategy_view in view["strategies"]:
        strategy_triggers = collect_triggers(strategy_view)
        triggers.extend(strategy_triggers)
        sent = bool(send_thread_message(build_message(strategy_view, strategy_triggers))) and sent
    return {
        "sent": sent,
        "reason": "테스트 발송" if sent else "슬랙 전송에 실패했습니다.",
        "triggers": triggers,
    }


def _send_order_messages(strategy_view: dict[str, Any], triggers: list[dict[str, Any]], signal_ts: str) -> int:
    """신호 메시지(signal_ts)에 실린 신호의 회차별 주문 메시지를 그 스레드로 보낸다(주문이 이미 있는 회차는 제외) — 신호 메시지는 이미 나갔으므로 실패해도 신호는 유지하고 실패 건수를 돌려준다."""
    from utils.slack_mock_orders import send_signal_order_messages

    if not triggers:
        return 0
    try:
        send_signal_order_messages(strategy_view, triggers, signal_ts)
    except Exception:
        logger.exception("사고팔기 주문 메시지 발송 실패: %s", strategy_view["strategy_id"])
        return 1
    return 0


def _log_entry(trigger: dict[str, Any], held_counts: dict[str, int], stamp: str, signal_ts: str) -> dict[str, Any]:
    """발송 이력 한 건 — signal_ts 는 이 신호를 처음 알린 메시지(재알림이 달리는 스레드의 뿌리)다."""
    return {
        "at": stamp,
        "action": trigger["action"],
        "strategy_id": trigger["strategy_id"],
        "ticker": trigger["ticker"],
        "name": trigger["name"],
        "round": trigger["round"],
        "index_text": _index_text(trigger["index"]),
        "signal_ts": signal_ts,
        # 발송 시점의 보유 회차 수 — 체결 확인은 회차 수 증감으로 판정한다(참고 기록).
        "held_count": held_counts.get(trigger["strategy_id"], 0),
    }


def _send_confirms(
    strategy_view: dict[str, Any], pendings: list[dict[str, Any]], sent_log: dict[str, dict[str, Any]]
) -> tuple[int, int]:
    """체결 확인을 신호를 알린 메시지의 스레드로 보낸다 — (보낸 메시지 수, 실패 수). 못 보낸 확인은 기록에 되돌린다."""
    from utils.strategy_trade_threads import send_thread_message

    groups: dict[str, list[dict[str, Any]]] = {}
    for pending in pendings:
        groups.setdefault(pending["signal_ts"], []).append(pending)
    sent = failed = 0
    for root, group in groups.items():
        text = "\n".join(_confirm_line(pending, strategy_view["index"]) for pending in group)
        if send_thread_message(text, thread_ts=root):
            sent += 1
        else:
            failed += 1
            sent_log.update({_trigger_key(pending): pending for pending in group})
    return sent, failed


def _send_repeats(
    strategy_view: dict[str, Any],
    repeats: list[dict[str, Any]],
    sent_log: dict[str, dict[str, Any]],
    held_counts: dict[str, int],
    stamp: str,
) -> tuple[int, int, int, list[dict[str, Any]]]:
    """재알림을 신호를 처음 알린 메시지의 스레드로 보낸다 — (보낸 메시지 수, 실패 수, 주문 메시지 실패 수, 보낸 신호).

    뿌리 메시지가 기록에 없는 옛 이력은 채널에 새로 올리고 그 메시지를 뿌리로 삼는다.
    """
    from utils.strategy_trade_threads import send_thread_message

    groups: dict[str | None, list[dict[str, Any]]] = {}
    for trigger in repeats:
        groups.setdefault(sent_log[_trigger_key(trigger)].get("signal_ts"), []).append(trigger)
    sent = failed = order_failed = 0
    delivered: list[dict[str, Any]] = []
    for root, group in groups.items():
        ts = send_thread_message("\n".join(build_repeat_line(trigger) for trigger in group), thread_ts=root)
        if not ts:
            failed += 1
            continue
        sent += 1
        thread_ts = root or ts
        order_failed += _send_order_messages(strategy_view, group, thread_ts)
        for trigger in group:
            sent_log[_trigger_key(trigger)] = _log_entry(trigger, held_counts, stamp, thread_ts)
        delivered.extend(group)
    return sent, failed, order_failed, delivered


def notify_strategy_trade() -> dict[str, Any]:
    """트리거를 판정해 필요할 때만 슬랙을 보낸다 — 슬리브가 붙은 계좌의 합성 슬랙이 켜진 종목풀만.

    Returns:
        ``{"sent": bool, "reason": str, "triggers": [...]}``
    """
    from utils.strategy_trade_orders import active_order_keys
    from utils.strategy_trade_threads import send_thread_message

    # 화면이 읽는 회차 원장을 슬랙 스위치와 무관하게 먼저 새로 계산해 저장한다 — 아래 판정도 이 저장본을 쓴다.
    for ticker, message in refresh_all_ledgers().items():
        logger.error("사고팔기 회차 원장 실패(%s) — 이 종목은 건너뜁니다: %s", ticker, message)

    enabled = [
        pool
        for pool in _pools_with_strategies()
        if (sleeve := sleeve_for_pool(pool)) is not None and sleeve["slack_enabled"]
    ]
    if not enabled:
        return {"sent": False, "reason": "슬랙이 켜진 사고팔기 슬리브가 없습니다.", "triggers": []}

    view = load_views_for_pools(enabled, fresh=False)
    triggers = [t for s in view["strategies"] for t in collect_triggers(s)]

    now = datetime.now(KST)
    today = now.strftime("%Y-%m-%d")
    sent_log = _load_sent_log()

    # 처음 보는 신호는 전 회차 현황이 담긴 본문으로, 이미 보낸 신호는 아직 살아 있는 동안
    # 한 시간마다 한 줄 리마인더로 보낸다. 지정가에 닿아 있는데 주문을 안 넣은 채로
    # 하루가 지나가는 것을 막기 위한 것이다. 메시지는 **전략(종목)마다 따로**, 보낼 것이 있는
    # 전략에만 보낸다.
    # ① 요청해 둔 주문이 체결됐는지 먼저 본다. 체결됐으면 확인 한 줄을 보내고 기록을 지운다 —
    #    지워야 다음에 같은 회차 신호가 나면 다시 전체 메시지로 나간다.
    #    이 검사는 트리거 유무와 무관하게 돌아야 한다: 주문을 넣는 순간 그 회차가 보유로 바뀌어
    #    바로 그 트리거가 사라지므로, 신호가 있을 때만 검사하면 확인은 영영 나가지 못한다.
    #    지금 전략 목록에 없는 전략의 기록은 체결 여부를 알 수 없어 건드리지 않는다.
    held_counts = _held_counts(view)
    confirmed = pop_filled(
        sent_log,
        lambda pending: str(pending.get("strategy_id")) in held_counts
        and _round_filled(pending, held_counts[str(pending.get("strategy_id"))]),
    )

    # 주문이 이미 진행 중인 회차는 재알림하지 않는다 — 사람이 이미 처리 중이다.
    in_flight = (
        active_order_keys([strategy_view["strategy_id"] for strategy_view in view["strategies"]]) if triggers else set()
    )
    sent_count = failed_count = order_failed = 0
    fresh_all: list[dict[str, Any]] = []
    repeats_all: list[dict[str, Any]] = []
    stamp = now.isoformat()
    for strategy_view in view["strategies"]:
        sid = strategy_view["strategy_id"]
        # ② 남은 신호 — 처음이면 전체 메시지, 이미 보냈으면 한 시간 간격으로 재알림.
        fresh: list[dict[str, Any]] = []
        repeats: list[dict[str, Any]] = []
        for trigger in (t for t in triggers if t["strategy_id"] == sid):
            last = _parse_sent_at((sent_log.get(_trigger_key(trigger)) or {}).get("at"))
            if last is None or last.strftime("%Y-%m-%d") != today:
                fresh.append(trigger)
            elif (now - last).total_seconds() >= _REPEAT_INTERVAL_SECONDS and (
                trigger["strategy_id"],
                trigger["round"],
                trigger["action"],
            ) not in in_flight:
                repeats.append(trigger)

        own_confirmed = [pending for pending in confirmed if str(pending.get("strategy_id")) == sid]
        # 신호 메시지가 기록된 확인은 그 스레드로, 없는 확인(옛 기록)은 새 신호 본문과 함께 채널로 보낸다.
        threaded = [pending for pending in own_confirmed if pending.get("signal_ts")]
        own_confirmed = [pending for pending in own_confirmed if not pending.get("signal_ts")]
        sent, failed = _send_confirms(strategy_view, threaded, sent_log)
        sent_count += sent
        failed_count += failed
        lines = [_confirm_line(pending, strategy_view["index"]) for pending in own_confirmed]
        if fresh:
            lines.append(build_message(strategy_view, fresh))
        if lines:
            signal_ts = send_thread_message("\n".join(lines))
            if not signal_ts:
                failed_count += 1
                # 못 보낸 확인은 기록에 되돌려 다음 배치에서 다시 보낸다.
                sent_log.update({_trigger_key(pending): pending for pending in own_confirmed})
                continue
            sent_count += 1
            order_failed += _send_order_messages(strategy_view, fresh, signal_ts)
            fresh_all.extend(fresh)
            for trigger in fresh:
                sent_log[_trigger_key(trigger)] = _log_entry(trigger, held_counts, stamp, signal_ts)
        sent, failed, order_failed_here, delivered = _send_repeats(strategy_view, repeats, sent_log, held_counts, stamp)
        sent_count += sent
        failed_count += failed
        order_failed += order_failed_here
        repeats_all.extend(delivered)
        _keep_latest_buy(sent_log, sid)

    if not sent_count and not failed_count:
        if not triggers and not confirmed:
            # 확인할 체결도 신호도 없다 — 10분마다 도는 배치라 여기서 DB 를 건드리면 무의미한 쓰기만 쌓인다.
            return {"sent": False, "reason": "매수·매도 신호가 없습니다.", "triggers": []}
        _save_sent_log(sent_log)
        return {"sent": False, "reason": "오늘 이미 발송한 신호입니다.", "triggers": triggers}

    # 지난 날짜 이력은 정리해 문서가 커지지 않게 한다.
    sent_log = {k: v for k, v in sent_log.items() if str(v.get("at", "")).startswith(today)}
    _save_sent_log(sent_log)

    parts = [f"메시지 {sent_count}건 발송"]
    if confirmed:
        parts.append(f"체결확인 {len(confirmed)}건")
    if fresh_all:
        parts.append(f"신호 {len(fresh_all)}건")
    if repeats_all:
        parts.append(f"재알림 {len(repeats_all)}건")
    if failed_count:
        parts.append(f"슬랙 전송 실패 {failed_count}건")
    if order_failed:
        parts.append(f"주문 메시지 실패 {order_failed}건")
    return {"sent": sent_count > 0, "reason": " · ".join(parts), "triggers": fresh_all or repeats_all or triggers}
