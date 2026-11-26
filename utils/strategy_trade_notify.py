"""사고팔기 자동·수동 Slack 신호의 공통 생명주기.

자동 신호는 거래 시간과 가격 조건을 따르고 세션·가격 복귀 때 미사용 메시지를 회수한다.
수동 신호는 직접 회수 전까지 유지한다. 발송·회수는 DB 선점으로 직렬화하고,
발송 전에 예약을 기록해 응답 불명 상태의 재전송을 막는다.
체결은 주문 기록과 원장 체결 순서로 판정하며 완료된 메시지의 재주문을 금지한다.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from config import INSTRUMENT_TRADING_WINDOWS
from utils.logger import get_app_logger
from utils.market_session import is_instrument_trading_time
from utils.order_confirm import build_confirm_line, pop_filled
from utils.strategy_trade_market import format_level, format_money
from utils.strategy_trade_service import load_settings, load_views_where, refresh_all_ledgers
from utils.strategy_trade_signal_lock import SignalBusyError, batch_signal_lock, signal_lock

KST = ZoneInfo("Asia/Seoul")
logger = get_app_logger()
_CONFIG_COLLECTION = "system_config"
_SETTINGS_KEY = "strategy_trade_settings"


class _SentLog(dict):
    """조회한 원본과 비교해 이 작업에서 바꾼 항목만 저장한다."""

    def __init__(self, entries):
        super().__init__(entries)
        self.saved = deepcopy(entries)


def _load_sent_log() -> _SentLog:
    from utils.db_manager import get_db_connection

    db = get_db_connection()
    if db is None:
        return _SentLog({})
    doc = db[_CONFIG_COLLECTION].find_one({"_id": _SETTINGS_KEY}) or {}
    stored = doc.get("sent_log")
    if not isinstance(stored, dict):
        return _SentLog({})
    # 옛 형식(값이 날짜 문자열)은 확인 판정에 쓸 티커가 없어 버린다 — 하루면 자연히 정리된다.
    return _SentLog({str(k): v for k, v in stored.items() if isinstance(v, dict)})


def _save_sent_log(sent_log: _SentLog) -> None:
    """다른 종목이 동시에 저장해도 그 발송 기록을 덮어쓰지 않는다."""
    from utils.db_manager import get_db_connection

    changed = {f"sent_log.{key}": value for key, value in sent_log.items() if sent_log.saved.get(key) != value}
    removed = {f"sent_log.{key}": "" for key in sent_log.saved if key not in sent_log}
    if not changed and not removed:
        return
    db = get_db_connection()
    if db is None:
        raise RuntimeError("DB 연결에 실패했습니다.")
    update = {"$set": {**changed, "sent_log_updated_at": datetime.now(KST).isoformat()}}
    if removed:
        update["$unset"] = removed
    db[_CONFIG_COLLECTION].update_one({"_id": _SETTINGS_KEY}, update, upsert=True)
    sent_log.saved = deepcopy(dict(sent_log))


_WITHDRAWAL_GAP_FRACTION = 0.20


def _withdrawal_price(base: float, gap_pct: float, *, buy: bool) -> float:
    distance = base * gap_pct / 100 * (1 - _WITHDRAWAL_GAP_FRACTION)
    return round(base - distance if buy else base + distance, 6)


def _trigger_key(trigger: dict[str, Any]) -> str:
    target = trigger.get("arbitrary_id") or trigger["round"]
    return f"{trigger['strategy_id']}:{target}-{trigger['action']}"


def _matching_signal(sent_log: dict[str, dict[str, Any]], trigger: dict[str, Any]) -> dict[str, Any] | None:
    """전략 ID가 바뀌어도 계좌·종목·회차·방향이 같으면 활성 신호 하나만 허용한다."""
    for entry in sent_log.values():
        same_target = entry.get("strategy_id") == trigger["strategy_id"] or (
            entry.get("account_id") is not None
            and entry["account_id"] == trigger.get("account_id")
            and entry["ticker"] == trigger["ticker"]
        )
        if (
            same_target
            and entry["round"] == trigger["round"]
            and entry["action"] == trigger["action"]
            and entry.get("arbitrary_id") == trigger.get("arbitrary_id")
        ):
            return entry
    return None


def _level(value: float | None, country: str) -> str:
    """가격 표기 — 국장은 원 단위 정수, 미장은 `$` 붙인 달러 소수 둘째 자리(Slack 볼드). 없으면 '-'."""
    text = format_level(value, country)
    if text == "-":
        return "-"
    formatted = text if country == "kor" else f"${text}"
    return f"*{formatted}*"


def _index_text(index: dict[str, Any], country: str) -> str:
    """`*262,870* (-2.34%)` — 종목 현재가와 일간 변동률."""
    change = index.get("change_pct")
    base = _level(index["close"], country)
    return f"{base} ({change:+.2f}%)" if change is not None else base


def _held_counts(view: dict[str, Any]) -> dict[str, int]:
    """전략별 보유 회차 수 — 전략마다 종목이 하나라 체결 확인은 회차 수 증감으로 본다."""
    return {
        strategy["strategy_id"]: int(strategy["status"]["held_count"])
        for strategy in view["strategies"]
        if "error" not in strategy
    }


def _held_fills(view: dict[str, Any]) -> dict[str, dict[int, int]]:
    """전략별 보유 회차의 체결가 {회차: 체결가} — 매도 후 재매수의 체결 확인에 쓴다."""
    return {
        strategy["strategy_id"]: {row["round"]: row["fill_price"] for row in strategy["rounds"] if row["held"]}
        for strategy in view["strategies"]
        if "error" not in strategy
    }


def _round_filled(pending: dict[str, Any], held_count: int, fills: dict[int, int]) -> bool:
    """대기 중인 주문이 체결됐는지 판정한다.

    매수 N호는 **보유 회차 수가 N 에 닿았는지**, 매도 N호는 **N 미만으로 줄었는지**로
    본다 — 같은 종목을 회차로 나눠 담으므로(마지막 회차부터 팔림) 회차 수가 곧 상태다.
    매도 후 재매수(``rebuy``)는 둘 다 체결되면 회차 수가 그대로라, 그 회차의 체결가가 신호 때와 달라졌는지로 본다.
    """
    round_no = pending.get("round")
    if not isinstance(round_no, int):
        return False
    if pending.get("action") != "buy":
        if held_count < round_no:
            return True
        return bool(pending.get("rebuy")) and fills.get(round_no) != pending.get("fill_price")
    return held_count >= round_no


def _entry_expired(pending: dict[str, Any]) -> bool:
    """만료된 주문 신호만 끝낸다. 미발주·취소·접수 불명·진행 중 주문은 보존한다."""
    from utils.strategy_trade_orders import ACTIVE_STATUSES, _orders
    from utils.strategy_trade_threads import _messages

    root = pending.get("signal_ts")
    if not root:
        return False
    if _messages().find_one({"thread_ts": root, "ts": root, "signal_expired": True}):
        return True
    query = {"slack.thread_ts": root, "round": pending["round"]}
    if not pending.get("rebuy"):
        query["side"] = pending["action"]
    if pending.get("arbitrary_id") is not None:
        query["arbitrary_id"] = pending["arbitrary_id"]
    records = list(_orders().find(query))
    return any(record["status"] == "expired" for record in records) and not any(
        record["status"] in ACTIVE_STATUSES for record in records
    )


def retire_expired_signals(sent_log: _SentLog, target: tuple[str, str]) -> int:
    """예약한 종목의 만료 신호만 활성 기록에서 빼고 원본 메시지·주문 이력은 보존한다."""
    from utils.strategy_trade_threads import expire_signal_thread

    keys = [
        key
        for key, entry in sent_log.items()
        if (entry["account_id"], entry["ticker"]) == target and _entry_expired(entry)
    ]
    roots = {sent_log[key]["signal_ts"] for key in keys}
    for key in keys:
        del sent_log[key]
    remaining = {entry.get("signal_ts") for entry in sent_log.values()}
    for root in roots - remaining:
        expire_signal_thread(root)
    return len(keys)


def _entry_filled(
    pending: dict[str, Any], held_count: int, fills: dict[int, int], events: list[dict[str, Any]]
) -> bool:
    """부분 체결은 완료가 아니다. 주문 기록이 없을 때만 공통 회차 판정을 쓴다."""
    from utils.strategy_trade_orders import ACTIVE_STATUSES, _orders
    from utils.strategy_trade_threads import _messages

    root = pending.get("signal_ts")
    if root:
        if _messages().find_one({"thread_ts": root, "ts": root, "signal_completed": True}):
            return True
        query = {"slack.thread_ts": root, "round": pending["round"]}
        if not pending.get("rebuy"):
            query["side"] = pending["action"]
        orders = list(_orders().find(query))
        if any(o["status"] in ACTIVE_STATUSES for o in orders):
            return False
        if any(o["status"] == "filled" for o in orders):
            return True
    # 이전 형식의 기록은 회차 상태로 판정하고 새 신호는 주문번호로 사이클을 구분한다.
    if "fill_cursor" in pending and not pending.get("rebuy") and not pending.get("arbitrary_id"):
        filled = sum(
            event["quantity"]
            for event in events
            if event["cursor"] > pending["fill_cursor"] and event["buy"] == (pending["action"] == "buy")
        )
        if filled >= (1 if pending["action"] == "buy" else pending["quantity"]):
            return True
    return _round_filled(pending, held_count, fills)


def _confirm_line(pending: dict[str, Any], index: dict[str, Any] | None, country: str) -> str:
    """`✅ 2호 KODEX 200 매수 확인 · 요청 시 … → 지금 …`."""
    now_text = _index_text(index, country) if index else "-"
    asked = pending.get("index_text")
    detail = f"요청 시 {asked} → 지금 {now_text}" if asked else now_text
    if pending.get("arbitrary_id"):
        return f"✅ {pending['name']} 임의 매도 확인 · {detail}"
    return build_confirm_line(f"{pending['round']}호 {pending['name']}", pending["action"], detail)


def collect_triggers(strategy_view: dict[str, Any]) -> list[dict[str, Any]]:
    """전략 하나에서 지정가에 닿은 회차를 모은다. 없으면 빈 리스트."""
    sid = strategy_view["strategy_id"]
    label = strategy_view["label"]
    country = strategy_view["country"]
    index = strategy_view["index"]
    triggers: list[dict[str, Any]] = []
    for row in strategy_view["rounds"]:
        if row["held"] and row["sell_reached"]:
            triggers.append(
                {
                    "strategy_id": sid,
                    "account_id": strategy_view["account_id"],
                    "strategy_label": label,
                    "country": country,
                    "action": "sell",
                    "round": row["round"],
                    "ticker": row["ticker"],
                    "name": row["name"],
                    "limit_price": row["sell_limit"],
                    "close": row["close"],
                    "quantity": row["quantity"],
                    "index_level": row["sell_index"],
                    "index_name": index["name"],
                    "index": index,
                    "profit_pct": row["profit_pct"],
                    # 팔면 바로 다시 사게 되는 회차 — 매도 후 재매수 한 짝으로 낸다.
                    "rebuy": row["rebuy_after_sell"],
                    "fill_price": row["fill_price"],
                }
            )
        # 매수는 '다음 진입 대상' 회차만 의미가 있다(회차는 순차 진행).
        elif row["is_next"] and row["buy_reached"]:
            triggers.append(
                {
                    "strategy_id": sid,
                    "account_id": strategy_view["account_id"],
                    "strategy_label": label,
                    "country": country,
                    "action": "buy",
                    "round": row["round"],
                    "ticker": row["ticker"],
                    "name": row["name"],
                    "limit_price": row["buy_limit"],
                    "close": row["close"],
                    "quantity": row["quantity"],
                    "index_level": row["buy_index"],
                    "index_name": index["name"],
                    "index": index,
                    "profit_pct": None,
                    "close_entry": row["close_entry"],
                    # 정해진 시각에 사는 1호의 그 시각(한국시간) — 신호 문구에 쓴다.
                    "entry_time": strategy_view["status"]["first_entry_time"],
                    "rebuy": False,
                    "fill_price": None,
                }
            )
    triggers.extend(
        {
            "strategy_id": sid,
            "account_id": strategy_view["account_id"],
            "strategy_label": label,
            "country": country,
            "ticker": strategy_view["ticker"],
            "name": label,
            "action": "sell",
            "round": None,
            "arbitrary_id": lot["id"],
            "quantity": lot["quantity"],
            "limit_price": lot["sell_limit"],
            "index_level": lot["sell_limit"],
            "close": index["close"],
            "index": index,
            "index_name": index["name"],
            "profit_pct": (index["close"] / lot["fill_price"] - 1) * 100,
            "rebuy": False,
            "fill_price": lot["fill_price"],
        }
        for lot in strategy_view["arbitrary_lots"]
        if index["close"] >= lot["sell_limit"]
    )
    for trigger in triggers:
        trigger["fill_cursor"] = max((event["cursor"] for event in strategy_view["fill_events"]), default=["", 0])
    return triggers


def mark_sent_rounds(view: dict[str, Any]) -> dict[str, Any]:
    """화면용 — 회차마다 활성 슬랙 신호의 방향·발송 기록 키·회수 가능 여부(`slack_signals`)와
    아직 끝나지 않은 주문의 방향(`order_sides`)을 행에 적는다. 회차 원장을 못 만든 전략(`error`)은 건너뛴다.

    지정가에 닿아 있는지와 무관하다 — 신호·주문이 나간 뒤 가격이 물러나도 표시가 남아야 한다.
    활성 상태는 배치와 같은 체결 판정으로 걸러낸다.
    """
    from utils.strategy_trade_orders import active_order_keys
    from utils.strategy_trade_threads import can_withdraw_signal

    held_counts = _held_counts(view)
    held_fills = _held_fills(view)
    events = {s["strategy_id"]: s["fill_events"] for s in view["strategies"] if "error" not in s}
    sent_log = _load_sent_log()
    for strategy_id in held_counts:
        _keep_latest_buy(sent_log, strategy_id)
    # (전략, 회차, 구분) → 발송 기록 — 화면이 기록 키로 신호를 회수한다.
    slack_keys = {}
    for key, entry in sent_log.items():
        sid = entry.get("strategy_id")
        if sid not in held_counts or entry.get("session_closed"):
            continue
        if _entry_expired(entry) or _entry_filled(entry, held_counts[sid], held_fills[sid], events[sid]):
            continue
        signal_ts = entry.get("signal_ts")
        slack_keys[(sid, entry["round"], entry["action"])] = {
            "key": key,
            "thread_ts": signal_ts,
            "can_withdraw": bool(signal_ts and can_withdraw_signal(signal_ts)),
            "manual": entry.get("origin") == "manual",
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
        if entry.get("strategy_id") == strategy_id and entry.get("action") == "buy" and entry.get("origin") != "manual"
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
    country = strategy_view["country"]

    # 실제 신호와 테스트 발송을 상단에서 바로 구분한다.
    # 실제 신호는 주문이 필요하므로 <!channel> 로 채널 전체에 알린다(테스트는 제외).
    header_line: str | None = None
    if triggers:
        from utils.slack_mock_orders import AUTO_ORDER_TAG, is_auto_trigger

        # 자동 주문 전략의 자동 신호는 머리말로 수동 신호와 구분한다.
        auto = all(is_auto_trigger(strategy_view, trigger) for trigger in triggers)
        head = f"{AUTO_ORDER_TAG}전략 사고팔기 — 실제 신호" if auto else "⚡ 전략 사고팔기 — 실제 신호"
        header_line = f"*{head}* ({now}) · {strategy_view['account_name']}"
        lines = []
    else:
        lines = [
            f"*🧪 전략 사고팔기 테스트* ({now}) · {strategy_view['account_name']}",
            "현재 매수·매도 신호는 없습니다.",
        ]

    index = strategy_view["index"]
    status = strategy_view["status"]
    rounds_total = strategy_view["config"]["rounds"]

    triggered = {(t["round"], t["action"]) for t in triggers}
    buy_count = sum(1 for t in triggers if t["action"] == "buy")
    # 매도 후 재매수는 한 회차를 다시 채운다.
    sell_count = sum(1 for t in triggers if t["action"] == "sell" and not t["rebuy"] and t["round"] is not None)
    after_held = status["held_count"] + buy_count - sell_count

    # 가장 중요한 정보 — 몇 % 움직이면 사고/파는지 헤더에 바로 보여준다.
    parts = []
    drop = status.get("next_buy_drop_pct")
    next_row = next((row for row in strategy_view["rounds"] if row["is_next"]), None)
    if status["next_round"] is None:
        parts.append("매수 없음(회차 소진)")
    elif status["budget_short"]:
        parts.append(f"⏸ {status['next_round']}호 회차 금액으로 1주 매수 불가")
    elif status["cash_short"]:
        parts.append(f"⏸ {status['next_round']}호 한도 소진 — 현금 부족")
    elif next_row is not None and next_row["close_entry"]:
        parts.append(f"📉 *{status['next_round']}호 {status['first_entry_time']} 매수!*")
    elif status["waiting_first_entry"]:
        parts.append(f"⏳ 1호 {status['first_entry_time']} 매수 예정")
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
    summary_line = (
        f"*[{strategy_view['label']}]* {_index_text(index, country)} · {after_held}/{rounds_total}회차 · "
        + " · ".join(parts)
    )
    if header_line is not None:
        # 모바일 푸시 알림 앞부분에 종목·가격·도달 요약이 먼저 보이도록 종목 요약 줄을 첫 줄에 둔다.
        lines.extend([f"<!channel> {summary_line}", header_line])
    else:
        lines.append(summary_line)
    for lot in strategy_view["arbitrary_lots"]:
        word = "*임의 매도*" if any(t.get("arbitrary_id") == lot["id"] for t in triggers) else "임의 매수"
        lines.append(
            f"• {word} · {lot['quantity']:,}주 · 매수 {_level(lot['fill_price'], country)} · 매도 목표 {_level(lot['sell_limit'], country)}"
        )

    # 1~6호 전부 나열 — 상태를 이모지로 즉시 구분한다.
    #   🔴 지금 팔 것 / 🔵 지금 살 것 / 🟢 보유 중 / ⏳ 다음 매수 대기 / ⚪ 이후 회차
    # 회차별 줄 — 종목 가격이 아니라 **지수 환산값**으로 적는다. ETF 단가는 종목마다
    # 달라 서로 비교가 안 되고, 판단은 결국 지수가 어디까지 왔느냐로 하기 때문이다.
    def plan_text(row: dict[str, Any]) -> str:
        """매수 예정 — 보유가 없으면 회차 가격이 없어 다음 1호 매수 시각(한국시간)을 적는다."""
        if row["buy_index"] is None:
            return f"{status['first_entry_time']} 매수 예정"
        return f"매수 예정 {_level(row['buy_index'], country)}"

    for row in strategy_view["rounds"]:
        quantity = f"{row['quantity']:,}주"
        if row["held"]:
            profit_text = f"{row['profit_pct']:+.2f}%" if row["profit_pct"] is not None else "-"
            buy_text = f" · 매수 {_level(row['fill_price'], country)}" if row.get("fill_price") is not None else ""
            if row["sell_reached"]:
                word = "지금 매도 후 재매수!" if row["rebuy_after_sell"] else "지금 매도!"
                lines.append(
                    f"🔴 {row['round']}호 {quantity} *{word}* {_level(row['sell_index'], country)} 도달"
                    f"{buy_text} ({profit_text}){_already_sent_note(row['round'], 'sell', triggered)}"
                )
            else:
                lines.append(
                    f"🟢 {row['round']}호 {quantity} 보유 중{buy_text} ({profit_text}) · 매도 목표 {_level(row['sell_index'], country)}"
                )
        elif row["budget_short"]:
            lines.append(f"⏸ {row['round']}호 회차 금액으로 1주 매수 불가 · {plan_text(row)}")
        elif row["is_next"] and row["cash_short"]:
            lines.append(f"⏸ {row['round']}호 {quantity} 한도 소진 — 현금 부족 · {plan_text(row)}")
        elif row["is_next"]:
            if row["buy_reached"]:
                reach = (
                    f"{status['first_entry_time']} 매수 · 현재가 {_level(row['close'], country)}"
                    if row["close_entry"]
                    else f"{_level(row['buy_index'], country)} 도달"
                )
                lines.append(
                    f"🔵 {row['round']}호 {quantity} *지금 매수!* {reach}"
                    f"{_already_sent_note(row['round'], 'buy', triggered)}"
                )
            else:
                lines.append(f"⏳ {row['round']}호 {quantity} 다음 매수 대기 · {plan_text(row)}")
        else:
            after = f" · {plan_text(row)}" if row["buy_index"] is not None else ""
            lines.append(f"⚪ {row['round']}호 {quantity} 이후 회차{after}")

    return "\n".join(lines)


def send_test(account_id: str) -> dict[str, Any]:
    """그 계좌를 고른 사고팔기 전략의 현재 상태를 즉시 보낸다(사고팔기 화면의 슬랙 테스트 버튼).

    스위치·중복 방지와 무관하고 상태는 건드리지 않는다. 신호가 없으면 현황만 보낸다.
    """
    from utils.strategy_trade_threads import send_thread_message

    view = load_views_where(lambda entry: entry["account_id"] == account_id, fresh=False)
    if not view["strategies"] and not view["errors"]:
        return {"sent": False, "reason": "이 계좌를 고른 사고팔기 전략이 없습니다.", "triggers": []}
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
    """발송 이력 한 건 — signal_ts 는 이 신호를 처음 알린 메시지(체결 확인이 달리는 스레드의 뿌리)다."""
    return {
        "at": stamp,
        "origin": trigger.get("origin", "auto"),
        "fill_cursor": trigger["fill_cursor"],
        "account_id": trigger.get("account_id"),
        "action": trigger["action"],
        "strategy_id": trigger["strategy_id"],
        "ticker": trigger["ticker"],
        "name": trigger["name"],
        "round": trigger["round"],
        "arbitrary_id": trigger.get("arbitrary_id"),
        "quantity": trigger["quantity"],
        "limit_price": trigger["limit_price"],
        "index_text": _index_text(trigger["index"], trigger["country"]),
        "signal_ts": signal_ts,
        # 발송 시점의 보유 회차 수 — 체결 확인은 회차 수 증감으로 판정한다(참고 기록).
        "held_count": held_counts.get(trigger["strategy_id"], 0),
        # 매도 후 재매수 — 체결 확인은 그 회차의 체결가가 이 값에서 바뀌었는지로 본다.
        "rebuy": trigger["rebuy"],
        "fill_price": trigger["fill_price"],
    }


def _can_notify(strategy_view: dict[str, Any], now: datetime) -> bool:
    return is_instrument_trading_time(strategy_view["country"], strategy_view["instrument_kind"], now=now)


def _session_closed(strategy_view: dict[str, Any], now: datetime) -> bool:
    """그 종목의 거래 세션이 끝났는지 — 국장은 시간표 마감 이후, 미장은 거래 시간이 아니면 끝난 것이다."""
    country = strategy_view["country"]
    if country == "kor":
        close = max(end for _, end in INSTRUMENT_TRADING_WINDOWS["kor"][strategy_view["instrument_kind"]])
        return now.time() >= close
    return not _can_notify(strategy_view, now)


def _send_live_message(strategy_view: dict[str, Any], text: str, *, thread_ts: str | None = None) -> str | None:
    from utils.strategy_trade_threads import send_thread_message

    if not _can_notify(strategy_view, datetime.now(KST)):
        return None
    return send_thread_message(text, thread_ts=thread_ts)


def _send_confirms(
    strategy_view: dict[str, Any], pendings: list[dict[str, Any]], sent_log: dict[str, dict[str, Any]]
) -> tuple[int, int]:
    """체결 확인을 신호를 알린 메시지의 스레드로 보낸다 — (보낸 메시지 수, 실패 수). 못 보낸 확인은 기록에 되돌린다.

    그 스레드의 슬랙 주문이 체결됐으면 주문 결과 알림이 이미 체결을 알렸으므로 확인을 보내지 않는다 — 증권사 앱에서
    직접 산 것처럼 슬랙 주문 없이 회차가 채워졌을 때만 확인이 나간다.
    """
    from utils.strategy_trade_orders import thread_has_filled_order
    from utils.strategy_trade_threads import complete_signal_thread, mark_signal_thread_used, signal_is_manual

    groups: dict[str, list[dict[str, Any]]] = {}
    for pending in pendings:
        groups.setdefault(pending["signal_ts"], []).append(pending)
    sent = failed = 0
    for root, group in groups.items():
        if signal_is_manual(root):
            complete_signal_thread(root)
        else:
            mark_signal_thread_used(root)
        if thread_has_filled_order(root):
            continue
        text = "\n".join(_confirm_line(pending, strategy_view["index"], strategy_view["country"]) for pending in group)
        if _send_live_message(strategy_view, text, thread_ts=root):
            sent += 1
        else:
            failed += 1
            sent_log.update({_trigger_key(pending): pending for pending in group})
    return sent, failed


def _entry_withdrawal_bounds(entry: dict[str, Any], strategy: dict[str, Any]) -> tuple[float, float] | None:
    """현재 전략 설정(`trigger_pct`)과 회차 현황 기준으로 (회수 기준가, 신호 기준가)를 실시간 계산한다."""
    gap_pct = float(strategy["config"]["trigger_pct"])
    if entry.get("arbitrary_id") is not None:
        lot = next((item for item in strategy["arbitrary_lots"] if item["id"] == entry["arbitrary_id"]), None)
        if lot is None:
            return None
        return _withdrawal_price(lot["fill_price"], gap_pct, buy=False), lot["sell_limit"]
    if entry["action"] == "buy":
        row = next((r for r in strategy["rounds"] if r["round"] == entry["round"]), None)
        if row is None or row["close_entry"] or row["buy_limit"] is None:
            return None
        prev = next((r for r in strategy["rounds"] if r["round"] == entry["round"] - 1 and r["held"]), None)
        if prev is None:
            return None
        return _withdrawal_price(prev["fill_price"], gap_pct, buy=True), row["buy_limit"]
    row = next((r for r in strategy["rounds"] if r["round"] == entry["round"] and r["held"]), None)
    if row is None or row["sell_limit"] is None:
        return None
    return _withdrawal_price(row["fill_price"], gap_pct, buy=False), row["sell_limit"]


def _price_returned(entries: list[dict[str, Any]], strategy: dict[str, Any]) -> str | None:
    """보낸 신호들이 모두 현재 기준 회수가까지 되돌아왔으면 회수 한 줄의 상세(현재가), 아니면 None."""
    country = strategy["country"]
    current = strategy["index"]["close"]
    if current <= 0:
        return None
    for entry in entries:
        bounds = _entry_withdrawal_bounds(entry, strategy)
        if bounds is None:
            return None
        threshold, target = bounds
        buy = entry["action"] == "buy"
        if buy and not (current >= threshold and current > target):
            return None
        if not buy and not (current <= threshold and current < target):
            return None
    return f"현재가 {format_money(current, country)}"


def _settle_unused_signals(
    strategies: list[dict[str, Any]],
    sent_log: dict[str, dict[str, Any]],
    now: datetime,
    configured: set[str],
    target: tuple[str, str],
) -> tuple[int, int]:
    """미사용 신호를 회수한다 — 사고팔기에서 빠진 전략은 바로, 장중에는 가격이 되돌아오면, 마감 후에는 모두.

    ``configured`` 는 지금 사고팔기 설정에 있는 전략 id 다. 회수는 모두 `withdraw_signal`(표준 한 줄)을 거친다.
    돌려주는 값은 (장 마감 회수 수, 그 밖의 회수 수)다.
    """
    from utils.strategy_trade_threads import (
        mark_signal_thread_used,
        open_signal_threads,
        resume_withdrawal,
        withdraw_signal,
    )

    by_strategy = {strategy["strategy_id"]: strategy for strategy in strategies}
    active_roots = {entry["signal_ts"] for entry in sent_log.values() if entry.get("signal_ts")}
    expired = set()
    withdrawn = set()
    for thread in open_signal_threads(active_roots):
        if not any(
            entry["strategy_id"] != "mix" and (entry["account_id"], entry["ticker"]) == target
            for entry in thread["signal_entries"]
        ):
            continue
        root = thread["thread_ts"]
        if thread.get("withdrawal_text"):
            withdrawn.add(root)
            if not thread.get("withdrawal_done"):
                try:
                    resume_withdrawal(root)
                except ValueError:
                    logger.exception("사고팔기 신호 회수 마무리 실패: %s", root)
            continue
        entries = thread["signal_entries"]
        if any(entry.get("origin") == "manual" for entry in entries):
            continue
        if all(entry["strategy_id"] not in configured for entry in entries):
            try:
                if withdraw_signal(root, "removed", None):
                    withdrawn.add(root)
            except ValueError:
                logger.exception("사고팔기에서 빠진 전략의 신호 회수 실패: %s", root)
            continue
        strategy = by_strategy.get(entries[0]["strategy_id"])
        if strategy is None:
            continue
        after_close = _session_closed(strategy, now)
        detail = None if after_close else _price_returned(entries, strategy)
        if not after_close and (detail is None or not _can_notify(strategy, now)):
            continue
        held = strategy["status"]["held_count"]
        fills = {row["round"]: row["fill_price"] for row in strategy["rounds"] if row["held"]}
        held_quantities = {row["round"]: row["quantity"] for row in strategy["rounds"] if row["held"]}
        held_quantities.update({lot["id"]: lot["quantity"] for lot in strategy["arbitrary_lots"]})
        if any(
            _round_filled(entry, held, fills)
            or (entry.get("arbitrary_id") is not None and entry["arbitrary_id"] not in held_quantities)
            or (
                entry["action"] == "sell"
                and entry.get("quantity") is not None
                and held_quantities.get(entry.get("arbitrary_id") or entry["round"], 0) < entry["quantity"]
            )
            for entry in entries
        ):
            mark_signal_thread_used(root)
            continue
        try:
            if after_close:
                if withdraw_signal(root, "session_closed", None):
                    expired.add(root)
            elif withdraw_signal(root, "price_returned", detail):
                withdrawn.add(root)
        except ValueError:
            logger.exception("사고팔기 신호 회수 실패: %s", root)
    inactive = expired | withdrawn
    for key in [key for key, entry in sent_log.items() if entry.get("signal_ts") in inactive]:
        del sent_log[key]
    return len(expired), len(withdrawn)


def _close_sessions(
    strategies: list[dict[str, Any]], sent_log: dict[str, dict[str, Any]], now: datetime, configured: set[str]
) -> bool:
    """신호의 수명을 날짜가 아니라 거래 세션으로 끊는다 — 기록이 바뀌었으면 True.

    같은 신호는 세션이 이어지는 동안 날짜가 바뀌어도 처음 보낸 스레드 하나로 이어진다. 세션이 닫힌 전략의 기록에는
    `session_closed` 표시만 남기고(닫힌 동안 체결 확인을 보낼 수 없으므로 지우지 않는다), 다음 세션이 열리면
    체결 확인을 먼저 처리한 뒤 `_drop_closed_session_signals` 가 지운다. 사고팔기 설정에서 빠진 전략의 기록은 지운다.
    """
    changed = False
    for key in [
        key
        for key, entry in sent_log.items()
        if entry.get("strategy_id") not in configured and entry.get("origin") != "manual"
    ]:
        del sent_log[key]
        changed = True
    closed = {strategy["strategy_id"] for strategy in strategies if _session_closed(strategy, now)}
    for entry in sent_log.values():
        if entry.get("origin") != "manual" and entry.get("strategy_id") in closed and not entry.get("session_closed"):
            entry["session_closed"] = True
            changed = True
    return changed


def _drop_closed_session_signals(sent_log: dict[str, dict[str, Any]], open_ids: set[str]) -> bool:
    """지난 세션에 보낸 신호 기록을 지운다(다시 열린 전략만) — 이번 세션의 같은 회차 신호는 새 스레드로 나간다."""
    stale = [
        key
        for key, entry in sent_log.items()
        if entry.get("origin") != "manual" and entry.get("session_closed") and entry.get("strategy_id") in open_ids
    ]
    for key in stale:
        del sent_log[key]
    return bool(stale)


def notify_strategy_trade() -> dict[str, Any]:
    """증권사 원장과 운용 현황을 먼저 준비한 뒤 Slack 신호를 처리한다."""
    snapshot_at = datetime.now(KST).isoformat()
    for key, message in refresh_all_ledgers().items():
        logger.error("사고팔기 회차 원장 실패(%s) — 이 종목은 건너뜁니다: %s", key, message)

    settings_at_read = load_settings()
    view = load_views_where(lambda entry: True, fresh=False)
    from utils.strategy_trade_threads import open_signal_threads

    targets = {(s["account_id"], s["ticker"]) for s in view["strategies"]}
    entries = [*_load_sent_log().values()]
    entries.extend(
        entry
        for thread in open_signal_threads(set())
        for entry in thread["signal_entries"]
        if entry["strategy_id"] != "mix"
    )
    targets.update((entry["account_id"], entry["ticker"]) for entry in entries)
    results, busy = [], 0
    for account_id, ticker in sorted(targets):
        try:
            with batch_signal_lock(account_id, ticker):
                result = _notify_prepared_view(view, settings_at_read, snapshot_at, (account_id, ticker))
                results.append(result)
        except SignalBusyError:
            busy += 1
            logger.info("사고팔기 알림 연기 — 예약 중인 종목: %s · %s", account_id, ticker)
    reasons = list(dict.fromkeys(result["reason"] for result in results))
    if busy:
        reasons.append(f"예약 중인 종목 {busy}개는 다음 배치에서 재확인")
    if not reasons:
        reasons.append("슬랙이 켜진 계좌를 고른 사고팔기 전략이 없습니다.")
    for sid, error in view["errors"].items():
        logger.error("사고팔기 운용 현황 실패(%s): %s", sid, error)
    return {
        "sent": any(result["sent"] for result in results),
        "reason": " · ".join(reasons),
        "triggers": [trigger for result in results for trigger in result["triggers"]],
    }


def _notify_prepared_view(
    view: dict[str, Any], settings_at_read: dict[str, Any], snapshot_at: str, target: tuple[str, str]
) -> dict[str, Any]:
    """예약한 계좌·종목의 최신 기록만 읽어 회수·체결 확인·발송을 처리한다."""
    from utils.strategy_trade_orders import active_order_keys, finished_since_keys
    from utils.strategy_trade_threads import mark_signal_thread_used, record_signal_thread

    now = datetime.now(KST)
    sent_log = _SentLog(
        {key: entry for key, entry in _load_sent_log().items() if (entry["account_id"], entry["ticker"]) == target}
    )
    if retire_expired_signals(sent_log, target):
        _save_sent_log(sent_log)
    current_settings = load_settings()
    enabled = {account_id for account_id, entry in current_settings["accounts"].items() if entry["slack_enabled"]}
    configured = set(current_settings["strategies"])
    # 준비하는 동안 바뀐 전략은 이전 현황을 사용하지 않고 다음 배치에서 다시 계산한다.
    stable_ids = {
        sid
        for sid, entry in settings_at_read["strategies"].items()
        if entry == current_settings["strategies"].get(sid)
        and settings_at_read["accounts"].get(entry["account_id"])
        == current_settings["accounts"].get(entry["account_id"])
    }
    view = {
        "strategies": [
            s for s in view["strategies"] if s["strategy_id"] in stable_ids and (s["account_id"], s["ticker"]) == target
        ],
        "errors": {sid: message for sid, message in view["errors"].items() if sid in stable_ids},
    }
    expired, withdrawn = _settle_unused_signals(view["strategies"], sent_log, now, configured, target)
    if _close_sessions(view["strategies"], sent_log, now, configured) or expired or withdrawn:
        _save_sent_log(sent_log)
    if expired or withdrawn:
        logger.info("사고팔기 신호 회수 — 장중·제외 %s건, 장 마감 미사용 %s건", withdrawn, expired)
    # 주문이 있던(회수 대상이 아닌) 신호 스레드의 남은 버튼은 세션이 닫히면 걷는다.
    from utils.slack_mock_orders import close_session_threads

    closed_messages = close_session_threads([s for s in view["strategies"] if _session_closed(s, now)])
    if closed_messages:
        logger.info("사고팔기 장 마감 — 남은 주문 버튼 %s건 정리", closed_messages)
    if not view["strategies"] and not view["errors"]:
        return {"sent": False, "reason": "슬랙이 켜진 계좌를 고른 사고팔기 전략이 없습니다.", "triggers": []}
    for sid, message in view["errors"].items():
        logger.error("사고팔기 운용 현황 실패(%s) — 이 전략은 건너뜁니다: %s", sid, message)

    open_strategies = [strategy for strategy in view["strategies"] if _can_notify(strategy, now)]
    if not open_strategies:
        reason = (
            "거래 가능 시간인 사고팔기 종목이 없습니다." if view["strategies"] else "운용 현황을 확인할 수 없습니다."
        )
        return {"sent": False, "reason": reason, "triggers": []}
    triggers = [
        trigger
        for strategy in open_strategies
        if strategy["account_id"] in enabled
        for trigger in collect_triggers(strategy)
    ]
    if triggers:
        # 주문이 진행 중이면 신호를 생략한다. 매도 후 재매수는 양쪽을 확인한다.
        # 원장을 계산한 뒤(`snapshot_at` 이후) 끝난 주문도 생략한다 — 원장이 그 체결을 아직 몰라 같은 회차 트리거가
        # 다시 나오는데, 신호 기록은 주문 기록으로 체결 판정돼 지워지므로 거르지 않으면 같은 신호가 또 나간다.
        strategy_ids = [strategy["strategy_id"] for strategy in open_strategies]
        in_flight = active_order_keys(strategy_ids) | finished_since_keys(strategy_ids, snapshot_at)
        triggers = [
            trigger
            for trigger in triggers
            if (trigger["strategy_id"], trigger.get("arbitrary_id") or trigger["round"], trigger["action"])
            not in in_flight
            and not (trigger["rebuy"] and (trigger["strategy_id"], trigger["round"], "buy") in in_flight)
        ]

    # 이번 세션에 처음 보는 신호만 전 회차 현황이 담긴 본문으로 보낸다 — 이미 보낸 신호는 날짜가 바뀌어도
    # 다시 알리지 않고 처음 스레드를 그대로 쓴다. 메시지는 **전략(종목)마다 따로**, 보낼 것이 있는 전략에만 보낸다.
    # ① 요청해 둔 주문이 체결됐는지 먼저 본다. 체결됐으면 확인 한 줄을 보내고 기록을 지운다 —
    #    지워야 다음에 같은 회차 신호가 나면 다시 전체 메시지로 나간다.
    #    이 검사는 트리거 유무와 무관하게 돌아야 한다: 주문을 넣는 순간 그 회차가 보유로 바뀌어
    #    바로 그 트리거가 사라지므로, 신호가 있을 때만 검사하면 확인은 영영 나가지 못한다.
    #    지금 전략 목록에 없는 전략의 기록은 체결 여부를 알 수 없어 건드리지 않는다.
    held_counts = _held_counts({"strategies": open_strategies})
    held_fills = _held_fills({"strategies": open_strategies})
    events = {s["strategy_id"]: s["fill_events"] for s in open_strategies}
    arbitrary_ids = {
        strategy["strategy_id"]: {lot["id"] for lot in strategy["arbitrary_lots"]} for strategy in open_strategies
    }
    confirmed = pop_filled(
        sent_log,
        # 준비 이후 수동으로 보낸 신호는 이전 원장으로 체결 판정하지 않는다.
        lambda pending: str(pending.get("strategy_id")) in held_counts
        and pending.get("at", "") <= snapshot_at
        and (
            pending["arbitrary_id"] not in arbitrary_ids[pending["strategy_id"]]
            if pending.get("arbitrary_id")
            else _entry_filled(
                pending,
                held_counts[pending["strategy_id"]],
                held_fills[pending["strategy_id"]],
                events[pending["strategy_id"]],
            )
        ),
    )
    dropped = _drop_closed_session_signals(sent_log, {strategy["strategy_id"] for strategy in open_strategies})

    sent_count = failed_count = order_failed = 0
    fresh_all: list[dict[str, Any]] = []
    stamp = now.isoformat()
    for strategy_view in open_strategies:
        sid = strategy_view["strategy_id"]
        # ② 남은 신호 — 기록에 없는(이번 세션에 처음 보는) 신호만 보낸다.
        fresh = [t for t in triggers if t["strategy_id"] == sid and _matching_signal(sent_log, t) is None]

        own_confirmed = [pending for pending in confirmed if str(pending.get("strategy_id")) == sid]
        # 신호 메시지가 기록된 확인은 그 스레드로, 없는 확인(옛 기록)은 새 신호 본문과 함께 채널로 보낸다.
        threaded = [pending for pending in own_confirmed if pending.get("signal_ts")]
        own_confirmed = [pending for pending in own_confirmed if not pending.get("signal_ts")]
        sent, failed = _send_confirms(strategy_view, threaded, sent_log)
        sent_count += sent
        failed_count += failed
        lines = [_confirm_line(pending, strategy_view["index"], strategy_view["country"]) for pending in own_confirmed]
        if fresh:
            lines.append(build_message(strategy_view, fresh))
        if lines:
            for trigger in fresh:
                sent_log[_trigger_key(trigger)] = _log_entry(trigger, held_counts, stamp, "")
            if fresh:
                _save_sent_log(sent_log)
            signal_ts = _send_live_message(strategy_view, "\n".join(lines))
            if not signal_ts:
                failed_count += 1
                # 못 보낸 확인은 기록에 되돌려 다음 배치에서 다시 보낸다.
                sent_log.update({_trigger_key(pending): pending for pending in own_confirmed})
                continue
            sent_count += 1
            entries = [_log_entry(trigger, held_counts, stamp, signal_ts) for trigger in fresh]
            if entries:
                record_signal_thread(signal_ts, entries)
                if own_confirmed:
                    mark_signal_thread_used(signal_ts)
            for entry in entries:
                sent_log[_trigger_key(entry)] = entry
            _save_sent_log(sent_log)
            order_failed += _send_order_messages(strategy_view, fresh, signal_ts)
            fresh_all.extend(fresh)
            for entry in entries:
                sent_log[_trigger_key(entry)] = entry
        _keep_latest_buy(sent_log, sid)

    if not sent_count and not failed_count:
        if not triggers and not confirmed and not dropped:
            # 확인할 체결도 신호도 없다 — 2분마다 도는 배치라 여기서 DB 를 건드리면 무의미한 쓰기만 쌓인다.
            return {"sent": False, "reason": "매수·매도 신호가 없습니다.", "triggers": []}
        _save_sent_log(sent_log)
        return {"sent": False, "reason": "이번 세션에 이미 발송한 신호입니다.", "triggers": triggers}

    _save_sent_log(sent_log)

    parts = [f"메시지 {sent_count}건 발송"]
    if confirmed:
        parts.append(f"체결확인 {len(confirmed)}건")
    if fresh_all:
        parts.append(f"신호 {len(fresh_all)}건")
    if failed_count:
        parts.append(f"슬랙 전송 실패 {failed_count}건")
    if order_failed:
        parts.append(f"주문 메시지 실패 {order_failed}건")
    return {"sent": sent_count > 0, "reason": " · ".join(parts), "triggers": fresh_all or triggers}


def _manual_signal(payload: dict[str, Any], *, fresh: bool) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """공통 회차 판정·트리거 생성으로 수동 신호와 서버 확인 내용을 만든다."""
    import json
    from copy import deepcopy
    from hashlib import sha256

    from utils.strategy_trade_orders import (
        _order_strategy,
        _require_no_manual_order,
        _require_round_state,
        require_order_session,
    )

    account_id, sid = payload.get("account_id"), payload.get("strategy_id")
    side, round_no = payload.get("side"), payload.get("round")
    if (
        not isinstance(account_id, str)
        or not isinstance(sid, str)
        or side not in ("buy", "sell")
        or type(round_no) is not int
    ):
        raise ValueError("account_id·strategy_id·매매 방향·정수 회차가 필요합니다.")
    setting = load_settings()["strategies"].get(sid)
    if setting is None or setting["account_id"] != account_id:
        raise ValueError("계좌의 전략을 찾을 수 없습니다.")
    if setting.get("mode") == "disabled":
        raise ValueError("비활성화된 전략에서는 슬랙 신호를 보낼 수 없습니다.")
    strategy = _order_strategy(account_id, setting["pool"], sid, fresh=fresh)
    require_order_session(strategy, now=datetime.now(KST))
    row = _require_round_state(strategy, side, round_no)
    _require_no_manual_order(strategy, side)
    if row["quantity"] <= 0 or (side == "buy" and (row["cash_short"] or row["budget_short"])):
        raise ValueError("주문 가능한 수량 또는 예수금이 부족합니다.")
    forced = deepcopy(strategy)
    forced["arbitrary_lots"] = []
    for item in forced["rounds"]:
        item["buy_reached"] = item["round"] == round_no and side == "buy"
        item["sell_reached"] = item["round"] == round_no and side == "sell"
        item["rebuy_after_sell"] = False
        if item["round"] == round_no and side == "buy" and item["buy_limit"] is None:
            item["close_entry"] = True
    trigger = collect_triggers(forced)[0]
    trigger.update(origin="manual", account_id=account_id)
    price = trigger["limit_price"] if trigger["limit_price"] is not None else trigger["close"]
    if price is None or price <= 0:
        raise ValueError("신호 가격을 확인할 수 없습니다.")
    preview = {
        "account_id": account_id,
        "account_name": strategy["account_name"],
        "strategy_id": sid,
        "ticker": strategy["ticker"],
        "name": row["name"],
        "round": round_no,
        "side": side,
        "price": price,
        "quantity": row["quantity"],
    }
    preview["token"] = sha256(json.dumps([preview, trigger["fill_cursor"]], sort_keys=True).encode()).hexdigest()
    return strategy, trigger, preview


def preview_manual_signal(payload: dict[str, Any]) -> dict[str, Any]:
    return _manual_signal(payload, fresh=False)[2]


def send_manual_signal(payload: dict[str, Any]) -> dict[str, Any]:
    setting = load_settings()["strategies"].get(payload["strategy_id"])
    if setting is None:
        raise ValueError("사고팔기 전략을 찾을 수 없습니다.")
    with signal_lock(setting["account_id"], setting["ticker"]):
        return _send_manual_signal(payload)


def _send_manual_signal(payload: dict[str, Any]) -> dict[str, Any]:
    """서버 선점 뒤 Slack 메시지만 발송한다. 증권사 주문은 Slack 확인 경로만 사용한다."""
    from utils.slack_mock_orders import send_signal_order_messages
    from utils.strategy_trade_orders import find_duplicate_order, require_order_session
    from utils.strategy_trade_threads import complete_signal_thread, record_signal_thread, send_thread_message

    strategy, trigger, preview = _manual_signal(payload, fresh=True)
    if payload.get("token") != preview["token"]:
        raise ValueError("회차 가격·수량이 변경됐습니다. 확인창을 다시 열어 확인하세요.")
    key = _trigger_key(trigger)
    sent_log = _load_sent_log()
    if retire_expired_signals(sent_log, (strategy["account_id"], strategy["ticker"])):
        _save_sent_log(sent_log)
    previous = _matching_signal(sent_log, trigger)
    if previous is not None:
        held = strategy["status"]["held_count"]
        fills = {r["round"]: r["fill_price"] for r in strategy["rounds"] if r["held"]}
        if not _entry_filled(previous, held, fills, strategy["fill_events"]):
            return {"sent": False, "duplicate": True}
        if previous.get("signal_ts"):
            complete_signal_thread(previous["signal_ts"])
        sent_log.pop(_trigger_key(previous), None)
    if find_duplicate_order(strategy["account_id"], strategy["strategy_id"], trigger["action"]) is not None:
        raise ValueError("같은 방향의 주문이 진행 중입니다.")
    require_order_session(strategy, now=datetime.now(KST))
    held_counts = {strategy["strategy_id"]: strategy["status"]["held_count"]}
    entry = _log_entry(trigger, held_counts, datetime.now(KST).isoformat(), "")
    sent_log[key] = entry
    _save_sent_log(sent_log)
    word = "매수" if trigger["action"] == "buy" else "매도"
    text = build_message(strategy, [trigger]) + (
        f"\n*수동 슬랙 발송* · {trigger['round']}호 {word} · "
        f"{_level(preview['price'], strategy['country'])} · {preview['quantity']:,}주 · 직접 회수 전까지 유지됩니다."
    )
    signal_ts = send_thread_message(text)
    if not signal_ts:
        raise ValueError(
            "Slack 발송 결과를 확인하지 못했습니다. 중복 방지를 위해 재발송을 막았습니다. 발송 기록을 확인하세요."
        )
    entry["signal_ts"] = signal_ts
    record_signal_thread(signal_ts, [entry])
    _save_sent_log(sent_log)
    send_signal_order_messages(strategy, [trigger], signal_ts)
    return {"sent": True, "duplicate": False}
